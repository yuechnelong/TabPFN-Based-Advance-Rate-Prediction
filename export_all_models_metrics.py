"""
=============================================================================
12-Model Comparison Metrics (80/10/10 Sequential Split)
Export: All_Models_Metrics.xlsx
=============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from hpelm import ELM
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import torch
from tabpfn import TabPFNRegressor
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"

# ============================================================================
# Data
# ============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
X = df.iloc[:, :-1].values.astype(np.float64)
y = df.iloc[:, -1].values.astype(np.float64)
n = len(y)

# 80/10/10 sequential
t_end = int(n * 0.8)
v_end = int(n * 0.9)

X_train, y_train = X[:t_end], y[:t_end]
X_valid, y_valid = X[t_end:v_end], y[t_end:v_end]
X_test,  y_test  = X[v_end:],    y[v_end:]

# Standardize
scaler_X = StandardScaler(); scaler_y = StandardScaler()
X_train_nor = scaler_X.fit_transform(X_train)
X_valid_nor = scaler_X.transform(X_valid)
X_test_nor  = scaler_X.transform(X_test)
y_train_nor = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

print("Train:{}, Valid:{}, Test:{}".format(len(y_train), len(y_valid), len(y_test)))

def denorm(yp_nor):
    return scaler_y.inverse_transform(yp_nor.reshape(-1,1)).ravel()

def calc_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    return mae, mape, mse, rmse, r2

# ============================================================================
# Models
# ============================================================================
models = {}

# TabPFN
print("Training TabPFN...")
m = TabPFNRegressor(model_path=MODEL_PATH, n_estimators=3, device="cpu", ignore_pretraining_limits=True)
m.fit(X_train_nor, y_train_nor)
models['TabPFN'] = m

# XGBoost
print("Training XGBoost...")
m = XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)
m.fit(X_train_nor, y_train_nor)
models['XGBoost'] = m

# CatBoost
print("Training CatBoost...")
m = CatBoostRegressor(iterations=150, learning_rate=0.1, depth=5, random_seed=42, verbose=0)
m.fit(X_train_nor, y_train_nor)
models['CatBoost'] = m

# LightGBM
print("Training LightGBM...")
m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
m.fit(X_train_nor, y_train_nor)
models['LightGBM'] = m

# SVR
print("Training SVR...")
m = SVR(kernel='rbf', C=1.0, epsilon=0.1)
m.fit(X_train_nor, y_train_nor)
models['SVR'] = m

# RF
print("Training RF...")
m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
m.fit(X_train_nor, y_train_nor)
models['RF'] = m

# MLP
print("Training MLP...")
m = MLPRegressor(hidden_layer_sizes=(64,64), activation='relu', solver='adam', max_iter=500, random_state=42)
m.fit(X_train_nor, y_train_nor)
models['MLP'] = m

# DT
print("Training DT...")
m = DecisionTreeRegressor(max_depth=5, random_state=42)
m.fit(X_train_nor, y_train_nor)
models['DT'] = m

# ELM
print("Training ELM...")
m = ELM(X_train_nor.shape[1], 1)
m.add_neurons(50, 'sigm')
m.train(X_train_nor, y_train_nor, 'r')
models['ELM'] = m

# BayesianRidge
print("Training BayesianRidge...")
m = BayesianRidge()
m.fit(X_train_nor, y_train_nor)
models['Bay'] = m

# GBR
print("Training GBR...")
m = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
m.fit(X_train_nor, y_train_nor)
models['GBR'] = m

# TABM placeholder (if not available)
try:
    from TabM_model1 import TabMConfig, TabMModel1
    from typing import NamedTuple
    class RegressionLabelStats(NamedTuple):
        mean: float; std: float
    data_numpy = {
        'train': {'x_num': X_train_nor.astype(np.float32), 'y': y_train_nor.astype(np.float32)},
        'val': {'x_num': X_valid_nor.astype(np.float32), 'y': scaler_y.transform(y_valid.reshape(-1,1)).ravel().astype(np.float32)},
        'test': {'x_num': X_test_nor.astype(np.float32), 'y': scaler_y.transform(y_test.reshape(-1,1)).ravel().astype(np.float32)},
    }
    rls = RegressionLabelStats(float(scaler_y.mean_[0]), float(scaler_y.scale_[0]))
    config = TabMConfig(task_type='regression', n_num_features=X_train_nor.shape[1],
                        cat_cardinalities=None, n_classes=None,
                        num_embedding_type='piecewise',
                        regression_label_stats=rls,
                        lr=2e-3, weight_decay=3e-4,
                        batch_size=256, num_epochs=50,
                        gradient_clipping_norm=1.0, enable_amp=False, compile_model=False)
    print("Training TABM...")
    m = TabMModel1(config, data_numpy)
    m.train()
    models['TABM'] = m
except Exception as e:
    print("TABM not available: {}".format(e))

# ============================================================================
# Evaluate
# ============================================================================
model_order = ['LightGBM','XGBoost','CatBoost','TabPFN','TABM','SVR','RF','MLP','DT','ELM','Bay','GBR']

# Filter to available
available = [m for m in model_order if m in models]
print("\nAvailable models: {}".format(available))

rows = []
for name in available:
    model = models[name]

    if name == 'TABM':
        y_tp = denorm(model.predict('train').flatten())
        y_vp = denorm(model.predict('val').flatten())
        y_tep = denorm(model.predict('test').flatten())
    elif name == 'ELM':
        y_tp = denorm(model.predict(X_train_nor))
        y_vp = denorm(model.predict(X_valid_nor))
        y_tep = denorm(model.predict(X_test_nor))
    else:
        y_tp = denorm(model.predict(X_train_nor))
        y_vp = denorm(model.predict(X_valid_nor))
        y_tep = denorm(model.predict(X_test_nor))

    y_vt_true = np.concatenate([y_valid, y_test])
    y_vt_pred = np.concatenate([y_vp, y_tep])

    # Valid+Test only
    mae, mape, mse, rmse, r2 = calc_metrics(y_vt_true, y_vt_pred)
    rows.append({
        'Model': name,
        'MAE': round(mae, 4), 'MAPE(%)': round(mape, 2),
        'MSE': round(mse, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4),
    })
    print("{:<12}  MAE={:.4f}  MAPE={:.2f}%  MSE={:.4f}  RMSE={:.4f}  R2={:.4f}".format(
        name, mae, mape, mse, rmse, r2))

df_out = pd.DataFrame(rows)
# Sort by R2 desc
df_out = df_out.sort_values('R2', ascending=False).reset_index(drop=True)
df_out.index = df_out.index + 1
df_out.index.name = 'Rank'
df_out.to_excel("All_Models_Metrics.xlsx")
print("\nExported: All_Models_Metrics.xlsx")
