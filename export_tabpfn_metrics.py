"""
=============================================================================
Export TabPFN prediction metrics
- Sequential split: Train(60%) / Valid(20%) / Test(20%)
- Metrics: MAE, MAPE(%), MSE, RMSE, R2
- Export: TabPFN_Metrics.xlsx
=============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabpfn import TabPFNRegressor
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"

# ============================================================================
# Load data
# ============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
X = df.iloc[:, :-1].values.astype(np.float64)
y = df.iloc[:, -1].values.astype(np.float64)
n = len(y)

# Sequential split: 80% train / 10% valid / 10% test (matching Fig_d)
t_end = int(n * 0.8)
v_end = int(n * 0.9)

X_train, y_train = X[:t_end], y[:t_end]
X_valid, y_valid = X[t_end:v_end], y[t_end:v_end]
X_test,  y_test  = X[v_end:],    y[v_end:]

print("Train: {} (ring 1-{})".format(len(y_train), t_end))
print("Valid: {} (ring {}-{})".format(len(y_valid), t_end+1, v_end))
print("Test:  {} (ring {}-{})".format(len(y_test),  v_end+1, n))

# Standardize
scaler_X = StandardScaler(); scaler_y = StandardScaler()
X_train_nor = scaler_X.fit_transform(X_train)
X_valid_nor = scaler_X.transform(X_valid)
X_test_nor  = scaler_X.transform(X_test)
y_train_nor = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()

# ============================================================================
# Train TabPFN
# ============================================================================
model = TabPFNRegressor(model_path=MODEL_PATH, n_estimators=3,
                        device="cpu", ignore_pretraining_limits=True)
model.fit(X_train_nor, y_train_nor)

# Predict
def pred(X_nor):
    p_nor = model.predict(X_nor)
    return scaler_y.inverse_transform(p_nor.reshape(-1,1)).ravel()

y_train_pred = pred(X_train_nor)
y_valid_pred = pred(X_valid_nor)
y_test_pred  = pred(X_test_nor)

# ============================================================================
# Metrics
# ============================================================================
def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    return mae, mape, mse, rmse, r2

datasets = {
    'Train': (y_train, y_train_pred),
    'Valid': (y_valid, y_valid_pred),
    'Test':  (y_test,  y_test_pred),
    'Valid+Test': (np.concatenate([y_valid, y_test]),
                   np.concatenate([y_valid_pred, y_test_pred])),
}

rows = []
for name, (yt, yp) in datasets.items():
    mae, mape, mse, rmse, r2 = metrics(yt, yp)
    rows.append({
        'Dataset': name, 'Samples': len(yt),
        'MAE': round(mae, 4), 'MAPE(%)': round(mape, 2),
        'MSE': round(mse, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4),
    })
    print("{:<12} {:>3} samples  MAE={:.4f}  MAPE={:.2f}%  MSE={:.4f}  RMSE={:.4f}  R2={:.4f}".format(
        name, len(yt), mae, mape, mse, rmse, r2))

df_out = pd.DataFrame(rows)
df_out.to_excel("TabPFN_Metrics.xlsx", index=False)
print("\nExported: TabPFN_Metrics.xlsx")
