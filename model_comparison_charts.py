"""
=============================================================================
多算法对比 + 雷达图 + 泰勒图
- 顺序划分 80/20
- 所有模型统一标准化+评价
- 雷达图: R², RMSE, MAE, MAPE 多指标对比
- 泰勒图: 标准差比 + 相关系数 + RMSE
- 论文级高清输出 (1200 DPI, Times New Roman)
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from tabpfn import TabPFNRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 配置
# ============================================================================
MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
print(f"设备: {DEVICE}")

# ============================================================================
# 1. 加载数据 + 顺序划分
# ============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
X_all = df.iloc[:, :-1].values.astype(np.float64)
y_all = df.iloc[:, -1].values.astype(np.float64)
n_samples = len(y_all)

# 顺序划分 80/20
split_idx = int(n_samples * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_nor = scaler_X.fit_transform(X_train)
X_test_nor = scaler_X.transform(X_test)
y_train_nor = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

print(f"训练: {len(y_train)} 样本, 测试: {len(y_test)} 样本")

# ============================================================================
# 2. 定义模型
# ============================================================================
def build_models():
    return {
        'TabPFN': TabPFNRegressor(
            model_path=MODEL_PATH, n_estimators=3,
            device=DEVICE, ignore_pretraining_limits=True),

        'XGBoost': XGBRegressor(
            n_estimators=150, learning_rate=0.1, max_depth=5,
            random_state=SEED, verbosity=0),

        'CatBoost': CatBoostRegressor(
            iterations=150, learning_rate=0.1, depth=5,
            random_seed=SEED, verbose=0),

        'LightGBM': lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=SEED, verbose=-1),

        'RF': RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=SEED),

        'GBR': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=SEED),

        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),

        'MLP': MLPRegressor(
            hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
            max_iter=500, random_state=SEED),

        'DT': DecisionTreeRegressor(max_depth=5, random_state=SEED),

        'BayesianRidge': BayesianRidge(),
    }

# ============================================================================
# 3. 训练所有模型 + 评估
# ============================================================================
print(f"\n{'='*60}")
print("训练所有模型...")
print(f"{'='*60}")

models_dict = build_models()
results = {}

for name, model in models_dict.items():
    print(f"  训练 {name}...", end=" ")
    try:
        if name == 'LightGBM':
            model.fit(X_train_nor, y_train_nor)
        else:
            model.fit(X_train_nor, y_train_nor)

        y_pred_nor = model.predict(X_test_nor)
        if name in ['TabPFN']:
            y_pred = scaler_y.inverse_transform(y_pred_nor.reshape(-1, 1)).ravel()
        else:
            y_pred = scaler_y.inverse_transform(y_pred_nor.reshape(-1, 1)).ravel()

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        # 相关系数和标准差
        corr = np.corrcoef(y_test, y_pred)[0, 1]
        std_ratio = np.std(y_pred) / np.std(y_test)

        results[name] = {
            'R²': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape,
            'Corr': corr, 'Std_Ratio': std_ratio,
            'y_pred': y_pred,
        }
        print(f"R²={r2:.4f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    except Exception as e:
        print(f"FAILED: {e}")

# ============================================================================
# 4. 结果汇总表
# ============================================================================
print(f"\n{'='*60}")
print("多模型对比结果")
print(f"{'='*60}")
print(f"{'模型':<14} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE(%)':>10} {'Corr':>8} {'Std_Ratio':>10}")
print("-" * 72)

for name in sorted(results.keys(), key=lambda x: results[x]['R²'], reverse=True):
    r = results[name]
    print(f"{name:<14} {r['R²']:>8.4f} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} "
          f"{r['MAPE(%)']:>9.2f} {r['Corr']:>8.4f} {r['Std_Ratio']:>10.4f}")

# ============================================================================
# 5. 全局绘图设置
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 1200,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.1,
})

# 模型配色（10个模型）
MODEL_COLORS = {
    'TabPFN': '#e05263',
    'XGBoost': '#3b5b92',
    'CatBoost': '#36a168',
    'LightGBM': '#e0a040',
    'RF': '#7b4ea3',
    'GBR': '#c4622d',
    'SVR': '#4daf4a',
    'MLP': '#984ea3',
    'DT': '#a65628',
    'BayesianRidge': '#f781bf',
}

MODEL_MARKERS = {
    'TabPFN': 'D', 'XGBoost': 's', 'CatBoost': '^', 'LightGBM': 'v',
    'RF': 'o', 'GBR': 'P', 'SVR': 'h', 'MLP': 'X', 'DT': '*', 'BayesianRidge': 'p',
}

def save_fig(fig, name):
    fig.savefig(f"{name}.png", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    fig.savefig(f"{name}.pdf", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"  ✓ {name}.png / .pdf")

# ============================================================================
# 6. 雷达图 (Radar Chart) — 4维指标对比
# ============================================================================
print("\n绘制雷达图...")

# 选取指标（归一化到 [0,1]，R²越高越好，其他越低越好）
metrics_radar = ['R²', 'RMSE', 'MAE', 'MAPE(%)']
directions = ['max', 'min', 'min', 'min']  # 优化方向

# 归一化函数
def normalize_for_radar(vals, direction='max'):
    arr = np.array(vals)
    if direction == 'max':
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        return (arr.max() - arr) / (arr.max() - arr.min() + 1e-8)

# 准备雷达数据
model_names_sorted = sorted(results.keys(), key=lambda x: results[x]['R²'], reverse=True)
n_metrics = len(metrics_radar)
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 创建极坐标
fig_radar, ax_radar = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for name in model_names_sorted:
    values = [results[name][m] for m in metrics_radar]
    normed = [normalize_for_radar([results[n][m] for n in model_names_sorted],
                                  directions[i])[model_names_sorted.index(name)]
              for i, m in enumerate(metrics_radar)]
    normed += normed[:1]

    color = MODEL_COLORS.get(name, '#888888')
    ax_radar.fill(angles, normed, alpha=0.06, color=color)
    ax_radar.plot(angles, normed, 'o-', color=color, linewidth=2.0, markersize=6,
                  label=f'{name} (R²={results[name]["R²"]:.3f})', zorder=3)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(metrics_radar, fontsize=14, fontweight='bold')
ax_radar.set_ylim(0, 1.1)
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='#888888')
ax_radar.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12), fontsize=10,
                framealpha=0.85, edgecolor='#cccccc')
ax_radar.grid(True, alpha=0.3, linestyle='--')
ax_radar.spines['polar'].set_visible(False)

# 标注：箭头方向更优
ax_radar.annotate('Better →', xy=(angles[0], 1.05), fontsize=9, ha='center', color='#555555')
for i, (ang, d) in enumerate(zip(angles[:-1], directions)):
    label = '↑ Higher' if d == 'max' else '↓ Lower'
    ax_radar.annotate(label, xy=(ang, 1.08), fontsize=8, ha='center', color='#999999', fontstyle='italic')

save_fig(fig_radar, "Fig_Radar_Model_Comparison")
plt.close(fig_radar)


# ============================================================================
# 7. 泰勒图 (Taylor Diagram)
# ============================================================================
print("绘制泰勒图...")

fig_taylor = plt.figure(figsize=(9.5, 8))
ax_taylor = fig_taylor.add_subplot(111, polar=True)

# 参考点（观测值）
ref_std = 1.0
max_std = max(max(results[n]['Std_Ratio'] for n in model_names_sorted), 1.3)

# 绘制标准差弧
std_grid = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
std_grid = std_grid[std_grid <= max_std * 1.05]

# 相关系数弧
corr_angles = np.arccos(np.linspace(0, 1, 10))
corr_labels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

# RMSE 同心半圆
rmse_max = max(results[n]['RMSE'] for n in model_names_sorted) * 1.1
for rms in np.arange(0.5, rmse_max + 0.5, 0.5):
    circle_angles = np.linspace(0, np.pi / 2, 100)
    x_rms = rms * np.cos(circle_angles)
    y_rms = rms * np.sin(circle_angles)
    ax_taylor.plot(circle_angles, np.full_like(circle_angles, rms) / ref_std,
                   color='gray', alpha=0.3, linewidth=0.6, linestyle='--')

# 绘制标准差弧线
for s in std_grid:
    arc_angles = np.linspace(0, np.pi / 2, 100)
    ax_taylor.plot(arc_angles, np.full_like(arc_angles, s),
                   color='#cccccc', linewidth=0.8, alpha=0.7)
    ax_taylor.text(np.pi / 2 + 0.02, s, f'{s:.2f}', fontsize=8,
                   va='center', color='#888888')

# 相关系数标签
for ca in np.arccos(np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])):
    ax_taylor.plot([ca, ca], [0, max_std], color='#cccccc', linewidth=0.5, alpha=0.5)
    ax_taylor.text(ca, max_std + 0.04, f'{np.cos(ca):.2f}', fontsize=9,
                   ha='center', va='bottom', color='#888888')

# 绘制 RMSE 同心圆标注
for rms in [0.5, 1.0, 1.5, 2.0]:
    if rms <= rmse_max + 0.3:
        ang = np.arctan2(0.4, rms)
        ax_taylor.annotate(f'{rms:.1f}', xy=(ang, rms / ref_std),
                           fontsize=8, color='#aaaaaa', rotation=np.degrees(ang) - 90)

# 绘制模型点
for name in model_names_sorted:
    r = results[name]
    angle_val = np.arccos(r['Corr'])
    radius_val = r['Std_Ratio']

    color = MODEL_COLORS.get(name, '#888888')
    marker = MODEL_MARKERS.get(name, 'o')

    ax_taylor.plot(angle_val, radius_val, marker=marker, color=color,
                   markersize=11, markeredgewidth=0.8, markeredgecolor='white',
                   label=f'{name}', zorder=4)

# 参考点 (REF)
ax_taylor.plot(0, ref_std, 'k*', markersize=18, zorder=5, label='Reference (Observed)')

# 标签：相关系数
ax_taylor.text(np.pi / 4, max_std * 1.15, 'Correlation\nCoefficient',
               fontsize=11, ha='center', fontstyle='italic', color='#666666')

# 设置
ax_taylor.set_thetamin(0)
ax_taylor.set_thetamax(90)
ax_taylor.set_ylim(0, max_std * 1.1)
ax_taylor.set_yticks([])
ax_taylor.set_xticks([])
ax_taylor.grid(False)

# 手动添加标准差标注
for s in std_grid:
    ax_taylor.text(np.deg2rad(89), s, f'{s:.1f}', fontsize=8,
                   va='center', ha='left', color='#888888')

# Legend
ax_taylor.legend(loc='upper right', bbox_to_anchor=(1.42, 1.05), fontsize=10,
                 framealpha=0.85, edgecolor='#cccccc', ncol=1)

save_fig(fig_taylor, "Fig_Taylor_Diagram")
plt.close(fig_taylor)

# ============================================================================
# 8. 简化的柱状图对比（R² + RMSE 双轴）
# ============================================================================
print("绘制R²/RMSE对比柱状图...")
fig_bar, ax1 = plt.subplots(figsize=(12, 5.5))

model_order = sorted(results.keys(), key=lambda x: results[x]['R²'], reverse=True)
x_pos = np.arange(len(model_order))
width = 0.35

r2_vals = [results[n]['R²'] for n in model_order]
rmse_vals = [results[n]['RMSE'] for n in model_order]
colors_models = [MODEL_COLORS.get(n, '#888888') for n in model_order]

ax1.bar(x_pos - width/2, r2_vals, width, color=colors_models, alpha=0.85,
        edgecolor='white', linewidth=0.8, zorder=3)
ax1.set_ylabel('$R^2$')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_order, rotation=30, ha='right', fontsize=12)

# R²值标注
for i, v in enumerate(r2_vals):
    ax1.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=8,
             fontweight='bold', color='#333333')
    if i == 0:
        # TabPFN最优标注
        ax1.annotate('Best', xy=(i - width/2, v), xytext=(i - width/2, v + 0.12),
                     arrowprops=dict(arrowstyle='->', color=MODEL_COLORS['TabPFN'], lw=1.5),
                     fontsize=10, fontweight='bold', color=MODEL_COLORS['TabPFN'], ha='center')

ax1.set_ylim(0, 1.0)
ax1.spines['top'].set_visible(False)
ax1.grid(axis='y', alpha=0.2, linestyle='--')

# RMSE标注
for i, v in enumerate(rmse_vals):
    ax1.text(i + width/2 - 0.02, -0.06, f'RMSE={v:.2f}', ha='center', fontsize=7.5,
             color='#888888', rotation=0)

save_fig(fig_bar, "Fig_Bar_Model_R2_Comparison")
plt.close(fig_bar)

# ============================================================================
# 9. 综合对比矩阵图（散点矩阵：R² vs RMSE vs MAE）
# ============================================================================
print("绘制综合对比散点图...")
fig_scatter, axes_sc = plt.subplots(1, 3, figsize=(18, 5.5))

# R² vs RMSE
ax = axes_sc[0]
for name in model_order:
    r = results[name]
    ax.scatter(r['R²'], r['RMSE'], c=MODEL_COLORS.get(name, '#888888'),
               s=100, edgecolors='white', linewidths=0.8, zorder=3, label=name)
ax.set_xlabel('$R^2$')
ax.set_ylabel('RMSE')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.2, linestyle='--')
ax.set_title('$R^2$ vs RMSE', fontsize=14, fontweight='bold', loc='left')

# R² vs MAE
ax = axes_sc[1]
for name in model_order:
    r = results[name]
    ax.scatter(r['R²'], r['MAE'], c=MODEL_COLORS.get(name, '#888888'),
               s=100, edgecolors='white', linewidths=0.8, zorder=3, label=name)
ax.set_xlabel('$R^2$')
ax.set_ylabel('MAE')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.2, linestyle='--')
ax.set_title('$R^2$ vs MAE', fontsize=14, fontweight='bold', loc='left')

# Corr vs Std_Ratio (Taylor空间)
ax = axes_sc[2]
for name in model_order:
    r = results[name]
    ax.scatter(r['Corr'], r['Std_Ratio'], c=MODEL_COLORS.get(name, '#888888'),
               s=100, edgecolors='white', linewidths=0.8, zorder=3, label=name)
# 标注最优位置
ax.scatter(1, 1, c='black', marker='*', s=250, zorder=5, label='Optimal (1,1)')
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Std Ratio')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.2, linestyle='--')
ax.set_title('Correlation vs Std Ratio', fontsize=14, fontweight='bold', loc='left')

# 统一legend
handles, labels = axes_sc[0].get_legend_handles_labels()
fig_scatter.legend(handles, labels, loc='center right', bbox_to_anchor=(1.08, 0.5),
                   fontsize=9, framealpha=0.9, edgecolor='#cccccc', ncol=1)
plt.tight_layout()
save_fig(fig_scatter, "Fig_Scatter_Model_Metrics")
plt.close(fig_scatter)

# ============================================================================
# 10. 导出 Excel
# ============================================================================
print("\n导出模型对比结果...")
output_excel = "TabPFN_Model_Comparison.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: 主指标对比
    comp_data = []
    for name in model_order:
        r = results[name]
        comp_data.append({
            '模型': name, 'R²': round(r['R²'], 4), 'RMSE': round(r['RMSE'], 4),
            'MAE': round(r['MAE'], 4), 'MAPE(%)': round(r['MAPE(%)'], 2),
            'Correlation': round(r['Corr'], 4), 'Std_Ratio': round(r['Std_Ratio'], 4),
            '排序': model_order.index(name) + 1,
        })
    pd.DataFrame(comp_data).to_excel(writer, sheet_name='多模型对比', index=False)

    # Sheet 2: 相对于TabPFN的提升
    r_tabpfn = results['TabPFN']
    relative_data = []
    for name in model_order:
        if name == 'TabPFN':
            continue
        r = results[name]
        relative_data.append({
            '模型': name,
            'ΔR²': round(r['R²'] - r_tabpfn['R²'], 4),
            'ΔRMSE': round(r['RMSE'] - r_tabpfn['RMSE'], 4),
            'ΔMAE': round(r['MAE'] - r_tabpfn['MAE'], 4),
            'ΔMAPE(%)': round(r['MAPE(%)'] - r_tabpfn['MAPE(%)'], 2),
            'R²相对提升%': round((r_tabpfn['R²'] - r['R²']) / max(abs(r['R²']), 0.001) * 100, 1),
        })
    pd.DataFrame(relative_data).to_excel(writer, sheet_name='相对TabPFN差异', index=False)

print(f"结果已导出: {output_excel}")

# ============================================================================
# 11. 论文 LaTeX 表格
# ============================================================================
print(f"\n{'='*60}")
print("论文 LaTeX 表格")
print(f"{'='*60}")

print("""
\\begin{table}[htbp]
\\centering
\\caption{Predictive performance comparison among different models on sequential split test set}
\\label{tab:model_comparison}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Model} & \\textbf{$R^2$} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{MAPE(\\%)} & \\textbf{$r$} & \\textbf{Rank} \\\\
\\hline""")

for i, name in enumerate(model_order):
    r = results[name]
    bold = "\\textbf{" if name == 'TabPFN' else ""
    end = "}" if name == 'TabPFN' else ""
    print(f"{bold}{name}{end} & {bold if name=='TabPFN' else ''}{r['R²']:.4f}{end} & "
          f"{bold if name=='TabPFN' else ''}{r['RMSE']:.4f}{end} & "
          f"{bold if name=='TabPFN' else ''}{r['MAE']:.4f}{end} & "
          f"{r['MAPE(%)']:.2f} & {r['Corr']:.4f} & {i+1} \\\\")

print("""\\hline
\\end{tabular}
\\end{table}
""")

print("\n✅ 多算法对比 + 雷达图 + 泰勒图 完成")
