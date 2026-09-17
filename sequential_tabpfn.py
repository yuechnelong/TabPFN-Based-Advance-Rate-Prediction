"""
=============================================================================
顺序划分 TabPFN 预测（无随机打乱，模拟真实工程预测场景）
- 按环号顺序划分：前80%训练 → 中间10%验证 → 后10%测试
- 同时对比随机划分，揭露信息泄露
- 仅运行 TabPFN，不包含 SHAP
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import torch
from tabpfn import TabPFNRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 配置
# ============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)

print(f"设备: {DEVICE}")
print(f"模型: {MODEL_PATH}")
print(f"模型存在: {__import__('os').path.exists(MODEL_PATH)}")

# ============================================================================
# 1. 加载数据
# ============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
target_name = df.columns[-1]

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
n_samples = len(y)

print(f"\n数据: {n_samples} 个样本, {X.shape[1]} 个特征")
print(f"特征: {feature_names}")
print(f"目标: {target_name}")

# ============================================================================
# 2. 顺序划分（按环号，不打乱）
# ============================================================================
train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
train_end = int(n_samples * train_ratio)
valid_end = int(n_samples * (train_ratio + valid_ratio))

# 顺序划分：前80%训练，中间10%验证，后10%测试
X_train_seq = X[:train_end]
y_train_seq = y[:train_end]
X_valid_seq = X[train_end:valid_end]
y_valid_seq = y[train_end:valid_end]
X_test_seq = X[valid_end:]
y_test_seq = y[valid_end:]

print(f"\n=== 顺序划分（按环号，无打乱）===")
print(f"训练集: 环 0 ~ {train_end-1}, {len(y_train_seq)} 样本")
print(f"验证集: 环 {train_end} ~ {valid_end-1}, {len(y_valid_seq)} 样本")
print(f"测试集: 环 {valid_end} ~ {n_samples-1}, {len(y_test_seq)} 样本")

# 标准化（只用训练集计算 mean/std）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_nor = scaler_X.fit_transform(X_train_seq)
X_valid_nor = scaler_X.transform(X_valid_seq)
X_test_nor = scaler_X.transform(X_test_seq)

y_train_nor = scaler_y.fit_transform(y_train_seq.reshape(-1, 1)).ravel()
y_valid_nor = scaler_y.transform(y_valid_seq.reshape(-1, 1)).ravel()
y_test_nor = scaler_y.transform(y_test_seq.reshape(-1, 1)).ravel()

# ============================================================================
# 3. 训练 TabPFN（顺序划分）
# ============================================================================
print(f"\n{'='*60}")
print("训练 TabPFN（顺序划分）...")
print(f"{'='*60}")

model_seq = TabPFNRegressor(
    model_path=MODEL_PATH,
    n_estimators=3,
    device=DEVICE,
    ignore_pretraining_limits=True,
)

model_seq.fit(X_train_nor, y_train_nor)

# 预测（标准化空间 → 反标准化）
y_train_pred_nor = model_seq.predict(X_train_nor)
y_valid_pred_nor = model_seq.predict(X_valid_nor)
y_test_pred_nor = model_seq.predict(X_test_nor)

y_train_pred = scaler_y.inverse_transform(y_train_pred_nor.reshape(-1, 1)).ravel()
y_valid_pred = scaler_y.inverse_transform(y_valid_pred_nor.reshape(-1, 1)).ravel()
y_test_pred = scaler_y.inverse_transform(y_test_pred_nor.reshape(-1, 1)).ravel()


def calc_metrics(y_true, y_pred):
    """计算回归评价指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1e-6
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {'MAE': mae, 'MAPE(%)': mape, 'MSE': mse, 'RMSE': rmse, 'R²': r2}


metrics_seq_train = calc_metrics(y_train_seq, y_train_pred)
metrics_seq_valid = calc_metrics(y_valid_seq, y_valid_pred)
metrics_seq_test = calc_metrics(y_test_seq, y_test_pred)

print(f"\n顺序划分 - 训练集: R²={metrics_seq_train['R²']:.4f}, RMSE={metrics_seq_train['RMSE']:.4f}, MAE={metrics_seq_train['MAE']:.4f}")
print(f"顺序划分 - 验证集: R²={metrics_seq_valid['R²']:.4f}, RMSE={metrics_seq_valid['RMSE']:.4f}, MAE={metrics_seq_valid['MAE']:.4f}")
print(f"顺序划分 - 测试集: R²={metrics_seq_test['R²']:.4f}, RMSE={metrics_seq_test['RMSE']:.4f}, MAE={metrics_seq_test['MAE']:.4f}")

# ============================================================================
# 4. 对比：随机划分（当前做法）
# ============================================================================
print(f"\n{'='*60}")
print("对比实验：随机划分（有信息泄露风险）")
print(f"{'='*60}")

X_train_rnd, X_test_rnd, y_train_rnd, y_test_rnd = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

scaler_X_r = StandardScaler()
scaler_y_r = StandardScaler()
X_train_r_nor = scaler_X_r.fit_transform(X_train_rnd)
X_test_r_nor = scaler_X_r.transform(X_test_rnd)
y_train_r_nor = scaler_y_r.fit_transform(y_train_rnd.reshape(-1, 1)).ravel()

model_rnd = TabPFNRegressor(
    model_path=MODEL_PATH, n_estimators=3,
    device=DEVICE, ignore_pretraining_limits=True,
)
model_rnd.fit(X_train_r_nor, y_train_r_nor)

y_test_r_pred_nor = model_rnd.predict(X_test_r_nor)
y_test_r_pred = scaler_y_r.inverse_transform(y_test_r_pred_nor.reshape(-1, 1)).ravel()

metrics_rnd = calc_metrics(y_test_rnd, y_test_r_pred)
print(f"随机划分 - 测试集: R²={metrics_rnd['R²']:.4f}, RMSE={metrics_rnd['RMSE']:.4f}, MAE={metrics_rnd['MAE']:.4f}")

# ============================================================================
# 5. 预测结果汇总表
# ============================================================================
print(f"\n{'='*60}")
print("预测结果汇总")
print(f"{'='*60}")

results_df = pd.DataFrame({
    '数据集': ['训练集(顺序)', '验证集(顺序)', '测试集(顺序)', '测试集(随机)'],
    '样本数': [len(y_train_seq), len(y_valid_seq), len(y_test_seq), len(y_test_rnd)],
    'MAE': [
        metrics_seq_train['MAE'], metrics_seq_valid['MAE'],
        metrics_seq_test['MAE'], metrics_rnd['MAE']
    ],
    'MAPE(%)': [
        metrics_seq_train['MAPE(%)'], metrics_seq_valid['MAPE(%)'],
        metrics_seq_test['MAPE(%)'], metrics_rnd['MAPE(%)']
    ],
    'MSE': [
        metrics_seq_train['MSE'], metrics_seq_valid['MSE'],
        metrics_seq_test['MSE'], metrics_rnd['MSE']
    ],
    'RMSE': [
        metrics_seq_train['RMSE'], metrics_seq_valid['RMSE'],
        metrics_seq_test['RMSE'], metrics_rnd['RMSE']
    ],
    'R²': [
        metrics_seq_train['R²'], metrics_seq_valid['R²'],
        metrics_seq_test['R²'], metrics_rnd['R²']
    ],
})
print(results_df.to_string(index=False))

# 信息泄露量化
delta_r2 = metrics_rnd['R²'] - metrics_seq_test['R²']
print(f"\n⚠ R² 虚高幅度 (随机-顺序): ΔR² = {delta_r2:+.4f}")
print(f"  随机划分 R² = {metrics_rnd['R²']:.4f}")
print(f"  顺序划分 R² = {metrics_seq_test['R²']:.4f}")

# ============================================================================
# 6. 拼接全环预测数组
# ============================================================================
all_y_pred = np.concatenate([y_train_pred, y_valid_pred, y_test_pred])
all_y_true = np.concatenate([y_train_seq, y_valid_seq, y_test_seq])

# ============================================================================
# 7. 论文级高清拟合图（4张独立图，Times New Roman，无标题，1200 DPI）
# ============================================================================
print("\n绘制拟合图（4张独立高清图）...")

# --- 配色方案 ---
COLORS = {
    'train': '#3b5b92',
    'valid': '#36a168',
    'test': '#e05263',
    'line': '#333333',
    'band': '#e0e0e0',
}

# --- 全局字体（Times New Roman）---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 1200,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})


def save_fig(fig, name):
    """保存为高清 PNG + PDF"""
    fig.savefig(f"{name}.png", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    fig.savefig(f"{name}.pdf", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"  ✓ {name}.png / .pdf")


# ========================================================================
# 图 a：全环时序预测线（Measured vs Predicted）
# ========================================================================
print("  图 a: 全环时序预测线...")
fig_a, ax_a = plt.subplots(figsize=(16, 4.5))

# 分区背景
ax_a.axvspan(0, train_end - 0.5, alpha=0.07, color=COLORS['train'], zorder=0)
ax_a.axvspan(train_end - 0.5, valid_end - 0.5, alpha=0.07, color=COLORS['valid'], zorder=0)
ax_a.axvspan(valid_end - 0.5, n_samples, alpha=0.07, color=COLORS['test'], zorder=0)

# 实测值——实心圆点
ax_a.scatter(range(n_samples), all_y_true, c='#555555', s=12, alpha=0.65,
             edgecolors='none', label='Measured', zorder=3)
# 预测值——彩色三角
ax_a.scatter(range(len(y_train_seq)), y_train_pred, c=COLORS['train'], s=14, alpha=0.75,
             marker='^', edgecolors='none', label='Predicted (Train)', zorder=4)
ax_a.scatter(range(train_end, valid_end), y_valid_pred, c=COLORS['valid'], s=16, alpha=0.80,
             marker='^', edgecolors='none', label='Predicted (Valid)', zorder=4)
ax_a.scatter(range(valid_end, n_samples), y_test_pred, c=COLORS['test'], s=18, alpha=0.85,
             marker='^', edgecolors='white', linewidths=0.3, label='Predicted (Test)', zorder=4)

# 划分边界
for x_pos in [train_end - 0.5, valid_end - 0.5]:
    ax_a.axvline(x=x_pos, color='#444444', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)

# 区域标签
y_bottom = ax_a.get_ylim()[0] + 0.8
ax_a.text(train_end / 2, y_bottom, 'Training Set', ha='center', fontsize=13,
          color=COLORS['train'], fontweight='bold')
ax_a.text((train_end + valid_end) / 2, y_bottom, 'Validation', ha='center', fontsize=12,
          color=COLORS['valid'], fontweight='bold')
ax_a.text((valid_end + n_samples) / 2, y_bottom, 'Test Set', ha='center', fontsize=13,
          color=COLORS['test'], fontweight='bold')

ax_a.set_xlabel('Ring Sequence')
ax_a.set_ylabel('Advance Rate (mm/min)')
ax_a.legend(loc='upper right', fontsize=12, ncol=2, framealpha=0.85,
           edgecolor='#cccccc', markerscale=1.2)
ax_a.grid(alpha=0.2, linestyle='--')
ax_a.set_xlim(-2, n_samples + 2)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

save_fig(fig_a, "Fig_a_Timeseries")
plt.close(fig_a)


# ========================================================================
# 图 b/c/d：训练集/验证集/测试集 拟合散点图（统一风格，无标题）
# ========================================================================
def plot_scatter_fitting(y_true, y_pred, color, metrics_dict, xlabel, ylabel, filename):
    """绘制单张拟合散点图——论文级美观"""
    r2 = metrics_dict['R²']
    rmse = metrics_dict['RMSE']
    mae = metrics_dict['MAE']

    fig, ax = plt.subplots(figsize=(5.5, 5.2))

    # 散点
    ax.scatter(y_true, y_pred, c=color, s=55, alpha=0.75, edgecolors='white',
               linewidths=0.6, zorder=3)

    # 完美拟合线
    vmin = min(y_true.min(), y_pred.min()) - 0.8
    vmax = max(y_true.max(), y_pred.max()) + 0.8
    ax.plot([vmin, vmax], [vmin, vmax], '--', color=COLORS['line'],
            linewidth=1.4, alpha=0.65, zorder=2, label='$y = x$')

    # ±RMSE 半透明带
    ax.fill_between([vmin, vmax],
                    [vmin - rmse, vmax - rmse],
                    [vmin + rmse, vmax + rmse],
                    alpha=0.10, color=color, zorder=1,
                    label=f'$\\pm$RMSE ({rmse:.2f})')

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    # 统计信息框（右上角，半透明）
    textstr = f'$R^2 = {r2:.4f}$\nRMSE = {rmse:.3f}\nMAE  = {mae:.3f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color,
                 alpha=0.92, linewidth=1.5)
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, family='monospace')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.85,
              edgecolor='#cccccc')
    ax.grid(alpha=0.18, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_fig(fig, filename)
    plt.close(fig)


print("  图 b: 训练集拟合散点图...")
plot_scatter_fitting(y_train_seq, y_train_pred, COLORS['train'], metrics_seq_train,
                     'Measured AR (mm/min)', 'Predicted AR (mm/min)',
                     'Fig_b_Train_Fitting')

print("  图 c: 验证集拟合散点图...")
plot_scatter_fitting(y_valid_seq, y_valid_pred, COLORS['valid'], metrics_seq_valid,
                     'Measured AR (mm/min)', 'Predicted AR (mm/min)',
                     'Fig_c_Valid_Fitting')

print("  图 d: 测试集拟合散点图...")
plot_scatter_fitting(y_test_seq, y_test_pred, COLORS['test'], metrics_seq_test,
                     'Measured AR (mm/min)', 'Predicted AR (mm/min)',
                     'Fig_d_Test_Fitting')

# ========================================================================
# 图 e：随机 vs 顺序 R² 对比图（无标题）
# ========================================================================
print("  图 e: R² 对比图...")
fig_e, ax_e = plt.subplots(figsize=(7, 5))

methods = ['Random Split\n(Potential Leakage)', 'Sequential Split\n(Time-Ordered)']
r2_vals = [metrics_rnd['R²'], metrics_seq_test['R²']]
colors_bar = ['#e05263', '#36a168']

bars = ax_e.bar(methods, r2_vals, color=colors_bar, alpha=0.85, width=0.42,
                edgecolor='white', linewidth=1.5)

for bar, val in zip(bars, r2_vals):
    ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.07,
              f'$R^2 = {val:.4f}$', ha='center', va='top',
              fontweight='bold', fontsize=15, color='white')

# ΔR² 注释
mid_y = (r2_vals[0] + r2_vals[1]) / 2
ax_e.annotate('', xy=(0.62, r2_vals[1] + 0.005), xytext=(0.62, r2_vals[0] - 0.005),
              arrowprops=dict(arrowstyle='<->', color='#333333', lw=2.0))
ax_e.text(0.70, mid_y, f'$\\Delta R^2 = {delta_r2:+.4f}$', fontsize=13,
          va='center', fontweight='bold', color='#333333',
          bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff9c4',
                    edgecolor='#bbbbbb', alpha=0.9))

ax_e.set_ylabel('$R^2$ Score')
ax_e.set_ylim(0, max(r2_vals) * 1.06)
ax_e.grid(axis='y', alpha=0.2, linestyle='--')
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)
ax_e.tick_params(axis='x', labelsize=14)

save_fig(fig_e, "Fig_e_R2_Comparison")
plt.close(fig_e)

print("\n全部拟合图已生成（1200 DPI, Times New Roman, 无标题）")

# ============================================================================
# 9. 导出结果 Excel（多Sheet分数据集）
# ============================================================================
print("\n导出结果表格...")
output_excel = "TabPFN_顺序预测结果.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # ---- Sheet 1: 评价指标汇总 ----
    results_df.to_excel(writer, sheet_name='评价指标汇总', index=False)

    # ---- Sheet 2: 全环预测值 ----
    pred_all = pd.DataFrame({
        '环序号': range(1, n_samples + 1),
        '真实值(mm/min)': np.round(all_y_true, 4),
        '预测值(mm/min)': np.round(all_y_pred, 4),
        '残差': np.round(all_y_true - all_y_pred, 4),
        '相对误差(%)': np.round((all_y_true - all_y_pred) / all_y_true * 100, 2),
        '数据集': (['训练集'] * len(y_train_seq) +
                   ['验证集'] * len(y_valid_seq) +
                   ['测试集'] * len(y_test_seq)),
    })
    pred_all.to_excel(writer, sheet_name='全环预测值', index=False)

    # ---- Sheet 3-5: 分数据集详细预测 ----
    dataset_configs = [
        ('训练集预测值', y_train_seq, y_train_pred, 0, metrics_seq_train),
        ('验证集预测值', y_valid_seq, y_valid_pred, train_end, metrics_seq_valid),
        ('测试集预测值', y_test_seq, y_test_pred, valid_end, metrics_seq_test),
    ]

    for sheet_name, y_t, y_p, offset, met in dataset_configs:
        n = len(y_t)
        df_detail = pd.DataFrame({
            '环序号': range(offset + 1, offset + n + 1),
            '真实值(mm/min)': np.round(y_t, 4),
            '预测值(mm/min)': np.round(y_p, 4),
            '残差': np.round(y_t - y_p, 4),
            '相对误差(%)': np.round((y_t - y_p) / y_t * 100, 2),
        })
        # 在末尾加一行统计指标
        stats_row = pd.DataFrame({
            '环序号': ['统计指标'],
            '真实值(mm/min)': [f'R²={met["R²"]:.4f}'],
            '预测值(mm/min)': [f'RMSE={met["RMSE"]:.4f}'],
            '残差': [f'MAE={met["MAE"]:.4f}'],
            '相对误差(%)': [f'MAPE={met["MAPE(%)"]:.2f}%'],
        })
        df_detail = pd.concat([df_detail, stats_row], ignore_index=True)
        df_detail.to_excel(writer, sheet_name=sheet_name, index=False)

    # ---- Sheet 6: 随机划分对比 ----
    n_rnd = len(y_test_rnd)
    df_rnd = pd.DataFrame({
        '序号': range(1, n_rnd + 1),
        '真实值(mm/min)': np.round(y_test_rnd, 4),
        '预测值(mm/min)': np.round(y_test_r_pred, 4),
        '残差': np.round(y_test_rnd - y_test_r_pred, 4),
        '相对误差(%)': np.round((y_test_rnd - y_test_r_pred) / y_test_rnd * 100, 2),
    })
    stats_rnd = pd.DataFrame({
        '序号': ['统计指标'],
        '真实值(mm/min)': [f'R²={metrics_rnd["R²"]:.4f}'],
        '预测值(mm/min)': [f'RMSE={metrics_rnd["RMSE"]:.4f}'],
        '残差': [f'MAE={metrics_rnd["MAE"]:.4f}'],
        '相对误差(%)': [f'MAPE={metrics_rnd["MAPE(%)"]:.2f}%'],
    })
    df_rnd = pd.concat([df_rnd, stats_rnd], ignore_index=True)
    df_rnd.to_excel(writer, sheet_name='随机划分对比(测试集)', index=False)

print(f"结果已导出: {output_excel}")
print(f"  ├── Sheet 1: 评价指标汇总")
print(f"  ├── Sheet 2: 全环预测值")
print(f"  ├── Sheet 3: 训练集预测值")
print(f"  ├── Sheet 4: 验证集预测值")
print(f"  ├── Sheet 5: 测试集预测值")
print(f"  └── Sheet 6: 随机划分对比(测试集)")

print("\n✅ 顺序划分 TabPFN 预测完成")
