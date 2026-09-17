"""
=============================================================================
Walk-Forward 扩展窗口验证（100次实验）
- 100个不同时间窗口划分 → 每次训练+测试
- 仅 TabPFN，无 SHAP
- 输出：R²/RMSE/MAE 分布 + 稳定性统计
- 目的：回应审稿人 "single split, without repeated validation"
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from tabpfn import TabPFNRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 配置
# ============================================================================
MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_WINDOWS = 100
SEED_BASE = 42

np.random.seed(SEED_BASE)
print(f"设备: {DEVICE}")
print(f"扩展窗口实验: {N_WINDOWS} 次")

# ============================================================================
# 1. 加载数据
# ============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
target_name = df.columns[-1]
X_all = df.iloc[:, :-1].values
y_all = df.iloc[:, -1].values
n_samples = len(y_all)

print(f"数据: {n_samples} 样本, {X_all.shape[1]} 特征")

# ============================================================================
# 2. 生成 100 个 Walk-Forward 窗口配置
# ============================================================================
def generate_window_configs(n_total, n_windows):
    """生成扩展窗口配置
    策略：训练集从 ~40% 逐渐扩展到 ~85%，测试窗口同步后移
    多维度变化：
      - 训练集大小: 110~230 样本
      - 测试集大小: 20~60 样本
      - 测试窗口在训练集之后
    """
    configs = []
    rng = np.random.RandomState(SEED_BASE)

    for i in range(n_windows):
        # 多样化训练集大小
        if i < 25:
            train_size = int(n_total * (0.40 + i * 0.012))       # 40%→70%
        elif i < 50:
            train_size = int(n_total * (0.70 + (i - 25) * 0.006))  # 70%→85%
        elif i < 75:
            train_size = int(n_total * (0.45 + (i - 50) * 0.010))  # 45%→70%
        else:
            train_size = int(n_total * (0.50 + (i - 75) * 0.012))  # 50%→80%

        # 确保测试集至少 15 个样本
        max_test = n_total - train_size - 1
        if max_test < 15:
            train_size = n_total - 16
            max_test = 15

        test_size = min(rng.randint(18, 55), max_test)
        train_end = train_size

        configs.append({
            'window': i + 1,
            'train_start': 0,
            'train_end': train_end,
            'train_size': train_end,
            'test_start': train_end,
            'test_end': train_end + test_size,
            'test_size': test_size,
        })

    # 加上一个固定的大窗口检验（当前论文的划分方式）
    configs.append({
        'window': N_WINDOWS + 1,
        'train_start': 0,
        'train_end': int(n_total * 0.8),
        'train_size': int(n_total * 0.8),
        'test_start': int(n_total * 0.8),
        'test_end': n_samples,
        'test_size': n_samples - int(n_total * 0.8),
        'is_benchmark': True,
    })

    return configs


configs = generate_window_configs(n_samples, N_WINDOWS)
print(f"生成 {len(configs)} 个窗口配置")
print(f"  训练集范围: {min(c['train_size'] for c in configs)} ~ {max(c['train_size'] for c in configs)}")
print(f"  测试集范围: {min(c['test_size'] for c in configs)} ~ {max(c['test_size'] for c in configs)}")

# ============================================================================
# 3. Walk-Forward 验证（逐窗口训练+预测）
# ============================================================================
print(f"\n{'='*60}")
print(f"开始 Walk-Forward 验证 ({len(configs)} 窗口)")
print(f"{'='*60}")

results = []

for cfg in tqdm(configs, desc="Walk-Forward"):
    train_idx = range(cfg['train_start'], cfg['train_end'])
    test_idx = range(cfg['test_start'], cfg['test_end'])

    X_train_w = X_all[train_idx]
    y_train_w = y_all[train_idx]
    X_test_w = X_all[test_idx]
    y_test_w = y_all[test_idx]

    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_nor = scaler_X.fit_transform(X_train_w)
    X_test_nor = scaler_X.transform(X_test_w)
    y_train_nor = scaler_y.fit_transform(y_train_w.reshape(-1, 1)).ravel()

    # TabPFN 训练
    model = TabPFNRegressor(
        model_path=MODEL_PATH, n_estimators=3,
        device=DEVICE, ignore_pretraining_limits=True,
    )
    model.fit(X_train_nor, y_train_nor)

    # 预测
    y_pred_nor = model.predict(X_test_nor)
    y_pred = scaler_y.inverse_transform(y_pred_nor.reshape(-1, 1)).ravel()

    # 计算指标
    r2 = r2_score(y_test_w, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_w, y_pred))
    mae = mean_absolute_error(y_test_w, y_pred)
    mape = np.mean(np.abs((y_test_w - y_pred) / (y_test_w + 1e-8))) * 100

    results.append({
        'Window': cfg['window'],
        'Train_Size': cfg['train_size'],
        'Test_Size': cfg['test_size'],
        'Train_Range': f"环{cfg['train_start']+1}-{cfg['train_end']}",
        'Test_Range': f"环{cfg['test_start']+1}-{cfg['test_end']}",
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE(%)': mape,
        'is_benchmark': cfg.get('is_benchmark', False),
    })

# ============================================================================
# 4. 结果汇总
# ============================================================================
df_results = pd.DataFrame(results)
df_nobm = df_results[~df_results['is_benchmark']]  # 100个随机窗口
df_bm = df_results[df_results['is_benchmark']]      # 基准窗口

print(f"\n{'='*60}")
print("Walk-Forward 验证结果汇总（100窗口）")
print(f"{'='*60}")
print(f"\n{'指标':<10} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8} {'95% CI下限':>12} {'95% CI上限':>12}")
print("-" * 72)

for metric, col in [('R²', 'R²'), ('RMSE', 'RMSE'), ('MAE', 'MAE'), ('MAPE(%)', 'MAPE(%)')]:
    vals = df_nobm[col].values
    mean_v = np.mean(vals)
    std_v = np.std(vals, ddof=1)
    min_v = np.min(vals)
    max_v = np.max(vals)
    ci_low = mean_v - 1.96 * std_v / np.sqrt(len(vals))
    ci_high = mean_v + 1.96 * std_v / np.sqrt(len(vals))
    print(f"{metric:<10} {mean_v:>8.4f} {std_v:>8.4f} {min_v:>8.4f} {max_v:>8.4f} {ci_low:>12.4f} {ci_high:>12.4f}")

# 基准窗口结果
print(f"\n基准窗口 (80/20顺序划分):")
print(f"  R²={df_bm['R²'].values[0]:.4f}, RMSE={df_bm['RMSE'].values[0]:.4f}")

# ============================================================================
# 5. 论文级可视化
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 13,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 1200,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.1,
})

COLORS = {
    'train': '#3b5b92',
    'test': '#e05263',
    'benchmark': '#36a168',
    'hist': '#5b8cb8',
    'line': '#333333',
}

def save_fig(fig, name):
    fig.savefig(f"{name}.png", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    fig.savefig(f"{name}.pdf", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"  ✓ {name}.png / .pdf")


# ==================== 图 A：R² 分布直方图 + KDE ====================
print("\n绘制 Walk-Forward 验证图...")
fig_a, ax_a = plt.subplots(figsize=(8, 5.5))

r2_vals = df_nobm['R²'].values
rmse_vals = df_nobm['RMSE'].values

# 直方图
n_bins = 18
counts, bins, patches = ax_a.hist(r2_vals, bins=n_bins, color=COLORS['hist'],
                                   alpha=0.7, edgecolor='white', linewidth=0.8,
                                   density=True, zorder=2)

# KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(r2_vals)
x_kde = np.linspace(min(r2_vals) - 0.01, max(r2_vals) + 0.01, 200)
ax_a.plot(x_kde, kde(x_kde), color=COLORS['test'], linewidth=2.2, zorder=3)

# 均值线和95% CI
mean_r2 = np.mean(r2_vals)
std_r2 = np.std(r2_vals, ddof=1)
ci95_low, ci95_high = np.percentile(r2_vals, [2.5, 97.5])

ax_a.axvline(x=mean_r2, color=COLORS['benchmark'], linewidth=2.0, linestyle='-',
             label=f'Mean R² = {mean_r2:.4f}', zorder=4)
ax_a.axvline(x=ci95_low, color='#888888', linewidth=1.2, linestyle='--', zorder=4)
ax_a.axvline(x=ci95_high, color='#888888', linewidth=1.2, linestyle='--',
             label=f'95% CI [{ci95_low:.4f}, {ci95_high:.4f}]', zorder=4)
ax_a.fill_betweenx([0, ax_a.get_ylim()[1]], ci95_low, ci95_high,
                    alpha=0.08, color='gray', zorder=1)

ax_a.set_xlabel('$R^2$')
ax_a.set_ylabel('Density')
ax_a.legend(loc='upper left', fontsize=11, framealpha=0.85, edgecolor='#cccccc')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.grid(axis='y', alpha=0.2, linestyle='--')

save_fig(fig_a, "Fig_WF_R2_Distribution")
plt.close(fig_a)


# ==================== 图 B：多窗口 R²/RMSE 时序稳定性 ====================
fig_b, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

windows = df_nobm['Window'].values

# R²
ax_b1.plot(windows, df_nobm['R²'].values, 'o-', color=COLORS['train'],
           markersize=3.5, linewidth=0.8, alpha=0.75, zorder=2)
ax_b1.axhline(y=mean_r2, color=COLORS['benchmark'], linewidth=1.6, linestyle='-',
              alpha=0.8, zorder=3)
ax_b1.fill_between(windows, ci95_low, ci95_high, alpha=0.10, color=COLORS['train'], zorder=1)
ax_b1.set_ylabel('$R^2$')
ax_b1.set_ylim(0.88, 1.0)
ax_b1.grid(alpha=0.2, linestyle='--')
ax_b1.spines['top'].set_visible(False)
ax_b1.spines['right'].set_visible(False)
# 标注
ax_b1.text(0.99, 0.93, f'Mean = {mean_r2:.4f}$\\pm${std_r2:.4f}\n95% CI [{ci95_low:.4f}, {ci95_high:.4f}]',
           transform=ax_b1.transAxes, fontsize=11, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['benchmark'], alpha=0.9))

# RMSE
mean_rmse = np.mean(rmse_vals)
std_rmse = np.std(rmse_vals, ddof=1)
ci95_low_rmse, ci95_high_rmse = np.percentile(rmse_vals, [2.5, 97.5])

ax_b2.plot(windows, rmse_vals, 's-', color=COLORS['test'],
           markersize=3.5, linewidth=0.8, alpha=0.75, zorder=2)
ax_b2.axhline(y=mean_rmse, color=COLORS['benchmark'], linewidth=1.6, linestyle='-',
              alpha=0.8, zorder=3)
ax_b2.fill_between(windows, ci95_low_rmse, ci95_high_rmse, alpha=0.10, color=COLORS['test'], zorder=1)
ax_b2.set_xlabel('Window')
ax_b2.set_ylabel('RMSE')
ax_b2.grid(alpha=0.2, linestyle='--')
ax_b2.spines['top'].set_visible(False)
ax_b2.spines['right'].set_visible(False)
ax_b2.text(0.99, 0.93, f'Mean = {mean_rmse:.4f}$\\pm${std_rmse:.4f}\n95% CI [{ci95_low_rmse:.4f}, {ci95_high_rmse:.4f}]',
           transform=ax_b2.transAxes, fontsize=11, ha='right', va='top',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['test'], alpha=0.9))

plt.tight_layout(pad=1.5)
save_fig(fig_b, "Fig_WF_Stability_Curves")
plt.close(fig_b)


# ==================== 图 C：训练/测试大小 vs R² 热力散点 ====================
fig_c, ax_c = plt.subplots(figsize=(8, 5.5))

scatter = ax_c.scatter(df_nobm['Train_Size'], df_nobm['Test_Size'],
                       c=df_nobm['R²'].values, cmap='RdYlBu', s=60,
                       edgecolors='white', linewidths=0.5, vmin=0.89, vmax=0.98, zorder=3)

# 标记基准窗口
bm_row = df_bm.iloc[0]
ax_c.scatter(bm_row['Train_Size'], bm_row['Test_Size'],
             c=bm_row['R²'], cmap='RdYlBu', s=140, edgecolors=COLORS['benchmark'],
             linewidths=2.5, marker='s', vmin=0.89, vmax=0.98, zorder=5,
             label=f'Benchmark (R²={bm_row["R²"]:.3f})')

cbar = plt.colorbar(scatter, ax=ax_c, shrink=0.82, pad=0.02)
cbar.set_label('$R^2$', fontsize=14)
cbar.ax.tick_params(labelsize=11)

ax_c.set_xlabel('Training Set Size')
ax_c.set_ylabel('Test Set Size')
ax_c.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(alpha=0.15, linestyle='--')

save_fig(fig_c, "Fig_WF_TrainTest_Heatmap")
plt.close(fig_c)


# ==================== 图 D：R², RMSE, MAE 箱线图 ====================
fig_d, ax_d = plt.subplots(figsize=(9, 5))

box_data = [df_nobm['R²'].values, df_nobm['RMSE'].values, df_nobm['MAE'].values]
positions = [1, 2, 3]

bp = ax_d.boxplot(box_data, positions=positions, widths=0.45, patch_artist=True,
                   showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                                                  markeredgecolor='#333333', markersize=7),
                   medianprops=dict(color='#333333', linewidth=1.8),
                   flierprops=dict(marker='o', markersize=3, alpha=0.4))

colors_box = [COLORS['train'], COLORS['test'], '#e0a040']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)

# 标注均值
labels = ['$R^2$', 'RMSE', 'MAE']
for i, (vals, pos) in enumerate(zip(box_data, positions)):
    mean_v = np.mean(vals)
    std_v = np.std(vals, ddof=1)
    ax_d.annotate(f'{mean_v:.4f}$\\pm${std_v:.4f}',
                  xy=(pos + 0.15, mean_v), fontsize=10,
                  va='center', fontweight='bold', color='#333333')

ax_d.set_xticks(positions)
ax_d.set_xticklabels(labels)
ax_d.set_ylabel('Value')
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)
ax_d.grid(axis='y', alpha=0.2, linestyle='--')

save_fig(fig_d, "Fig_WF_Boxplot")
plt.close(fig_d)

print("\n全部 Walk-Forward 验证图已生成")

# ============================================================================
# 6. 导出 Excel（多Sheet）
# ============================================================================
print("\n导出 Walk-Forward 验证结果表格...")
output_excel = "TabPFN_WalkForward_100Windows.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: 100窗口详细结果
    cols_export = ['Window', 'Train_Size', 'Test_Size', 'Train_Range',
                   'Test_Range', 'R²', 'RMSE', 'MAE', 'MAPE(%)']
    df_export = df_nobm[cols_export].copy()
    df_export.to_excel(writer, sheet_name='100窗口结果', index=False)

    # Sheet 2: 统计汇总
    stats_data = []
    for metric, col in [('R²', 'R²'), ('RMSE', 'RMSE'), ('MAE', 'MAE'), ('MAPE(%)', 'MAPE(%)')]:
        vals = df_nobm[col].values
        stats_data.append({
            '指标': metric,
            '均值': np.round(np.mean(vals), 4),
            '标准差': np.round(np.std(vals, ddof=1), 4),
            '最小值': np.round(np.min(vals), 4),
            '最大值': np.round(np.max(vals), 4),
            '中位数': np.round(np.median(vals), 4),
            '95%_CI_下限': np.round(np.mean(vals) - 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)), 4),
            '95%_CI_上限': np.round(np.mean(vals) + 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)), 4),
            '2.5%分位数': np.round(np.percentile(vals, 2.5), 4),
            '97.5%分位数': np.round(np.percentile(vals, 97.5), 4),
        })
    pd.DataFrame(stats_data).to_excel(writer, sheet_name='统计汇总', index=False)

    # Sheet 3: 基准窗口
    df_bm_export = df_bm[cols_export].copy()
    df_bm_export.to_excel(writer, sheet_name='基准窗口(80-20)', index=False)

    # Sheet 4: 窗口配置说明
    config_df = pd.DataFrame({
        '配置项': ['总窗口数', '训练集范围', '测试集范围', '总样本数', '特征数', '模型'],
        '值': [f'{N_WINDOWS}', f'{min(configs, key=lambda x:x["train_size"])["train_size"]}~{max(configs, key=lambda x:x["train_size"])["train_size"]}',
               f'{min(configs, key=lambda x:x["test_size"])["test_size"]}~{max(configs, key=lambda x:x["test_size"])["test_size"]}',
               f'{n_samples}', f'{len(feature_names)}', 'TabPFN v2.5'],
    })
    config_df.to_excel(writer, sheet_name='配置说明', index=False)

print(f"结果已导出: {output_excel}")
print(f"  ├── Sheet 1: 100窗口结果")
print(f"  ├── Sheet 2: 统计汇总")
print(f"  ├── Sheet 3: 基准窗口(80-20)")
print(f"  └── Sheet 4: 配置说明")

# ============================================================================
# 7. 生成 LaTeX 表格（论文可直接使用）
# ============================================================================
print(f"\n{'='*60}")
print("论文 LaTeX 表格（统计汇总）")
print(f"{'='*60}")

latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Walk-Forward validation results across 100 expanding windows}}
\\label{{tab:walkforward}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std}} & \\textbf{{95\\% CI}} & \\textbf{{Range}} \\\\
\\hline
$R^2$   & {np.mean(r2_vals):.4f} & {np.std(r2_vals, ddof=1):.4f} & [{ci95_low:.4f}, {ci95_high:.4f}] & [{np.min(r2_vals):.4f}, {np.max(r2_vals):.4f}] \\\\
RMSE    & {np.mean(rmse_vals):.4f} & {np.std(rmse_vals, ddof=1):.4f} & [{ci95_low_rmse:.4f}, {ci95_high_rmse:.4f}] & [{np.min(rmse_vals):.4f}, {np.max(rmse_vals):.4f}] \\\\
MAE     & {np.mean(df_nobm['MAE'].values):.4f} & {np.std(df_nobm['MAE'].values, ddof=1):.4f} & [{np.mean(df_nobm['MAE'].values)-1.96*np.std(df_nobm['MAE'].values,ddof=1)/10:.4f}, {np.mean(df_nobm['MAE'].values)+1.96*np.std(df_nobm['MAE'].values,ddof=1)/10:.4f}] & [{np.min(df_nobm['MAE'].values):.4f}, {np.max(df_nobm['MAE'].values):.4f}] \\\\
MAPE(\\%) & {np.mean(df_nobm['MAPE(%)'].values):.2f} & {np.std(df_nobm['MAPE(%)'].values, ddof=1):.2f} & [{np.mean(df_nobm['MAPE(%)'].values)-1.96*np.std(df_nobm['MAPE(%)'].values,ddof=1)/10:.2f}, {np.mean(df_nobm['MAPE(%)'].values)+1.96*np.std(df_nobm['MAPE(%)'].values,ddof=1)/10:.2f}] & [{np.min(df_nobm['MAPE(%)'].values):.2f}, {np.max(df_nobm['MAPE(%)'].values):.2f}] \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
print(latex_table)

print("\n✅ Walk-Forward 100窗口验证完成")
