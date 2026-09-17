"""
=============================================================================
Bootstrap 置信区间检验
- Bootstrap 重采样 2000 次 → 95% CI for R², RMSE, MAE, MAPE
- 逐样本预测区间 (Prediction Interval)
- 多模型统计检验 (Wilcoxon + Friedman)
- 目的：回应审稿人 "without confidence intervals or statistical comparisons"
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats
import torch
from tabpfn import TabPFNRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 配置
# ============================================================================
MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOTSTRAP = 2000
ALPHA = 0.05  # 95% CI
SEED = 42

np.random.seed(SEED)

print(f"Bootstrap 置信区间检验: {N_BOOTSTRAP} 次重采样")
print(f"设备: {DEVICE}")

# ============================================================================
# 1. 加载数据 + 顺序划分
# ============================================================================
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
target_name = df.columns[-1]
X_all = df.iloc[:, :-1].values
y_all = df.iloc[:, -1].values
n_samples = len(y_all)

# 顺序划分 80/20
train_end = int(n_samples * 0.8)
X_train, X_test = X_all[:train_end], X_all[train_end:]
y_train, y_test = y_all[:train_end], y_all[train_end:]
n_train, n_test = len(y_train), len(y_test)

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_nor = scaler_X.fit_transform(X_train)
X_test_nor = scaler_X.transform(X_test)
y_train_nor = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_nor = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

print(f"训练集: {n_train} 样本 (环 1-{train_end})")
print(f"测试集: {n_test} 样本 (环 {train_end+1}-{n_samples})")

# ============================================================================
# 2. 训练 TabPFN + 基准预测
# ============================================================================
print(f"\n{'='*60}")
print("训练 TabPFN 基准模型...")
print(f"{'='*60}")

model = TabPFNRegressor(
    model_path=MODEL_PATH, n_estimators=3,
    device=DEVICE, ignore_pretraining_limits=True,
)
model.fit(X_train_nor, y_train_nor)

y_pred_nor = model.predict(X_test_nor)
y_pred = scaler_y.inverse_transform(y_pred_nor.reshape(-1, 1)).ravel()
residuals = y_test - y_pred

# 基准指标
def compute_metrics(y_true, y_pred):
    return {
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE(%)': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }

base_metrics = compute_metrics(y_test, y_pred)
print(f"基准指标: R²={base_metrics['R²']:.4f}, RMSE={base_metrics['RMSE']:.4f}, "
      f"MAE={base_metrics['MAE']:.4f}, MAPE={base_metrics['MAPE(%)']:.2f}%")

# ============================================================================
# 3. Bootstrap 置信区间 (配对残差法)
# ============================================================================
print(f"\n{'='*60}")
print(f"Bootstrap 重采样 ({N_BOOTSTRAP} 次)...")
print(f"{'='*60}")

rng_boot = np.random.RandomState(SEED)

# 存储所有 bootstrap 指标
boot_r2, boot_rmse, boot_mae, boot_mape = [], [], [], []
# 存储 bootstrap 预测分布（用于逐样本区间）
boot_preds_matrix = np.zeros((N_BOOTSTRAP, n_test))

for i in tqdm(range(N_BOOTSTRAP), desc="Bootstrapping"):
    # 配对残差 bootstrap：对 (y_true, y_pred) 对进行有放回抽样
    idx = rng_boot.choice(n_test, size=n_test, replace=True)
    y_t_boot = y_test[idx]
    y_p_boot = y_pred[idx]

    boot_r2.append(r2_score(y_t_boot, y_p_boot))
    boot_rmse.append(np.sqrt(mean_squared_error(y_t_boot, y_p_boot)))
    boot_mae.append(mean_absolute_error(y_t_boot, y_p_boot))
    boot_mape.append(np.mean(np.abs((y_t_boot - y_p_boot) / (y_t_boot + 1e-8))) * 100)

    # 逐样本：残差 bootstrap
    resid_boot = residuals[rng_boot.choice(n_test, size=n_test, replace=True)]
    boot_preds_matrix[i] = y_pred + resid_boot


def bootstrap_ci(values, alpha=0.05):
    """百分位数法 Bootstrap CI"""
    return np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])


print(f"\n{'='*60}")
print("Bootstrap 95% 置信区间")
print(f"{'='*60}")
print(f"{'指标':<10} {'基准值':>8} {'95% CI下限':>12} {'95% CI上限':>12} {'Bootstrap均值':>14}")
print("-" * 60)

ci_results = {}
for name, vals in [('R²', boot_r2), ('RMSE', boot_rmse), ('MAE', boot_mae), ('MAPE(%)', boot_mape)]:
    ci_low, ci_high = bootstrap_ci(vals)
    boot_mean = np.mean(vals)
    base_val = base_metrics[name]
    print(f"{name:<10} {base_val:>8.4f} {ci_low:>12.4f} {ci_high:>12.4f} {boot_mean:>14.4f}")
    ci_results[name] = {'low': ci_low, 'high': ci_high, 'mean': boot_mean, 'base': base_val, 'vals': vals}

# ============================================================================
# 4. 逐样本预测区间 (Prediction Interval)
# ============================================================================
pi_low = np.percentile(boot_preds_matrix, 2.5, axis=0)
pi_high = np.percentile(boot_preds_matrix, 97.5, axis=0)
pi_width = pi_high - pi_low
in_interval = np.mean((y_test >= pi_low) & (y_test <= pi_high)) * 100

print(f"\n预测区间覆盖度: {in_interval:.1f}% (期望 95%)")

# ============================================================================
# 5. 多模型统计检验
# ============================================================================
print(f"\n{'='*60}")
print("多模型统计检验 (Wilcoxon Signed-Rank + Friedman)")
print(f"{'='*60}")

# 训练对比模型
print("训练对比模型 (XGBoost, CatBoost, LightGBM)...")

comparison_models = {
    'XGBoost': XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=5,
                             random_state=SEED, verbosity=0),
    'CatBoost': CatBoostRegressor(iterations=150, learning_rate=0.1, depth=5,
                                   random_seed=SEED, verbose=0),
    'LightGBM': None,  # 特殊处理
}

preds = {'TabPFN': y_pred}

# XGBoost
comparison_models['XGBoost'].fit(X_train_nor, y_train_nor)
preds['XGBoost'] = scaler_y.inverse_transform(
    comparison_models['XGBoost'].predict(X_test_nor).reshape(-1, 1)).ravel()

# CatBoost
comparison_models['CatBoost'].fit(X_train_nor, y_train_nor)
preds['CatBoost'] = scaler_y.inverse_transform(
    comparison_models['CatBoost'].predict(X_test_nor).reshape(-1, 1)).ravel()

# LightGBM
lgb_model = lgb.train(
    {'objective': 'regression', 'metric': 'rmse', 'verbose': -1},
    lgb.Dataset(X_train_nor, y_train_nor), num_boost_round=200
)
preds['LightGBM'] = scaler_y.inverse_transform(
    lgb_model.predict(X_test_nor).reshape(-1, 1)).ravel()

# 计算每个模型的残差
model_names = list(preds.keys())
model_residuals = {name: y_test - preds[name] for name in model_names}
model_abs_residuals = {name: np.abs(y_test - preds[name]) for name in model_names}
model_metrics = {name: compute_metrics(y_test, preds[name]) for name in model_names}

print(f"\n各模型测试集指标:")
print(f"{'模型':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MAPE(%)':>10}")
print("-" * 52)
for name in model_names:
    m = model_metrics[name]
    print(f"{name:<12} {m['R²']:>8.4f} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} {m['MAPE(%)']:>9.2f}")

# === Wilcoxon 配对检验 (TabPFN vs Each) ===
print(f"\n--- Wilcoxon Signed-Rank Test (H₀: 无显著差异) ---")
wilcoxon_results = []
for name in ['XGBoost', 'CatBoost', 'LightGBM']:
    stat, p_val = stats.wilcoxon(
        model_abs_residuals['TabPFN'],
        model_abs_residuals[name],
        alternative='less'  # H₁: TabPFN 残差 < 其他模型残差
    )
    # 效应量 r = Z / sqrt(N)
    z = stats.norm.ppf(p_val / 2) if p_val < 0.5 else stats.norm.ppf(1 - p_val / 2)
    effect_size = abs(z) / np.sqrt(n_test)

    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"  TabPFN vs {name:<10}: W={stat:.0f}, p={p_val:.6f} {sig}, "
          f"effect size r={effect_size:.3f}")

    wilcoxon_results.append({
        'Comparison': f'TabPFN vs {name}',
        'Statistic': stat,
        'p-value': p_val,
        'Significance': sig,
        'Effect_Size_r': effect_size,
    })

# === Friedman 检验 (所有模型整体) ===
print(f"\n--- Friedman Test (多模型整体排序) ---")
all_rankings = np.zeros((n_test, len(model_names)))
for i in range(n_test):
    row = [model_abs_residuals[name][i] for name in model_names]
    all_rankings[i, :] = stats.rankdata(row)  # 残差越小排名越高

friedman_stat, friedman_p = stats.friedmanchisquare(
    *[all_rankings[:, j] for j in range(len(model_names))]
)
print(f"  Friedman χ² = {friedman_stat:.2f}, p = {friedman_p:.6f}")

# Nemenyi 事后检验的临界距离
from scipy.stats import studentized_range
q_alpha = studentized_range.ppf(1 - 0.05, len(model_names), 1e10)
CD = q_alpha * np.sqrt(len(model_names) * (len(model_names) + 1) / (6 * n_test))
print(f"  Nemenyi CD (α=0.05): {CD:.3f}")

# 平均排名
avg_ranks = np.mean(all_rankings, axis=0)
rank_df = pd.DataFrame({'Model': model_names, 'Avg_Rank': avg_ranks}).sort_values('Avg_Rank')
print(f"\n  模型平均排名 (越小越好):")
for _, row in rank_df.iterrows():
    print(f"    {row['Model']:<12}: {row['Avg_Rank']:.2f}")

# ============================================================================
# 6. 可视化
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 1200,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.1,
})

COLORS = {
    'tabpfn': '#3b5b92',
    'ci': '#8395b1',
    'line': '#333333',
    'xgb': '#e05263',
    'cat': '#36a168',
    'lgb': '#e0a040',
}

def save_fig(fig, name):
    fig.savefig(f"{name}.png", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    fig.savefig(f"{name}.pdf", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"  ✓ {name}.png / .pdf")


# ==================== 图 A：R² Bootstrap 分布 + 95% CI ====================
print("\n绘制置信区间检验图...")
fig_a, ax_a = plt.subplots(figsize=(8, 5))

r2_arr = np.array(boot_r2)
r2_ci_low, r2_ci_high = ci_results['R²']['low'], ci_results['R²']['high']

ax_a.hist(r2_arr, bins=45, color=COLORS['tabpfn'], alpha=0.65, edgecolor='white',
          linewidth=0.5, density=True, zorder=2)

# KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(r2_arr)
x_kde = np.linspace(r2_arr.min(), r2_arr.max(), 300)
ax_a.plot(x_kde, kde(x_kde), color=COLORS['xgb'], linewidth=2.2, zorder=3)

# 基准值
ax_a.axvline(x=base_metrics['R²'], color='black', linewidth=2.5, linestyle='-',
             label=f'Observed $R^2$ = {base_metrics["R²"]:.4f}', zorder=4)
# 95% CI
ax_a.axvline(x=r2_ci_low, color=COLORS['ci'], linewidth=1.8, linestyle='--', zorder=4)
ax_a.axvline(x=r2_ci_high, color=COLORS['ci'], linewidth=1.8, linestyle='--',
             label=f'95% CI [{r2_ci_low:.4f}, {r2_ci_high:.4f}]', zorder=4)
ax_a.fill_betweenx([0, ax_a.get_ylim()[1]], r2_ci_low, r2_ci_high,
                    alpha=0.08, color=COLORS['ci'], zorder=1)

ax_a.set_xlabel('$R^2$ (Bootstrap)')
ax_a.set_ylabel('Density')
ax_a.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc')
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.grid(axis='y', alpha=0.2, linestyle='--')

save_fig(fig_a, "Fig_CI_Bootstrap_R2")
plt.close(fig_a)


# ==================== 图 B：逐样本预测区间 (Prediction Intervals) ====================
fig_b, ax_b = plt.subplots(figsize=(13, 5.5))

# 按真实值排序
sort_idx = np.argsort(y_test)
x_pos = np.arange(n_test)

ax_b.plot(x_pos, y_test[sort_idx], 'o', color='#555555', markersize=6,
          label='Measured', zorder=3)
ax_b.plot(x_pos, y_pred[sort_idx], 's', color=COLORS['tabpfn'], markersize=5,
          label='Predicted (TabPFN)', zorder=3)
ax_b.fill_between(x_pos, pi_low[sort_idx], pi_high[sort_idx],
                   alpha=0.18, color=COLORS['tabpfn'], zorder=1,
                   label=f'95% PI (Coverage: {in_interval:.1f}%)')

# 标记不在区间内的点
outside = (y_test[sort_idx] < pi_low[sort_idx]) | (y_test[sort_idx] > pi_high[sort_idx])
if np.any(outside):
    ax_b.scatter(x_pos[outside], y_test[sort_idx][outside],
                 s=50, facecolors='none', edgecolors='red', linewidths=1.5,
                 zorder=5, label=f'Outside PI ({np.sum(outside)} pts)')

ax_b.set_xlabel('Test Sample (Sorted by Measured AR)')
ax_b.set_ylabel('Advance Rate (mm/min)')
ax_b.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='#cccccc', ncol=2)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.grid(alpha=0.2, linestyle='--')

save_fig(fig_b, "Fig_CI_Prediction_Interval")
plt.close(fig_b)


# ==================== 图 C：多模型残差对比 + Wilcoxon 显著性 ====================
fig_c, ax_c = plt.subplots(figsize=(9, 5.5))

model_colors = [COLORS['tabpfn'], COLORS['xgb'], COLORS['cat'], COLORS['lgb']]
positions = np.arange(len(model_names)) + 1

# 残差绝对值箱线图
bp_data = [model_abs_residuals[name] for name in model_names]
bp = ax_c.boxplot(bp_data, positions=positions, widths=0.4, patch_artist=True,
                   showmeans=True,
                   meanprops=dict(marker='D', markerfacecolor='white',
                                  markeredgecolor='#333333', markersize=7),
                   medianprops=dict(color='#333333', linewidth=1.8),
                   flierprops=dict(marker='o', markersize=3, alpha=0.4))

for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

# 显著性标注
max_y = ax_c.get_ylim()[1]
for i, name in enumerate(['XGBoost', 'CatBoost', 'LightGBM']):
    w_result = [w for w in wilcoxon_results if name in w['Comparison']][0]
    y_pos = max_y * (0.85 + i * 0.12)
    sig_text = '*' * (3 if w_result['p-value'] < 0.001 else (2 if w_result['p-value'] < 0.01 else 1))
    if w_result['p-value'] >= 0.05:
        sig_text = 'ns'
    ax_c.annotate('', xy=(1, y_pos), xytext=(i + 2, y_pos),
                  arrowprops=dict(arrowstyle='-', color='#666666', lw=0.8))
    ax_c.text(1.5 + (i + 1) / 2, y_pos + 0.02, f'p={w_result["p-value"]:.4f} {sig_text}',
              ha='center', fontsize=9, fontweight='bold')

ax_c.set_xticks(positions)
ax_c.set_xticklabels(model_names)
ax_c.set_ylabel('|Residual| (mm/min)')
ax_c.spines['top'].set_visible(False)
ax_c.spines['right'].set_visible(False)
ax_c.grid(axis='y', alpha=0.2, linestyle='--')

save_fig(fig_c, "Fig_CI_Model_Comparison")
plt.close(fig_c)


# ==================== 图 D：Bootstrap 指标综合面板 ====================
fig_d, axes_d = plt.subplots(2, 2, figsize=(12, 9))
axes_flat = axes_d.flatten()

metric_configs = [
    ('R²', boot_r2, 'Blues'),
    ('RMSE', boot_rmse, 'Reds'),
    ('MAE', boot_mae, 'Greens'),
    ('MAPE(%)', boot_mape, 'Oranges'),
]

for ax, (metric_name, boot_vals, cmap) in zip(axes_flat, metric_configs):
    arr = np.array(boot_vals)
    ci_low, ci_high = bootstrap_ci(arr)
    base_v = base_metrics[metric_name]
    boot_m = np.mean(arr)

    ax.hist(arr, bins=40, color=COLORS['tabpfn'], alpha=0.6, edgecolor='white',
            linewidth=0.4, density=True, zorder=2)

    kde = gaussian_kde(arr)
    x_kde = np.linspace(arr.min(), arr.max(), 200)
    ax.plot(x_kde, kde(x_kde), color=COLORS['xgb'], linewidth=1.8, zorder=3)

    ax.axvline(x=base_v, color='black', linewidth=2.0, linestyle='-', zorder=4)
    ax.axvline(x=ci_low, color='#888888', linewidth=1.2, linestyle='--', zorder=4)
    ax.axvline(x=ci_high, color='#888888', linewidth=1.2, linestyle='--', zorder=4)

    ax.set_xlabel(metric_name)
    ax.set_ylabel('Density')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.15, linestyle='--')

    # 标注
    textstr = f'Obs: {base_v:.4f}\n95% CI: [{ci_low:.4f}, {ci_high:.4f}]'
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.9))

plt.tight_layout(pad=2.5)
save_fig(fig_d, "Fig_CI_Bootstrap_Panel")
plt.close(fig_d)

print("\n全部置信区间检验图已生成")

# ============================================================================
# 7. 导出 Excel
# ============================================================================
print("\n导出置信区间检验结果...")
output_excel = "TabPFN_Confidence_Intervals.xlsx"

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    # Sheet 1: Bootstrap CI 汇总
    ci_summary = []
    for name, vals_dict in ci_results.items():
        ci_summary.append({
            '指标': name, '基准值': vals_dict['base'], 'Bootstrap均值': vals_dict['mean'],
            '95%CI下限': vals_dict['low'], '95%CI上限': vals_dict['high'],
            'CI宽度': vals_dict['high'] - vals_dict['low'],
            'Bootstrap次数': N_BOOTSTRAP,
        })
    pd.DataFrame(ci_summary).to_excel(writer, sheet_name='Bootstrap_CI汇总', index=False)

    # Sheet 2: 逐样本预测区间
    pi_df = pd.DataFrame({
        '样本序号': range(1, n_test + 1),
        '环号': range(train_end + 1, n_samples + 1),
        '真实值': np.round(y_test, 4),
        '预测值': np.round(y_pred, 4),
        '残差': np.round(residuals, 4),
        'PI下限(2.5%)': np.round(pi_low, 4),
        'PI上限(97.5%)': np.round(pi_high, 4),
        'PI宽度': np.round(pi_width, 4),
        '在区间内': (y_test >= pi_low) & (y_test <= pi_high),
    })
    pi_df.to_excel(writer, sheet_name='逐样本预测区间', index=False)

    # Sheet 3: 多模型比较
    model_comp = []
    for name in model_names:
        m = model_metrics[name]
        model_comp.append({
            '模型': name, 'R²': m['R²'], 'RMSE': m['RMSE'],
            'MAE': m['MAE'], 'MAPE(%)': m['MAPE(%)'],
            '平均排名': avg_ranks[model_names.index(name)],
        })
    pd.DataFrame(model_comp).to_excel(writer, sheet_name='多模型比较', index=False)

    # Sheet 4: Wilcoxon 检验
    pd.DataFrame(wilcoxon_results).to_excel(writer, sheet_name='Wilcoxon检验', index=False)

    # Sheet 5: Friedman 检验
    pd.DataFrame({
        '检验方法': ['Friedman Test'],
        '统计量': [friedman_stat],
        'p值': [friedman_p],
        '模型数': [len(model_names)],
        '样本数': [n_test],
        'Nemenyi_CD': [CD],
    }).to_excel(writer, sheet_name='Friedman检验', index=False)

print(f"结果已导出: {output_excel}")
print(f"  ├── Sheet 1: Bootstrap_CI汇总")
print(f"  ├── Sheet 2: 逐样本预测区间")
print(f"  ├── Sheet 3: 多模型比较")
print(f"  ├── Sheet 4: Wilcoxon检验")
print(f"  └── Sheet 5: Friedman检验")

# ============================================================================
# 8. 论文级 LaTeX 表格
# ============================================================================
print(f"\n{'='*60}")
print("论文 LaTeX 表格")
print(f"{'='*60}")

print(f"""
% ===== Bootstrap 置信区间表 =====
\\begin{{table}}[htbp]
\\centering
\\caption{{Bootstrap 95\\% confidence intervals for TabPFN prediction metrics ({N_BOOTSTRAP} resamples)}}
\\label{{tab:bootstrap_ci}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Observed}} & \\textbf{{Bootstrap Mean}} & \\textbf{{95\\% CI}} \\\\
\\hline
$R^2$   & {base_metrics['R²']:.4f} & {np.mean(boot_r2):.4f} & [{ci_results['R²']['low']:.4f}, {ci_results['R²']['high']:.4f}] \\\\
RMSE    & {base_metrics['RMSE']:.4f} & {np.mean(boot_rmse):.4f} & [{ci_results['RMSE']['low']:.4f}, {ci_results['RMSE']['high']:.4f}] \\\\
MAE     & {base_metrics['MAE']:.4f} & {np.mean(boot_mae):.4f} & [{ci_results['MAE']['low']:.4f}, {ci_results['MAE']['high']:.4f}] \\\\
MAPE(\\%) & {base_metrics['MAPE(%)']:.2f} & {np.mean(boot_mape):.2f} & [{ci_results['MAPE(%)']['low']:.2f}, {ci_results['MAPE(%)']['high']:.2f}] \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

% ===== Wilcoxon 检验表 =====
\\begin{{table}}[htbp]
\\centering
\\caption{{Wilcoxon signed-rank test: TabPFN vs comparison models}}
\\label{{tab:wilcoxon}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Comparison}} & \\textbf{{$W$}} & \\textbf{{$p$-value}} & \\textbf{{Effect Size $r$}} & \\textbf{{Significance}} \\\\
\\hline
""")
for wr in wilcoxon_results:
    print(f"TabPFN vs {wr['Comparison'].split('vs ')[1]:<10} & {wr['Statistic']:.0f} & {wr['p-value']:.4f} & {wr['Effect_Size_r']:.3f} & {wr['Significance']} \\\\")
print(f"""\\hline
\\end{{tabular}}
\\end{{table}}
""")

# 覆盖度总结
print(f"预测区间覆盖度: {in_interval:.1f}% (期望 95%) —— {'✓ 良好' if abs(in_interval - 95) < 5 else '⚠ 需检查'}")
print(f"Friedman 检验: χ²={friedman_stat:.2f}, p={friedman_p:.6f} —— {'✓ 模型间存在显著差异' if friedman_p < 0.05 else '模型间无显著差异'}")
print("\n✅ Bootstrap 置信区间检验完成")
