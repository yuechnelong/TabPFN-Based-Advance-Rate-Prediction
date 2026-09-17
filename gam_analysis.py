"""
=============================================================================
GAM SHAP 依赖分析 — 参照原代码格式 + 新增阈值不确定性
- 原格式: CI带 + 临界点 + p值 + R²
- 新增: 阈值 Bootstrap 95% CI 带 + 数值标注
- 无 SHAP 方向角标
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pygam import LinearGAM, s
from sklearn.preprocessing import StandardScaler
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ============================================================================
# 0. 数据
# ============================================================================
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
feature_names = df.columns[:-1].tolist()
X_all = df[feature_names].values.astype(np.float64)
y_all = df.iloc[:, -1].values.astype(np.float64)

split_idx = int(len(y_all) * 0.8)
scaler_X = StandardScaler(); scaler_y = StandardScaler()
scaler_X.fit(X_all[:split_idx])
scaler_y.fit(y_all[:split_idx].reshape(-1, 1))
y_std = scaler_y.scale_[0]
X_mean = scaler_X.mean_; X_std = scaler_X.scale_

# SHAP值 → 物理尺度
shap_df = pd.read_excel("TabPFN_SHAP_Analysis.xlsx", sheet_name="SHAP_Values")
shap_norm = shap_df[feature_names].values
shap_real = shap_norm * y_std

# 原始特征值
np.random.seed(42)
N_EXPLAIN = 27
explain_idx = np.sort(np.random.choice(len(X_all)-split_idx, N_EXPLAIN, replace=False))
X_explain_raw = X_all[split_idx:][explain_idx]

# 排序
imp_df = pd.read_excel("TabPFN_SHAP_Analysis.xlsx", sheet_name="Feature_Importance")
top_features = imp_df.sort_values('Mean_ABS_SHAP', ascending=False)['Feature'].tolist()[:8]

# ============================================================================
# 1. 配色（参照原代码）
# ============================================================================
COLORS = {
    'background': '#f9f9f9',
    'main': '#3b5b92',
    'ci': '#8395b1',
    'positive': '#36a168',
    'negative': '#e05263',
    'data': '#c5c5c5',
    'zero_line': '#666666',
    'tipping_point': '#f0746e',
    'threshold_ci': '#f5c6cb',
}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 10,
    'axes.unicode_minus': False, 'axes.linewidth': 2.0,
    'figure.dpi': 200, 'savefig.dpi': 1200,
})

def save_fig(fig, name):
    fig.savefig(f"{name}.png", dpi=1200, bbox_inches='tight', facecolor='white', pad_inches=0.12)
    fig.savefig(f"{name}.pdf", dpi=1200, bbox_inches='tight', facecolor='white', pad_inches=0.12)

# ============================================================================
# 2. GAM 拟合（参照原代码 y_shap ~ s(feature)）
# ============================================================================
N_BOOT = 500
gam_results = []

for feat_name in top_features:
    feat_idx = feature_names.index(feat_name)
    x = X_explain_raw[:, feat_idx]
    y = shap_real[:, feat_idx]

    # GAM拟合（参照原代码参数）
    gam = LinearGAM(s(0, n_splines=15, spline_order=3, lam=1.0)).fit(x, y)
    XX = gam.generate_X_grid(term=0, n=250)
    y_pred = gam.predict(XX)
    # 原代码用 prediction_intervals
    ci = gam.prediction_intervals(XX, width=0.95)
    x_grid, ci_l, ci_u = XX[:, 0], ci[:, 0], ci[:, 1]

    r2  = gam.statistics_['pseudo_r2']['explained_deviance']
    edof = gam.statistics_['edof']
    p_raw = gam.statistics_['p_values'][0]
    if p_raw < 0.001:
        p_show = 'p < 0.001'
    elif p_raw < 0.01:
        p_show = f'p = {p_raw:.4f}'
    else:
        p_show = f'p = {p_raw:.4f}'
    # 原代码风格p值展示
    p_text = 'p < 0.01' if p_raw < 0.01 else f'p = {p_raw:.3e}'

    # 临界点（原代码的 zero_crossing 逻辑）
    thresholds = []
    for i in range(1, len(x_grid)):
        if y_pred[i-1] * y_pred[i] < 0:
            t = x_grid[i-1] + (x_grid[i] - x_grid[i-1]) * abs(y_pred[i-1]) / (abs(y_pred[i-1]) + abs(y_pred[i]))
            # 在曲线上对应的y值约=0
            y_at_t = np.interp(t, x_grid, y_pred)
            thresholds.append({'x': t, 'y': y_at_t})

    # ---- 新增：Bootstrap 阈值 95% CI ----
    if thresholds:
        boot_vals = []
        rng = np.random.RandomState(42)
        for _ in range(N_BOOT):
            idx_b = rng.choice(len(x), size=len(x), replace=True)
            try:
                gb = LinearGAM(s(0, n_splines=15, spline_order=3, lam=1.0)).fit(x[idx_b], y[idx_b])
                XXb = gb.generate_X_grid(term=0, n=200)
                ypb = gb.predict(XXb)
                xgb = XXb[:, 0]
                for j in range(1, len(xgb)):
                    if ypb[j-1] * ypb[j] < 0:
                        boot_vals.append(xgb[j-1] + (xgb[j]-xgb[j-1]) * abs(ypb[j-1]) / (abs(ypb[j-1])+abs(ypb[j])))
                        break
            except:
                pass
        if len(boot_vals) >= 50:
            t_lo, t_hi = np.percentile(boot_vals, [2.5, 97.5])
            thresholds[0]['ci_low'] = t_lo
            thresholds[0]['ci_high'] = t_hi

    gam_results.append({
        'Feature': feat_name, 'R²': r2, 'p_text': p_text, 'p_show': p_show,
        'edof': edof, 'x_grid': x_grid, 'y_pred': y_pred,
        'ci_lower': ci_l, 'ci_upper': ci_u, 'x_raw': x, 'y_shap': y,
        'thresholds': thresholds,
    })

    t_str = ''
    if thresholds:
        tc = thresholds[0]
        t_str = f'CV={tc["x"]:.2f}'
        if 'ci_low' in tc:
            t_str += f' [{tc["ci_low"]:.2f}, {tc["ci_high"]:.2f}]'
    print(f"  {feat_name:8s}: R²={r2:.4f}, {p_show}, edof={edof:.1f}, {t_str}")


# ============================================================================
# 3. 单特征图（完全参照原代码 plot_scientific_style_unnormalized）
# ============================================================================
print("\n绘制单特征 GAM 图...")

def plot_gam_single(ax, res, sample_indices, is_panel=False):
    """参照原代码格式的GAM单图 + 新增阈值CI + 不确定性标注"""
    feat = res['Feature']
    xg = res['x_grid']; yp = res['y_pred']
    ci_l = res['ci_lower']; ci_u = res['ci_upper']
    xr = res['x_raw']; yr = res['y_shap']

    ax.set_facecolor(COLORS['background'])

    # 1. 零线
    ax.axhline(0, color=COLORS['zero_line'], linestyle=(0, (5, 2)),
               linewidth=1.2, alpha=0.9, zorder=2)

    # 2. 正负区域
    pos_mask = yp > 0
    if any(pos_mask):
        ax.fill_between(xg, 0, yp, where=pos_mask, color=COLORS['positive'],
                        alpha=0.15, zorder=0, label='Positive contribution')
    neg_mask = yp <= 0
    if any(neg_mask):
        ax.fill_between(xg, 0, yp, where=neg_mask, color=COLORS['negative'],
                        alpha=0.15, zorder=0, label='Negative contribution')

    # 3. 95% CI 带
    ax.fill_between(xg, ci_l, ci_u, color=COLORS['ci'], alpha=0.35,
                    edgecolor='none', zorder=3, label='95% CI (GAM fit)')

    # 4. 新增：阈值 95% CI 带
    for tc in res['thresholds']:
        if 'ci_low' in tc:
            ax.axvspan(tc['ci_low'], tc['ci_high'], alpha=0.15, color=COLORS['tipping_point'],
                       edgecolor='none', zorder=2, label='95% CI (threshold)')

    # 5. 主曲线
    ax.plot(xg, yp, color=COLORS['main'], linewidth=2.5, zorder=5)

    # 6. 数据散点
    ax.scatter(xr[sample_indices], yr[sample_indices], color=COLORS['data'],
               s=12, alpha=0.4, edgecolors='none', zorder=1)

    # 7. 临界点标记
    if res['thresholds']:
        for tc in res['thresholds']:
            ax.axvline(tc['x'], color=COLORS['tipping_point'], linestyle='--',
                       linewidth=1.5, alpha=0.8, zorder=6)
            ax.scatter(tc['x'], tc['y'], color=COLORS['tipping_point'], s=60,
                       zorder=6, edgecolor='white', linewidth=1)

            # 标注：临界值 + CI（智能避让位置）
            if 'ci_low' in tc:
                label = f'CV = {tc["x"]:.2f}\n95% CI [{tc["ci_low"]:.2f}, {tc["ci_high"]:.2f}]'
            else:
                label = f'CV = {tc["x"]:.2f}'

            # 根据阈值在x轴的位置选择标注偏移方向
            x_range = xg.max() - xg.min()
            if tc['x'] < xg.min() + x_range * 0.5:
                # 阈值在左侧，标注放右下
                xytext_offset = (20, -20)
                ha_align = 'left'
            else:
                # 阈值在右侧，标注放左下
                xytext_offset = (-20, -20)
                ha_align = 'right'

            if not is_panel:
                ax.annotate(label, xy=(tc['x'], tc['y']), xytext=xytext_offset,
                            textcoords='offset points', fontsize=9,
                            fontweight='bold', color=COLORS['tipping_point'],
                            ha=ha_align,
                            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                                      alpha=0.92, edgecolor=COLORS['tipping_point'], linewidth=1.2))

    # 8. 统计信息框（移到一个不拥挤的角落，根据曲线形状自适应）
    y_range = ci_u.max() - ci_l.min()
    # 找曲线上最空闲的角落
    if yp[-1] > 0:
        # 曲线右端在正区，放左上角
        ax.text(0.03, 0.97, f'$R^2$ = {res["R²"]:.3f}\n{res["p_text"]}\nedof = {res["edof"]:.1f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9.5, fontweight='bold', linespacing=1.6,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc',
                          boxstyle='round,pad=0.4', linewidth=0.8))
    else:
        # 放右上角
        ax.text(0.97, 0.97, f'$R^2$ = {res["R²"]:.3f}\n{res["p_text"]}\nedof = {res["edof"]:.1f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9.5, fontweight='bold', linespacing=1.6,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#cccccc',
                          boxstyle='round,pad=0.4', linewidth=0.8))

    # 9. 不确定性说明标注（面板图不显示，避免拥挤）
    if not is_panel:
        has_ci = any('ci_low' in tc for tc in res['thresholds'])
        if has_ci:
            unc_note = ('Uncertainty quantification:\n'
                        '  Blue band: 95% CI of GAM fit\n'
                        '  Pink band: 95% CI of threshold\n'
                        '  (Bootstrap N=500)')
            ax.text(0.03, 0.03, unc_note, transform=ax.transAxes,
                    fontsize=7, va='bottom', ha='left', color='#777777',
                    fontstyle='italic', linespacing=1.3,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='#dddddd',
                              boxstyle='round,pad=0.3', linewidth=0.5))

    # 10. 坐标轴
    ax.tick_params(which='minor', size=0)
    ax.set_xlabel(f'{feat}', labelpad=5, fontweight='bold')
    ax.set_ylabel('SHAP value (mm/min)', labelpad=5, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(2)


# ---- 逐特征绘制 ----
for res in gam_results:
    fig_s, ax_s = plt.subplots(figsize=(7.5, 5.8))
    sample_idx = np.random.choice(len(res['x_raw']), min(len(res['x_raw']), 27), replace=False)
    plot_gam_single(ax_s, res, sample_idx, is_panel=False)

    # 图例：避免与数据重叠，放最佳位置
    ax_s.legend(loc='best', fontsize=8.5, framealpha=0.85, edgecolor='#cccccc', ncol=2)

    plt.tight_layout(pad=1.2)
    save_fig(fig_s, f"Fig_GAM_{res['Feature'].replace(chr(8203),'').strip()}")
    plt.close(fig_s)
    print(f"  ✓ Fig_GAM_{res['Feature']}")

# ============================================================================
# 4. 综合面板图 2×4
# ============================================================================
print("绘制 GAM 面板图...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
axes_flat = axes.flatten()

for idx, (res, ax) in enumerate(zip(gam_results, axes_flat)):
    sample_idx = np.random.choice(len(res['x_raw']), min(len(res['x_raw']), 27), replace=False)
    plot_gam_single(ax, res, sample_idx, is_panel=True)
    # 面板图里不画图例和不确定性说明（避免拥挤）
    leg = ax.get_legend()
    if leg:
        leg.remove()

# 统一图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color=COLORS['main'], lw=2.5, label='GAM fit'),
    Patch(facecolor=COLORS['ci'], alpha=0.35, label='95% CI (GAM fit)'),
    Patch(facecolor=COLORS['positive'], alpha=0.2, label='Positive SHAP'),
    Patch(facecolor=COLORS['negative'], alpha=0.2, label='Negative SHAP'),
    Line2D([0],[0], color=COLORS['tipping_point'], lw=1.5, ls='--', label='Critical value (CV)'),
    Patch(facecolor=COLORS['tipping_point'], alpha=0.18, label='95% CI (CV, Bootstrap)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=9,
           frameon=True, edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.0))
plt.subplots_adjust(bottom=0.07, hspace=0.35, wspace=0.30)
save_fig(fig, "Fig_GAM_Panel")
plt.close()
print("  ✓ Fig_GAM_Panel")

# ============================================================================
# 5. 导出
# ============================================================================
export_rows = []
for res in gam_results:
    row = {'Feature': res['Feature'], 'R²': round(res['R²'], 4),
           'p_value': res['p_show'], 'edof': round(res['edof'], 1)}
    if res['thresholds']:
        tc = res['thresholds'][0]
        row['Threshold'] = round(tc['x'], 3)
        if 'ci_low' in tc:
            row['Thresh_95%_CI_Low'] = round(tc['ci_low'], 3)
            row['Thresh_95%_CI_High'] = round(tc['ci_high'], 3)
    export_rows.append(row)
pd.DataFrame(export_rows).to_excel("TabPFN_GAM_Results.xlsx", index=False)

print(f"\n{'Feature':<10} {'R²':>8} {'p':>12} {'edof':>5} {'CV':>10} {'95% CI':>28}")
print("-" * 82)
for res in gam_results:
    t_str, ci_str = '', ''
    if res['thresholds']:
        tc = res['thresholds'][0]
        t_str = f'{tc["x"]:.3f}'
        if 'ci_low' in tc:
            ci_str = f'[{tc["ci_low"]:.3f}, {tc["ci_high"]:.3f}]'
    print(f"{res['Feature']:<10} {res['R²']:>8.4f} {res['p_show']:>12} {res['edof']:>5.1f} {t_str:>10} {ci_str:>28}")

print("\n✅ GAM 完成")
