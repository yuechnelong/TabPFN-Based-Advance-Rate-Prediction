"""
=============================================================================
SHAP 蜂群图+柱状图 融合为一张图（复用已计算SHAP值，秒出）
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# 0. 读取已计算的 SHAP 值
# ============================================================================
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"

# 从之前导出的 Excel 读取 SHAP 值
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

shap_df = pd.read_excel("TabPFN_SHAP_Analysis.xlsx", sheet_name="SHAP_Values")
importance_df = pd.read_excel("TabPFN_SHAP_Analysis.xlsx", sheet_name="Feature_Importance")

feature_names = importance_df['Feature'].tolist()
base_values = shap_df['Base_Value'].values
shap_arr = shap_df[feature_names].values

# 读取原始特征值（用于着色）
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
X_all = df[feature_names].values.astype(np.float64)
split_idx = int(len(X_all) * 0.8)
n_test = len(X_all) - split_idx
np.random.seed(42)
N_EXPLAIN = 27
explain_idx = np.sort(np.random.choice(n_test, min(N_EXPLAIN, n_test), replace=False))
X_explain_raw = X_all[split_idx:][explain_idx]  # 原始尺度特征值

# ============================================================================
# 1. 排序 + 选 Top 特征
# ============================================================================
mean_abs = np.abs(shap_arr).mean(axis=0)
top_k = min(12, len(feature_names))
top_idx = np.argsort(mean_abs)[::-1][:top_k]
top_features = [feature_names[i] for i in top_idx]
top_mean_abs = mean_abs[top_idx]
top_shap = shap_arr[:, top_idx]
top_feat_raw = X_explain_raw[:, top_idx]

print(f"SHAP values: {shap_arr.shape}, Explain samples: {len(explain_idx)}")
print("Top features:")
for rank, (f, v) in enumerate(zip(top_features, top_mean_abs)):
    print(f"  {rank+1:2d}. {f:8s} = {v:.4f}")

# ============================================================================
# 2. 绘制融合图
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.labelsize': 15, 'xtick.labelsize': 10,
    'ytick.labelsize': 12, 'legend.fontsize': 10,
    'axes.unicode_minus': False, 'axes.linewidth': 1.0,
})

fig = plt.figure(figsize=(10, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 6.8], wspace=0.02)
ax_left = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)

# ============ 左侧：Mean|SHAP| 柱状图 ============
y_ticks = np.arange(top_k)
bar_colors = plt.cm.RdBu_r(
    plt.Normalize(top_mean_abs.min(), top_mean_abs.max())(top_mean_abs))

ax_left.barh(y_ticks, top_mean_abs, height=0.72, color=bar_colors,
             alpha=0.88, edgecolor='white', linewidth=0.5, zorder=3)

for i, v in enumerate(top_mean_abs):
    ax_left.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10,
                 color='#333333')

ax_left.set_xlabel('Mean |SHAP value|')
ax_left.set_yticks(y_ticks)
ax_left.set_yticklabels(top_features, fontsize=12)
ax_left.invert_yaxis()
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.spines['left'].set_visible(False)
ax_left.tick_params(left=False)
ax_left.grid(axis='x', alpha=0.2, linestyle='--')
ax_left.set_xlim(0, top_mean_abs.max() * 1.22)

# 排名小标
for i in range(top_k):
    ax_left.text(-0.005, i, f'#{i+1}', va='center', ha='right',
                 fontsize=7, color='#aaaaaa', fontstyle='italic')

# ============ 右侧：Beeswarm 散点 ============
for i in range(top_k):
    feat_shap_vals = top_shap[:, i]
    feat_raw_vals = top_feat_raw[:, i]
    n_pts = len(feat_shap_vals)

    jitter = np.zeros(n_pts)
    if n_pts > 1:
        order = np.argsort(feat_shap_vals)
        for j in range(n_pts):
            jitter[order[j]] = np.sin(j / n_pts * np.pi) * 0.38

    vmin, vmax = np.percentile(feat_raw_vals, [5, 95])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.RdYlBu_r(norm(feat_raw_vals))

    ax_right.scatter(feat_shap_vals, i + jitter, c=colors, s=28,
                     alpha=0.72, edgecolors='none', zorder=3)

# 零线
ax_right.axvline(x=0, color='#333333', linewidth=1.0, linestyle='-', alpha=0.5, zorder=2)

# Colorbar
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.55])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Feature\nValue', fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=9)
cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes,
             fontsize=8, ha='center', fontstyle='italic', color='#e05263')
cbar.ax.text(0.5, -0.06, 'Low', transform=cbar.ax.transAxes,
             fontsize=8, ha='center', fontstyle='italic', color='#3b5b92')

ax_right.set_xlabel('SHAP value')
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.tick_params(left=False)
ax_right.grid(axis='x', alpha=0.2, linestyle='--')

ax_right.set_ylim(-0.7, top_k - 0.3)

# 底部图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e05263',
           markersize=8, label='High feature value', markeredgewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b5b92',
           markersize=8, label='Low feature value', markeredgewidth=0),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=10, frameon=True, edgecolor='#cccccc',
           bbox_to_anchor=(0.55, -0.01))

plt.subplots_adjust(bottom=0.08, top=0.97, left=0.08, right=0.90)

fig.savefig("Fig_SHAP_Combined.png", dpi=1200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
fig.savefig("Fig_SHAP_Combined.pdf", dpi=1200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
print("\n  ✓ Fig_SHAP_Combined.png / .pdf")
plt.close()
print("✅ 完成（秒出）")
