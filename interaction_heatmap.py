"""
=============================================================================
Feature Interaction Heatmap - Standalone, Beautiful
- Full numerical values in cells
- Category separators (Geological / Operational)
- 1200 DPI, Times New Roman
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pygam import LinearGAM, s
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
OUT_DIR = os.path.join(SCRIPT_DIR, "Interaction")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# 0. Data & interaction matrix
# ============================================================================
SHAP_PATH = "TabPFN_SHAP_Analysis.xlsx"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"

shap_df = pd.read_excel(SHAP_PATH, sheet_name="SHAP_Values")
features_all = [c for c in shap_df.columns if c not in ['Base_Value','Sample_Index']]
n_all = len(features_all)
shap_vals = shap_df[features_all].values

df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
X_full = df[features_all].values.astype(np.float64)
split_idx = int(len(df)*0.8)
np.random.seed(42)
ex = np.sort(np.random.choice(len(df)-split_idx, 27, replace=False))
X_ex = X_full[split_idx:][ex]

GEO = ['Es\u200b\u200b','CQ','CCQ','FAO','QIK']
SHIELD = ['CS','TTF','MDFR','MDD','PS','MIFR','MID']

def cat(f):
    return 'G' if f in GEO else 'S'

# Compute interaction matrix
print("Computing interaction matrix...")
R2 = np.zeros((n_all, n_all))
for i in range(n_all):
    yi = shap_vals[:, i]
    for j in range(n_all):
        if i == j: R2[i,j] = 1.0
        else:
            try:
                g = LinearGAM(s(0,n_splines=10,lam=1.0)).fit(X_ex[:,j].reshape(-1,1), yi)
                R2[i,j] = max(0, g.statistics_['pseudo_r2']['explained_deviance'])
            except: R2[i,j] = 0.0

I_mat = np.maximum(R2, R2.T)

# Sort: geological first, then operational
order = sorted(range(n_all), key=lambda i: (0 if cat(features_all[i])=='G' else 1, features_all[i]))
feats_sorted = [features_all[i] for i in order]
I_sorted = I_mat[np.ix_(order, order)]
n = len(feats_sorted)
geo_n = len(GEO)

short = [f.replace('\u200b','') for f in feats_sorted]

print("Features: {}".format(short))

# ============================================================================
# 1. Plot
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.unicode_minus': False,
    'figure.dpi': 200, 'savefig.dpi': 1200,
})

fig, ax = plt.subplots(figsize=(11, 9.5))

# ---- Colormap ----
cmap = plt.cm.YlOrRd
im = ax.imshow(np.ones((n,n)), cmap=cmap, vmin=0, vmax=1, alpha=0)  # dummy

# ---- Draw cells manually for full control ----
for i in range(n):
    for j in range(n):
        val = I_sorted[i, j]
        if i == j:
            face = '#f0f0f0'
            edge = '#dddddd'
            txt = '1.00'
            tc = '#aaaaaa'
            fs = 8
        else:
            face = cmap(val)
            edge = 'white'
            txt = '{:.2f}'.format(val)
            tc = 'white' if val > 0.45 else '#333333'
            fs = 9

        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=face,
                             edgecolor=edge, linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(j, i, txt, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold', zorder=3)

# ---- Category separators ----
ax.axhline(y=geo_n-0.5, color='#333333', lw=3, alpha=0.8, zorder=4)
ax.axvline(x=geo_n-0.5, color='#333333', lw=3, alpha=0.8, zorder=4)

# ---- Axis labels with category colors ----
for i, (f, s) in enumerate(zip(feats_sorted, short)):
    c = '#c0392b' if cat(f) == 'S' else '#2471a3'
    # y-axis
    ax.text(-0.8, i, s, fontsize=12, va='center', ha='right',
            color=c, fontweight='bold')
    # x-axis
    ax.text(i, n+0.6, s, fontsize=12, ha='right', va='bottom',
            color=c, fontweight='bold', rotation=45)

# ---- Category labels ----
mid_g = geo_n/2 - 0.5
mid_s = geo_n + (n-geo_n)/2 - 0.5

ax.text(mid_g, n+2.2, 'Geological', fontsize=14, ha='center', va='bottom',
        fontweight='bold', color='#2471a3')
ax.text(mid_s, n+2.2, 'Operational', fontsize=14, ha='center', va='bottom',
        fontweight='bold', color='#c0392b')
ax.text(-2.5, mid_g, 'Geological', fontsize=14, ha='center', va='center',
        fontweight='bold', color='#2471a3', rotation=90)
ax.text(-2.5, mid_s, 'Operational', fontsize=14, ha='center', va='center',
        fontweight='bold', color='#c0392b', rotation=90)

# ---- Colorbar ----
cbar_ax = fig.add_axes([0.93, 0.15, 0.025, 0.60])
cb_vals = np.linspace(0, 1, 256).reshape(-1, 1)
cbar_ax.imshow(cb_vals, aspect='auto', cmap=cmap, origin='lower',
               extent=[0, 1, 0, 1])
cbar_ax.set_xticks([])
cbar_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
cbar_ax.set_yticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'], fontsize=11)
cbar_ax.set_ylabel('Interaction Strength (GAM R$^2$)', fontsize=13, labelpad=8)
cbar_ax.yaxis.set_label_position('right')
cbar_ax.yaxis.tick_right()

# ---- Clean up ----
ax.set_xlim(-1.2, n+0.3)
ax.set_ylim(n-0.2, -1.2)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)

fig.savefig(os.path.join(OUT_DIR, 'Fig_Interaction_Heatmap.png'), dpi=1200,
            bbox_inches='tight', facecolor='white', pad_inches=0.2)
fig.savefig(os.path.join(OUT_DIR, 'Fig_Interaction_Heatmap.pdf'), dpi=1200,
            bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.close(fig)

print('Done -> ' + OUT_DIR + '/Fig_Interaction_Heatmap.png/pdf')
