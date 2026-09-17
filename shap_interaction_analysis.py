"""
=============================================================================
SHAP Feature Interaction Analysis - Final
- Heatmap: beautiful, sorted by category
- Top 9 Dependence plots: labeled (a)-(i), sorted geo-geo, shield-geo, shield-shield
- 1200 DPI, Times New Roman
- Output: Interaction/
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pygam import LinearGAM, s
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
OUT_DIR = os.path.join(SCRIPT_DIR, "Interaction")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# 0. Data
# ============================================================================
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
SHAP_PATH = "TabPFN_SHAP_Analysis.xlsx"

shap_df = pd.read_excel(SHAP_PATH, sheet_name="SHAP_Values")
features_all = [c for c in shap_df.columns if c not in ['Base_Value', 'Sample_Index']]
n_all = len(features_all)
shap_vals = shap_df[features_all].values

df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
X_full = df[features_all].values.astype(np.float64)
split_idx = int(len(df) * 0.8)
np.random.seed(42)
N_EXPLAIN = 27
ex = np.sort(np.random.choice(len(df)-split_idx, N_EXPLAIN, replace=False))
X_ex = X_full[split_idx:][ex]

# Categorize features
GEO_FEATS = ['Es\u200b\u200b', 'CQ', 'CCQ', 'FAO', 'QIK']
SHIELD_FEATS = ['CS', 'TTF', 'MDFR', 'MDD', 'PS', 'MIFR', 'MID']

def feat_category(f):
    return 'G' if f in GEO_FEATS else 'S'

# ============================================================================
# 1. Global style
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 13, 'axes.labelsize': 15, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 10,
    'axes.unicode_minus': False, 'axes.linewidth': 1.3,
    'figure.dpi': 200, 'savefig.dpi': 1200,
})

def save_fig(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(OUT_DIR, name + '.' + ext), dpi=1200,
                    bbox_inches='tight', facecolor='white', pad_inches=0.15)

# ============================================================================
# 2. Interaction matrix (GAM R2)
# ============================================================================
print("Computing interaction matrix...")
R2 = np.zeros((n_all, n_all))
for i in range(n_all):
    yi = shap_vals[:, i]
    for j in range(n_all):
        if i == j:
            R2[i,j] = 1.0
        else:
            xj = X_ex[:, j]
            try:
                g = LinearGAM(s(0, n_splines=10, lam=1.0)).fit(xj.reshape(-1,1), yi)
                R2[i,j] = max(0, g.statistics_['pseudo_r2']['explained_deviance'])
            except:
                R2[i,j] = 0.0

I_strength = np.maximum(R2, R2.T)

# ---- Sort features by category ----
cat_order = [(f, feat_category(f)) for f in features_all]
sorted_idx = sorted(range(n_all), key=lambda i: (0 if cat_order[i][1]=='G' else 1, features_all[i]))
sorted_feats = [features_all[i] for i in sorted_idx]
sorted_I = I_strength[np.ix_(sorted_idx, sorted_idx)]

# Short labels
short_names = [f.replace('\u200b','') for f in sorted_feats]
n = len(sorted_feats)

# ---- Generate all interaction pairs ----
all_pairs = []
for i in range(n):
    for j in range(i+1, n):
        fi, fj = sorted_feats[i], sorted_feats[j]
        ci, cj = feat_category(fi), feat_category(fj)
        if ci == 'G' and cj == 'G':
            cat = 'geo-geo'
        elif (ci == 'G' and cj == 'S') or (ci == 'S' and cj == 'G'):
            cat = 'shield-geo'
        else:
            cat = 'shield-shield'
        all_pairs.append((i, j, fi, fj, sorted_I[i,j], cat))

# Sort: geo-geo first, then shield-geo, then shield-shield; within each, by strength desc
cat_rank = {'geo-geo': 0, 'shield-geo': 1, 'shield-shield': 2}
all_pairs_sorted = sorted(all_pairs, key=lambda x: (cat_rank[x[5]], -x[4]))

# Pick top 3 from each category
top_pairs = []
for cat in ['geo-geo', 'shield-geo', 'shield-shield']:
    cat_pairs = [p for p in all_pairs_sorted if p[5] == cat]
    top_pairs.extend(cat_pairs[:3])
# Ensure exactly 9
top_pairs = top_pairs[:9]

print("\nTop {} interactions (ordered by category):".format(len(top_pairs)))
for idx, (i, j, fi, fj, s, cat) in enumerate(top_pairs):
    print("  ({}) {} x {}: R2={:.3f} [{}]".format(
        chr(ord('a')+idx), fi, fj, s, cat))


# ============================================================================
# 3. Beautiful Heatmap
# ============================================================================
print("\nPlotting heatmap...")

fig_hm, ax_hm = plt.subplots(figsize=(10.5, 9))

# Category boundaries
geo_n = len(GEO_FEATS)
shield_n = len(SHIELD_FEATS)

# Draw category separators
ax_hm.axhline(y=geo_n - 0.5, color='#333333', lw=2.5, alpha=0.7)
ax_hm.axvline(x=geo_n - 0.5, color='#333333', lw=2.5, alpha=0.7)

# Mask diagonal
mask = np.eye(n, dtype=bool)
masked = np.ma.array(sorted_I, mask=mask)

im = ax_hm.imshow(masked, cmap='YlOrRd', vmin=0, vmax=0.8, aspect='auto')

# Labels with category coloring
for idx, (f, sname) in enumerate(zip(sorted_feats, short_names)):
    cat = feat_category(f)
    c = '#c0392b' if cat == 'S' else '#2471a3'
    ax_hm.text(ax_hm.get_xlim()[1] + 0.3, idx, sname, fontsize=11,
               va='center', ha='left', color=c, fontweight='bold')
    ax_hm.text(idx, ax_hm.get_ylim()[1] + 0.3, sname, fontsize=11,
               ha='right', va='bottom', color=c, fontweight='bold',
               rotation=45)

# Remove axis ticks (labels are above)
ax_hm.set_xticks([]); ax_hm.set_yticks([])

# Value annotations (only for top interactions)
top_set = set()
for i, j, fi, fj, s, cat in top_pairs:
    top_set.add((i, j))
    top_set.add((j, i))

for i in range(n):
    for j in range(n):
        if i != j and (i, j) in top_set:
            val = sorted_I[i, j]
            color = 'white' if val > 0.4 else '#333333'
            ax_hm.text(j, i, '{:.2f}'.format(val), ha='center', va='center',
                       fontsize=7.5, color=color, fontweight='bold')

# Category labels - use axes fraction coords
ax_hm.annotate('Geological', xy=(geo_n/2-0.5, n), xytext=(geo_n/2-0.5, n+1.2),
               ha='center', fontsize=13, fontweight='bold', color='#2471a3',
               annotation_clip=False)
ax_hm.annotate('Operational', xy=(geo_n+shield_n/2-0.5, n), xytext=(geo_n+shield_n/2-0.5, n+1.2),
               ha='center', fontsize=13, fontweight='bold', color='#c0392b',
               annotation_clip=False)
ax_hm.annotate('Geological', xy=(-1, geo_n/2-0.5), xytext=(-1.8, geo_n/2-0.5),
               ha='center', fontsize=13, fontweight='bold', color='#2471a3', rotation=90,
               annotation_clip=False, va='center')
ax_hm.annotate('Operational', xy=(-1, geo_n+shield_n/2-0.5), xytext=(-1.8, geo_n+shield_n/2-0.5),
               ha='center', fontsize=13, fontweight='bold', color='#c0392b', rotation=90,
               annotation_clip=False, va='center')

# Colorbar
cbar = plt.colorbar(im, ax=ax_hm, shrink=0.78, pad=0.03)
cbar.set_label('Interaction Strength (GAM R$^2$)', fontsize=14)
cbar.ax.tick_params(labelsize=11)

# Remove frame
for sp in ax_hm.spines.values():
    sp.set_visible(False)

plt.subplots_adjust(left=0.18, right=0.76, top=0.85, bottom=0.16)
save_fig(fig_hm, 'Fig_Interaction_Heatmap')
plt.close(fig_hm)
print('  + Fig_Interaction_Heatmap')

# ============================================================================
# 4. \u4e09\u5f20\u5206\u7c7b\u56fe + \u4e5d\u5f20\u5355\u56fe\uff08\u5927\u5b57\u4f53\uff0c\u65e0\u5185\u5d4ccolorbar\uff09
# ============================================================================

def draw_interaction_scatter(ax, i, j, fi, fj, s, cat, show_colorbar=False):
    """Single interaction scatter plot - large fonts, clean"""
    xi_vals = X_ex[:, sorted_idx[i]]
    shap_i = shap_vals[:, sorted_idx[i]]
    xj_vals = X_ex[:, sorted_idx[j]]

    vmin, vmax = np.percentile(xj_vals, [5, 95])
    colors = plt.cm.RdYlBu_r(plt.Normalize(vmin, vmax)(xj_vals))

    ax.scatter(xi_vals, shap_i, c=colors, s=38, alpha=0.72,
               edgecolors='white', linewidths=0.4, zorder=3)

    # GAM trend
    try:
        gam = LinearGAM(s(0, n_splines=8, lam=1.0)).fit(xi_vals.reshape(-1,1), shap_i)
        XX = gam.generate_X_grid(term=0, n=200)
        ax.plot(XX[:,0], gam.predict(XX), '-', color='#333333', lw=3.0, alpha=0.9, zorder=4)
    except: pass

    ax.axhline(0, color='#aaaaaa', lw=0.8, ls='-', alpha=0.4, zorder=2)

    # Category border
    cat_colors = {'geo-geo': '#2471a3', 'shield-geo': '#8e44ad', 'shield-shield': '#c0392b'}
    for sp_name in ['bottom', 'left']:
        ax.spines[sp_name].set_color(cat_colors[cat])
        ax.spines[sp_name].set_linewidth(4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(fi.replace('\u200b',''), fontsize=16, fontweight='bold')
    ax.set_ylabel('SHAP({})'.format(fi.replace('\u200b','')), fontsize=16, fontweight='bold')
    ax.grid(alpha=0.10, linestyle='--')
    ax.tick_params(labelsize=14)

    return vmin, vmax


# ---- 4a: Three category figures (1x3 each) ----
print("Plotting 3 category figures...")
categories = [
    ('geo-geo', 'Geological-Geological', [p for p in top_pairs if p[5]=='geo-geo']),
    ('shield-geo', 'Shield-Geological', [p for p in top_pairs if p[5]=='shield-geo']),
    ('shield-shield', 'Shield-Shield', [p for p in top_pairs if p[5]=='shield-shield']),
]
cat_colors = {'geo-geo': '#2471a3', 'shield-geo': '#8e44ad', 'shield-shield': '#c0392b'}

for cat_key, cat_title, cat_pairs in categories:
    n_p = len(cat_pairs)
    fig_cat, axes_cat = plt.subplots(1, n_p, figsize=(5.5*n_p, 5.2))
    if n_p == 1:
        axes_cat = [axes_cat]

    global_vmin, global_vmax = np.inf, -np.inf

    for ax, (i, j, fi, fj, s, cat) in zip(axes_cat, cat_pairs):
        vmin, vmax = draw_interaction_scatter(ax, i, j, fi, fj, s, cat)
        global_vmin = min(global_vmin, vmin)
        global_vmax = max(global_vmax, vmax)

    # Labels
    for k, ax in enumerate(axes_cat):
        sub_label = '({})'.format(chr(ord('a')+top_pairs.index(cat_pairs[k])))
        ax.text(0.02, 0.97, sub_label, transform=ax.transAxes, fontsize=16,
                ha='left', va='top', fontweight='bold', color='#111111')

    # Unified colorbar for the row
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                norm=plt.Normalize(global_vmin, global_vmax))
    sm.set_array([])
    cbar = fig_cat.colorbar(sm, ax=axes_cat, shrink=0.75, pad=0.02,
                            location='right')
    cbar.set_label('Interacting Feature Value', fontsize=13, labelpad=8)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout(pad=1.5)
    fname = 'Fig_Interaction_Category_{}'.format(cat_key)
    fig_cat.savefig(os.path.join(OUT_DIR, fname + '.png'), dpi=600,
                    bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close(fig_cat)
    print('  + ' + fname)


# ---- 4b: Nine individual large figures ----
print("Plotting 9 individual figures...")
for idx, (i, j, fi, fj, s, cat) in enumerate(top_pairs):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    vmin, vmax = draw_interaction_scatter(ax, i, j, fi, fj, s, cat, show_colorbar=True)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(fj.replace('\u200b',''), fontsize=16, labelpad=8)
    cb.ax.tick_params(labelsize=14)

    plt.tight_layout()
    fname = 'Fig_Interaction_{}_{}_vs_{}'.format(
        chr(ord('a')+idx), fi.replace(chr(8203),''), fj.replace(chr(8203),''))
    save_fig(fig, fname)
    plt.close(fig)
    print('  + ' + fname)

print('\nDone -> ' + OUT_DIR)
rows = []
for i, j, fi, fj, s, cat in all_pairs_sorted:
    rows.append({
        'Feature_1': fi, 'Feature_2': fj,
        'Category': cat,
        'Interaction_R2': round(s, 4),
    })
pd.DataFrame(rows).to_excel(os.path.join(OUT_DIR, 'Interaction_Matrix.xlsx'), index=False)

print('\nDone -> ' + OUT_DIR)
