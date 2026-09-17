"""
=============================================================================
SHAP 瀑布图 — 高速样本 + 低速样本（复用已计算SHAP值，秒出）
- 反标准化到真实物理尺度 (mm/min)
- 论文级美观 (1200 DPI, Times New Roman, 无标题)
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from sklearn.preprocessing import StandardScaler
import os

# ============================================================================
# 0. 读取已计算的 SHAP 和原始数据
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"

# SHAP值
shap_df = pd.read_excel("TabPFN_SHAP_Analysis.xlsx", sheet_name="SHAP_Values")
importance_df = pd.read_excel("TabPFN_SHAP_Analysis.xlsx", sheet_name="Feature_Importance")
feature_names = importance_df['Feature'].tolist()
shap_arr = shap_df[feature_names].values
base_arr = shap_df['Base_Value'].values

# 原始数据
df = pd.read_excel(DATA_PATH).dropna().reset_index(drop=True)
X_all = df[feature_names].values.astype(np.float64)
y_all = df.iloc[:, -1].values.astype(np.float64)
n_samples = len(y_all)
split_idx = int(n_samples * 0.8)
ring_test_start = split_idx + 1

X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
n_test = len(y_test)

# 反标准化参数
scaler_X = StandardScaler(); scaler_y = StandardScaler()
scaler_X.fit(X_train)
scaler_y.fit(y_train.reshape(-1, 1))
X_mean_arr = scaler_X.mean_; X_std_arr = scaler_X.scale_
y_mean_val = scaler_y.mean_[0]; y_std_val = scaler_y.scale_[0]

# 被解释样本的全局索引
np.random.seed(42)
N_EXPLAIN = 27
explain_idx = np.sort(np.random.choice(n_test, min(N_EXPLAIN, n_test), replace=False))

# TabPFN 预测值
import torch
from tabpfn import TabPFNRegressor
MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
X_train_nor = scaler_X.transform(X_train)
X_test_nor = scaler_X.transform(X_test)
y_train_nor = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
model = TabPFNRegressor(model_path=MODEL_PATH, n_estimators=3, device="cpu", ignore_pretraining_limits=True)
model.fit(X_train_nor, y_train_nor)
y_pred_nor = model.predict(X_test_nor)
y_pred = scaler_y.inverse_transform(y_pred_nor.reshape(-1, 1)).ravel()

print(f"SHAP: {shap_arr.shape}, 被解释样本: {len(explain_idx)}")

# ============================================================================
# 1. 反标准化函数
# ============================================================================
def unnormalize_shap(shap_norm, feat_norm, base_norm):
    shap_u = shap_norm * y_std_val
    feat_u = feat_norm * X_std_arr + X_mean_arr
    base_u = base_norm * y_std_val + y_mean_val
    pred_u = base_u + shap_u.sum()
    return shap_u, feat_u, base_u, pred_u

# ============================================================================
# 2. 找高速/低速样本
# ============================================================================
shap_sum = shap_arr.sum(axis=1)
high_local = np.argmax(shap_sum)
low_local = np.argmin(shap_sum)
high_global = explain_idx[high_local]
low_global = explain_idx[low_local]

print(f"高速: ring={ring_test_start+high_global}, pred={y_pred[high_global]:.1f}, actual={y_test[high_global]:.1f}")
print(f"低速: ring={ring_test_start+low_global}, pred={y_pred[low_global]:.1f}, actual={y_test[low_global]:.1f}")

# ============================================================================
# 3. 全局绘图设置
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 12, 'axes.labelsize': 15, 'xtick.labelsize': 11,
    'ytick.labelsize': 12, 'legend.fontsize': 11,
    'axes.unicode_minus': False, 'axes.linewidth': 1.1,
    'figure.dpi': 200, 'savefig.dpi': 1200,
})

COLOR_POS = '#d9404d'   # SHAP正贡献—暖红
COLOR_NEG = '#3568a5'   # SHAP负贡献—深蓝
COLOR_BASE = '#555555'   # 基准线
COLOR_FX = '#111111'     # f(x)线

def save_fig(fig, name):
    fig.savefig(f"{name}.png", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    fig.savefig(f"{name}.pdf", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"  ✓ {name}.png / .pdf")


def plot_waterfall(shap_norm, feat_norm, base_norm, global_idx,
                   y_actual, sample_label, filename):
    """绘制反标准化后美观 Waterfall 图"""
    shap_u, feat_u, base_u, pred_u = unnormalize_shap(shap_norm, feat_norm, base_norm)

    # 按SHAP绝对值排序，取Top 12
    sort_idx = np.argsort(np.abs(shap_u))
    n_disp = min(12, len(shap_u))
    shap_disp = shap_u[sort_idx[-n_disp:]]
    names_disp = [feature_names[i] for i in sort_idx[-n_disp:]]
    vals_disp = feat_u[sort_idx[-n_disp:]]

    # ========= 绘制 =========
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    y_pos = np.arange(n_disp, 0, -1)

    # ========= 瀑布累积绘制 =========
    # 从 base_value 开始，逐特征累积
    running = base_u
    for i, (sv, name, val) in enumerate(zip(shap_disp, names_disp, vals_disp)):
        color = COLOR_POS if sv >= 0 else COLOR_NEG
        bar_start = running if sv >= 0 else running + sv  # sv为负时，条从左边开始
        ax.barh(y_pos[i], abs(sv), left=bar_start, height=0.62,
                color=color, alpha=0.85, edgecolor='white', linewidth=0.4, zorder=3)
        # SHAP值标注
        if abs(sv) > 0.15:
            ax.text(bar_start + abs(sv)/2, y_pos[i], f'{sv:+.2f}',
                    ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        # 连接线（除最后一个外）
        if i < n_disp - 1:
            next_running = running + sv
            ax.plot([running + sv, running + sv], [y_pos[i]-0.35, y_pos[i+1]+0.35],
                    color='#888888', linewidth=0.6, alpha=0.5, zorder=2)
        running += sv

    # 累积终点 = f(x)
    ax.axvline(x=base_u, color=COLOR_BASE, linewidth=1.5, linestyle='--',
               alpha=0.65, zorder=2)
    ax.axvline(x=pred_u, color=COLOR_FX, linewidth=1.6, linestyle='-',
               alpha=0.75, zorder=2)

    # E[f(x)] 和 f(x) 标注
    ymin, ymax = -0.6, n_disp + 0.4
    ax.text(base_u, ymax, f'E[f(x)] = {base_u:.2f}', fontsize=10,
            ha='center', va='bottom', fontweight='bold', color=COLOR_BASE)
    ax.text(pred_u, ymin + 0.1, f'f(x) = {pred_u:.2f}', fontsize=10,
            ha='center', va='bottom', fontweight='bold', color=COLOR_FX)

    # Y轴特征标签：特征名 = 原始值
    y_labels = []
    for name, val in zip(names_disp, vals_disp):
        # 格式化数值
        if abs(val) >= 100:
            val_str = f'{val:.0f}'
        elif abs(val) >= 10:
            val_str = f'{val:.1f}'
        else:
            val_str = f'{val:.2f}'
        y_labels.append(f'{name} = {val_str}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('SHAP Contribution to Advance Rate (mm/min)', fontsize=13)

    # 边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.18, linestyle='--')

    # ========= 右上角信息框 =========
    is_high = 'High' in sample_label
    speed_label = 'High-AR Case' if is_high else 'Low-AR Case'
    speed_color = COLOR_POS if is_high else COLOR_NEG
    ring_num = ring_test_start + global_idx
    delta = pred_u - y_actual

    info_lines = [
        f'{speed_label}',
        f'Ring: {ring_num}  |  Pred = {pred_u:.2f}',
        f'Actual = {y_actual:.2f}  |  \u0394 = {delta:+.2f}',
    ]
    info_text = '\n'.join(info_lines)
    ax.text(0.98, 0.96, info_text, transform=ax.transAxes, fontsize=10.5,
            va='top', ha='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=speed_color, alpha=0.93, linewidth=1.5),
            linespacing=1.5)

    # ========= 图例 =========
    legend_patches = [
        Patch(color=COLOR_POS, alpha=0.85, label=f'SHAP > 0  (Increases AR)'),
        Patch(color=COLOR_NEG, alpha=0.85, label=f'SHAP < 0  (Decreases AR)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10,
              framealpha=0.85, edgecolor='#cccccc', ncol=1)

    plt.tight_layout(pad=0.8)
    save_fig(fig, filename)
    plt.close(fig)


# ============================================================================
# 4. 生成两张瀑布图
# ============================================================================
print("\n绘制瀑布图...")

# 获取原始特征值（用于标注）
X_explain_nor = X_test_nor[explain_idx]

plot_waterfall(
    shap_arr[high_local], X_explain_nor[high_local], base_arr[high_local],
    high_global, y_test[high_global],
    'High AR Sample', 'Fig_SHAP_Waterfall_HighSpeed',
)

plot_waterfall(
    shap_arr[low_local], X_explain_nor[low_local], base_arr[low_local],
    low_global, y_test[low_global],
    'Low AR Sample', 'Fig_SHAP_Waterfall_LowSpeed',
)

print("\n✅ 瀑布图完成（秒出）")
