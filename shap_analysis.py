"""
=============================================================================
SHAP 可解释性分析（顺序划分 + TabPFN 原生 SHAP 加速）
- 蜂群图 (Beeswarm)：全局特征重要性 + 影响方向
- 瀑布图 (Waterfall)：高速样本 + 低速样本各一例
- 论文级美观输出 (1200 DPI, Times New Roman, 无标题)
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
import torch
from tabpfn import TabPFNRegressor
from tabpfn_extensions.interpretability import shap as tabpfn_shap
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. 配置
# ============================================================================
MODEL_PATH = r"D:\岳的python机器学习3\tabpfn-v2.5-regressor-v2.5_real.ckpt"
DATA_PATH = r"D:\岳的python机器学习3\1.0tust返稿\测试.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_EXPLAIN = 27  # 被解释的测试样本数（采样）

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

split_idx = int(n_samples * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
ring_test_start = split_idx + 1

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_nor = scaler_X.fit_transform(X_train)
X_test_nor = scaler_X.transform(X_test)
y_train_nor = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

print(f"训练: 环 1-{split_idx} ({len(y_train)} samples)")
print(f"测试: 环 {split_idx+1}-{n_samples} ({len(y_test)} samples)")

# ============================================================================
# 2. 训练 TabPFN
# ============================================================================
print("训练 TabPFN...")
model = TabPFNRegressor(
    model_path=MODEL_PATH, n_estimators=3,
    device=DEVICE, ignore_pretraining_limits=True,
)
model.fit(X_train_nor, y_train_nor)

y_pred_nor = model.predict(X_test_nor)
y_pred = scaler_y.inverse_transform(y_pred_nor.reshape(-1, 1)).ravel()

# ============================================================================
# 3. 采样测试样本（用于 SHAP 解释）
# ============================================================================
n_test = len(y_test)
explain_indices = np.random.choice(n_test, min(N_EXPLAIN, n_test), replace=False)
explain_indices = np.sort(explain_indices)
X_explain_nor = X_test_nor[explain_indices]
print(f"SHAP 解释样本: {len(explain_indices)} 个")

# ============================================================================
# 4. SHAP 计算（TabPFN 原生 Permutation SHAP）
# ============================================================================
print("计算 SHAP 值 (TabPFN 原生 Permutation SHAP)...")

# 背景数据采样
n_bg = min(100, len(X_train_nor))
bg_idx_nor = np.random.choice(len(X_train_nor), n_bg, replace=False)
X_bg_nor = X_train_nor[bg_idx_nor]

shap_values_list = []
base_values_list = []

for i in tqdm(range(len(explain_indices)), desc="SHAP"):
    result = tabpfn_shap.get_shap_values(
        estimator=model,
        train_x=X_bg_nor,
        test_x=X_explain_nor[i:i + 1],
        attribute_names=feature_names,
        algorithm="permutation",
    )
    shap_values_list.append(result.values[0])
    base_values_list.append(result.base_values[0])

shap_values_array = np.vstack(shap_values_list)
base_values_array = np.array(base_values_list)

print(f"SHAP values shape: {shap_values_array.shape}")

# ============================================================================
# 5. 全局绘图设置
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 13,
    'axes.labelsize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 1200,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.1,
})

SAVE_DIR = "."

def save_fig(fig, name):
    fig.savefig(f"{SAVE_DIR}/{name}.png", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    fig.savefig(f"{SAVE_DIR}/{name}.pdf", dpi=1200, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.15)
    print(f"  ✓ {name}.png / .pdf")


# ============================================================================
# 6. 计算 mean|SHAP| + 选 Top 特征
# ============================================================================
mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
top_k = min(12, len(feature_names))
top_idx = np.argsort(mean_abs_shap)[::-1][:top_k]
top_features = [feature_names[i] for i in top_idx]

print(f"\nTop {top_k} 特征 (by mean|SHAP|):")
for rank, (f, v) in enumerate(zip(top_features, mean_abs_shap[top_idx])):
    print(f"  {rank+1:2d}. {f:20s} = {v:.4f}")

# ============================================================================
# 7. 蜂群图 (Beeswarm)
# ============================================================================
print("\n绘制 SHAP Beeswarm 蜂群图...")

shap_exp = shap.Explanation(
    values=shap_values_array[:, top_idx],
    base_values=base_values_array,
    data=X_explain_nor[:, top_idx],
    feature_names=top_features,
)

fig_bee = plt.figure(figsize=(9, 5.5))
shap.summary_plot(
    shap_values_array[:, top_idx],
    X_explain_nor[:, top_idx],
    feature_names=top_features,
    plot_type="dot",
    show=False,
    max_display=top_k,
)

ax_bee = plt.gca()
ax_bee.spines['top'].set_visible(False)
ax_bee.spines['right'].set_visible(False)
ax_bee.tick_params(axis='both', labelsize=11)
# 调整 colorbar label
for c in ax_bee.get_children():
    if isinstance(c, plt.cm.ScalarMappable):
        try:
            c.colorbar.set_label('Feature Value', fontsize=12)
        except:
            pass

plt.tight_layout()
save_fig(fig_bee, "Fig_SHAP_Beeswarm")
plt.close(fig_bee)

# ============================================================================
# 8. Bar 图（特征重要性）
# ============================================================================
print("绘制 SHAP Bar 图...")

sorted_idx_bar = np.argsort(mean_abs_shap[top_idx])
bar_features = [top_features[i] for i in sorted_idx_bar]
bar_values = mean_abs_shap[top_idx][sorted_idx_bar]

fig_bar, ax_bar = plt.subplots(figsize=(7, 5.5))
cmap = plt.get_cmap('RdBu_r')
norm = plt.Normalize(vmin=bar_values.min(), vmax=bar_values.max())
colors_bar = cmap(norm(bar_values))

ax_bar.barh(range(len(bar_values)), bar_values, color=colors_bar,
            alpha=0.85, edgecolor='white', linewidth=0.8, height=0.7)
ax_bar.set_yticks(range(len(bar_values)))
ax_bar.set_yticklabels(bar_features, fontsize=12)
ax_bar.set_xlabel('Mean |SHAP value|', fontsize=14)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.grid(axis='x', alpha=0.2, linestyle='--')

for i, v in enumerate(bar_values):
    ax_bar.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=10, color='#333333')

plt.tight_layout()
save_fig(fig_bar, "Fig_SHAP_Bar_Importance")
plt.close(fig_bar)

# ============================================================================
# 9. 反标准化函数
# ============================================================================
X_mean_arr = scaler_X.mean_
X_std_arr = scaler_X.scale_
y_mean_val = scaler_y.mean_[0]
y_std_val = scaler_y.scale_[0]


def unnormalize_shap(shap_vals_norm, feat_vals_norm, base_val_norm):
    shap_vals_unnorm = shap_vals_norm * y_std_val
    feat_vals_unnorm = feat_vals_norm * X_std_arr + X_mean_arr
    base_val_unnorm = base_val_norm * y_std_val + y_mean_val
    pred_unnorm = base_val_unnorm + shap_vals_unnorm.sum()
    return shap_vals_unnorm, feat_vals_unnorm, base_val_unnorm, pred_unnorm


# ============================================================================
# 10. 瀑布图 — 高速 + 低速样本
# ============================================================================
print("绘制 SHAP Waterfall 瀑布图...")

# 基于 SHAP 贡献总和找高速/低速样本
shap_sum = shap_values_array.sum(axis=1)
high_local_idx = np.argmax(shap_sum)
low_local_idx = np.argmin(shap_sum)

# 全局索引
high_global_idx = explain_indices[high_local_idx]
low_global_idx = explain_indices[low_local_idx]

print(f"  高速样本: ring={ring_test_start + high_global_idx}, "
      f"pred={y_pred[high_global_idx]:.2f}, actual={y_test[high_global_idx]:.2f} mm/min")
print(f"  低速样本: ring={ring_test_start + low_global_idx}, "
      f"pred={y_pred[low_global_idx]:.2f}, actual={y_test[low_global_idx]:.2f} mm/min")


def plot_waterfall(shap_norm, feat_norm, base_norm, feat_names,
                   global_idx, y_actual, y_pred_val, sample_label, filename):
    """绘制反标准化后美观 Waterfall 图"""
    shap_u, feat_u, base_u, pred_u = unnormalize_shap(shap_norm, feat_norm, base_norm)

    # 按贡献绝对值排序，取 Top 12
    sort_idx = np.argsort(np.abs(shap_u))
    n_disp = min(12, len(shap_u))
    shap_disp = shap_u[sort_idx[-n_disp:]]
    names_disp = [feat_names[i] for i in sort_idx[-n_disp:]]
    vals_disp = [f'{feat_u[i]:.2f}' for i in sort_idx[-n_disp:]]
    colors_wf = ['#e05263' if v > 0 else '#3b5b92' for v in shap_disp]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ypos = list(range(n_disp, 0, -1))
    ax.barh(ypos, shap_disp, color=colors_wf, alpha=0.85,
            edgecolor='white', linewidth=0.6, height=0.65, zorder=3)

    # 基准线 E[f(x)]
    ax.axvline(x=base_u, color='#444444', linewidth=1.6, linestyle='--', alpha=0.7, zorder=2)
    ax.text(base_u, n_disp + 0.55, f'E[f(x)]={base_u:.2f}', fontsize=10,
            ha='center', fontweight='bold', color='#555555')

    # 预测值 f(x)
    ax.axvline(x=pred_u, color='#111111', linewidth=1.8, linestyle='-', alpha=0.8, zorder=2)
    ax.text(pred_u, -0.15, f'f(x)={pred_u:.2f}', fontsize=10,
            ha='center', fontweight='bold', color='#111111')

    # Y 轴标签
    y_labels = [f'{n} = {v}' for n, v in zip(names_disp, vals_disp)]
    ax.set_yticks(ypos)
    ax.set_yticklabels(y_labels, fontsize=11)
    ax.set_xlabel('SHAP Contribution to AR (mm/min)', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2, linestyle='--')

    # 速度标签
    speed_label = 'High AR' if 'High' in sample_label else 'Low AR'
    speed_color = '#e05263' if 'High' in sample_label else '#3b5b92'

    # 信息框
    info = (f'{speed_label}\n'
            f'Ring: {ring_test_start + global_idx}\n'
            f'Predicted = {pred_u:.1f} mm/min\n'
            f'Measured = {y_actual:.1f} mm/min\n'
            f'Δ = {pred_u - y_actual:+.1f} mm/min')
    ax.text(0.98, 0.96, info, transform=ax.transAxes, fontsize=11,
            va='top', ha='right', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=speed_color, alpha=0.92, linewidth=1.5))

    # 图例
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color='#e05263', alpha=0.7, label='SHAP > 0 (Increases AR)'),
        Patch(color='#3b5b92', alpha=0.7, label='SHAP < 0 (Decreases AR)'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=10,
              framealpha=0.85, edgecolor='#cccccc')

    plt.tight_layout()
    save_fig(fig, filename)
    plt.close(fig)


# 高速样本
plot_waterfall(shap_values_array[high_local_idx], X_explain_nor[high_local_idx],
               base_values_array[high_local_idx], feature_names,
               high_global_idx, y_test[high_global_idx], y_pred[high_global_idx],
               'High AR Sample', 'Fig_SHAP_Waterfall_HighSpeed')

# 低速样本
plot_waterfall(shap_values_array[low_local_idx], X_explain_nor[low_local_idx],
               base_values_array[low_local_idx], feature_names,
               low_global_idx, y_test[low_global_idx], y_pred[low_global_idx],
               'Low AR Sample', 'Fig_SHAP_Waterfall_LowSpeed')

# ============================================================================
# 11. 导出结果文件
# ============================================================================
print("\n导出 SHAP 结果...")

# Excel
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_ABS_SHAP': mean_abs_shap,
    'Rank': np.argsort(np.argsort(-mean_abs_shap)) + 1,
}).sort_values('Mean_ABS_SHAP', ascending=False)

shap_df = pd.DataFrame(shap_values_array, columns=feature_names)
shap_df['Base_Value'] = base_values_array

with pd.ExcelWriter(f"{SAVE_DIR}/TabPFN_SHAP_Analysis.xlsx", engine='openpyxl') as writer:
    importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
    shap_df.to_excel(writer, sheet_name='SHAP_Values', index=False)

    for label, local_idx in [('HighSpeed', high_local_idx), ('LowSpeed', low_local_idx)]:
        shap_u, feat_u, base_u, pred_u = unnormalize_shap(
            shap_values_array[local_idx], X_explain_nor[local_idx],
            base_values_array[local_idx])
        detail = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Norm': np.round(shap_values_array[local_idx], 6),
            'SHAP_Real(mm/min)': np.round(shap_u, 4),
            'Feature_Value_Raw': np.round(X_test[explain_indices[local_idx]], 4),
        }).sort_values('SHAP_Real(mm/min)', key=abs, ascending=False)
        detail.to_excel(writer, sheet_name=f'{label}_Waterfall', index=False)

print(f"  ✓ TabPFN_SHAP_Analysis.xlsx")

# 特征重要性排序 LaTeX
print(f"\n论文 SHAP 特征重要性排序:")
print(f"{'Rank':<6} {'Feature':<25} {'Mean|SHAP|':<14}")
print("-" * 48)
for i, row in importance_df.iterrows():
    print(f"{int(row['Rank']):<6} {row['Feature']:<25} {row['Mean_ABS_SHAP']:.6f}")

print(f"\n✅ SHAP 分析完成")
