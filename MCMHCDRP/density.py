import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 读取数据
df = pd.read_csv("output_dir/final_all_predictions_real_name.csv")

# ============ 图1: 密度散点图（True vs Predicted） ============ #
true = df["true_ic50"]
pred = df["predicted_ic50"]

# 1. 计算统计指标
r, _ = pearsonr(true, pred)  # Pearson 相关系数
r2 = r2_score(true, pred)    # 决定系数 R2
rmse = np.sqrt(mean_squared_error(true, pred)) # 均方根误差
mae = mean_absolute_error(true, pred)          # 平均绝对误差

# 2. 计算密度
xy = np.vstack([true, pred])
z = gaussian_kde(xy)(xy)

# 排序（更好颜色渐变效果）
idx = z.argsort()
x, y, z = true.values[idx], pred.values[idx], z[idx]

# 3. 创建图
plt.figure(figsize=(9, 7))
sc = plt.scatter(x, y, c=z, cmap='coolwarm', s=10, alpha=0.7)

# 绘制 y=x 参照线
min_val = min(true.min(), pred.min())
max_val = max(true.max(), pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Identity line')

# 4. 在图中添加指标文本框
# 坐标 (0.05, 0.95) 表示在左上角，transform=plt.gca().transAxes 确保坐标是相对图表的
stats_text = (
    f"$R^2 = {r2:.3f}$\n"
    f"Pearson $r = {r:.3f}$\n"
    f"RMSE = {rmse:.3f}$\n"
    f"MAE = {mae:.3f}"
)

plt.text(0.05, 0.95, stats_text,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# 图形修饰
plt.xlabel("True Label (IC50)", fontsize=14, weight='bold')
plt.ylabel("Prediction (IC50)", fontsize=14, weight='bold')
plt.title("Density Scatter Plot with Metrics", fontsize=16, weight='bold')

cb = plt.colorbar(sc)
cb.set_label('Density', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存并显示
plt.savefig("12.29density_scatter_metrics.png", dpi=400)
plt.show()