import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'  # 同样指定字体
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
# plt.rc('font',family='Times New Roman')


# 数据
data = [
    {"x": -3.8, "y": 0.65, "label": "BERT-whitening (66.3)", "color": 66.3},
    {"x": -3.7, "y": 0.55, "label": "BERT-flow (66.6)", "color": 66.6},
    {"x": -3.8, "y": 0.5, "label": "SBERT-whitening (77.0)", "color": 77.0},
    {"x": -3.7, "y": 0.45, "label": "SBERT-flow (76.6)", "color": 76.6},
    {"x": -3.75, "y": 0.35, "label": "ConSERT (72.74)", "color": 72.74},
    {"x": -2.72, "y": 0.22, "label": "SimCSE (76.3)", "color": 76.3},
    {"x": -1.5, "y": 0.1, "label": "", "color": 78.49},
    {"x": -1.8, "y": 0.12, "label": "", "color": 80.36},
    {"x": -1.51, "y": 0.19, "label": "", "color": 56.70},
    {"x": -1.9, "y": 0.15, "label": "", "color": 82.02},
    {"x": -3.0, "y": 0.18, "label": "SBERT (74.9)", "color": 74.9}
]

# 提取数据
x = [point["x"] for point in data]
y = [point["y"] for point in data]
labels = [point["label"] for point in data]
colors = [point["color"] for point in data]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制散点图
sc = ax.scatter(x, y, c=colors, cmap="gist_heat_r", s=100, 
# edgecolor="black",
vmin = 50, vmax =80)

# 添加颜色条
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel("")  # 去掉颜色条的标签

cbar.set_ticks(np.arange(50, 85, 5))  # 设置颜色条刻度
cbar.ax.tick_params(labelsize=16)  # 调整颜色条刻度字体大小

# 标注每个点
for point in data:
    ax.text(point["x"] + 0.1, point["y"], point["label"], fontsize=12, ha="left")

# 添加边框标注 -1.64, "y": 0.15
ax.annotate(
    "MaCSE (82.02)",
    xy=(-1.9, 0.15),  # 箭头指向位置
    xytext=(-2.25,0.18),  # 文字位置
    arrowprops=dict(facecolor='black', arrowstyle='-'),
    size=12,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
)

# "x": -1.51, "y": 0.19, "label"
ax.annotate(
    "Avg.BERT (56.70)",
    xy=(-1.51, 0.19),  # 箭头指向位置
    xytext=(-1.7,0.23),  # 文字位置
    size=12,
    arrowprops=dict(facecolor='black', arrowstyle='-')
)
# "x": -1.5, "y": 0.1
ax.annotate(
    "DiffCSE (78.49)",
    xy=(-1.5, 0.1),  # 箭头指向位置
    xytext=(-1.84, 0.05),  # 文字位置
    size=12,
    arrowprops=dict(facecolor='black', arrowstyle='-')
)
# rankcse "x": -1.8, "y": 0.12
ax.annotate(
    "RankCSE (80.36)",
    xy=(-1.8,0.12),  # 箭头指向位置
    xytext=(-2.53,0.09),  # 文字位置
    size=12,
    arrowprops=dict(facecolor='black', arrowstyle='-')
)

# 设置网格
ax.grid(True, linestyle="--", alpha=0.6)


# 设置字体属性
font_prop = font_manager.FontProperties(family='Times New Roman', size=26)
# 设置轴标签
ax.set_xlabel("$\\ell_{\mathregular{uniform}}$", fontproperties=font_prop)
ax.set_ylabel("$\\ell_{\mathregular{align}}$", fontproperties=font_prop)

# 调整坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=18)

# 设置轴范围
ax.set_xlim([-4, -1])
ax.set_ylim([0, 0.7])

# 保存图像
plt.tight_layout()
plt.savefig("./alignment_uniform_plot3.png",dpi=1000, bbox_inches='tight')
plt.show()
