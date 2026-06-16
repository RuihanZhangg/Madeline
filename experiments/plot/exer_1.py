import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# ==========================================
# 1. 路径与数据加载 (Paths & Data Loading)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

df = pd.read_csv(os.path.join(data_dir, 'exer_1_data.csv'))
models = df['model'].tolist()
baseline_throughput = df['baseline_throughput'].tolist()
madeline_throughput = df['madeline_throughput'].tolist()

# 计算 Speedup
speedups = [m / b for m, b in zip(madeline_throughput, baseline_throughput)]

# ==========================================
# 2. 绘图设置 (Plotting Settings)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman' # 学术常用字体
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1.5
# 中文字体：使用宋体（SimSun），后面为跨平台回退
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun', 'STSong', 'Songti SC', 'Noto Serif CJK SC', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Noto Serif CJK SC', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示成方块

fig, ax = plt.subplots(figsize=(8, 5))

# 柱状图位置
x = np.arange(len(models))
width = 0.35  # 柱子宽度

# 绘制柱子
# 使用 Hatch (纹理) 区分，方便黑白打印
rects1 = ax.bar(x - width/2, baseline_throughput, width, 
                label='ZeRO-3 (基线)', color='#7CB9E8', edgecolor='black', hatch='/')
rects2 = ax.bar(x + width/2, madeline_throughput, width, 
                label='Madeline', color='#FFB347', edgecolor='black', hatch='\\')

# ==========================================
# 3. 标签与装饰 (Labels & Decoration)
# ==========================================
ax.set_ylabel('吞吐量 (词元 / 秒)', fontweight='bold', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.legend(frameon=False, loc='upper right')

# 添加网格线 (仅 Y 轴)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_axisbelow(True)
ax.set_ylim(0, 730)
# ==========================================
# 4. 在柱子上标注加速比 (Speedup Text)
# ==========================================
def autolabel(rects, is_baseline=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if not is_baseline:
            # 在 Madeline 的柱子上写加速比 (e.g., 1.32x)
            speedup_text = f"{speedups[i]:.2f}x"
            ax.annotate(speedup_text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=20, fontweight='bold', color='#b03060')

autolabel(rects2)

# 调整布局防止切边
plt.tight_layout()

# 保存图片
plt.savefig(os.path.join(script_dir, 'e2e_performance.pdf'), format='pdf', dpi=300)
plt.savefig(os.path.join(script_dir, 'e2e_performance.png'), format='png', dpi=300)
print("Figure saved as e2e_performance.pdf and e2e_performance.png")
plt.show()