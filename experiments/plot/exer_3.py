import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# ==========================================
# 1. 路径与数据加载 (Paths & Data Loading)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

df = pd.read_csv(os.path.join(data_dir, 'exer_3_data.csv'))
models = df['model'].tolist()
baseline_mem = df['baseline_mem'].tolist()
madeline_mem = df['madeline_mem'].tolist()
hardware_limit = float(df['hardware_limit'].iloc[0]) 

# ==========================================
# 2. 绘图设置
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1.5
# 中文字体：使用宋体（SimSun），后面为跨平台回退
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['SimSun', 'STSong', 'Songti SC', 'Noto Serif CJK SC', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['font.sans-serif'] = ['SimSun', 'STSong', 'Songti SC', 'Noto Serif CJK SC', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示成方块

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(models))
width = 0.35

# 绘制柱子
rects1 = ax.bar(x - width/2, baseline_mem, width, 
                label='ZeRO-3 (基线)', color='#7CB9E8', edgecolor='black', hatch='/')
rects2 = ax.bar(x + width/2, madeline_mem, width, 
                label='Madeline', color='#FFB347', edgecolor='black', hatch='\\')

# ==========================================
# 3. 辅助线与标签
# ==========================================
# 添加 32GB 硬件限制线
ax.axhline(y=hardware_limit, color='red', linestyle='--', linewidth=2, label='硬件上限')

ax.set_ylabel('显存使用峰值 (GB)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold')
ax.set_ylim(0, 55)  # 稍微留一点顶部空间

# 图例
ax.legend(frameon=False, loc='upper left', ncol=1)

# 网格
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_axisbelow(True)

# ==========================================
# 4. 标注文本
# ==========================================
def autolabel(rects, is_madeline=False):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        # 在柱子上方标注数值
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20, fontweight='bold')

autolabel(rects1)
autolabel(rects2, is_madeline=True)

# 特殊标注：在 13B 和 30B 的 Madeline 柱子上标注 "Max Util."
# 在 7B 上标注 "Full Cache"
# ax.text(x[0] + width/2, madeline_mem[0] + 1.5, "Full Cache", ha='center', color='darkred', fontsize=10, fontweight='bold')
# ax.text(x[1] + width/2, madeline_mem[1] + 1.5, "High Util.", ha='center', color='darkred', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'memory_usage.pdf'), format='pdf', dpi=300)
plt.savefig(os.path.join(script_dir, 'memory_usage.png'), format='png', dpi=300)
print("Figure saved as memory_usage.pdf/png")
plt.show()
