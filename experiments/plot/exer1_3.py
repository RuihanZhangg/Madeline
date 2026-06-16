import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# ==========================================
# 0. 路径与数据加载 (Paths & Data Loading)
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')

df = pd.read_csv(os.path.join(data_dir, 'exer1_3_data.csv'))
models = df['model'].tolist()
baseline_throughput = df['baseline_throughput'].tolist()
madeline_throughput = df['madeline_throughput'].tolist()
baseline_mem = df['baseline_mem'].tolist()
madeline_mem = df['madeline_mem'].tolist()
hardware_limit = float(df['hardware_limit'].iloc[0])

# ==========================================
# 1. 全局设置 (Global Settings)
# ==========================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20  # 稍微调整字号以适应双图布局
plt.rcParams['axes.linewidth'] = 1.5

# 准备画布：1行2列，宽16高6
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(models))
width = 0.35

# 颜色与纹理定义
color_base = '#7CB9E8'  # Blue
color_ours = '#FFB347'  # Orange
hatch_base = '/'
hatch_ours = '\\'

# ==========================================
# 1. 左图绘制：Throughput
# ==========================================
ax1 = axes[0]  # 获取左边的子图对象

speedups = [m / b for m, b in zip(madeline_throughput, baseline_throughput)]

# 绘制柱状图
rects1_1 = ax1.bar(x - width/2, baseline_throughput, width,
                   label='ZeRO-3 (Baseline)', color=color_base, edgecolor='black', hatch=hatch_base)
rects1_2 = ax1.bar(x + width/2, madeline_throughput, width,
                   label='Madeline (Ours)', color=color_ours, edgecolor='black', hatch=hatch_ours)

# 标签与装饰
ax1.set_ylabel('Throughput (Tokens / GPU / sec)', fontweight='bold', fontsize=18)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight='bold', fontsize=16)
ax1.legend(frameon=False, loc='upper right', fontsize=16)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax1.set_axisbelow(True)

# 标注加速比
def autolabel_speedup(rects):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        speedup_text = f"{speedups[i]:.2f}x"
        ax1.annotate(speedup_text,
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=16, fontweight='bold', color='#b03060')

autolabel_speedup(rects1_2)

# 添加底部子图标题 (a)
ax1.text(0.5, -0.2, '(a) End-to-End Throughput Comparison', transform=ax1.transAxes,
         fontsize=20, fontweight='bold', va='top', ha='center')

# ==========================================
# 2. 右图绘制：Memory Usage
# ==========================================
ax2 = axes[1]  # 获取右边的子图对象

# 绘制柱状图
rects2_1 = ax2.bar(x - width/2, baseline_mem, width,
                   label='ZeRO-3 (Baseline)', color=color_base, edgecolor='black', hatch=hatch_base)
rects2_2 = ax2.bar(x + width/2, madeline_mem, width,
                   label='Madeline (Ours)', color=color_ours, edgecolor='black', hatch=hatch_ours)

# 辅助线 (Hardware Limit)
ax2.axhline(y=hardware_limit, color='red', linestyle='--', linewidth=2, label='Hardware Limit (32GB)')

# 标签与装饰
ax2.set_ylabel('Peak Memory Usage (GB)', fontweight='bold', fontsize=18)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontweight='bold', fontsize=16)
ax2.set_ylim(0, 45)  # 调整上限以容纳文本
# 合并图例：这里把 ZeRO/Madeline 和 Limit 线放在一起
# 为了避免图例重复，我们重新收集 handles 和 labels
handles, labels = ax2.get_legend_handles_labels()
# 只需要显示 Limit 的图例，因为颜色已经在左图解释过了，或者保留全部
# 这里选择保留全部以便右图独立可读
ax2.legend(handles, labels, frameon=False, loc='upper left', ncol=1, fontsize=16)

ax2.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax2.set_axisbelow(True)

# 标注内存数值
def autolabel_mem(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=16, fontweight='bold')

autolabel_mem(rects2_1)
autolabel_mem(rects2_2)

# 添加底部子图标题 (b)
ax2.text(0.5, -0.2, '(b) Peak Memory Usage Analysis', transform=ax2.transAxes,
         fontsize=20, fontweight='bold', va='top', ha='center')

# ==========================================
# 3. 保存与展示
# ==========================================
plt.tight_layout()
# 预留一点底部空间给 text 标注
plt.subplots_adjust(bottom=0.15) 

plt.savefig(os.path.join(script_dir, 'merged_performance.pdf'), format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(script_dir, 'merged_performance.png'), format='png', dpi=300, bbox_inches='tight')
print("Figure saved as merged_performance.pdf and png")
plt.show()