import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os
import pandas as pd


def load_four_plot_data(data_dir):
    """从 CSV 文件加载四图数据"""
    df = pd.read_csv(os.path.join(data_dir, 'four_plot_data.csv'))
    data = {
        'values_topleft': df['throughput_7b'].tolist(),
        'values_topright': df['throughput_13b'].tolist(),
        'values_bottomleft': df['mfu_7b'].tolist(),
        'values_bottomright': df['mfu_13b'].tolist(),
        'methods': df['method'].tolist(),
        'x_labels': df['x_label'].tolist(),
    }
    # Load averages
    df_avg = pd.read_csv(os.path.join(data_dir, 'four_plot_averages.csv'))
    avg_tl = df_avg[df_avg['group'] == 'topleft']
    avg_tr = df_avg[df_avg['group'] == 'topright']
    data['average_topleft'] = [avg_tl[avg_tl['run'] == r][['val1', 'val2', 'val3']].values.flatten().tolist() for r in avg_tl['run'].unique()]
    data['average_topright'] = [avg_tr[avg_tr['run'] == r][['val1', 'val2', 'val3']].values.flatten().tolist() for r in avg_tr['run'].unique()]
    return data


def create_quad_bar_charts(
    # Labels
    subtitles=['7B', '13B', '7B', '13B'],
    ylabels=['Throughput (Tokens / sec)', 'MFU (%)'],
    y_ranges=[(450, 700), (250, 450), (20, 65), (20, 55)],  # Y-axis ranges for each subplot
    save_path=None
):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    if save_path is None:
        save_path = os.path.join(script_dir, 'fourplot.pdf')

    plot_data = load_four_plot_data(data_dir)
    values_topleft = plot_data['values_topleft']
    values_topright = plot_data['values_topright']
    values_bottomleft = plot_data['values_bottomleft']
    values_bottomright = plot_data['values_bottomright']
    methods = plot_data['methods']
    x_labels = plot_data['x_labels']
    average_topleft = plot_data['average_topleft']
    average_topright = plot_data['average_topright']

    myfontsize = 26
    titlefontsize = myfontsize-1

    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Bar width and positions
    width = 0.55
    x = np.arange(len(x_labels))
    
    # Colors for different methods - updated to lighter professional color scheme
    colors = ['#7CB9E8', '#E5E4E2', '#E5E4E2', '#FFB347']  # Blue, Gray, Orange, Green
    hatches = ['/', 'x', '...', '\\']
    
    def plot_single_bars(ax, values, subtitle='', add_ylabel=False, ylabel_index=0, y_range=None, show_ratio=False):
        vals = np.array(values)
        if vals.ndim == 2:
            data = vals.mean(axis=0).tolist()
        else:
            data = vals.tolist()
        bar = ax.bar(x, data, width,
                     color=colors[:len(x_labels)],
                     edgecolor='black',
                     linewidth=1.5)
        baseline = data[0] if show_ratio and len(data) > 0 else None
        for i, rect in enumerate(bar):
            rect.set_hatch(hatches[i % len(hatches)])
            height = rect.get_height()
            label_text = f'{height}'
            if baseline and baseline != 0:
                label_text = f'{data[i]/baseline:.2f}x' if show_ratio else f'{height}'
            ax.text(rect.get_x() + rect.get_width()/2, height,
                    label_text,
                    ha='center', va='bottom',
                    fontsize=myfontsize-4, fontweight='bold')
        ax.set_title(subtitle, fontsize=myfontsize, pad=15, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=myfontsize)
        ax.grid(True, linestyle='--', alpha=0.4)
        if y_range:
            ax.set_ylim(y_range)
        ax.tick_params(axis='y', labelsize=myfontsize)
        if add_ylabel:
            ax.set_ylabel(ylabels[ylabel_index], fontsize=myfontsize, weight='bold')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
    
    # Plot all subplots with single bars per tick (3 ticks per subplot)
    plot_single_bars(ax1, values_topleft, subtitles[0], True, 0, y_ranges[0], True)
    plot_single_bars(ax2, values_topright, subtitles[1], False, 0, y_ranges[1], True)
    plot_single_bars(ax3, values_bottomleft, subtitles[2], True, 1, y_ranges[2], True)
    plot_single_bars(ax4, values_bottomright, subtitles[3], False, 1, y_ranges[3], True)
    
    # Restore legend for methods
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor='black', hatch=hatches[i], label=methods[i])
        for i in range(len(methods))
    ]
    fig.legend(legend_handles, methods,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=len(methods),
               fontsize=myfontsize)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, wspace=0.2, hspace=0.3)
    
    # Save figure
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.show()

if __name__ == "__main__":
    create_quad_bar_charts()
