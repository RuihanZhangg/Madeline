import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def load_two_plot_data(data_dir):
    """从 CSV 文件加载双图数据"""
    df = pd.read_csv(os.path.join(data_dir, 'two_plot_data.csv'))
    return {
        'value_left': df['mfu_13b'].tolist(),
        'value_right': df['mfu_30b'].tolist(),
        'objects': df['object'].tolist(),
    }


def create_combined_bar_charts(
    # Labels
    subtitles=['Model Size: 13B', 'Model Size: 30B'],
    y_range=[10, 50],   # [min, max] for MFU y-axis
    save_path=None
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    if save_path is None:
        save_path = os.path.join(script_dir, 'two_plot.pdf')

    plot_data = load_two_plot_data(data_dir)
    value_left = plot_data['value_left']
    value_right = plot_data['value_right']
    objects = plot_data['objects']

    myfontsize = 26
    titlefontsize = myfontsize-1
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Bar positions
    x = np.arange(len(objects))
    width = 0.35
    
    def plot_bars(ax, values, subtitle='', add_ylabel=True):
        # Plot MFU bars
        rects = ax.bar(x, values, width,
                      color='#FFB347', hatch='\\',
                      label='MFU',
                      edgecolor='black',
                      linewidth=1.5)
        
        # Set y-axis range
        ax.set_ylim(y_range)
        
        # Customize subplot
        ax.set_title(format_subtitle(subtitle), fontsize=myfontsize, pad=25)
        ax.set_xticks(x)
        ax.set_xticklabels(objects, fontsize=titlefontsize, rotation=15, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Add y-label only for left plot
        if add_ylabel:
            ax.set_ylabel('MFU (%)', fontsize=myfontsize, weight='bold')
        
        # Set y-axis tick label size
        ax.tick_params(axis='y', labelsize=myfontsize-2)
        
        # Add value labels on bars
        baseline = values[0] if len(values) > 0 else None
        for i, rect in enumerate(rects):
            height = rect.get_height()
            label_text = f'{height}'
            if baseline and baseline != 0:
                label_text = f'{values[i]/baseline:.2f}x'
            ax.annotate(label_text,
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=myfontsize-4,
                       weight='bold')
        
        # Enhance spines
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
            
        return rects
    
    # Add subtitles with line breaks
    def format_subtitle(subtitle):
        parts = subtitle.split(',')
        return '\n'.join(parts)
    
    # Plot both subplots
    rects_left = plot_bars(ax1, value_left, subtitles[0], True)
    rects_right = plot_bars(ax2, value_right, subtitles[1], False)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, wspace=0.15)
    
    # Save figure
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.2)
    plt.show()

if __name__ == "__main__":
    create_combined_bar_charts()
