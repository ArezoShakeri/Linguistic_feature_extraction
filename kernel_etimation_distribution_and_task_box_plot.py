import matplotlib.pyplot as plt
import seaborn as sns
import math



def plot_significant_feature_distributions(df, feature_list,save_path, group_col='Diagnosis', figsize=(15, 10), palette='Set2'):
    """
    Plots the distribution of significant linguistic features for two groups (e.g., MCI and NC).

    """
    num_features = len(feature_list)
    ncols = 2
    nrows = (num_features + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for idx, feature in enumerate(feature_list):
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in DataFrame. Skipping.")
            continue

        sns.kdeplot(data=df, x=feature, hue=group_col, fill=True, common_norm=False,
                    palette=palette, ax=axes[idx], alpha=0.5)
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Density')

    # Hide unused subplots if any
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Distribution of Significant Linguistic Features by Diagnosis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()






    

def plot_feature_boxplots_by_task(df, feature_list, task_col='task_index', save_path="task_box_plts.png"):
    """
    Plot all features in feature_list as boxplots grouped by task_col in a single figure
    with subplots. 
    """
    sns.set(style="whitegrid", font_scale=1.2)
    
    unique_tasks = sorted(df[task_col].unique())
    palette = sns.color_palette("Set2", n_colors=len(unique_tasks))
    
    n_features = len(feature_list)
    n_cols = 2  # adjust layout here if you want
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows), constrained_layout=True)
    axes = axes.flatten()
    
    for i, feature in enumerate(feature_list):
        ax = axes[i]
        sns.boxplot(
            x=task_col,
            y=feature,
            data=df,
            order=unique_tasks,
            palette=palette,
            linewidth=1.2,
            fliersize=3,
            ax=ax
        )
        ax.set_title(f'Distribution of "{feature}" Across Different Tasks', fontsize=14, weight='bold')
        ax.set_xlabel('Task', fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='x', rotation=0)
        sns.despine(ax=ax, trim=True)
    
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Boxplots of Linguistic Features by Task', fontsize=18, weight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
