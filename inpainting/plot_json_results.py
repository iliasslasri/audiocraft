import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse


def main(path, output_dir):
    with open(path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")

    df_avg = df.groupby(['min_cfg_scale', 'max_cfg_scale']).mean(numeric_only=True).reset_index()

    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=df_avg, 
        x='max_cfg_scale', 
        y='consistency_gap_score_targetdesc', 
        hue='min_cfg_scale', 
        marker='o', 
        palette='viridis',
        linewidth=2.5
    )

    baseline_val = df['consistency_gap_original_targetdesc'].mean()
    plt.axhline(baseline_val, ls='--', color='red', alpha=0.7, label='Baseline (Original Audio)')

    plt.title('Inpainting Consistency vs. CFG Scales', fontsize=16, pad=20)
    plt.xlabel('Max CFG Scale', fontsize=12)
    plt.ylabel('Consistency Gap Score (Target Desc)', fontsize=12)
    plt.legend(title='Min CFG Scale', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/inpainting_gap_cfg_consistency_plot.png', dpi=300)
    plt.show()


    plt.figure(figsize=(12, 7))

    sns.lineplot(
        data=df_avg, 
        x='max_cfg_scale', 
        y='consistency_score_targetdesc', 
        hue='min_cfg_scale', 
        marker='o', 
        palette='viridis',
        linewidth=2.5
    )

    baseline_val = df['consistency_score_original_targetdesc'].mean()
    plt.axhline(baseline_val, ls='--', color='red', alpha=0.7, label='Baseline (Original Audio)')

    plt.title('Inpainting Consistency vs. CFG Scales', fontsize=16, pad=20)
    plt.xlabel('Max CFG Scale', fontsize=12)
    plt.ylabel('Consistency Score (Target Desc)', fontsize=12)
    plt.legend(title='Min CFG Scale', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/inpainting_cfg_consistency_plot.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot consistency scores vs CFG scales')
    parser.add_argument('--path', type=str, default='inpainting_cfg_consistency_results.json', help='Path to the JSON results file')
    parser.add_argument('--output', type=str, default='inpainting_consistency_plots', help='Directory to save the plots')
    args = parser.parse_args()
    main(args.path, args.output)