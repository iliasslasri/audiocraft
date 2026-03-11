"""Visualization for MAGNET audio inpainting metrics.

Reads the new JSON format from inpaint.py:
  { config, fad, per_sample_metrics[] }

Generates multi-panel plots for CLAP consistency, boundary smoothness,
and semantic shift, all vs. CFG scales.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(path: str) -> pd.DataFrame:
    """Load metrics JSON (supports both old flat-list and new nested format)."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Old format: flat list of dicts
        return pd.DataFrame(data), None, None
    else:
        # New format: { config, fad, per_sample_metrics }
        df = pd.DataFrame(data["per_sample_metrics"])
        return df, data.get("fad"), data.get("config")


def plot_clap_consistency(df: pd.DataFrame, output_dir: str):
    """CLAP consistency vs. CFG scales — gap region and full audio."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, col, title, baseline_col in [
        (axes[0],
         "clap_gap_inpainted_vs_target",
         "Gap Region — CLAP vs Target Desc",
         "clap_gap_original_vs_target"),
        (axes[1],
         "clap_inpainted_vs_target",
         "Full Audio — CLAP vs Target Desc",
         "clap_original_vs_target"),
    ]:
        avg = df.groupby(["min_cfg_scale", "max_cfg_scale"])[col].mean().reset_index()
        sns.lineplot(data=avg, x="max_cfg_scale", y=col,
                     hue="min_cfg_scale", marker="o", palette="viridis",
                     linewidth=2.5, ax=ax)

        if baseline_col in df.columns:
            baseline = df[baseline_col].mean()
            ax.axhline(baseline, ls="--", color="red", alpha=0.7, label="Baseline (Original)")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Max CFG Scale")
        ax.set_ylabel("CLAP Cosine Similarity")
        ax.legend(title="Min CFG", fontsize=9)

    fig.suptitle("CLAP Text-Audio Consistency vs. CFG Scales", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/clap_consistency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved clap_consistency.png")


def plot_boundary_smoothness(df: pd.DataFrame, output_dir: str):
    """Boundary smoothness (flux ratio) vs. CFG scales."""
    if "boundary_avg_flux_ratio" not in df.columns:
        print("  Skipping boundary smoothness (not in data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Average flux ratio
    avg = df.groupby(["min_cfg_scale", "max_cfg_scale"])["boundary_avg_flux_ratio"].mean().reset_index()
    sns.lineplot(data=avg, x="max_cfg_scale", y="boundary_avg_flux_ratio",
                 hue="min_cfg_scale", marker="o", palette="magma",
                 linewidth=2.5, ax=axes[0])
    axes[0].axhline(1.0, ls="--", color="green", alpha=0.7, label="Perfect (ratio=1.0)")
    axes[0].set_title("Boundary Flux Ratio (lower = smoother)", fontsize=14)
    axes[0].set_xlabel("Max CFG Scale")
    axes[0].set_ylabel("Avg Flux Ratio (inpainted / original)")
    axes[0].legend(title="Min CFG", fontsize=9)

    # Left vs right boundary
    for side, color in [("left", "tab:blue"), ("right", "tab:orange")]:
        col = f"boundary_{side}_peak_flux"
        if col in df.columns:
            avg_side = df.groupby(["min_cfg_scale", "max_cfg_scale"])[col].mean().reset_index()
            sns.lineplot(data=avg_side, x="max_cfg_scale", y=col,
                         marker="o", linewidth=2, ax=axes[1],
                         label=f"{side.title()} boundary", color=color)
    axes[1].set_title("Peak Spectral Flux at Boundaries", fontsize=14)
    axes[1].set_xlabel("Max CFG Scale")
    axes[1].set_ylabel("Peak Spectral Flux")
    axes[1].legend(fontsize=9)

    fig.suptitle("Boundary Smoothness vs. CFG Scales", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/boundary_smoothness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved boundary_smoothness.png")


def plot_semantic_shift(df: pd.DataFrame, output_dir: str):
    """Semantic shift ratio vs. CFG scales."""
    if "semantic_shift_to_target" not in df.columns:
        print("  Skipping semantic shift (not in data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Shift to target
    avg = df.groupby(["min_cfg_scale", "max_cfg_scale"])["semantic_shift_to_target"].mean().reset_index()
    sns.lineplot(data=avg, x="max_cfg_scale", y="semantic_shift_to_target",
                 hue="min_cfg_scale", marker="o", palette="coolwarm",
                 linewidth=2.5, ax=axes[0])
    axes[0].axhline(1.0, ls="--", color="gray", alpha=0.7, label="No shift (1.0)")
    axes[0].set_title("Semantic Shift to Target (>1 = improved)", fontsize=14)
    axes[0].set_xlabel("Max CFG Scale")
    axes[0].set_ylabel("CLAP(gap,target) ratio: inpainted / original")
    axes[0].legend(title="Min CFG", fontsize=9)

    # Drift from original
    avg2 = df.groupby(["min_cfg_scale", "max_cfg_scale"])["semantic_drift_from_original"].mean().reset_index()
    sns.lineplot(data=avg2, x="max_cfg_scale", y="semantic_drift_from_original",
                 hue="min_cfg_scale", marker="o", palette="coolwarm",
                 linewidth=2.5, ax=axes[1])
    axes[1].axhline(1.0, ls="--", color="gray", alpha=0.7, label="No drift (1.0)")
    axes[1].set_title("Semantic Drift from Original (<1 = changed)", fontsize=14)
    axes[1].set_xlabel("Max CFG Scale")
    axes[1].set_ylabel("CLAP(gap,original) ratio: inpainted / original")
    axes[1].legend(title="Min CFG", fontsize=9)

    fig.suptitle("Semantic Shift Analysis vs. CFG Scales", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/semantic_shift.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved semantic_shift.png")


def plot_summary_dashboard(df: pd.DataFrame, fad_score, output_dir: str):
    """Single-figure dashboard with key metrics across CFG scales."""
    key_metrics = [
        ("clap_gap_inpainted_vs_target", "CLAP Gap vs Target", "viridis"),
        ("boundary_avg_flux_ratio", "Boundary Flux Ratio", "magma"),
        ("semantic_shift_to_target", "Semantic Shift to Target", "coolwarm"),
    ]
    available = [(c, t, p) for c, t, p in key_metrics if c in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (col, title, pal) in zip(axes, available):
        avg = df.groupby(["min_cfg_scale", "max_cfg_scale"])[col].mean().reset_index()
        sns.lineplot(data=avg, x="max_cfg_scale", y=col,
                     hue="min_cfg_scale", marker="o", palette=pal,
                     linewidth=2.5, ax=ax)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Max CFG Scale")
        ax.legend(title="Min CFG", fontsize=8)

    fad_text = f"FAD = {fad_score:.2f}" if fad_score is not None else "FAD = N/A"
    fig.suptitle(f"Inpainting Metrics Dashboard — {fad_text}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/metrics_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved metrics_dashboard.png")


def main(path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df, fad_score, config = load_metrics(path)
    print(f"Loaded {len(df)} entries from {path}")
    if config:
        print(f"  Config: mask={config.get('start_time','?')}s–{config.get('end_time','?')}s, "
              f"soft_mask={config.get('soft_mask_transition', 0)}, "
              f"pos_cfg={config.get('position_aware_cfg', False)}")
    if fad_score is not None:
        print(f"  FAD = {fad_score:.4f}")

    plot_clap_consistency(df, output_dir)
    plot_boundary_smoothness(df, output_dir)
    plot_semantic_shift(df, output_dir)
    plot_summary_dashboard(df, fad_score, output_dir)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MAGNET inpainting metrics")
    parser.add_argument("--path", type=str, default="inpainting_metrics.json",
                        help="Path to metrics JSON")
    parser.add_argument("--output", type=str, default="plots",
                        help="Directory to save plots")
    args = parser.parse_args()
    main(args.path, args.output)