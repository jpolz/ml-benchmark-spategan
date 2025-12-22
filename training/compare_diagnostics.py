#!/usr/bin/env python3
"""
Script to compare diagnostic histories across multiple training runs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_diagnostic_history(checkpoint_path):
    """Load diagnostic history from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return checkpoint.get("diagnostic_history", None)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def get_run_label(run_dir):
    """Generate a label for a run based on its directory name."""
    run_id = run_dir.name
    return run_id


def plot_diagnostic_comparison(runs_data, output_path="diagnostic_comparison.png"):
    """
    Plot diagnostic metrics comparison across multiple runs.

    Parameters
    ----------
    runs_data : list of dict
        List of dictionaries with keys 'label', 'history', 'color'
    output_path : str
        Path to save the output figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    # Define metrics to plot
    metrics = [
        ("rmse", "RMSE (spatial mean)", 0),
        ("mae", "MAE (spatial mean)", None),
        ("bias_mean", "Bias Mean (spatial mean)", 0),
        ("bias_q95", "Bias Q95 (spatial mean)", 0),
        ("bias_q98", "Bias Q98 (spatial mean)", 0),
        ("std_ratio", "Std Ratio (spatial mean)", 1),
        ("correlation", "Correlation (spatial mean)", None),
        ("anomaly_correlation", "Anomaly Correlation (spatial mean)", None),
        ("fss", "FSS (Fractions Skill Score)", None),
    ]

    # Plot each metric
    for idx, (key, ylabel, hline) in enumerate(metrics):
        ax = axes[idx]
        has_data = False

        # Plot each run
        for run_data in runs_data:
            history = run_data["history"]
            label = run_data["label"]
            color = run_data["color"]

            if key in history and len(history[key]) > 0:
                epochs = history.get("epochs", list(range(1, len(history[key]) + 1)))
                ax.plot(
                    epochs,
                    history[key],
                    "o-",
                    linewidth=2,
                    markersize=4,
                    color=color,
                    label=label,
                    alpha=0.8,
                )
                has_data = True

        if has_data:
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel} Comparison", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")

            # Add horizontal reference line if specified
            if hline is not None:
                ax.axhline(y=hline, color="k", linestyle="--", alpha=0.3, linewidth=1)
        else:
            ax.axis("off")

    plt.suptitle(
        "Diagnostic Metrics Comparison Across Runs",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {output_path}")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare diagnostic histories across multiple training runs"
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="Paths to run directories to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagnostic_comparison.png",
        help="Output file path (default: diagnostic_comparison.png)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_models.pt",
        help="Checkpoint filename to load (default: final_models.pt)",
    )

    args = parser.parse_args()

    # Define colors for different runs
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    runs_data = []

    for i, run_path in enumerate(args.run_dirs):
        run_dir = Path(run_path)

        if not run_dir.exists():
            print(f"Warning: {run_dir} does not exist, skipping")
            continue

        # Try to load checkpoint
        checkpoint_path = run_dir / "checkpoints" / args.checkpoint

        if not checkpoint_path.exists():
            # Try to find the latest checkpoint_epoch_XX.pt
            checkpoints_dir = run_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
                if checkpoints:
                    # Sort by epoch number and get the highest
                    epochs_and_paths = [
                        (int(cp.stem.split("_")[-1]), cp) for cp in checkpoints
                    ]
                    epochs_and_paths.sort(reverse=True)
                    checkpoint_path = epochs_and_paths[0][1]
                    print(
                        f"Using checkpoint: {checkpoint_path.name} for {run_dir.name}"
                    )
                else:
                    print(f"Warning: No checkpoints found in {run_dir}, skipping")
                    continue
            else:
                print(f"Warning: No checkpoints directory in {run_dir}, skipping")
                continue

        # Load diagnostic history
        history = load_diagnostic_history(checkpoint_path)

        if history is None:
            print(f"Warning: No diagnostic history in {checkpoint_path}, skipping")
            continue

        # Get run label and color
        label = get_run_label(run_dir)
        color = colors[i % len(colors)]

        runs_data.append({"label": label, "history": history, "color": color})

    if not runs_data:
        print("Error: No valid runs found")
        return

    # Generate comparison plot
    plot_diagnostic_comparison(runs_data, output_path=args.output)


if __name__ == "__main__":
    main()
