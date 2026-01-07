#!/usr/bin/env python3
"""
Script to compare diagnostic histories across multiple training runs.
"""

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
import yaml


def load_diagnostic_history(checkpoint_path):
    """Load diagnostic history from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return checkpoint.get("diagnostic_history", None)
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def load_run_config(run_dir):
    """Load configuration from a run directory."""
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {run_dir}: {e}")
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
        List of dictionaries with keys 'label', 'history', 'color', 'config'
    output_path : str
        Path to save the output figure
    """
    # Create figure with space for table at bottom
    fig = plt.figure(figsize=(18, 16))

    # Create gridspec: 9 plots in 3x3 grid (top), 1 table (bottom)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.4], hspace=0.3, wspace=0.3)

    # Create subplots for metrics
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Define metrics to plot
    metrics = [
        ("rmse", "RMSE (spatial mean)", None, 0, 12),
        ("mae", "MAE (spatial mean)", None, 0, 5),
        ("bias_mean", "Bias Mean (spatial mean)", 0, -0.5, 0.5),
        ("bias_q95", "Bias Q95 (spatial mean)", 0, -0.5, 0.5),
        ("bias_q98", "Bias Q98 (spatial mean)", 0, -0.5, 0.5),
        ("std_ratio", "Std Ratio (spatial mean)", 1, 0.5, 1.5),
        ("correlation", "Correlation (spatial mean)", None, 0.4, None),
        ("anomaly_correlation", "Anomaly Correlation (spatial mean)", None, 0.3, None),
        ("fss", "FSS (Fractions Skill Score)", None, 0, None),
    ]

    # Plot each metric
    for idx, (key, ylabel, hline, preset_vmin, preset_vmax) in enumerate(metrics):
        ax = axes[idx]
        has_data = False
        all_values = []

        # Plot each run and collect values
        for run_data in runs_data:
            history = run_data["history"]
            label = run_data["label"]
            color = run_data["color"]

            if key in history and len(history[key]) > 0:
                epochs = history.get("epochs", list(range(1, len(history[key]) + 1)))
                values = history[key]
                all_values.extend(values)
                ax.plot(
                    epochs,
                    values,
                    "o-",
                    linewidth=0.6,
                    markersize=1,
                    color=color,
                    label=label,
                    alpha=1,
                )
                has_data = True

        if has_data:
            # Compute actual data range
            actual_min = min(all_values)
            actual_max = max(all_values)

            # Determine final limits
            if hline == 0:
                # For plots centered at 0, use symmetrical limits
                max_abs = max(abs(actual_min), abs(actual_max))
                if preset_vmax is not None:
                    max_abs = min(max_abs, preset_vmax)
                if preset_vmin is not None:
                    max_abs = min(max_abs, abs(preset_vmin))
                vmin, vmax = -max_abs, max_abs
            else:
                # Use preset limits but constrain to actual data range
                vmin = (
                    max(preset_vmin, actual_min)
                    if preset_vmin is not None
                    else actual_min
                )
                vmax = (
                    min(preset_vmax, actual_max)
                    if preset_vmax is not None
                    else actual_max
                )

            ax.set_ylim(bottom=vmin, top=vmax)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel} Comparison", fontsize=12, fontweight="bold")
            # 20 yticks
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.grid(True, alpha=0.3)
            # ax.legend(fontsize=8, loc="best")

            # Add horizontal reference line if specified
            if hline is not None:
                ax.axhline(y=hline, color="k", linestyle="--", alpha=0.3, linewidth=1)
        else:
            ax.axis("off")

    # Create configuration table at the bottom
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")

    # Extract key settings from configs
    table_data = []
    header = [
        "Run ID",
        "Domain",
        "Variable",
        "Experiment",
        "Architecture",
        "Oro",
        "GAN λ",
    ]

    for run_data in runs_data:
        config = run_data.get("config")
        if config:
            model_cfg = config.get("model", {})
            training_cfg = config.get("training", {})
            data_cfg = config.get("data", {})

            run_id = run_data["label"][:22]  # Truncate for space
            domain = data_cfg.get("domain", "?")[:5]
            var_target = data_cfg.get("var_target", "?")[:6]
            experiment = data_cfg.get("training_experiment", "?")[:15]
            architecture = model_cfg.get("architecture", "?")[:15]
            use_oro = "✓" if data_cfg.get("use_orography", False) else "✗"
            gan_weight = (
                f"{training_cfg.get('loss_weights', None).get('gan', None):.1e}"
                if training_cfg.get("loss_weights") is not None
                else "?"
            )

            table_data.append(
                [
                    run_id,
                    domain,
                    var_target,
                    experiment,
                    architecture,
                    use_oro,
                    gan_weight,
                ]
            )

    if table_data:
        # Create table with colors matching the plot lines
        cell_colors = []
        for i, run_data in enumerate(runs_data):
            color = run_data["color"]
            # Convert to RGBA and set alpha for better text visibility
            rgba = mcolors.to_rgba(color, alpha=0.2)
            cell_colors.append([rgba] * len(header))

        table = ax_table.table(
            cellText=table_data,
            colLabels=header,
            cellLoc="center",
            loc="center",
            colWidths=[0.15, 0.08, 0.08, 0.15, 0.15, 0.05, 0.10],
            cellColours=cell_colors,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header
        for i in range(len(header)):
            table[(0, i)].set_facecolor("#cccccc")
            table[(0, i)].set_text_props(weight="bold")

    plt.suptitle(
        "Diagnostic Metrics Comparison Across Runs",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
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

        # Load run configuration
        config = load_run_config(run_dir)

        # Get run label and color
        label = get_run_label(run_dir)
        color = colors[i % len(colors)]

        runs_data.append(
            {
                "label": label,
                "history": history,
                "color": color,
                "config": config,
            }
        )

    if not runs_data:
        print("Error: No valid runs found")
        return

    # Generate comparison plot
    plot_diagnostic_comparison(runs_data, output_path=args.output)


if __name__ == "__main__":
    main()
