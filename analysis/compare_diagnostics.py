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
    # Define metrics to plot - base metrics common to all variables
    base_metrics = [
        ("rmse", "RMSE (spatial mean)", None, 0, 12),
        ("mae", "MAE (spatial mean)", None, 0, 5),
        ("bias_mean", "Bias Mean (spatial mean)", 0, -0.5, 0.5),
        ("bias_q95", "Bias Q95 (spatial mean)", 0, -0.5, 0.5),
        ("bias_q98", "Bias Q98 (spatial mean)", 0, -0.5, 0.5),
        ("std_ratio", "Std Ratio (spatial mean)", 1, 0.5, 1.5),
        ("correlation", "Correlation (spatial mean)", None, 0.4, None),
        ("anomaly_correlation", "Anomaly Correlation (spatial mean)", None, 0.3, None),
        ("psd_distance", "PSD Distance (log RMSE)", None, 0, None),
        ("fss", "FSS (Fractions Skill Score)", None, 0, None),
        ("ensemble_std", "Ensemble Variability (std)", None, 0, None),
        ("lag1_corr_bias", "Lag-1 Autocorr Bias", 0, -0.2, 0.2),
        ("interannual_var_bias", "Interannual Var Bias", 0, None, None),
    ]

    # Variable-specific metrics - will try to plot these if available
    var_specific_metrics = {
        "tasmax": [
            ("su_bias", "Summer Days Bias", 0, None, None),
            ("txx_bias", "TXx (Annual Max) Bias", 0, None, None),
            ("txn_bias", "TXn (Annual Min) Bias", 0, None, None),
        ],
        "pr": [
            ("rx1day_bias", "Rx1day Bias", 0, None, None),
            ("sdii_bias", "SDII Bias", 0, None, None),
            ("cdd_bias", "CDD (Dry Spell) Bias", 0, None, None),
            ("cwd_bias", "CWD (Wet Spell) Bias", 0, None, None),
        ],
    }

    # Determine which variable we're working with from first run
    var_target = None
    if runs_data:
        first_config = runs_data[0].get("config", {})
        var_target = first_config.get("data", {}).get("var_target")

    # Build final metrics list
    metrics = base_metrics.copy()

    # Check if any runs have variable-specific metrics in their history
    if var_target and var_target in var_specific_metrics:
        var_metrics = var_specific_metrics[var_target]
        # Only add metrics that exist in at least one run's history
        for metric_key, metric_label, hline, vmin, vmax in var_metrics:
            has_metric = any(
                metric_key in run_data.get("history", {}) for run_data in runs_data
            )
            if has_metric:
                metrics.append((metric_key, metric_label, hline, vmin, vmax))

    # Determine grid size dynamically based on number of metrics
    n_metrics = len(metrics)
    n_cols = 3  # Fixed at 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with space for table at bottom
    fig = plt.figure(figsize=(18, 4 * n_rows + 4))

    # Create gridspec: n_rows for plots, 1 row for table at bottom
    gs = fig.add_gridspec(
        n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.4], hspace=0.3, wspace=0.3
    )

    # Create subplots for metrics
    axes = []
    for i in range(n_rows):
        for j in range(n_cols):
            axes.append(fig.add_subplot(gs[i, j]))

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
            marker = run_data.get("marker", "o")
            linestyle = run_data.get("linestyle", "-")

            if key in history and len(history[key]) > 0:
                epochs = history.get("epochs", list(range(1, len(history[key]) + 1)))
                values = history[key]
                all_values.extend(values)
                ax.plot(
                    epochs,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=0.3,
                    markersize=0.75,
                    color=color,
                    label=label,
                    alpha=0.85,
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

    # Hide any unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    # Create configuration table at the bottom (last row spans all columns)
    ax_table = fig.add_subplot(gs[n_rows, :])
    ax_table.axis("off")

    # Extract key settings from configs
    table_data = []
    # Check if any run has sweep parameters (from CLI args) OR has ablation parameters in config
    # to decide on whether to show the detailed ablation table
    has_sweep_params = any("sweep_params" in run_data for run_data in runs_data)

    # Determine sweep type: "loss_weights", "discriminator", or None
    sweep_type = None
    if has_sweep_params:
        # Check first run with sweep_params to determine type
        for run_data in runs_data:
            if "sweep_params" in run_data:
                params = run_data["sweep_params"]
                if "l1_weight" in params or "mse_weight" in params:
                    sweep_type = "loss_weights"
                elif "gp_weight" in params or "use_lr_path" in params:
                    sweep_type = "discriminator"
                break

    # Also check if configs contain ablation study parameters (gp_weight, spectral_norm, etc.)
    if not has_sweep_params:
        for run_data in runs_data:
            config = run_data.get("config")
            if config:
                training_cfg = config.get("training", {})
                model_cfg = config.get("model", {})
                disc_cfg = model_cfg.get("discriminator", {})
                # Check if ablation parameters are present and varying
                if (
                    "gradient_penalty_weight" in training_cfg
                    or "use_lr_path" in disc_cfg
                    or "spectral_norm" in disc_cfg
                ):
                    has_sweep_params = True
                    sweep_type = "discriminator"
                    break

    if sweep_type == "loss_weights":
        header = [
            "M",  # Marker column
            "Run ID",
            "Domain",
            "Var",
            "Exp",
            "Arch",
            "L1",
            "MSE",
            "GAN",
        ]
    elif sweep_type == "discriminator":
        header = [
            "M",  # Marker column
            "Run ID",
            "Domain",
            "Var",
            "Exp",
            "Arch",
            "GAN λ",
            "GP",
            "LR_in",
            "SpecN",
            "n_crit",
        ]
    else:
        header = [
            "M",  # Marker column
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

            # Get marker symbol for this run
            marker = run_data.get("marker", "o")

            if sweep_type == "loss_weights":
                # Shorter labels for sweep comparison
                experiment = data_cfg.get("training_experiment", "?")[:6]
                architecture = model_cfg.get("architecture", "?")[:8]

                # Get sweep parameters
                sweep_params = run_data.get("sweep_params", {})
                loss_weights_cfg = training_cfg.get("loss_weights", {})

                # Loss weights
                l1_weight = sweep_params.get(
                    "l1_weight", loss_weights_cfg.get("l1", "?")
                )
                mse_weight = sweep_params.get(
                    "mse_weight", loss_weights_cfg.get("mse", "?")
                )
                gan_weight_val = sweep_params.get(
                    "gan_weight", loss_weights_cfg.get("gan", "?")
                )

                # Format values
                l1_str = (
                    f"{l1_weight:.2f}"
                    if isinstance(l1_weight, (float, int))
                    else str(l1_weight)
                )
                mse_str = (
                    f"{mse_weight:.2f}"
                    if isinstance(mse_weight, (float, int))
                    else str(mse_weight)
                )
                gan_str = (
                    f"{gan_weight_val:.3f}"
                    if isinstance(gan_weight_val, (float, int))
                    else str(gan_weight_val)
                )

                table_data.append(
                    [
                        marker,
                        run_id,
                        domain,
                        var_target,
                        experiment,
                        architecture,
                        l1_str,
                        mse_str,
                        gan_str,
                    ]
                )
            elif sweep_type == "discriminator":
                # Shorter labels for sweep comparison
                experiment = data_cfg.get("training_experiment", "?")[:6]
                architecture = model_cfg.get("architecture", "?")[:8]

                # Get sweep parameters (new ablation study format)
                sweep_params = run_data.get("sweep_params", {})

                # Gradient penalty weight
                gp_weight = sweep_params.get(
                    "gp_weight",
                    training_cfg.get("gradient_penalty_weight", "?"),
                )
                if isinstance(gp_weight, (float, int)):
                    gp_weight = f"{gp_weight:.1f}"
                else:
                    gp_weight = str(gp_weight)

                # LR path (coarse input) usage
                use_lr_path = sweep_params.get(
                    "use_lr_path",
                    model_cfg.get("discriminator", {}).get("use_lr_path", "?"),
                )
                lr_in = (
                    "✓"
                    if use_lr_path in [True, "true"]
                    else "✗"
                    if use_lr_path in [False, "false"]
                    else str(use_lr_path)
                )

                # Spectral normalization
                spec_norm = sweep_params.get(
                    "spectral_norm",
                    model_cfg.get("discriminator", {}).get("spectral_norm", "?"),
                )
                spec_n = (
                    "✓"
                    if spec_norm in [True, "true"]
                    else "✗"
                    if spec_norm in [False, "false"]
                    else str(spec_norm)
                )

                # n_critic
                n_critic = sweep_params.get(
                    "n_critic", training_cfg.get("n_critic", "?")
                )

                table_data.append(
                    [
                        marker,
                        run_id,
                        domain,
                        var_target,
                        experiment,
                        architecture,
                        gan_weight,
                        gp_weight,
                        lr_in,
                        spec_n,
                        str(n_critic),
                    ]
                )
            else:
                table_data.append(
                    [
                        marker,
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

        if sweep_type == "loss_weights":
            # M, Run ID, Domain, Var, Exp, Arch, L1, MSE, GAN (9 columns)
            col_widths = [
                0.03,
                0.14,
                0.05,
                0.05,
                0.05,
                0.07,
                0.05,
                0.05,
                0.05,
            ]
        elif sweep_type == "discriminator":
            # M, Run ID, Domain, Var, Exp, Arch, GAN λ, GP, LR_in, SpecN, n_crit (11 columns)
            col_widths = [
                0.03,
                0.14,
                0.05,
                0.05,
                0.05,
                0.07,
                0.05,
                0.05,
                0.05,
                0.05,
                0.05,
            ]
        else:
            # M, Run ID, Domain, Variable, Experiment, Architecture, Oro, GAN λ (8 columns)
            col_widths = [0.03, 0.13, 0.08, 0.08, 0.13, 0.13, 0.04, 0.08]

        table = ax_table.table(
            cellText=table_data,
            colLabels=header,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
            cellColours=cell_colors,
        )
        table.auto_set_font_size(False)

        # Adjust font size and row height based on number of runs
        num_runs = len(table_data)
        if num_runs > 25:
            table.set_fontsize(6)
            row_height = 1.3
        elif num_runs > 15:
            table.set_fontsize(7)
            row_height = 1.5
        else:
            table.set_fontsize(8)
            row_height = 1.8

        table.scale(1, row_height)

        # Style header
        for i in range(len(header)):
            table[(0, i)].set_facecolor("#cccccc")
            table[(0, i)].set_text_props(weight="bold")

        # For very large numbers of runs, add alternating row colors for readability
        if num_runs > 20:
            for row in range(1, num_runs + 1):
                if row % 2 == 0:
                    for col in range(len(header)):
                        current_color = table[(row, col)].get_facecolor()
                        # Slightly darken alternating rows
                        darker = tuple(max(0, c * 0.9) for c in current_color[:3]) + (
                            current_color[3],
                        )
                        table[(row, col)].set_facecolor(darker)

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
    parser.add_argument(
        "--sweep-params",
        type=str,
        nargs="*",
        default=None,
        help="Sweep parameters for each run in format 'lambda,disc_lr,n_critic' (optional)",
    )

    args = parser.parse_args()

    # Extended color palette for many runs
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
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Marker styles to vary when colors repeat
    markers = ["o", "s", "^", "v", "D", "p", "*", "h", "X", "P"]

    # Line styles to further vary appearance
    linestyles = ["-", "--", "-.", ":"]

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
        # Markers cycle independently: change marker after each full color cycle
        marker = markers[(i // len(colors)) % len(markers)]
        # Linestyles cycle even more slowly: change after full color*marker cycles
        linestyle = linestyles[(i // (len(colors) * len(markers))) % len(linestyles)]

        run_info = {
            "label": label,
            "history": history,
            "color": color,
            "marker": marker,
            "linestyle": linestyle,
            "config": config,
        }

        # Add sweep parameters if provided
        if args.sweep_params and i < len(args.sweep_params):
            try:
                params = args.sweep_params[i].split(",")
                if len(params) == 3:
                    # Check if it's loss weights (all numeric) or old discriminator format
                    try:
                        # Try parsing as loss weights: l1, mse, gan
                        l1 = float(params[0])
                        mse = float(params[1])
                        gan = float(params[2])
                        run_info["sweep_params"] = {
                            "l1_weight": l1,
                            "mse_weight": mse,
                            "gan_weight": gan,
                        }
                    except ValueError:
                        # Old format: lambda_disc, disc_lr, n_critic
                        run_info["sweep_params"] = {
                            "lambda_disc": params[0],
                            "disc_lr": float(params[1]),
                            "n_critic": params[2],
                        }
                elif len(params) == 4:
                    # New discriminator ablation format: gp_weight, use_lr_path, spectral_norm, n_critic
                    run_info["sweep_params"] = {
                        "gp_weight": float(params[0]),
                        "use_lr_path": params[1],
                        "spectral_norm": params[2],
                        "n_critic": int(params[3]),
                    }
            except (ValueError, IndexError) as e:
                print(
                    f"Warning: Could not parse sweep params '{args.sweep_params[i]}': {e}"
                )

        runs_data.append(run_info)

    if not runs_data:
        print("Error: No valid runs found")
        return

    # Generate comparison plot
    plot_diagnostic_comparison(runs_data, output_path=args.output)


if __name__ == "__main__":
    main()
