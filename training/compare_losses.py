#!/usr/bin/env python3
"""
Script to compare loss histories across multiple training runs.
Parses training.log files to extract loss curves.
"""

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import yaml


def parse_training_log(log_path):
    """
    Parse training.log file to extract loss histories.

    Returns dict with lists for each loss type across epochs.
    """
    losses = {
        "epochs": [],
        "gen_train": [],
        "disc_train": [],
        "gen_total_test": [],
        "disc_total_test": [],
        "disc_real_test": [],
        "disc_fake_test": [],
        "l1_test": [],
        "mse_test": [],
        "gan_test": [],
        "fss_test": [],
    }

    try:
        with open(log_path, "r") as f:
            current_epoch = None
            for line in f:
                # Match epoch line: "Epoch 1/150"
                epoch_match = re.search(r"Epoch (\d+)/\d+", line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    losses["epochs"].append(current_epoch)
                    continue

                if current_epoch is None:
                    continue

                # Match generator loss: "  Generator Loss:     0.463650 (LR: 2.08e-06)"
                gen_match = re.search(r"Generator Loss:\s+(\d+\.\d+)", line)
                if gen_match:
                    losses["gen_train"].append(float(gen_match.group(1)))
                    continue

                # Match discriminator loss: "  Discriminator Loss: 1.410671 (LR: 2.00e-04)"
                disc_match = re.search(r"Discriminator Loss:\s+(\d+\.\d+)", line)
                if disc_match:
                    losses["disc_train"].append(float(disc_match.group(1)))
                    continue

                # Match test loss: "  Test Loss (Gen Total): 0.390243"
                test_total_match = re.search(
                    r"Test Loss \(Gen Total\):\s+(\d+\.\d+)", line
                )
                if test_total_match:
                    losses["gen_total_test"].append(float(test_total_match.group(1)))
                    continue

                # Match discriminator test loss: "  Test Disc Total:    1.234567"
                disc_test_match = re.search(r"Test Disc Total:\s+(\d+\.\d+)", line)
                if disc_test_match:
                    losses["disc_total_test"].append(float(disc_test_match.group(1)))
                    continue

                # Match discriminator real test loss: "  Test Disc Real:     1.234567"
                disc_real_match = re.search(r"Test Disc Real:\s+(\d+\.\d+)", line)
                if disc_real_match:
                    losses["disc_real_test"].append(float(disc_real_match.group(1)))
                    continue

                # Match discriminator fake test loss: "  Test Disc Fake:     1.234567"
                disc_fake_match = re.search(r"Test Disc Fake:\s+(\d+\.\d+)", line)
                if disc_fake_match:
                    losses["disc_fake_test"].append(float(disc_fake_match.group(1)))
                    continue

                # Match L1 loss: "  Test L1:            0.353805"
                l1_match = re.search(r"Test L1:\s+(\d+\.\d+)", line)
                if l1_match:
                    losses["l1_test"].append(float(l1_match.group(1)))
                    continue

                # Match MSE loss: "  Test MSE:           0.213260"
                mse_match = re.search(r"Test MSE:\s+(\d+\.\d+)", line)
                if mse_match:
                    losses["mse_test"].append(float(mse_match.group(1)))
                    continue

                # Match GAN loss: "  Test GAN:           0.799941"
                gan_match = re.search(r"Test GAN:\s+(\d+\.\d+)", line)
                if gan_match:
                    losses["gan_test"].append(float(gan_match.group(1)))
                    continue

                # Match FSS loss: "  Test FSS:           0.123456"
                fss_match = re.search(r"Test FSS:\s+(\d+\.\d+)", line)
                if fss_match:
                    losses["fss_test"].append(float(fss_match.group(1)))
                    continue

        # Validate that we have matching lengths
        n_epochs = len(losses["epochs"])
        for key in [
            "gen_train",
            "disc_train",
            "gen_total_test",
            "disc_total_test",
            "disc_real_test",
            "disc_fake_test",
            "l1_test",
            "mse_test",
            "gan_test",
        ]:
            if len(losses[key]) != n_epochs:
                print(
                    f"Warning: Mismatch in {key} length ({len(losses[key])}) vs epochs ({n_epochs})"
                )

        return losses

    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
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
    return run_dir.name


def plot_loss_comparison(runs_data, output_path="loss_comparison.png"):
    """
    Plot loss comparison across multiple runs.

    Parameters
    ----------
    runs_data : list of dict
        List of dictionaries with keys 'label', 'losses', 'color', 'marker', 'linestyle', 'config'
    output_path : str
        Path to save the output figure
    """
    # Create figure with space for table at bottom
    fig = plt.figure(figsize=(18, 14))

    # Create gridspec: 9 plots in 3x3 grid (top), 1 table (bottom)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.4], hspace=0.35, wspace=0.3)

    # Create subplots for metrics
    axes = []
    for i in range(3):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Define loss metrics to plot
    loss_metrics = [
        ("gen_train", "Generator Training Loss", None, None),
        ("disc_train", "Discriminator Training Loss", None, None),
        ("gen_total_test", "Generator Test Loss (Total)", None, None),
        ("disc_total_test", "Discriminator Test Loss (Total)", None, None),
        ("disc_real_test", "Discriminator Test Loss (Real)", None, None),
        ("disc_fake_test", "Discriminator Test Loss (Fake)", None, None),
        ("l1_test", "L1 Test Loss", None, None),
        ("mse_test", "MSE Test Loss", None, None),
        ("gan_test", "GAN Test Loss", None, None),
    ]

    # Collect all losses from training.log for each run
    for run_data in runs_data:
        run_dir = Path(run_data["run_dir"])
        log_path = run_dir / "training.log"

        if not log_path.exists():
            print(f"Warning: No training.log found in {run_dir}, skipping")
            run_data["losses"] = {}
            continue

        # Parse log file
        losses = parse_training_log(log_path)
        if losses:
            run_data["losses"] = losses
        else:
            print(f"Warning: Could not parse losses from {log_path}")
            run_data["losses"] = {}

    # Plot each metric
    for idx, (key, ylabel, preset_vmin, preset_vmax) in enumerate(loss_metrics):
        ax = axes[idx]
        has_data = False
        all_values = []

        # Plot each run
        for run_data in runs_data:
            losses = run_data.get("losses", {})
            label = run_data["label"]
            color = run_data["color"]
            marker = run_data.get("marker", "o")
            linestyle = run_data.get("linestyle", "-")

            if (
                key in losses
                and len(losses[key]) > 0
                and "epochs" in losses
                and len(losses["epochs"]) > 0
            ):
                epochs = losses["epochs"]
                values = losses[key]

                # Filter out zero or negative values for better visualization
                if all(v > 0 for v in values):
                    all_values.extend(values)
                    ax.plot(
                        epochs,
                        values,
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=0.4,
                        markersize=1.0,
                        color=color,
                        label=label,
                        alpha=0.85,
                    )
                    has_data = True

        if has_data and all_values:
            # Set limits
            actual_min = min(all_values)
            actual_max = max(all_values)

            vmin = preset_vmin if preset_vmin is not None else actual_min * 0.9
            vmax = preset_vmax if preset_vmax is not None else actual_max * 1.1

            ax.set_ylim(bottom=vmin, top=vmax)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel}", fontsize=12, fontweight="bold")
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
            ax.tick_params(axis="both", which="major", labelsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")  # Log scale often better for losses
        else:
            ax.axis("off")

    # Hide remaining empty subplots if we have fewer than 9 metrics
    for idx in range(len(loss_metrics), 9):
        axes[idx].axis("off")

    # Create configuration table at the bottom
    ax_table = fig.add_subplot(gs[3, :])
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
            architecture = model_cfg.get("architecture", "?")[:15]
            use_oro = "✓" if data_cfg.get("use_orography", False) else "✗"
            gan_weight = f"{training_cfg.get('loss_weights', {}).get('gan', 0):.1e}"

            # Get marker symbol for this run
            marker = run_data.get("marker", "o")

            if sweep_type == "loss_weights":
                # Shorter labels for sweep comparison
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
                        architecture,
                        l1_str,
                        mse_str,
                        gan_str,
                    ]
                )
            elif sweep_type == "discriminator":
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
            # M, Run ID, Domain, Var, Arch, L1, MSE, GAN (8 columns)
            col_widths = [0.03, 0.16, 0.07, 0.06, 0.09, 0.06, 0.06, 0.06]
        elif sweep_type == "discriminator":
            # M, Run ID, Domain, Var, Arch, GAN λ, GP, LR_in, SpecN, n_crit (10 columns)
            col_widths = [0.03, 0.16, 0.07, 0.06, 0.09, 0.06, 0.06, 0.06, 0.06, 0.05]
        else:
            # M, Run ID, Domain, Variable, Architecture, Oro, GAN λ (7 columns)
            col_widths = [0.03, 0.18, 0.10, 0.11, 0.16, 0.05, 0.10]

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

        # For very large numbers of runs, add alternating row colors
        if num_runs > 20:
            for row in range(1, num_runs + 1):
                if row % 2 == 0:
                    for col in range(len(header)):
                        current_color = table[(row, col)].get_facecolor()
                        darker = tuple(max(0, c * 0.9) for c in current_color[:3]) + (
                            current_color[3],
                        )
                        table[(row, col)].set_facecolor(darker)

    plt.suptitle(
        "Training Loss Comparison Across Runs",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Loss comparison plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare loss histories across multiple training runs"
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
        default="loss_comparison.png",
        help="Output file path (default: loss_comparison.png)",
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

        log_path = run_dir / "training.log"
        if not log_path.exists():
            print(f"Warning: No training.log in {run_dir}, skipping")
            continue

        # Load run configuration
        config = load_run_config(run_dir)

        # Get run label and visual properties
        label = get_run_label(run_dir)
        color = colors[i % len(colors)]
        # Markers cycle independently: change marker after each full color cycle
        marker = markers[(i // len(colors)) % len(markers)]
        # Linestyles cycle even more slowly: change after full color*marker cycles
        linestyle = linestyles[(i // (len(colors) * len(markers))) % len(linestyles)]

        run_info = {
            "label": label,
            "run_dir": str(run_dir),
            "losses": {},
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
    plot_loss_comparison(runs_data, output_path=args.output)


if __name__ == "__main__":
    main()
