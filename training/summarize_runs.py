#!/usr/bin/env python3
"""
Script to parse all run directories and summarize configuration changes in a markdown table.
"""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path):
    """Load a YAML config file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config_path}: {e}")
        return None


def extract_key_params(config):
    """Extract key parameters from config for comparison."""
    if config is None:
        return {}

    params = {}

    # Model parameters
    model = config.get("model", {})
    params["architecture"] = model.get(
        "architecture", model.get("generator_architecture", "N/A")
    )
    params["filter_size"] = model.get("filter_size", "N/A")
    params["dropout_ratio"] = model.get("dropout_ratio", "N/A")

    # Training parameters
    training = config.get("training", {})
    params["batch_size"] = training.get("batch_size", "N/A")
    params["epochs"] = training.get("epochs", "N/A")
    params["batches_per_epoch"] = training.get("batches_per_epoch", "N/A")

    # Generator optimizer
    gen = training.get("generator", {})
    params["gen_lr"] = gen.get("learning_rate", "N/A")
    params["gen_optimizer"] = gen.get("optimizer", "N/A")
    params["gen_weight_decay"] = gen.get("weight_decay", "N/A")

    # Discriminator optimizer
    disc = training.get("discriminator", {})
    params["disc_lr"] = disc.get("learning_rate", "N/A")
    params["disc_optimizer"] = disc.get("optimizer", "N/A")
    params["disc_weight_decay"] = disc.get("weight_decay", "N/A")

    # Loss weights
    loss_weights = training.get("loss_weights", {})
    params["loss_l1"] = loss_weights.get("l1", "N/A")
    params["loss_gan"] = loss_weights.get("gan", "N/A")

    # Early stopping
    params["early_stopping"] = training.get("early_stopping", "N/A")
    params["patience"] = training.get("patience", "N/A")

    # Data parameters
    data = config.get("data", {})
    params["domain"] = data.get("domain", "N/A")
    params["var_target"] = data.get("var_target", "N/A")
    params["normalization"] = data.get("normalization", "N/A")

    return params


def get_run_info(run_dir):
    """Get information about a run directory."""
    config_path = run_dir / "config.yaml"

    if not config_path.exists():
        return None

    config = load_config(config_path)
    if config is None:
        return None

    params = extract_key_params(config)

    # Extract timestamp from directory name
    dir_name = run_dir.name
    try:
        timestamp_str = dir_name[:13]  # YYYYMMDD_HHMM
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
        params["timestamp"] = timestamp
        params["run_id"] = dir_name
    except:
        params["timestamp"] = None
        params["run_id"] = dir_name

    # Check for final model in checkpoints subfolder
    checkpoints_dir = run_dir / "checkpoints"
    params["has_final_model"] = (checkpoints_dir / "final_models.pt").exists()

    # Count checkpoints in checkpoints subfolder
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob("checkpoint_epoch_*.pt"))
        params["num_checkpoints"] = len(checkpoints)
        if checkpoints:
            epochs = [int(cp.stem.split("_")[-1]) for cp in checkpoints]
            params["max_checkpoint_epoch"] = max(epochs)
        else:
            params["max_checkpoint_epoch"] = "N/A"
    else:
        params["num_checkpoints"] = 0
        params["max_checkpoint_epoch"] = "N/A"

    return params


def format_value(value):
    """Format a value for markdown display."""
    if value == "N/A" or value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    return str(value)


def detect_changes(runs):
    """Detect which parameters changed across runs."""
    if not runs:
        return set()

    changed_params = set()
    first_run = runs[0]

    for param in first_run.keys():
        if param in [
            "timestamp",
            "run_id",
            "has_final_model",
            "num_checkpoints",
            "max_checkpoint_epoch",
        ]:
            continue

        values = [run.get(param) for run in runs]
        if len(set(str(v) for v in values)) > 1:
            changed_params.add(param)

    return changed_params


def generate_markdown_table(runs, show_all=False):
    """Generate a markdown table summarizing the runs."""
    if not runs:
        return "No runs found."

    # Sort runs by timestamp
    runs = sorted(runs, key=lambda x: x.get("timestamp") or datetime.min)

    # Detect which parameters changed
    changed_params = detect_changes(runs)

    # Define parameter groups and their display order
    always_show = ["run_id", "timestamp", "has_final_model", "max_checkpoint_epoch"]

    model_params = ["architecture", "filter_size", "dropout_ratio"]
    training_params = [
        "batch_size",
        "epochs",
        "batches_per_epoch",
        "early_stopping",
        "patience",
    ]
    optimizer_params = [
        "gen_lr",
        "gen_optimizer",
        "gen_weight_decay",
        "disc_lr",
        "disc_optimizer",
        "disc_weight_decay",
    ]
    loss_params = ["loss_l1", "loss_gan"]
    data_params = ["domain", "var_target", "normalization"]

    # Select parameters to show
    if show_all:
        params_to_show = (
            always_show
            + model_params
            + training_params
            + optimizer_params
            + loss_params
            + data_params
        )
    else:
        params_to_show = always_show + sorted(changed_params)

    # Generate header
    headers = []
    header_names = {
        "run_id": "Run ID",
        "timestamp": "Timestamp",
        "has_final_model": "Final Model",
        "max_checkpoint_epoch": "Max Epoch",
        "architecture": "Architecture",
        "filter_size": "Filters",
        "dropout_ratio": "Dropout",
        "batch_size": "Batch Size",
        "epochs": "Epochs",
        "batches_per_epoch": "Batches/Epoch",
        "gen_lr": "Gen LR",
        "gen_optimizer": "Gen Opt",
        "gen_weight_decay": "Gen WD",
        "disc_lr": "Disc LR",
        "disc_optimizer": "Disc Opt",
        "disc_weight_decay": "Disc WD",
        "loss_l1": "L1 Loss",
        "loss_gan": "GAN Loss",
        "early_stopping": "Early Stop",
        "patience": "Patience",
        "domain": "Domain",
        "var_target": "Variable",
        "normalization": "Normalization",
    }

    for param in params_to_show:
        headers.append(header_names.get(param, param))

    # Build table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for run in runs:
        row = []
        for param in params_to_show:
            value = run.get(param, "N/A")
            if param == "timestamp" and value:
                value = value.strftime("%Y-%m-%d %H:%M")
            row.append(format_value(value))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_summary_stats(runs):
    """Generate summary statistics about the runs."""
    if not runs:
        return ""

    lines = []
    lines.append("\n## Summary Statistics\n")
    lines.append(f"- Total runs: {len(runs)}")
    lines.append(
        f"- Runs with final models: {sum(1 for r in runs if r.get('has_final_model'))}"
    )
    lines.append(
        f"- Runs without final models: {sum(1 for r in runs if not r.get('has_final_model'))}"
    )

    # Timestamp range
    timestamps = [r.get("timestamp") for r in runs if r.get("timestamp")]
    if timestamps:
        lines.append(f"- First run: {min(timestamps).strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"- Last run: {max(timestamps).strftime('%Y-%m-%d %H:%M')}")

    # Architecture counts
    architectures = [
        r.get("architecture") for r in runs if r.get("architecture") != "N/A"
    ]
    if architectures:
        arch_counts = defaultdict(int)
        for arch in architectures:
            arch_counts[arch] += 1
        lines.append("\n### Architectures Used")
        for arch, count in sorted(arch_counts.items()):
            lines.append(f"- {arch}: {count} runs")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize ML experiment runs")
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="./runs",
        help="Directory containing run subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs_summary.md",
        help="Output markdown file (default: runs_summary.md)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Show all parameters, not just changed ones"
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Filter runs by substring in run_id"
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    if not runs_dir.exists():
        print(f"Error: {runs_dir} does not exist")
        return

    # Collect all runs
    runs = []
    for run_subdir in sorted(runs_dir.iterdir()):
        if run_subdir.is_dir():
            if args.filter and args.filter not in run_subdir.name:
                continue
            run_info = get_run_info(run_subdir)
            if run_info:
                runs.append(run_info)

    if not runs:
        print("No valid runs found")
        return

    # Generate output
    output = []
    output.append("# ML Benchmark SpaGAN - Run Summary\n")
    output.append(generate_summary_stats(runs))
    output.append("\n## Runs Comparison\n")
    output.append(generate_markdown_table(runs, show_all=args.all))
    output.append("\n")

    result = "\n".join(output)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Summary written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
