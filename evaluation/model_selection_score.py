#!/usr/bin/env python3
"""
Model Selection Score Calculator

Computes a weighted composite score from diagnostic metrics for model selection.
Lower scores indicate better models.

Usage:
    python model_selection_score.py <run_dir>
    python model_selection_score.py <run_dir> --weights-config weights.yaml
    python model_selection_score.py runs/*/diagnostic_history.json --compare
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Default weights for each metric
# Positive weights: lower is better (RMSE, MAE, biases, PSD distance)
# Negative weights: higher is better (correlation, FSS)
DEFAULT_WEIGHTS = {
    # Core performance metrics (high importance)
    "rmse": 1.0,
    "mae": 1.0,
    "bias_mean": 1.0,
    # Distribution metrics (medium-high importance)
    "bias_q95": 1.5,
    "bias_q98": 1.5,
    "std_ratio": 1.0,  # deviation from 1.0
    # Correlation metrics (high importance, negative = higher is better)
    "correlation": 1.0,
    "anomaly_correlation": 1.0,
    # Spatial pattern metrics (medium importance)
    "psd_distance": 5.0,
    "fss": 1.0,  # negative = higher is better
    "ensemble_std": 1.0,  # ensemble variability (lower = more consistent)
    # Temporal metrics (medium importance)
    "lag1_corr_bias": 1.0,
    "interannual_var_bias": 1.0,
    # Variable-specific metrics (lower importance)
    "su_bias": 0.1,
    "txx_bias": 0.5,
    "txn_bias": 0.5,
    "rx1day_bias": 0.5,
    "sdii_bias": 0.5,
    "cdd_bias": 0.5,
    "cwd_bias": 0.5,
}


def load_diagnostic_history(json_path: Path) -> Dict:
    """Load diagnostic history from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def load_weights_config(config_path: Optional[Path]) -> Dict[str, float]:
    """Load custom weights from YAML config file."""
    if config_path is None:
        return DEFAULT_WEIGHTS.copy()

    with open(config_path, "r") as f:
        custom_weights = yaml.safe_load(f)

    # Merge with defaults
    weights = DEFAULT_WEIGHTS.copy()
    weights.update(custom_weights)
    return weights


def normalize_metric(
    values: List[float], metric_name: str, weights: Dict[str, float]
) -> float:
    """
    Normalize a metric to a comparable scale.

    For all metrics, returns absolute value of deviation from ideal.
    For std_ratio and correlations, measures deviation from 1.0.
    For other metrics where higher is better (negative weight), negates first.

    Returns the last epoch value (assumes later epochs are better trained).
    """
    if len(values) == 0:
        return 0.0

    # Use last epoch value (most trained)
    value = values[-1]

    # Metrics where ideal value is 1.0
    if metric_name in ["std_ratio", "correlation", "anomaly_correlation", "fss"]:
        value = abs(value - 1.0)
    else:
        # For metrics where higher is better (negative weight), negate first
        if weights.get(metric_name, 0) < 0:
            value = -value
        # Then take absolute value so all contributions are positive
        value = abs(value)

    return value


def compute_score_at_epoch(
    diagnostic_history: Dict, weights: Dict[str, float], epoch_idx: int
) -> float:
    """
    Compute composite score at a specific epoch index.

    Args:
        diagnostic_history: Full diagnostic history
        weights: Metric weights
        epoch_idx: Index into the epochs list

    Returns:
        Composite score at that epoch
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for metric_name, weight in weights.items():
        if metric_name not in diagnostic_history:
            continue

        values = diagnostic_history[metric_name]
        if len(values) == 0 or epoch_idx >= len(values):
            continue

        value = values[epoch_idx]

        # Metrics where ideal value is 1.0
        if metric_name in ["std_ratio", "correlation", "anomaly_correlation"]:
            value = abs(value - 1.0)
        else:
            # For metrics where higher is better (negative weight), negate first
            if weight < 0:
                value = -value
            # Then take absolute value so all contributions are positive
            value = abs(value)

        # Apply weight
        weighted_value = abs(weight) * value
        weighted_sum += weighted_value
        total_weight += abs(weight)

    if total_weight > 0:
        return weighted_sum / total_weight
    return float("inf")


def compute_score_evolution(
    diagnostic_history: Dict, weights: Dict[str, float]
) -> tuple[List[int], List[float]]:
    """
    Compute composite score evolution across all epochs.

    Returns:
        Tuple of (epochs, scores)
    """
    epochs = diagnostic_history.get("epochs", [])
    if not epochs:
        return [], []

    scores = []
    for i in range(len(epochs)):
        score = compute_score_at_epoch(diagnostic_history, weights, i)
        scores.append(score)

    return epochs, scores


def compute_composite_score(
    diagnostic_history: Dict, weights: Dict[str, float], verbose: bool = False
) -> Dict:
    """
    Compute weighted composite score from diagnostic history.

    Returns:
        Dictionary with 'score', 'components', and 'epoch'
    """
    epochs = diagnostic_history.get("epochs", [])
    if not epochs:
        raise ValueError("No epochs found in diagnostic history")

    last_epoch = epochs[-1]

    # Compute weighted components
    components = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for metric_name, weight in weights.items():
        if metric_name not in diagnostic_history:
            continue

        values = diagnostic_history[metric_name]
        if len(values) == 0:
            continue

        # Normalize and get last epoch value
        normalized_value = normalize_metric(values, metric_name, weights)

        # Apply weight
        weighted_value = abs(weight) * normalized_value

        components[metric_name] = {
            "raw_value": values[-1],
            "normalized_value": normalized_value,
            "weight": weight,
            "weighted_contribution": weighted_value,
        }

        weighted_sum += weighted_value
        total_weight += abs(weight)

    # Compute final score (normalized by total weight)
    if total_weight > 0:
        score = weighted_sum / total_weight
    else:
        score = float("inf")

    result = {
        "score": score,
        "epoch": last_epoch,
        "total_weight": total_weight,
        "components": components,
    }

    if verbose:
        print(f"\nComposite Score: {score:.4f} (epoch {last_epoch})")
        print(f"Total weight: {total_weight:.1f}")
        print("\nTop contributors:")
        sorted_components = sorted(
            components.items(),
            key=lambda x: x[1]["weighted_contribution"],
            reverse=True,
        )
        for metric, data in sorted_components[:10]:
            print(
                f"  {metric:25s}: {data['weighted_contribution']:8.4f} "
                f"(raw={data['raw_value']:8.4f}, weight={data['weight']:6.2f})"
            )

    return result


def plot_score_evolution(
    run_paths: List[Path], weights: Dict[str, float], output_path: Optional[Path] = None
):
    """
    Plot composite score evolution over epochs for one or more runs.

    Args:
        run_paths: List of run directories or JSON files
        weights: Metric weights
        output_path: Where to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, run_path in enumerate(run_paths):
        try:
            if run_path.is_file() and run_path.name == "diagnostic_history.json":
                json_path = run_path
                run_dir = run_path.parent
            else:
                json_path = run_path / "diagnostic_history.json"
                run_dir = run_path

            if not json_path.exists():
                print(f"Warning: {json_path} not found, skipping")
                continue

            diag_history = load_diagnostic_history(json_path)
            epochs, scores = compute_score_evolution(diag_history, weights)

            if not epochs:
                continue

            color = colors[idx % len(colors)]
            label = run_dir.name[:30]  # Truncate long names
            plt.plot(
                epochs,
                scores,
                "o-",
                linewidth=2,
                markersize=4,
                color=color,
                label=label,
                alpha=0.8,
            )

        except Exception as e:
            print(f"Error processing {run_path}: {e}")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Composite Score (lower is better)", fontsize=12)
    plt.title("Model Selection Score Evolution", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.savefig("score_evolution.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to: score_evolution.png")

    plt.close()


def compare_runs(run_paths: List[Path], weights: Dict[str, float]):
    """Compare multiple runs and rank by composite score."""
    results = []

    for run_path in run_paths:
        try:
            if run_path.is_file() and run_path.name == "diagnostic_history.json":
                json_path = run_path
                run_dir = run_path.parent
            else:
                json_path = run_path / "diagnostic_history.json"
                run_dir = run_path

            if not json_path.exists():
                print(f"Warning: {json_path} not found, skipping")
                continue

            diag_history = load_diagnostic_history(json_path)
            score_data = compute_composite_score(diag_history, weights, verbose=False)

            results.append(
                {
                    "run_dir": run_dir.name,
                    "run_path": str(run_dir),
                    "score": score_data["score"],
                    "epoch": score_data["epoch"],
                }
            )
        except Exception as e:
            print(f"Error processing {run_path}: {e}")

    # Sort by score (lower is better)
    results.sort(key=lambda x: x["score"])

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL SELECTION RANKING (lower score = better)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Score':<12} {'Epoch':<8} {'Run Directory'}")
    print("-" * 80)

    for rank, result in enumerate(results, 1):
        print(
            f"{rank:<6} {result['score']:<12.4f} {result['epoch']:<8} {result['run_dir']}"
        )

    print("=" * 80)

    # Show best run details
    if results:
        print(f"\n🏆 Best run: {results[0]['run_dir']}")
        print("\nDetailed breakdown for best run:")
        best_json = Path(results[0]["run_path"]) / "diagnostic_history.json"
        diag_history = load_diagnostic_history(best_json)
        compute_composite_score(diag_history, weights, verbose=True)


def save_weights_template(output_path: Path):
    """Save a template weights configuration file."""
    with open(output_path, "w") as f:
        f.write("# Model Selection Weights Configuration\n")
        f.write("# Positive weights: lower is better (RMSE, MAE, biases)\n")
        f.write("# Negative weights: higher is better (correlation, FSS)\n")
        f.write("# Set weight to 0 to exclude a metric\n\n")

        yaml.dump(DEFAULT_WEIGHTS, f, default_flow_style=False, sort_keys=False)

    print(f"Weights template saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute composite score for model selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directory or diagnostic_history.json file(s)",
    )
    parser.add_argument(
        "--weights-config", type=Path, help="YAML file with custom weights (optional)"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple runs and show ranking"
    )
    parser.add_argument(
        "--save-weights-template",
        type=Path,
        help="Save a template weights config file and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed component breakdown"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plot of score evolution over epochs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for plot (default: score_evolution.png)",
    )

    args = parser.parse_args()

    # Save template and exit if requested
    if args.save_weights_template:
        save_weights_template(args.save_weights_template)
        return

    # Load weights
    weights = load_weights_config(args.weights_config)

    # Generate plot if requested
    if args.plot:
        plot_score_evolution(args.paths, weights, args.output)
        if not args.compare:
            return

    # Process paths
    if args.compare or len(args.paths) > 1:
        compare_runs(args.paths, weights)
    else:
        # Single run
        run_path = args.paths[0]

        if run_path.is_file() and run_path.name == "diagnostic_history.json":
            json_path = run_path
        else:
            json_path = run_path / "diagnostic_history.json"

        if not json_path.exists():
            print(f"Error: {json_path} not found")
            return

        diag_history = load_diagnostic_history(json_path)
        score_data = compute_composite_score(
            diag_history, weights, verbose=args.verbose
        )

        print(f"\nRun: {run_path.name if run_path.is_dir() else run_path.parent.name}")
        print(f"Composite Score: {score_data['score']:.4f}")
        print(f"Epoch: {score_data['epoch']}")

        if not args.verbose:
            print("\nUse --verbose for detailed breakdown")


if __name__ == "__main__":
    main()
