"""Comparison and evaluation script for multiple models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch.utils.data import DataLoader

from ml_benchmark_spategan.analysis.data_utils import (
    TestDataset,
    load_cordex_data,
    normalize_predictors,
    prepare_torch_data,
    split_train_test,
    standardize_predictors,
)
from ml_benchmark_spategan.analysis.model_loader import load_model
from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.visualization.plot_results import (
    plot_prediction_comparison,
    plot_psd_comparison,
)

# Add evaluation directory to path to import diagnostics
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "evaluation"))
import diagnostics


def evaluate_model(
    model_wrapper,
    test_loader: DataLoader,
    y_test: xr.Dataset,
    y_train: xr.Dataset,
    var_target: str,
    domain: str,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a model on test data with comprehensive metrics.

    Args:
        model_wrapper: Model wrapper instance
        test_loader: DataLoader for test data
        y_test: Test target data (xarray)
        y_train: Training target data (xarray) for climatology
        var_target: Target variable name
        domain: Domain name
        model_name: Name for logging

    Returns:
        Dictionary with predictions and metrics
    """
    print(f"\n=== Evaluating {model_name} ===")

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for batch_x in test_loader:
            outputs = model_wrapper.predict(batch_x)
            predictions.append(outputs.cpu().numpy())

    # Concatenate predictions
    predictions = np.concatenate(predictions, axis=0)

    # Reshape if needed (for generators: (B, 1, H, W) -> (B, H*W))
    if predictions.ndim == 4:
        predictions = rearrange(predictions, "b 1 h w -> b (h w)")

    print(f"Predictions shape: {predictions.shape}")

    # Determine spatial dimensions
    if domain == "ALPS":
        spatial_dims = ("x", "y")
    elif domain in ["NZ", "SA"]:
        spatial_dims = ("lat", "lon")
    else:
        raise ValueError(f"Invalid domain: {domain}")

    # Convert predictions to xarray
    y_pred_stack = y_test.stack(gridpoint=spatial_dims).copy(deep=True)
    y_pred_stack[var_target].values = predictions
    y_pred = y_pred_stack.unstack()

    # Calculate metrics
    metrics = {}

    # RMSE
    rmse = diagnostics.rmse(x0=y_test, x1=y_pred, var=var_target, dim="time")
    metrics["mean_rmse"] = float(rmse[var_target].mean().values.item())

    # Bias (mean error)
    bias = (y_pred[var_target] - y_test[var_target]).mean(dim="time")
    metrics["mean_bias"] = float(bias.mean().values.item())

    # MAE (Mean Absolute Error)
    mae = np.abs(y_pred[var_target] - y_test[var_target]).mean(dim="time")
    metrics["mean_mae"] = float(mae.mean().values.item())

    # Correlation
    corr = xr.corr(y_pred[var_target], y_test[var_target], dim="time")
    metrics["mean_correlation"] = float(corr.mean().values.item())

    # Anomaly Correlation (subtract climatology from training data)
    y_train_clim = y_train[var_target].mean(dim="time")
    y_test_anom = y_test[var_target] - y_train_clim
    y_pred_anom = y_pred[var_target] - y_train_clim
    # compute correlation in space
    anom_corr = xr.corr(y_pred_anom, y_test_anom, dim=spatial_dims)
    metrics["mean_anomaly_correlation"] = float(anom_corr.mean().values.item())

    # Quantiles (95th percentile)
    q95_pred = y_pred[var_target].quantile(0.95, dim="time")
    q95_test = y_test[var_target].quantile(0.95, dim="time")
    metrics["q95_bias"] = float((q95_pred - q95_test).mean().values.item())

    # Quantiles (98th percentile)
    q98_pred = y_pred[var_target].quantile(0.98, dim="time")
    q98_test = y_test[var_target].quantile(0.98, dim="time")
    metrics["q98_bias"] = float((q98_pred - q98_test).mean().values.item())

    # Standard deviation ratio
    std_pred = y_pred[var_target].std(dim="time")
    std_test = y_test[var_target].std(dim="time")
    metrics["std_ratio"] = float((std_pred / std_test).mean().values.item())

    # Power Spectral Density
    psd_test, psd_pred = diagnostics.psd(x0=y_test, x1=y_pred, var=var_target)

    # Print metrics
    print(f"Mean RMSE: {metrics['mean_rmse']:.4f}")
    print(f"Mean Bias: {metrics['mean_bias']:.4f}")
    print(f"Mean MAE: {metrics['mean_mae']:.4f}")
    print(f"Mean Correlation: {metrics['mean_correlation']:.4f}")
    print(f"Mean Anomaly Correlation: {metrics['mean_anomaly_correlation']:.4f}")
    print(f"Q95 Bias: {metrics['q95_bias']:.4f}")
    print(f"Q98 Bias: {metrics['q98_bias']:.4f}")
    print(f"Std Ratio: {metrics['std_ratio']:.4f}")

    return {
        "predictions": y_pred,
        "rmse": rmse,
        "metrics": metrics,
        "psd_test": psd_test,
        "psd_pred": psd_pred,
        **metrics,  # Include individual metrics for backward compatibility
    }


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models")
    parser.add_argument(
        "--domain",
        type=str,
        default="SA",
        choices=["SA", "NZ", "ALPS"],
        help="Domain for evaluation",
    )
    parser.add_argument(
        "--var-target", type=str, default="tasmax", help="Target variable"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ESD_pseudo_reality",
        help="Training experiment name",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/bg/fast/aihydromet/cordexbench/",
        help="Path to CORDEX data",
    )
    parser.add_argument(
        "--deepesd-model",
        type=str,
        default="./training/models/model.pt",
        help="Path to DeepESD model weights",
    )
    parser.add_argument(
        "--gan-runs", type=str, nargs="+", help="List of GAN run directories to compare"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Checkpoint epochs to load for each GAN run (default: final model). Must match length of --gan-runs if provided.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n=== Loading Data ===")
    predictor, predictand = load_cordex_data(
        domain=args.domain,
        training_experiment=args.experiment,
        var_target=args.var_target,
        data_path=args.data_path,
    )

    # Split train/test
    x_train, y_train, x_test, y_test = split_train_test(
        predictor, predictand, args.experiment
    )

    # Results storage
    results = {}

    # Evaluate DeepESD if model exists (uses standardization)
    if Path(args.deepesd_model).exists():
        print("\n=== Loading DeepESD Model ===")

        # DeepESD uses standardization
        x_train_stand, x_test_stand = standardize_predictors(x_train, x_test)
        x_train_tensor, y_train_tensor = prepare_torch_data(
            x_train_stand, y_train, args.domain
        )
        x_test_tensor, _ = prepare_torch_data(x_test_stand, y_test, args.domain)

        # Create test dataloader
        test_dataset = TestDataset(x_test_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )

        deepesd = load_model(
            "deepesd",
            model_path=args.deepesd_model,
            x_shape=x_train_tensor.shape,
            y_shape=y_train_tensor.shape,
            device=device,
        )
        results["DeepESD"] = evaluate_model(
            deepesd,
            test_loader,
            y_test,
            y_train,
            args.var_target,
            args.domain,
            "DeepESD",
        )
    else:
        print(f"\nWarning: DeepESD model not found at {args.deepesd_model}")

    # Evaluate each GAN run
    if args.gan_runs:
        # Validate checkpoint_epochs if provided
        if args.checkpoint_epochs is not None:
            if len(args.checkpoint_epochs) != len(args.gan_runs):
                print(
                    f"Warning: Number of checkpoint epochs ({len(args.checkpoint_epochs)}) "
                    f"does not match number of runs ({len(args.gan_runs)}). Using final models."
                )
                checkpoint_epochs = [None] * len(args.gan_runs)
            else:
                checkpoint_epochs = args.checkpoint_epochs
        else:
            checkpoint_epochs = [None] * len(args.gan_runs)

        for run_dir, checkpoint_epoch in zip(args.gan_runs, checkpoint_epochs):
            run_path = Path(run_dir)
            if not run_path.exists():
                print(f"\nWarning: Run directory not found: {run_dir}")
                continue

            run_id = run_path.name
            if checkpoint_epoch is not None:
                model_name = f"{run_id}_epoch{checkpoint_epoch}"
                print(f"\n=== Loading GAN from {run_id} (epoch {checkpoint_epoch}) ===")
            else:
                model_name = run_id
                print(f"\n=== Loading GAN from {run_id} (final model) ===")

            # Load config
            config_path = run_path / "config.yaml"
            if not config_path.exists():
                print(f"Warning: Config not found in {run_dir}")
                continue

            cf = config.load_config_from_yaml(str(config_path))

            # Get normalization method from config and prepare data accordingly
            normalization = cf.data.get("normalization", "standardization")
            print(f"Using normalization: {normalization}")

            x_train_norm, x_test_norm = normalize_predictors(
                x_train, x_test, normalization
            )
            x_train_tensor, y_train_tensor = prepare_torch_data(
                x_train_norm, y_train, args.domain
            )
            x_test_tensor, _ = prepare_torch_data(x_test_norm, y_test, args.domain)

            # Create test dataloader with model-specific normalization
            test_dataset = TestDataset(x_test_tensor)
            test_loader = DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False
            )

            # Load model
            try:
                gan = load_model(
                    "gan",
                    run_dir=str(run_path),
                    config=cf,
                    checkpoint_epoch=checkpoint_epoch,
                    device=device,
                )
                results[model_name] = evaluate_model(
                    gan,
                    test_loader,
                    y_test,
                    y_train,
                    args.var_target,
                    args.domain,
                    model_name,
                )
            except Exception as e:
                print(f"Error loading GAN from {run_dir}: {e}")
                continue

    # Print summary
    print("\n" + "=" * 95)
    print("SUMMARY")
    print("=" * 95)
    print(
        f"{'Model':<30} | {'RMSE':<8} | {'MAE':<8} | {'Corr':<6} | {'AnoCorr':<7} | {'Bias':<8} | {'StdRatio':<8}"
    )
    print("-" * 95)
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{model_name:30s} | "
            f"{metrics['mean_rmse']:8.4f} | "
            f"{metrics['mean_mae']:8.4f} | "
            f"{metrics['mean_correlation']:6.4f} | "
            f"{metrics['mean_anomaly_correlation']:7.4f} | "
            f"{metrics['mean_bias']:8.4f} | "
            f"{metrics['std_ratio']:8.4f}"
        )
    print("=" * 95)

    # Create PSD comparison plot
    if len(results) > 0:
        print("\nGenerating PSD comparison plot...")
        plot_psd_comparison(results, output_dir, args.var_target)

        # Create prediction comparison plots for each model
        print("\nGenerating prediction comparison plots...")
        for model_name, result in results.items():
            plot_prediction_comparison(
                model_name=model_name,
                y_test=y_test,
                y_pred=result["predictions"],
                var_target=args.var_target,
                domain=args.domain,
                output_dir=output_dir,
            )

    # Save detailed results
    summary = {}
    for name, res in results.items():
        summary[name] = res["metrics"]

    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed metrics saved to {output_dir / 'comparison_summary.json'}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
