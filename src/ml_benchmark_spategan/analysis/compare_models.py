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

from ml_benchmark_spategan.analysis.data_utils import prepare_torch_data
from ml_benchmark_spategan.analysis.model_loader import load_model
from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.dataloader.dataloader import (
    EmulationTestDataset,
    load_cordex_data,
    load_orography,
    split_train_test,
)
from ml_benchmark_spategan.utils.normalize import normalize_predictors
from ml_benchmark_spategan.visualization.plot_results import (
    plot_lag1_autocorr_maps,
    plot_prediction_comparison,
    plot_psd_comparison,
)

# Add evaluation directory to path to import diagnostics
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "evaluation"))
import diagnostics
import indices


def compute_psd_score(
    psd_test: xr.DataArray,
    psd_pred: xr.DataArray,
    wavenumber_min: int = 1,
    wavenumber_max: int = 60,
) -> float:
    """
    Compute PSD similarity score as RMSE in log space.

    Lower values indicate better match to the test data's spectral characteristics.

    Args:
        psd_test: Power spectral density of test data
        psd_pred: Power spectral density of predicted data
        wavenumber_min: Minimum wavenumber to include (default: 1)
        wavenumber_max: Maximum wavenumber to include (default: 60)

    Returns:
        RMSE in log10 space
    """
    # Filter by wavenumber range
    wavenumber = psd_test["wavenumber"].values
    mask = (wavenumber >= wavenumber_min) & (wavenumber <= wavenumber_max)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_psd_test = np.log10(psd_test.values[mask] + eps)
    log_psd_pred = np.log10(psd_pred.values[mask] + eps)

    # Compute RMSE in log space
    log_rmse = np.sqrt(np.mean((log_psd_test - log_psd_pred) ** 2))

    return float(log_rmse)


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
    elif predictions.ndim == 3:
        # Shape is (b, h, w) - flatten to (b, h*w)
        predictions = rearrange(predictions, "b h w -> b (h w)")

    print(f"Predictions shape: {predictions.shape}")

    # Determine spatial dimensions
    if domain == "ALPS":
        spatial_dims = ("x", "y")
    elif domain in ["NZ", "SA"]:
        spatial_dims = ("lat", "lon")
    else:
        raise ValueError(f"Invalid domain: {domain}")

    # Convert predictions to xarray
    # Check if we have 2D coordinates (meshgrid) like in ALPS rotated pole
    coord_is_2d = y_test[spatial_dims[0]].ndim > 1 or y_test[spatial_dims[1]].ndim > 1

    if coord_is_2d:
        # For 2D coordinates, we can't use stack (MultiIndex requires 1D coords)
        # Instead, directly assign to reshaped data
        y_pred = y_test.copy(deep=True)
        # Reshape predictions from (time, gridpoint) to (time, dim0, dim1)
        spatial_shape = y_test[var_target].shape[
            1:
        ]  # Get spatial shape (e.g., 128, 128)
        predictions_reshaped = predictions.reshape(predictions.shape[0], *spatial_shape)
        y_pred[var_target].values = predictions_reshaped
    else:
        # For 1D coordinates, use the standard stack/unstack method
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

    # Compute PSD score (RMSE in log space)
    metrics["psd_score"] = compute_psd_score(psd_test, psd_pred)

    # Variable-specific climate indices
    if var_target == "tasmax":
        # Temperature-specific indices
        # Summer days (days > 25°C, threshold=300K for data in Kelvin)
        su_test = indices.su(y_test, var_target, threshold=298.15)  # 25°C in Kelvin
        su_pred = indices.su(y_pred, var_target, threshold=298.15)
        metrics["su_bias"] = float(
            (su_pred[var_target] - su_test[var_target]).mean().values.item()
        )

        # Mean annual maximum temperature
        txx_test = indices.txx(y_test, var_target)
        txx_pred = indices.txx(y_pred, var_target)
        metrics["txx_bias"] = float(
            (txx_pred[var_target] - txx_test[var_target]).mean().values.item()
        )

        # Mean annual minimum temperature
        txn_test = indices.txn(y_test, var_target)
        txn_pred = indices.txn(y_pred, var_target)
        metrics["txn_bias"] = float(
            (txn_pred[var_target] - txn_test[var_target]).mean().values.item()
        )

        print(f"Summer Days Bias: {metrics['su_bias']:.4f}")
        print(f"TXx (Annual Max) Bias: {metrics['txx_bias']:.4f}")
        print(f"TXn (Annual Min) Bias: {metrics['txn_bias']:.4f}")

    elif var_target == "pr":
        # Precipitation-specific indices
        # Maximum 1-day precipitation
        rx1day_test = indices.rx1day(y_test, var_target)
        rx1day_pred = indices.rx1day(y_pred, var_target)
        metrics["rx1day_bias"] = float(
            (rx1day_pred[var_target] - rx1day_test[var_target]).mean().values.item()
        )

        # Simple precipitation intensity (mean precip on wet days)
        sdii_test = indices.sdii(y_test, var_target, wet_threshold=1.0)
        sdii_pred = indices.sdii(y_pred, var_target, wet_threshold=1.0)
        metrics["sdii_bias"] = float(
            (sdii_pred[var_target] - sdii_test[var_target]).mean().values.item()
        )

        # Consecutive dry days
        cdd_test = indices.cdd(y_test, var_target, dry_threshold=1.0)
        cdd_pred = indices.cdd(y_pred, var_target, dry_threshold=1.0)
        metrics["cdd_bias"] = float(
            (cdd_pred[var_target] - cdd_test[var_target]).mean().values.item()
        )

        # Consecutive wet days
        cwd_test = indices.cwd(y_test, var_target, wet_threshold=1.0)
        cwd_pred = indices.cwd(y_pred, var_target, wet_threshold=1.0)
        metrics["cwd_bias"] = float(
            (cwd_pred[var_target] - cwd_test[var_target]).mean().values.item()
        )

        print(f"Rx1day (Max 1-day Precip) Bias: {metrics['rx1day_bias']:.4f}")
        print(f"SDII (Precip Intensity) Bias: {metrics['sdii_bias']:.4f}")
        print(f"CDD (Max Dry Spell) Bias: {metrics['cdd_bias']:.4f}")
        print(f"CWD (Max Wet Spell) Bias: {metrics['cwd_bias']:.4f}")

    # Universal indices (applicable to both variables)
    # Lag-1 autocorrelation
    lag1_test = indices.lag1_corr(y_test, var_target)
    lag1_pred = indices.lag1_corr(y_pred, var_target)
    metrics["lag1_corr_bias"] = float(
        (lag1_pred[var_target] - lag1_test[var_target]).mean().values.item()
    )

    # Store spatial lag-1 fields for visualization
    lag1_test_spatial = lag1_test
    lag1_pred_spatial = lag1_pred

    # Interannual variability
    interann_test = indices.interannual_var(y_test, var_target)
    interann_pred = indices.interannual_var(y_pred, var_target)
    metrics["interannual_var_bias"] = float(
        (interann_pred[var_target] - interann_test[var_target]).mean().values.item()
    )

    # Print metrics
    print(f"Mean RMSE: {metrics['mean_rmse']:.4f}")
    print(f"Mean Bias: {metrics['mean_bias']:.4f}")
    print(f"Mean MAE: {metrics['mean_mae']:.4f}")
    print(f"Mean Correlation: {metrics['mean_correlation']:.4f}")
    print(f"Mean Anomaly Correlation: {metrics['mean_anomaly_correlation']:.4f}")
    print(f"Q95 Bias: {metrics['q95_bias']:.4f}")
    print(f"Q98 Bias: {metrics['q98_bias']:.4f}")
    print(f"Std Ratio: {metrics['std_ratio']:.4f}")
    print(f"PSD Score (log RMSE): {metrics['psd_score']:.4f}")
    print(f"Lag-1 Autocorr Bias: {metrics['lag1_corr_bias']:.4f}")
    print(f"Interannual Var Bias: {metrics['interannual_var_bias']:.4f}")

    return {
        "predictions": y_pred,
        "rmse": rmse,
        "metrics": metrics,
        "psd_test": psd_test,
        "psd_pred": psd_pred,
        "lag1_test": lag1_test_spatial,
        "lag1_pred": lag1_pred_spatial,
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
        x_train_stand, x_test_stand, _, _, _ = normalize_predictors(
            x_train, x_test, y_train, y_test, "standardization"
        )
        x_train_tensor, y_train_tensor = prepare_torch_data(
            x_train_stand, y_train, args.domain
        )
        x_test_tensor, _ = prepare_torch_data(x_test_stand, y_test, args.domain)

        # Create test dataset
        test_dataset = EmulationTestDataset(x_test_tensor)
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

            # Load orography if needed
            orography_da = None
            if cf.data.get("use_orography", False):
                print("Loading orography data...")
                orography_da = load_orography(
                    domain=args.domain,
                    training_experiment=args.experiment,
                    data_path=args.data_path,
                )
                print(f"Orography loaded with shape: {orography_da.shape}")

            x_train_norm, x_test_norm, y_train_norm, y_test_norm, norm_params = (
                normalize_predictors(
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    normalization,
                    orography=orography_da,
                )
            )

            # Get normalized orography from norm_params if it was provided
            orography = None
            if orography_da is not None:
                orography_norm = norm_params.get("orography_norm")
                if orography_norm is not None:
                    orography = torch.from_numpy(orography_norm.values).float()
                    print(f"Normalized orography shape: {orography.shape}")
                else:
                    print("Warning: Orography was not normalized!")

            x_train_tensor, y_train_tensor = prepare_torch_data(
                x_train_norm, y_train_norm, args.domain
            )
            x_test_tensor, _ = prepare_torch_data(x_test_norm, y_test_norm, args.domain)

            # Create test dataloader with model-specific normalization
            test_dataset = EmulationTestDataset(x_test_tensor)
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
                    orography=orography,
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
    print("\n" + "=" * 200)
    print("SUMMARY - ALL METRICS")
    print("=" * 200)

    # Determine which variable-specific metrics to show
    first_result = next(iter(results.values()))
    var_target = args.var_target

    if var_target == "tasmax":
        # Temperature metrics
        print(
            f"{'Model':<30} | {'RMSE':<8} | {'MAE':<8} | {'Corr':<6} | {'AnoCorr':<7} | {'Bias':<8} | "
            f"{'Q95':<8} | {'Q98':<8} | {'StdRatio':<8} | {'PSD':<8} | "
            f"{'SU':<8} | {'TXx':<8} | {'TXn':<8} | {'Lag1':<8} | {'InterAnn':<8}"
        )
        print("-" * 200)
        for model_name, result in results.items():
            metrics = result["metrics"]
            print(
                f"{model_name:30s} | "
                f"{metrics['mean_rmse']:8.4f} | "
                f"{metrics['mean_mae']:8.4f} | "
                f"{metrics['mean_correlation']:6.4f} | "
                f"{metrics['mean_anomaly_correlation']:7.4f} | "
                f"{metrics['mean_bias']:8.4f} | "
                f"{metrics['q95_bias']:8.4f} | "
                f"{metrics['q98_bias']:8.4f} | "
                f"{metrics['std_ratio']:8.4f} | "
                f"{metrics['psd_score']:8.4f} | "
                f"{metrics['su_bias']:8.4f} | "
                f"{metrics['txx_bias']:8.4f} | "
                f"{metrics['txn_bias']:8.4f} | "
                f"{metrics['lag1_corr_bias']:8.4f} | "
                f"{metrics['interannual_var_bias']:8.4f}"
            )
    else:
        # Precipitation metrics
        print(
            f"{'Model':<30} | {'RMSE':<8} | {'MAE':<8} | {'Corr':<6} | {'AnoCorr':<7} | {'Bias':<8} | "
            f"{'Q95':<8} | {'Q98':<8} | {'StdRatio':<8} | {'PSD':<8} | "
            f"{'Rx1day':<8} | {'SDII':<8} | {'CDD':<8} | {'CWD':<8} | {'Lag1':<8} | {'InterAnn':<8}"
        )
        print("-" * 200)
        for model_name, result in results.items():
            metrics = result["metrics"]
            print(
                f"{model_name:30s} | "
                f"{metrics['mean_rmse']:8.4f} | "
                f"{metrics['mean_mae']:8.4f} | "
                f"{metrics['mean_correlation']:6.4f} | "
                f"{metrics['mean_anomaly_correlation']:7.4f} | "
                f"{metrics['mean_bias']:8.4f} | "
                f"{metrics['q95_bias']:8.4f} | "
                f"{metrics['q98_bias']:8.4f} | "
                f"{metrics['std_ratio']:8.4f} | "
                f"{metrics['psd_score']:8.4f} | "
                f"{metrics['rx1day_bias']:8.4f} | "
                f"{metrics['sdii_bias']:8.4f} | "
                f"{metrics['cdd_bias']:8.4f} | "
                f"{metrics['cwd_bias']:8.4f} | "
                f"{metrics['lag1_corr_bias']:8.4f} | "
                f"{metrics['interannual_var_bias']:8.4f}"
            )
    print("=" * 200)

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

            # Plot lag-1 autocorrelation maps
            plot_lag1_autocorr_maps(
                model_name=model_name,
                y_test=y_test,
                y_pred=result["predictions"],
                var_target=args.var_target,
                domain=args.domain,
                output_dir=output_dir,
                lag1_test=result["lag1_test"],
                lag1_pred=result["lag1_pred"],
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
