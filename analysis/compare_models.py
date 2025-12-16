"""Comparison and evaluation script for multiple models."""

import argparse
import json
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import xarray as xr
from data_utils import (
    TestDataset,
    load_cordex_data,
    normalize_predictors,
    prepare_torch_data,
    split_train_test,
    standardize_predictors,
)
from einops import rearrange
from model_loader import load_model
from torch.utils.data import DataLoader

from ml_benchmark_spategan.config import config

# Add evaluation directory to path to import diagnostics
sys.path.append("evaluation")
import diagnostics


def evaluate_model(
    model_wrapper,
    test_loader: DataLoader,
    y_test: xr.Dataset,
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

    # Quantiles (95th percentile)
    q95_pred = y_pred[var_target].quantile(0.95, dim="time")
    q95_test = y_test[var_target].quantile(0.95, dim="time")
    metrics["q95_bias"] = float((q95_pred - q95_test).mean().values.item())

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
    print(f"Q95 Bias: {metrics['q95_bias']:.4f}")
    print(f"Std Ratio: {metrics['std_ratio']:.4f}")

    return {
        "predictions": y_pred,
        "rmse": rmse,
        "metrics": metrics,
        "psd_test": psd_test,
        "psd_pred": psd_pred,
        **metrics,  # Include individual metrics for backward compatibility
    }


def plot_psd_comparison(results: dict, output_dir: Path, var_target: str):
    """
    Plot power spectral density comparison for all models.

    Args:
        results: Dictionary with model results including psd_test and psd_pred
        output_dir: Directory to save plots
        var_target: Target variable name
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot test data (only once, it's the same for all models)
    first_result = next(iter(results.values()))
    psd_test = first_result["psd_test"]

    wavenumber = psd_test["wavenumber"].values
    psd_test_vals = psd_test.values

    ax.loglog(
        wavenumber, psd_test_vals, "k-", linewidth=2, label="Test Data", alpha=0.8
    )

    # Plot each model's predictions
    colors = plt.cm.tab10(range(len(results)))
    for i, (model_name, result) in enumerate(results.items()):
        psd_pred = result["psd_pred"]
        psd_pred_vals = psd_pred.values
        ax.loglog(
            wavenumber,
            psd_pred_vals,
            "--",
            linewidth=2,
            label=model_name,
            color=colors[i],
            alpha=0.8,
        )

    ax.set_xlabel("Wavenumber", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(f"Power Spectral Density Comparison - {var_target}", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "psd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nPSD comparison plot saved to {output_dir / 'psd_comparison.png'}")


def plot_prediction_comparison(
    model_name: str,
    y_test: xr.Dataset,
    y_pred: xr.Dataset,
    var_target: str,
    domain: str,
    output_dir: Path,
):
    """
    Plot comparison of predictions vs targets.

    3 rows:
    - Row 1: First sample of target and prediction
    - Row 2: Difference (prediction - target) for first sample
    - Row 3: Climatology (annual average) for target, prediction, and difference

    Args:
        model_name: Name of the model
        y_test: Test target data
        y_pred: Model predictions
        var_target: Target variable name
        domain: Domain name (SA, NZ, ALPS)
        output_dir: Directory to save plot
    """
    # Select colormap and scaling based on variable
    if var_target == "tasmax":
        cmap = "RdYlBu_r"
        diff_cmap = "RdBu_r"
        use_log_scale = False
    elif var_target == "pr":
        # Use better colormap for precipitation with log scale
        # WhiteBlueGreenYellowRed scheme from NCL
        colors_pr = ['#FFFFFF', '#E0F0FF', '#B3D9FF', '#66B3FF', '#3399FF',
                     '#00FF00', '#66FF66', '#99FF33', '#CCFF00', '#FFFF00',
                     '#FFCC00', '#FF9900', '#FF6600', '#FF3300', '#CC0000']
        cmap = mcolors.LinearSegmentedColormap.from_list("precipitation", colors_pr)
        diff_cmap = "BrBG"
        use_log_scale = True
    else:
        cmap = "viridis"
        diff_cmap = "RdBu_r"
        use_log_scale = False

    # Setup projection
    central_longitude = 180 if domain == "NZ" else 0
    projection = ccrs.PlateCarree(central_longitude=central_longitude)

    # Get first sample
    y_test_sample = y_test[var_target].isel(time=0)
    y_pred_sample = y_pred[var_target].isel(time=0)
    diff_sample = y_pred_sample - y_test_sample

    # Compute climatology (annual average over test set)
    y_test_clim = y_test[var_target].mean(dim="time")
    y_pred_clim = y_pred[var_target].mean(dim="time")
    diff_clim = y_pred_clim - y_test_clim

    # Create figure with 3 rows, 3 columns
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Determine vmin/vmax for each row
    if use_log_scale:
        # For precipitation, use log scale with minimum threshold
        min_threshold = 0.01  # mm/day
        vmin_sample = min_threshold
        vmax_sample = max(y_test_sample.max().values, y_pred_sample.max().values)
        vmin_clim = min_threshold
        vmax_clim = max(y_test_clim.max().values, y_pred_clim.max().values)
        
        # Use LogNorm for precipitation
        from matplotlib.colors import LogNorm
        norm_sample = LogNorm(vmin=vmin_sample, vmax=vmax_sample)
        norm_clim = LogNorm(vmin=vmin_clim, vmax=vmax_clim)
    else:
        vmin_sample = min(y_test_sample.min().values, y_pred_sample.min().values)
        vmax_sample = max(y_test_sample.max().values, y_pred_sample.max().values)
        vmin_clim = min(y_test_clim.min().values, y_pred_clim.min().values)
        vmax_clim = max(y_test_clim.max().values, y_pred_clim.max().values)
        norm_sample = None
        norm_clim = None

    # Symmetric range for differences
    diff_max_sample = max(abs(diff_sample.min().values), abs(diff_sample.max().values))
    diff_max_clim = max(abs(diff_clim.min().values), abs(diff_clim.max().values))

    # Row 1: First sample - Target, Prediction, Difference
    titles_row1 = [
        "Target (Sample 1)",
        "Prediction (Sample 1)",
        "Difference (Sample 1)",
    ]
    data_row1 = [y_test_sample, y_pred_sample, diff_sample]
    cmaps_row1 = [cmap, cmap, diff_cmap]
    norms_row1 = [norm_sample, norm_sample, None]

    for col, (title, data, cmap_i, norm_i) in enumerate(
        zip(titles_row1, data_row1, cmaps_row1, norms_row1)
    ):
        ax = fig.add_subplot(gs[0, col], projection=projection)
        
        if col < 2:  # Target and Prediction
            if norm_i is not None:
                im = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap_i,
                    norm=norm_i,
                    add_colorbar=False,
                )
            else:
                im = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap_i,
                    vmin=vmin_sample,
                    vmax=vmax_sample,
                    add_colorbar=False,
                )
        else:  # Difference
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_i,
                vmin=-diff_max_sample,
                vmax=diff_max_sample,
                add_colorbar=False,
            )
        ax.coastlines()
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)

    # Row 2: Climatology - Target, Prediction, Difference
    titles_row2 = [
        "Target Climatology",
        "Prediction Climatology",
        "Difference Climatology",
    ]
    data_row2 = [y_test_clim, y_pred_clim, diff_clim]
    cmaps_row2 = [cmap, cmap, diff_cmap]
    norms_row2 = [norm_clim, norm_clim, None]

    for col, (title, data, cmap_i, norm_i) in enumerate(
        zip(titles_row2, data_row2, cmaps_row2, norms_row2)
    ):
        ax = fig.add_subplot(gs[1, col], projection=projection)
        
        if col < 2:  # Target and Prediction
            if norm_i is not None:
                im = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap_i,
                    norm=norm_i,
                    add_colorbar=False,
                )
            else:
                im = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap_i,
                    vmin=vmin_clim,
                    vmax=vmax_clim,
                    add_colorbar=False,
                )
        else:  # Difference
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_i,
                vmin=-diff_max_clim,
                vmax=diff_max_clim,
                add_colorbar=False,
            )
        ax.coastlines()
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)

    # Row 3: Statistics text
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")

    stats_text = f"""
    Model: {model_name}
    
    Sample Statistics (First Sample):
      Target range: [{y_test_sample.min().values:.2f}, {y_test_sample.max().values:.2f}]
      Prediction range: [{y_pred_sample.min().values:.2f}, {y_pred_sample.max().values:.2f}]
      Difference range: [{diff_sample.min().values:.2f}, {diff_sample.max().values:.2f}]
      Mean difference: {diff_sample.mean().values:.4f}
    
    Climatology Statistics (Mean over {len(y_test.time)} time steps):
      Target range: [{y_test_clim.min().values:.2f}, {y_test_clim.max().values:.2f}]
      Prediction range: [{y_pred_clim.min().values:.2f}, {y_pred_clim.max().values:.2f}]
      Difference range: [{diff_clim.min().values:.2f}, {diff_clim.max().values:.2f}]
      Mean difference: {diff_clim.mean().values:.4f}
    """

    ax_stats.text(
        0.1,
        0.5,
        stats_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
    )

    fig.suptitle(
        f"Prediction Comparison: {model_name} - {var_target}",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Save plot
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"prediction_comparison_{safe_model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Prediction comparison plot saved to {output_path}")


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
        default="./analysis/results",
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
            deepesd, test_loader, y_test, args.var_target, args.domain, "DeepESD"
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
                    gan, test_loader, y_test, args.var_target, args.domain, model_name
                )
            except Exception as e:
                print(f"Error loading GAN from {run_dir}: {e}")
                continue

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<30} | {'RMSE':<8} | {'MAE':<8} | {'Corr':<6} | {'Bias':<8} | {'StdRatio':<8}"
    )
    print("-" * 80)
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(
            f"{model_name:30s} | "
            f"{metrics['mean_rmse']:8.4f} | "
            f"{metrics['mean_mae']:8.4f} | "
            f"{metrics['mean_correlation']:6.4f} | "
            f"{metrics['mean_bias']:8.4f} | "
            f"{metrics['std_ratio']:8.4f}"
        )
    print("=" * 80)

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
