from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


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
        colors_pr = [
            "#FFFFFF",
            "#E0F0FF",
            "#B3D9FF",
            "#66B3FF",
            "#3399FF",
            "#00FF00",
            "#66FF66",
            "#99FF33",
            "#CCFF00",
            "#FFFF00",
            "#FFCC00",
            "#FF9900",
            "#FF6600",
            "#FF3300",
            "#CC0000",
        ]
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

    # Compute temporal variance for each pixel
    y_test_var = y_test[var_target].var(dim="time")
    y_pred_var = y_pred[var_target].var(dim="time")
    var_ratio = (
        y_pred_var / y_test_var
    )  # Variance ratio (>1 = overestimation, <1 = underestimation)

    # Compute pixel-wise RMSE
    pixel_rmse = ((y_pred[var_target] - y_test[var_target]) ** 2).mean(
        dim="time"
    ) ** 0.5

    # Create figure with 5 rows, 3 columns
    fig = plt.figure(figsize=(18, 20))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.3, wspace=0.3)

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
    if not use_log_scale:
        diff_max_sample = 5
        diff_max_clim = 2
    else:
        diff_max_sample = 20
        diff_max_clim = 5

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

    # Row 3: Temporal Variance - Target, Prediction, Ratio
    titles_row3 = [
        "Target Variance",
        "Prediction Variance",
        "Variance Ratio (Pred/Target)",
    ]
    data_row3 = [y_test_var, y_pred_var, var_ratio]
    cmaps_row3 = ["YlOrRd", "YlOrRd", "RdBu_r"]

    for col, (title, data, cmap_i) in enumerate(
        zip(titles_row3, data_row3, cmaps_row3)
    ):
        ax = fig.add_subplot(gs[2, col], projection=projection)

        if col < 2:  # Variance plots
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_i,
                add_colorbar=False,
            )
        else:  # Variance ratio (centered at 1.0)
            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_i,
                vmin=0.5,
                vmax=1.5,
                add_colorbar=False,
            )
        ax.coastlines()
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)

    # Row 4: Pixel-wise RMSE and standard deviations
    titles_row4 = [
        "Pixel-wise RMSE",
        "Target Std Dev",
        "Prediction Std Dev",
    ]
    y_test_std = y_test[var_target].std(dim="time")
    y_pred_std = y_pred[var_target].std(dim="time")
    data_row4 = [pixel_rmse, y_test_std, y_pred_std]
    cmaps_row4 = ["Reds", "viridis", "viridis"]

    for col, (title, data, cmap_i) in enumerate(
        zip(titles_row4, data_row4, cmaps_row4)
    ):
        ax = fig.add_subplot(gs[3, col], projection=projection)
        im = data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap_i,
            add_colorbar=False,
        )
        ax.coastlines()
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, fraction=0.046)

    # Row 5: Statistics text
    ax_stats = fig.add_subplot(gs[4, :])
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
    
    Variance Statistics:
      Target variance range: [{y_test_var.min().values:.4f}, {y_test_var.max().values:.4f}]
      Prediction variance range: [{y_pred_var.min().values:.4f}, {y_pred_var.max().values:.4f}]
      Mean variance ratio: {var_ratio.mean().values:.4f} (1.0 = perfect match)
      
    Pixel-wise RMSE Statistics:
      Min RMSE: {pixel_rmse.min().values:.4f}
      Max RMSE: {pixel_rmse.max().values:.4f}
      Mean RMSE: {pixel_rmse.mean().values:.4f}
      Median RMSE: {pixel_rmse.median().values:.4f}
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


def plot_multi_experiment_comparison(
    results: dict,
    y_test: xr.Dataset,
    var_target: str,
    domain: str,
    output_dir: Path,
):
    """
    Create aggregate comparison plots across multiple experiments.

    Focuses on spatial pattern variability and consistency across experiments.

    Args:
        results: Dictionary with all model results
        y_test: Test target data (same for all models)
        var_target: Target variable name
        domain: Domain name (SA, NZ, ALPS)
        output_dir: Directory to save plot
    """
    print("\n=== Generating multi-experiment comparison ===")

    # Compute spatial statistics for each experiment
    spatial_stats = {}
    pixel_rmse_maps = {}
    variance_ratios = {}
    pred_std_maps = {}

    for model_name, result in results.items():
        y_pred = result["predictions"]

        # Pixel-wise RMSE
        pixel_rmse = ((y_pred[var_target] - y_test[var_target]) ** 2).mean(
            dim="time"
        ) ** 0.5
        pixel_rmse_maps[model_name] = pixel_rmse

        # Variance ratio
        y_test_var = y_test[var_target].var(dim="time")
        y_pred_var = y_pred[var_target].var(dim="time")
        variance_ratio = y_pred_var / y_test_var
        variance_ratios[model_name] = variance_ratio

        # Prediction std dev
        pred_std = y_pred[var_target].std(dim="time")
        pred_std_maps[model_name] = pred_std

        # Aggregate statistics
        spatial_stats[model_name] = {
            "mean_rmse": float(pixel_rmse.mean().values),
            "std_rmse": float(pixel_rmse.std().values),
            "mean_var_ratio": float(variance_ratio.mean().values),
            "std_var_ratio": float(variance_ratio.std().values),
            "mean_pred_std": float(pred_std.mean().values),
            "std_pred_std": float(pred_std.std().values),
        }

    # Extract metric names and values
    model_names = list(spatial_stats.keys())
    n_models = len(model_names)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Heatmap of RMSE statistics across experiments
    ax1 = fig.add_subplot(gs[0, 0])
    rmse_means = [spatial_stats[m]["mean_rmse"] for m in model_names]
    rmse_stds = [spatial_stats[m]["std_rmse"] for m in model_names]
    x_pos = np.arange(n_models)
    ax1.bar(x_pos, rmse_means, yerr=rmse_stds, capsize=3, alpha=0.7, color="steelblue")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(range(1, n_models + 1), fontsize=8)
    ax1.set_xlabel("Experiment ID", fontsize=10)
    ax1.set_ylabel("Mean Pixel-wise RMSE ± Std", fontsize=10)
    ax1.set_title("Spatial RMSE: Mean and Variability", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Variance ratio distribution
    ax2 = fig.add_subplot(gs[0, 1])
    var_ratio_means = [spatial_stats[m]["mean_var_ratio"] for m in model_names]
    var_ratio_stds = [spatial_stats[m]["std_var_ratio"] for m in model_names]
    ax2.bar(
        x_pos, var_ratio_means, yerr=var_ratio_stds, capsize=3, alpha=0.7, color="coral"
    )
    ax2.axhline(y=1.0, color="red", linestyle="--", linewidth=2, label="Perfect match")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(range(1, n_models + 1), fontsize=8)
    ax2.set_xlabel("Experiment ID", fontsize=10)
    ax2.set_ylabel("Mean Variance Ratio ± Std", fontsize=10)
    ax2.set_title(
        "Temporal Variance Ratio (Pred/Target)", fontsize=12, fontweight="bold"
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Prediction std dev consistency
    ax3 = fig.add_subplot(gs[0, 2])
    pred_std_means = [spatial_stats[m]["mean_pred_std"] for m in model_names]
    pred_std_stds = [spatial_stats[m]["std_pred_std"] for m in model_names]
    ax3.bar(
        x_pos,
        pred_std_means,
        yerr=pred_std_stds,
        capsize=3,
        alpha=0.7,
        color="mediumseagreen",
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(range(1, n_models + 1), fontsize=8)
    ax3.set_xlabel("Experiment ID", fontsize=10)
    ax3.set_ylabel("Mean Prediction Std Dev ± Spatial Std", fontsize=10)
    ax3.set_title("Prediction Variability Across Space", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Spatial pattern consistency: Std dev of pixel-wise RMSE across experiments
    ax4 = fig.add_subplot(gs[1, :])
    # Stack all pixel_rmse maps
    rmse_stack = np.stack([pixel_rmse_maps[m].values for m in model_names])
    rmse_across_exp_mean = np.mean(rmse_stack, axis=0)
    rmse_across_exp_std = np.std(rmse_stack, axis=0)

    # Plot as 2D heatmap
    projection = ccrs.PlateCarree(central_longitude=180 if domain == "NZ" else 0)

    ax4a = plt.subplot(gs[1, 0], projection=projection)
    im = ax4a.pcolormesh(
        y_test.lon.values if "lon" in y_test else y_test.x.values,
        y_test.lat.values if "lat" in y_test else y_test.y.values,
        rmse_across_exp_mean,
        transform=ccrs.PlateCarree(),
        cmap="Reds",
    )
    ax4a.coastlines()
    ax4a.set_title(
        "Mean Pixel-wise RMSE\n(across experiments)", fontsize=11, fontweight="bold"
    )
    plt.colorbar(im, ax=ax4a, orientation="horizontal", pad=0.05, fraction=0.046)

    ax4b = plt.subplot(gs[1, 1], projection=projection)
    im = ax4b.pcolormesh(
        y_test.lon.values if "lon" in y_test else y_test.x.values,
        y_test.lat.values if "lat" in y_test else y_test.y.values,
        rmse_across_exp_std,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
    )
    ax4b.coastlines()
    ax4b.set_title(
        "Std Dev of Pixel-wise RMSE\n(across experiments)",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax4b, orientation="horizontal", pad=0.05, fraction=0.046)

    # Coefficient of variation (CV = std/mean)
    rmse_cv = rmse_across_exp_std / (rmse_across_exp_mean + 1e-10)
    ax4c = plt.subplot(gs[1, 2], projection=projection)
    im = ax4c.pcolormesh(
        y_test.lon.values if "lon" in y_test else y_test.x.values,
        y_test.lat.values if "lat" in y_test else y_test.y.values,
        rmse_cv,
        transform=ccrs.PlateCarree(),
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.5,
    )
    ax4c.coastlines()
    ax4c.set_title(
        "RMSE Coefficient of Variation\n(lower = more consistent)",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax4c, orientation="horizontal", pad=0.05, fraction=0.046)

    # 5. Variance ratio spatial consistency
    ax5 = fig.add_subplot(gs[2, :])
    var_ratio_stack = np.stack([variance_ratios[m].values for m in model_names])
    var_ratio_across_exp_mean = np.mean(var_ratio_stack, axis=0)
    var_ratio_across_exp_std = np.std(var_ratio_stack, axis=0)

    ax5a = plt.subplot(gs[2, 0], projection=projection)
    im = ax5a.pcolormesh(
        y_test.lon.values if "lon" in y_test else y_test.x.values,
        y_test.lat.values if "lat" in y_test else y_test.y.values,
        var_ratio_across_exp_mean,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=0.5,
        vmax=1.5,
    )
    ax5a.coastlines()
    ax5a.set_title(
        "Mean Variance Ratio\n(across experiments)", fontsize=11, fontweight="bold"
    )
    plt.colorbar(im, ax=ax5a, orientation="horizontal", pad=0.05, fraction=0.046)

    ax5b = plt.subplot(gs[2, 1], projection=projection)
    im = ax5b.pcolormesh(
        y_test.lon.values if "lon" in y_test else y_test.x.values,
        y_test.lat.values if "lat" in y_test else y_test.y.values,
        var_ratio_across_exp_std,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
    )
    ax5b.coastlines()
    ax5b.set_title(
        "Std Dev of Variance Ratio\n(across experiments)",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax5b, orientation="horizontal", pad=0.05, fraction=0.046)

    # Identify regions with consistent over/under-estimation
    over_under_consistency = np.mean((var_ratio_stack > 1.0).astype(float), axis=0)
    ax5c = plt.subplot(gs[2, 2], projection=projection)
    im = ax5c.pcolormesh(
        y_test.lon.values if "lon" in y_test else y_test.x.values,
        y_test.lat.values if "lat" in y_test else y_test.y.values,
        over_under_consistency,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
    )
    ax5c.coastlines()
    ax5c.set_title(
        "Fraction of Experiments\nwith Variance Ratio > 1",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax5c, orientation="horizontal", pad=0.05, fraction=0.046)

    fig.suptitle(
        f"Multi-Experiment Spatial Pattern Comparison - {var_target} ({n_models} experiments)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Save plot
    output_path = output_dir / f"multi_experiment_comparison_{var_target}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Multi-experiment comparison plot saved to {output_path}")

    # Also save spatial statistics to JSON
    import json

    stats_path = output_dir / f"multi_experiment_spatial_stats_{var_target}.json"
    with open(stats_path, "w") as f:
        json.dump(spatial_stats, f, indent=2)
    print(f"Spatial statistics saved to {stats_path}")


def plot_lag1_autocorr_maps(
    model_name: str,
    y_test: xr.Dataset,
    y_pred: xr.Dataset,
    var_target: str,
    domain: str,
    output_dir: Path,
    lag1_test: xr.Dataset = None,
    lag1_pred: xr.Dataset = None,
):
    """
    Plot spatial maps of lag-1 autocorrelation.

    Shows:
    - Target lag-1 autocorrelation map
    - Prediction lag-1 autocorrelation map
    - Bias map (prediction - target)

    Args:
        model_name: Name of the model
        y_test: Test target data
        y_pred: Model predictions
        var_target: Target variable name
        domain: Domain name (SA, NZ, ALPS)
        output_dir: Directory to save plot
        lag1_test: Pre-computed lag-1 for test data (optional, computed if None)
        lag1_pred: Pre-computed lag-1 for prediction (optional, computed if None)
    """
    # Compute lag-1 autocorrelation if not provided
    if lag1_test is None or lag1_pred is None:
        import sys
        from pathlib import Path

        # Add evaluation directory to path
        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent.parent / "evaluation")
        )
        import indices

        lag1_test = indices.lag1_corr(y_test, var_target)
        lag1_pred = indices.lag1_corr(y_pred, var_target)

    # Extract data arrays
    lag1_test_da = lag1_test[var_target]
    lag1_pred_da = lag1_pred[var_target]
    lag1_bias = lag1_pred_da - lag1_test_da

    # Setup projection
    central_longitude = 180 if domain == "NZ" else 0
    projection = ccrs.PlateCarree(central_longitude=central_longitude)

    # Create figure with 1 row, 3 columns
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.25, wspace=0.3)

    # Common colormap for lag-1 autocorrelation (0 to 1 range typically)
    lag1_cmap = "YlOrRd"
    bias_cmap = "RdBu_r"

    # Determine vmin/vmax
    vmin_lag1 = min(lag1_test_da.min().values, lag1_pred_da.min().values)
    vmax_lag1 = max(lag1_test_da.max().values, lag1_pred_da.max().values)

    # Ensure lag-1 range includes typical values (0 to 1)
    vmin_lag1 = max(0, vmin_lag1)  # Lag-1 typically non-negative
    vmax_lag1 = min(1, vmax_lag1)  # Lag-1 typically <= 1

    # Symmetric range for bias
    bias_max = max(abs(lag1_bias.min().values), abs(lag1_bias.max().values))
    bias_max = max(0.1, bias_max)  # Ensure minimum range for visualization

    # Column 1: Target lag-1 autocorrelation
    ax1 = fig.add_subplot(gs[0, 0], projection=projection)
    im1 = lag1_test_da.plot(
        ax=ax1,
        transform=ccrs.PlateCarree(),
        cmap=lag1_cmap,
        vmin=vmin_lag1,
        vmax=vmax_lag1,
        add_colorbar=False,
    )
    ax1.coastlines()
    ax1.set_title("Target Lag-1 Autocorrelation", fontsize=12, fontweight="bold")
    cbar1 = plt.colorbar(
        im1, ax=ax1, orientation="horizontal", pad=0.05, fraction=0.046
    )
    cbar1.set_label("Autocorrelation", fontsize=10)

    # Add statistics text
    mean_test = float(lag1_test_da.mean().values)
    std_test = float(lag1_test_da.std().values)
    ax1.text(
        0.02,
        0.98,
        f"Mean: {mean_test:.3f}\nStd: {std_test:.3f}",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Column 2: Prediction lag-1 autocorrelation
    ax2 = fig.add_subplot(gs[0, 1], projection=projection)
    im2 = lag1_pred_da.plot(
        ax=ax2,
        transform=ccrs.PlateCarree(),
        cmap=lag1_cmap,
        vmin=vmin_lag1,
        vmax=vmax_lag1,
        add_colorbar=False,
    )
    ax2.coastlines()
    ax2.set_title("Prediction Lag-1 Autocorrelation", fontsize=12, fontweight="bold")
    cbar2 = plt.colorbar(
        im2, ax=ax2, orientation="horizontal", pad=0.05, fraction=0.046
    )
    cbar2.set_label("Autocorrelation", fontsize=10)

    # Add statistics text
    mean_pred = float(lag1_pred_da.mean().values)
    std_pred = float(lag1_pred_da.std().values)
    ax2.text(
        0.02,
        0.98,
        f"Mean: {mean_pred:.3f}\nStd: {std_pred:.3f}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Column 3: Bias (Prediction - Target)
    ax3 = fig.add_subplot(gs[0, 2], projection=projection)
    im3 = lag1_bias.plot(
        ax=ax3,
        transform=ccrs.PlateCarree(),
        cmap=bias_cmap,
        vmin=-bias_max,
        vmax=bias_max,
        add_colorbar=False,
    )
    ax3.coastlines()
    ax3.set_title("Lag-1 Autocorr Bias (Pred - Target)", fontsize=12, fontweight="bold")
    cbar3 = plt.colorbar(
        im3, ax=ax3, orientation="horizontal", pad=0.05, fraction=0.046
    )
    cbar3.set_label("Bias", fontsize=10)

    # Add statistics text
    mean_bias = float(lag1_bias.mean().values)
    rmse_bias = float((lag1_bias**2).mean().values ** 0.5)
    ax3.text(
        0.02,
        0.98,
        f"Mean Bias: {mean_bias:.3f}\nRMSE: {rmse_bias:.3f}",
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        f"Lag-1 Autocorrelation Spatial Patterns: {model_name} - {var_target}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Save plot
    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    output_path = output_dir / f"lag1_autocorr_maps_{safe_model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Lag-1 autocorrelation maps saved to {output_path}")
