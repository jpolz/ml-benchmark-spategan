import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path


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