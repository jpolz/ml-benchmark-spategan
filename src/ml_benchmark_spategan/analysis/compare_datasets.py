"""
Compare test target data characteristics across different domain/experiment/variable combinations.

This script analyzes the inherent spatial and temporal variability in the test datasets
to understand if certain domains or variables have more/less variable spatial patterns.
"""

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ml_benchmark_spategan.evaluate import scores
from ml_benchmark_spategan.train.dataloader.dataloader import (
    load_cordex_data,
    split_train_test,
)


def load_test_data(domain, training_experiment, var_target, data_path):
    """
    Load test data for a specific configuration.

    Args:
        domain: Domain name (SA, NZ, ALPS)
        training_experiment: Experiment name
        var_target: Target variable
        data_path: Path to data directory

    Returns:
        Tuple of (x_test, y_test) - predictor and target test datasets
    """
    print(f"\nLoading {domain} / {training_experiment} / {var_target}...")

    # Load data
    predictor, predictand = load_cordex_data(
        domain=domain,
        training_experiment=training_experiment,
        var_target=var_target,
        data_path=data_path,
    )

    # Split into train/test
    x_train, y_train, x_test, y_test = split_train_test(
        predictor=predictor,
        predictand=predictand,
        training_experiment=training_experiment,
    )

    return x_test, y_test


def compute_predictor_target_correlation(x_test, y_test, var_target):
    """
    Compute correlation metrics between predictors and target.

    Args:
        x_test: Predictor test dataset
        y_test: Target test dataset
        var_target: Target variable name

    Returns:
        Dictionary with correlation metrics
    """
    from scipy.ndimage import zoom

    corr_stats = {}

    # Get target data
    target = y_test[var_target].values  # Shape: (time, lat, lon)
    target_shape = target.shape

    # Upsample each predictor variable to target resolution and compute correlation
    predictor_vars = [v for v in x_test.data_vars]
    correlations = []
    explained_variances = []

    for var in predictor_vars:
        pred_data = x_test[var].values  # Shape: (time, lat_lr, lon_lr)

        # Upsample predictor to target resolution using bilinear interpolation
        zoom_factors = (
            1,
            target_shape[1] / pred_data.shape[1],
            target_shape[2] / pred_data.shape[2],
        )
        pred_upsampled = zoom(pred_data, zoom_factors, order=1)

        # Flatten spatial dimensions for correlation
        target_flat = target.reshape(target_shape[0], -1)
        pred_flat = pred_upsampled.reshape(pred_upsampled.shape[0], -1)

        # Compute pixel-wise temporal correlation
        pixel_corrs = []
        for i in range(target_flat.shape[1]):
            if np.std(target_flat[:, i]) > 1e-10 and np.std(pred_flat[:, i]) > 1e-10:
                corr = np.corrcoef(target_flat[:, i], pred_flat[:, i])[0, 1]
                if not np.isnan(corr):
                    pixel_corrs.append(corr)

        if pixel_corrs:
            mean_corr = np.mean(pixel_corrs)
            correlations.append(mean_corr)

            # Explained variance (R²)
            explained_var = np.mean([c**2 for c in pixel_corrs])
            explained_variances.append(explained_var)

    # Aggregate statistics
    corr_stats["mean_predictor_correlation"] = (
        float(np.mean(correlations)) if correlations else 0.0
    )
    corr_stats["max_predictor_correlation"] = (
        float(np.max(correlations)) if correlations else 0.0
    )
    corr_stats["mean_explained_variance"] = (
        float(np.mean(explained_variances)) if explained_variances else 0.0
    )
    corr_stats["max_explained_variance"] = (
        float(np.max(explained_variances)) if explained_variances else 0.0
    )

    # Compute spatial correlation (correlation of spatial patterns across time)
    # For each timestep, compute correlation between spatial pattern of predictors (averaged) and target
    spatial_correlations = []
    for t in range(target_shape[0]):
        target_t = target[t].flatten()

        # Average all upsampled predictors for this timestep
        pred_avg_t = np.zeros_like(target_t)
        for var in predictor_vars:
            pred_data = x_test[var].values
            pred_upsampled = zoom(
                pred_data[t],
                (
                    target_shape[1] / pred_data.shape[1],
                    target_shape[2] / pred_data.shape[2],
                ),
                order=1,
            )
            pred_avg_t += pred_upsampled.flatten()
        pred_avg_t /= len(predictor_vars)

        if np.std(target_t) > 1e-10 and np.std(pred_avg_t) > 1e-10:
            spatial_corr = np.corrcoef(target_t, pred_avg_t)[0, 1]
            if not np.isnan(spatial_corr):
                spatial_correlations.append(spatial_corr)

    corr_stats["mean_spatial_correlation"] = (
        float(np.mean(spatial_correlations)) if spatial_correlations else 0.0
    )
    corr_stats["std_spatial_correlation"] = (
        float(np.std(spatial_correlations)) if spatial_correlations else 0.0
    )

    # Temporal autocorrelation of target (persistence)
    target_autocorr = []
    target_flat = target.reshape(target_shape[0], -1)
    for i in range(target_flat.shape[1]):
        if np.std(target_flat[:, i]) > 1e-10:
            # Lag-1 autocorrelation
            if target_shape[0] > 1:
                corr = np.corrcoef(target_flat[:-1, i], target_flat[1:, i])[0, 1]
                if not np.isnan(corr):
                    target_autocorr.append(corr)

    corr_stats["mean_temporal_autocorr"] = (
        float(np.mean(target_autocorr)) if target_autocorr else 0.0
    )
    corr_stats["std_temporal_autocorr"] = (
        float(np.std(target_autocorr)) if target_autocorr else 0.0
    )

    return corr_stats


def compute_dataset_statistics(x_test, y_test, var_target, domain):
    """
    Compute comprehensive statistics for a test dataset.

    Args:
        x_test: Predictor test dataset
        y_test: Target test dataset
        var_target: Variable name
        domain: Domain name

    Returns:
        Dictionary with statistics
    """
    data = y_test[var_target]

    stats = {}

    # Compute predictor-target correlations
    print("  Computing predictor-target correlations...")
    corr_stats = compute_predictor_target_correlation(x_test, y_test, var_target)
    stats.update(corr_stats)

    # Temporal statistics (for each pixel)
    stats["temporal_mean"] = data.mean(dim="time")
    stats["temporal_std"] = data.std(dim="time")
    stats["temporal_var"] = data.var(dim="time")
    stats["temporal_cv"] = stats["temporal_std"] / (stats["temporal_mean"] + 1e-10)

    # Spatial statistics (for each timestep)
    stats["spatial_mean"] = data.mean(
        dim=["lat" if "lat" in data.dims else "x", "lon" if "lon" in data.dims else "y"]
    )
    stats["spatial_std"] = data.std(
        dim=["lat" if "lat" in data.dims else "x", "lon" if "lon" in data.dims else "y"]
    )
    stats["spatial_var"] = data.var(
        dim=["lat" if "lat" in data.dims else "x", "lon" if "lon" in data.dims else "y"]
    )

    # Overall statistics
    stats["overall_mean"] = float(data.mean().values)
    stats["overall_std"] = float(data.std().values)
    stats["overall_min"] = float(data.min().values)
    stats["overall_max"] = float(data.max().values)

    # Spatial pattern variability (how much do spatial patterns vary over time?)
    # Std dev across time of spatial std dev
    stats["spatial_pattern_variability"] = float(stats["spatial_std"].std().values)

    # Temporal pattern variability (how much do temporal patterns vary across space?)
    # Std dev across space of temporal std dev
    stats["temporal_pattern_variability"] = float(stats["temporal_std"].std().values)

    # Quantiles
    stats["q05"] = float(data.quantile(0.05).values)
    stats["q25"] = float(data.quantile(0.25).values)
    stats["q50"] = float(data.quantile(0.50).values)
    stats["q75"] = float(data.quantile(0.75).values)
    stats["q95"] = float(data.quantile(0.95).values)
    stats["q99"] = float(data.quantile(0.99).values)

    # Compute power spectral density using scores.psd (requires two datasets)
    # We'll use the same dataset twice to just get one PSD
    psd_data, _ = scores.psd(x0=y_test, x1=y_test, var=var_target)
    stats["psd"] = psd_data

    return stats


def plot_dataset_comparison(datasets_stats, output_dir):
    """
    Create comprehensive comparison plots across datasets.

    Args:
        datasets_stats: Dictionary mapping dataset names to their statistics
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = list(datasets_stats.keys())
    n_datasets = len(dataset_names)

    # Create summary statistics comparison
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)

    # 1. Predictor-target correlation comparison
    ax1 = fig.add_subplot(gs[0, 0])
    pred_corrs = [
        datasets_stats[d]["mean_predictor_correlation"] for d in dataset_names
    ]
    max_corrs = [datasets_stats[d]["max_predictor_correlation"] for d in dataset_names]
    x_pos = np.arange(n_datasets)
    width = 0.35
    ax1.bar(
        x_pos - width / 2, pred_corrs, width, label="Mean", alpha=0.7, color="steelblue"
    )
    ax1.bar(x_pos + width / 2, max_corrs, width, label="Max", alpha=0.7, color="navy")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax1.set_ylabel("Correlation", fontsize=10)
    ax1.set_title("Predictor-Target Correlation", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # 2. Explained variance
    ax2 = fig.add_subplot(gs[0, 1])
    exp_vars = [datasets_stats[d]["mean_explained_variance"] for d in dataset_names]
    max_exp_vars = [datasets_stats[d]["max_explained_variance"] for d in dataset_names]
    ax2.bar(x_pos - width / 2, exp_vars, width, label="Mean", alpha=0.7, color="coral")
    ax2.bar(
        x_pos + width / 2, max_exp_vars, width, label="Max", alpha=0.7, color="darkred"
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax2.set_ylabel("R² (Explained Variance)", fontsize=10)
    ax2.set_title("Predictor-Target Explained Variance", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # 3. Spatial correlation
    ax3 = fig.add_subplot(gs[0, 2])
    spatial_corrs = [
        datasets_stats[d]["mean_spatial_correlation"] for d in dataset_names
    ]
    spatial_stds = [datasets_stats[d]["std_spatial_correlation"] for d in dataset_names]
    ax3.bar(
        x_pos,
        spatial_corrs,
        yerr=spatial_stds,
        capsize=3,
        alpha=0.7,
        color="mediumseagreen",
    )
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax3.set_ylabel("Spatial Correlation ± Std", fontsize=10)
    ax3.set_title("Spatial Pattern Correlation", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    # 4. Temporal autocorrelation
    ax4 = fig.add_subplot(gs[0, 3])
    temp_autocorrs = [
        datasets_stats[d]["mean_temporal_autocorr"] for d in dataset_names
    ]
    temp_autocorr_stds = [
        datasets_stats[d]["std_temporal_autocorr"] for d in dataset_names
    ]
    ax4.bar(
        x_pos,
        temp_autocorrs,
        yerr=temp_autocorr_stds,
        capsize=3,
        alpha=0.7,
        color="mediumpurple",
    )
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax4.set_ylabel("Lag-1 Autocorrelation ± Std", fontsize=10)
    ax4.set_title("Temporal Persistence", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # 5. Overall mean comparison
    ax5 = fig.add_subplot(gs[1, 0])
    means = [datasets_stats[d]["overall_mean"] for d in dataset_names]
    stds = [datasets_stats[d]["overall_std"] for d in dataset_names]
    ax5.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7, color="steelblue")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax5.set_ylabel("Mean ± Std Dev", fontsize=10)
    ax5.set_title("Overall Statistics", fontsize=12, fontweight="bold")
    ax5.grid(axis="y", alpha=0.3)

    # 6. Data range comparison
    ax6 = fig.add_subplot(gs[1, 1])
    mins = [datasets_stats[d]["overall_min"] for d in dataset_names]
    maxs = [datasets_stats[d]["overall_max"] for d in dataset_names]
    ranges = [maxs[i] - mins[i] for i in range(n_datasets)]
    ax6.bar(x_pos, ranges, alpha=0.7, color="coral")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax6.set_ylabel("Max - Min", fontsize=10)
    ax6.set_title("Data Range", fontsize=12, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)

    # 7. Spatial pattern variability
    ax7 = fig.add_subplot(gs[1, 2])
    spatial_var = [
        datasets_stats[d]["spatial_pattern_variability"] for d in dataset_names
    ]
    ax7.bar(x_pos, spatial_var, alpha=0.7, color="mediumseagreen")
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax7.set_ylabel("Std(Spatial Std)", fontsize=10)
    ax7.set_title("Spatial Pattern Variability", fontsize=12, fontweight="bold")
    ax7.grid(axis="y", alpha=0.3)

    # 8. Temporal pattern variability
    ax8 = fig.add_subplot(gs[1, 3])
    temporal_var = [
        datasets_stats[d]["temporal_pattern_variability"] for d in dataset_names
    ]
    ax8.bar(x_pos, temporal_var, alpha=0.7, color="mediumpurple")
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_names], fontsize=8, rotation=0
    )
    ax8.set_ylabel("Std(Temporal Std)", fontsize=10)
    ax8.set_title("Temporal Pattern Variability", fontsize=12, fontweight="bold")
    ax8.grid(axis="y", alpha=0.3)

    # 9. Quantile comparison
    ax9 = fig.add_subplot(gs[2, :2])
    quantiles = ["q05", "q25", "q50", "q75", "q95", "q99"]
    for i, dataset in enumerate(dataset_names):
        q_vals = [datasets_stats[dataset][q] for q in quantiles]
        ax9.plot(
            quantiles, q_vals, "o-", linewidth=2, markersize=8, label=dataset, alpha=0.8
        )
    ax9.set_xlabel("Quantile", fontsize=10)
    ax9.set_ylabel("Value", fontsize=10)
    ax9.set_title("Quantile Comparison", fontsize=12, fontweight="bold")
    ax9.legend(fontsize=8, loc="best")
    ax9.grid(alpha=0.3)

    # 10. PSD comparison
    ax10 = fig.add_subplot(gs[2, 2:])
    for i, dataset in enumerate(dataset_names):
        psd = datasets_stats[dataset]["psd"]
        wavenumber = psd["wavenumber"].values
        psd_vals = psd.values
        ax10.loglog(wavenumber, psd_vals, linewidth=2, label=dataset, alpha=0.8)
    ax10.set_xlabel("Wavenumber", fontsize=10)
    ax10.set_ylabel("Power Spectral Density", fontsize=10)
    ax10.set_title("Power Spectral Density Comparison", fontsize=12, fontweight="bold")
    ax10.legend(fontsize=8, loc="best")
    ax10.grid(which="both", alpha=0.3)

    # 11. Spatial std distribution boxplot
    ax11 = fig.add_subplot(gs[3, :2])
    spatial_std_data = [datasets_stats[d]["spatial_std"].values for d in dataset_names]
    bp = ax11.boxplot(
        spatial_std_data,
        labels=[d.replace("_", "\n") for d in dataset_names],
        patch_artist=True,
        showmeans=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    ax11.set_ylabel("Spatial Std Dev (per timestep)", fontsize=10)
    ax11.set_title("Spatial Variability Distribution", fontsize=12, fontweight="bold")
    ax11.grid(axis="y", alpha=0.3)

    # 12. Temporal std distribution boxplot
    ax12 = fig.add_subplot(gs[3, 2:])
    temporal_std_data = [
        datasets_stats[d]["temporal_std"].values.flatten() for d in dataset_names
    ]
    bp = ax12.boxplot(
        temporal_std_data,
        labels=[d.replace("_", "\n") for d in dataset_names],
        patch_artist=True,
        showmeans=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightcoral")
        patch.set_alpha(0.7)
    ax12.set_ylabel("Temporal Std Dev (per pixel)", fontsize=10)
    ax12.set_title("Temporal Variability Distribution", fontsize=12, fontweight="bold")
    ax12.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Dataset Characteristics Comparison - Test Data (with Predictor-Target Analysis)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    output_path = output_dir / "dataset_comparison_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nDataset comparison plot saved to {output_path}")


def plot_spatial_maps(
    datasets_stats, y_test_dict, var_target_dict, domain_dict, output_dir
):
    """
    Create spatial maps of temporal statistics for each dataset.

    Args:
        datasets_stats: Dictionary mapping dataset names to their statistics
        y_test_dict: Dictionary mapping dataset names to their test data
        var_target_dict: Dictionary mapping dataset names to their variable names
        domain_dict: Dictionary mapping dataset names to their domains
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)

    for dataset_name in datasets_stats.keys():
        y_test = y_test_dict[dataset_name]
        var_target = var_target_dict[dataset_name]
        domain = domain_dict[dataset_name]
        stats = datasets_stats[dataset_name]

        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        projection = ccrs.PlateCarree(central_longitude=180 if domain == "NZ" else 0)

        # Get coordinates
        if "lat" in y_test.coords:
            lon_coord = y_test.lon.values
            lat_coord = y_test.lat.values
        else:
            lon_coord = y_test.x.values
            lat_coord = y_test.y.values

        # 1. Temporal mean
        ax1 = plt.subplot(gs[0, 0], projection=projection)
        im = ax1.pcolormesh(
            lon_coord,
            lat_coord,
            stats["temporal_mean"].values,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
        )
        ax1.coastlines()
        ax1.set_title("Temporal Mean", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax1, orientation="horizontal", pad=0.05, fraction=0.046)

        # 2. Temporal std
        ax2 = plt.subplot(gs[0, 1], projection=projection)
        im = ax2.pcolormesh(
            lon_coord,
            lat_coord,
            stats["temporal_std"].values,
            transform=ccrs.PlateCarree(),
            cmap="YlOrRd",
        )
        ax2.coastlines()
        ax2.set_title("Temporal Std Dev", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax2, orientation="horizontal", pad=0.05, fraction=0.046)

        # 3. Temporal coefficient of variation
        ax3 = plt.subplot(gs[0, 2], projection=projection)
        im = ax3.pcolormesh(
            lon_coord,
            lat_coord,
            stats["temporal_cv"].values,
            transform=ccrs.PlateCarree(),
            cmap="RdYlGn_r",
        )
        ax3.coastlines()
        ax3.set_title("Temporal CV (Std/Mean)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax3, orientation="horizontal", pad=0.05, fraction=0.046)

        # 4. Spatial mean over time
        ax4 = plt.subplot(gs[1, 0])
        ax4.plot(stats["spatial_mean"].values, linewidth=1, alpha=0.7)
        ax4.set_xlabel("Time Step", fontsize=10)
        ax4.set_ylabel("Spatial Mean", fontsize=10)
        ax4.set_title("Spatial Mean Time Series", fontsize=11, fontweight="bold")
        ax4.grid(alpha=0.3)

        # 5. Spatial std over time
        ax5 = plt.subplot(gs[1, 1])
        ax5.plot(stats["spatial_std"].values, linewidth=1, alpha=0.7, color="coral")
        ax5.set_xlabel("Time Step", fontsize=10)
        ax5.set_ylabel("Spatial Std Dev", fontsize=10)
        ax5.set_title("Spatial Std Dev Time Series", fontsize=11, fontweight="bold")
        ax5.grid(alpha=0.3)

        # 6. Histogram of all values
        ax6 = plt.subplot(gs[1, 2])
        all_values = y_test[var_target].values.flatten()
        ax6.hist(all_values, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax6.set_xlabel("Value", fontsize=10)
        ax6.set_ylabel("Frequency", fontsize=10)
        ax6.set_title("Value Distribution", fontsize=11, fontweight="bold")
        ax6.set_yscale("log")
        ax6.grid(alpha=0.3)

        fig.suptitle(
            f"Dataset: {dataset_name}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        safe_name = dataset_name.replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"dataset_spatial_maps_{safe_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Spatial maps saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare test data characteristics across different configurations"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["SA", "NZ"],
        help="List of domains to compare (default: SA NZ)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["ESD_pseudo_reality"],
        help="List of training experiments (default: ESD_pseudo_reality)",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["pr", "tasmax"],
        help="List of target variables (default: pr tasmax)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/bg/fast/aihydromet/cordexbench/",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis/dataset_comparison",
        help="Output directory for plots and statistics",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyze all dataset combinations
    datasets_stats = {}
    y_test_dict = {}
    var_target_dict = {}
    domain_dict = {}

    for domain in args.domains:
        for experiment in args.experiments:
            for var_target in args.variables:
                dataset_name = f"{domain}_{experiment}_{var_target}"

                try:
                    x_test, y_test = load_test_data(
                        domain, experiment, var_target, args.data_path
                    )
                    stats = compute_dataset_statistics(
                        x_test, y_test, var_target, domain
                    )

                    datasets_stats[dataset_name] = stats
                    y_test_dict[dataset_name] = y_test
                    var_target_dict[dataset_name] = var_target
                    domain_dict[dataset_name] = domain

                except Exception as e:
                    print(f"Error loading {dataset_name}: {e}")
                    continue

    if len(datasets_stats) == 0:
        print("No datasets loaded successfully!")
        return

    # Create comparison plots
    print("\n=== Creating comparison plots ===")
    plot_dataset_comparison(datasets_stats, output_dir)

    # Create spatial maps for each dataset
    print("\n=== Creating spatial maps ===")
    plot_spatial_maps(
        datasets_stats, y_test_dict, var_target_dict, domain_dict, output_dir
    )

    # Save statistics to JSON (excluding xarray objects)
    print("\n=== Saving statistics ===")
    stats_to_save = {}
    for dataset_name, stats in datasets_stats.items():
        stats_to_save[dataset_name] = {
            k: v
            for k, v in stats.items()
            if k
            not in [
                "temporal_mean",
                "temporal_std",
                "temporal_var",
                "temporal_cv",
                "spatial_mean",
                "spatial_std",
                "spatial_var",
                "psd",
            ]
        }

    with open(output_dir / "dataset_statistics.json", "w") as f:
        json.dump(stats_to_save, f, indent=2)

    print(f"\nStatistics saved to {output_dir / 'dataset_statistics.json'}")
    print(f"\nAll results saved to {output_dir}")

    # Print summary table with predictor-target correlations
    print("\n" + "=" * 120)
    print("SUMMARY TABLE - Predictor-Target Relationships")
    print("=" * 120)
    print(
        f"{'Dataset':<30} {'Pred Corr':<12} {'Exp Var (R²)':<15} {'Spatial Corr':<15} {'Autocorr':<12}"
    )
    print("-" * 120)
    for dataset_name in datasets_stats.keys():
        stats = datasets_stats[dataset_name]
        print(
            f"{dataset_name:<30} "
            f"{stats['mean_predictor_correlation']:<12.4f} "
            f"{stats['mean_explained_variance']:<15.4f} "
            f"{stats['mean_spatial_correlation']:<15.4f} "
            f"{stats['mean_temporal_autocorr']:<12.4f}"
        )
    print("=" * 120)

    print("\n" + "=" * 100)
    print("SUMMARY TABLE - Dataset Variability")
    print("=" * 100)
    print(
        f"{'Dataset':<30} {'Mean':<12} {'Std':<12} {'Spatial Var':<15} {'Temporal Var':<15}"
    )
    print("-" * 100)
    for dataset_name in datasets_stats.keys():
        stats = datasets_stats[dataset_name]
        print(
            f"{dataset_name:<30} "
            f"{stats['overall_mean']:<12.4f} "
            f"{stats['overall_std']:<12.4f} "
            f"{stats['spatial_pattern_variability']:<15.4f} "
            f"{stats['temporal_pattern_variability']:<15.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
