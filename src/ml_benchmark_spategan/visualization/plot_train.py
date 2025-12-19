import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec


def plot_losses(loss_train, loss_test, cf):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(loss_train, label="Train Loss")
    axs.plot(loss_test, label="Test Loss")
    axs.legend()
    axs.set_yscale("log")
    axs.grid(True)
    plt.savefig(cf.logging.run_dir + "/losses.png", dpi=150)
    plt.close()


def plot_adversarial_losses(loss_gen_train, loss_disc_train, loss_test_history, cf):
    """
    Plot GAN training losses including generator, discriminator, and individual validation components.

    Parameters
    ----------
    loss_gen_train : list
        Training generator losses per epoch
    loss_disc_train : list
        Training discriminator losses per epoch
    loss_test_history : dict
        Dictionary containing test loss components with keys:
        - 'gen_total': Total generator test loss
        - 'disc_total': Total discriminator test loss
        - 'l1': L1 test loss
        - 'mse': MSE test loss
        - 'gan': GAN adversarial test loss
        - 'fss': FSS test loss
        - 'disc_real': Discriminator loss on real samples
        - 'disc_fake': Discriminator loss on fake samples
    cf : Config
        Configuration object
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Training losses
    ax1.plot(loss_gen_train, label="Generator", linewidth=2)
    ax1.plot(loss_disc_train, label="Discriminator", linewidth=2)
    ax1.legend()
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Training Losses", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Validation - Total losses
    ax2.plot(
        loss_test_history["gen_total"], label="Gen Total", linewidth=2, color="tab:blue"
    )
    ax2.plot(
        loss_test_history["disc_total"],
        label="Disc Total",
        linewidth=2,
        color="tab:orange",
    )
    ax2.set_yscale("log")
    ax2.legend()
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Validation Total Losses", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Validation - Individual components
    components_to_plot = []
    if any(v > 0 for v in loss_test_history.get("l1", [])):
        components_to_plot.append(("l1", "L1", "tab:green"))
    if any(v > 0 for v in loss_test_history.get("mse", [])):
        components_to_plot.append(("mse", "MSE", "tab:red"))
    if any(v > 0 for v in loss_test_history.get("fss", [])):
        components_to_plot.append(("fss", "FSS", "tab:purple"))
    if any(v > 0 for v in loss_test_history.get("gan", [])):
        components_to_plot.append(("gan", "GAN", "tab:brown"))

    for key, label, color in components_to_plot:
        ax3.plot(loss_test_history[key], label=label, linewidth=2, color=color)

    if components_to_plot:
        ax3.set_yscale("log")
        ax3.legend()
        ax3.set_xlabel("Epoch", fontsize=11)
        ax3.set_ylabel("Loss", fontsize=11)
        ax3.set_title("Validation Loss Components", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis("off")

    plt.tight_layout()
    plt.savefig(f"{cf.logging.run_dir}/losses.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_diagnostic_history(diagnostic_history, cf):
    """
    Plot diagnostic metrics evolution over training epochs.

    Parameters
    ----------
    diagnostic_history : dict
        Dictionary containing diagnostic metrics with keys:
        - 'epochs': list of epoch numbers
        - 'rmse': list of RMSE values
        - 'bias_mean': list of bias mean values
        - 'bias_q95': list of 95th percentile bias values
        - 'bias_q98': list of 98th percentile bias values
        - 'std_ratio': list of standard deviation ratio values
        - 'mae': list of MAE values
        - 'correlation': list of correlation values
        - 'anomaly_correlation': list of anomaly correlation values
    cf : Config
        Configuration object with logging settings
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    epochs = diagnostic_history["epochs"]

    # Define plots with their properties
    plots = [
        ("rmse", "RMSE (spatial mean)", "tab:blue", None),
        ("mae", "MAE (spatial mean)", "tab:orange", None),
        ("bias_mean", "Bias Mean (spatial mean)", "tab:green", 0),
        ("bias_q95", "Bias Q95 (spatial mean)", "tab:red", 0),
        ("bias_q98", "Bias Q98 (spatial mean)", "tab:purple", 0),
        ("std_ratio", "Std Ratio (spatial mean)", "tab:brown", 1),
        ("correlation", "Correlation (spatial mean)", "tab:pink", None),
        ("anomaly_correlation", "Anomaly Correlation (spatial mean)", "tab:gray", None),
        ("fss", "FSS (Fractions Skill Score)", "tab:cyan", None),
    ]

    for idx, (key, ylabel, color, hline) in enumerate(plots):
        ax = axes[idx]
        if key in diagnostic_history and len(diagnostic_history[key]) > 0:
            ax.plot(
                epochs,
                diagnostic_history[key],
                "o-",
                linewidth=2,
                markersize=6,
                color=color,
            )
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f"{ylabel} Evolution", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add horizontal reference line if specified
            if hline is not None:
                ax.axhline(y=hline, color="k", linestyle="--", alpha=0.3)
        else:
            ax.axis("off")

    # The 9th subplot will now have FSS, so no need to hide it

    plt.suptitle(
        "Diagnostic Metrics Evolution", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout()
    plt.savefig(
        f"{cf.logging.run_dir}/diagnostics_history.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    plt.close()


def plot_predictions(
    generator,
    x_batch,
    y_batch,
    cf,
    epoch,
    device,
    sample_idx=0,
    input_cols=3,
    norm_params=None,
):
    """
    Plot a single sample with grid layout:
      - Left: grid of all input channels
      - Right: True (top) and Prediction (bottom) with cartopy maps

    Parameters
    ----------
    norm_params : dict, optional
        Normalization parameters for denormalizing predictions before plotting
    """
    generator.eval()

    # Select colormap based on target variable
    if cf.data.var_target == "tasmax":
        cmap = "RdYlBu_r"
    elif cf.data.var_target == "pr":
        # Create custom colormap: white for 0 and below, jet for positive values

        # Get jet colormap
        jet = plt.cm.get_cmap("jet")

        # Create custom colormap with white at bottom
        colors = ["white"] + [jet(i) for i in np.linspace(0, 1, 255)]
        cmap = mcolors.LinearSegmentedColormap.from_list("jet_white", colors, N=256)

        # Set values below/at 0 to map to white
        cmap.set_under("white")
    else:
        cmap = "viridis"

    input_cmap = "viridis"

    # Select sample
    x = x_batch[sample_idx : sample_idx + 1]
    y = y_batch[sample_idx : sample_idx + 1]

    with torch.no_grad():
        match cf.model.architecture:
            case "spategan":
                y_pred = generator(x)
            case "diffusion_unet":
                y_pred = generator(x, torch.zeros([1]).to(device))
            case "deepesd":
                y_pred = generator(x)
            case _:
                raise ValueError(f"Invalid option: {cf.model.architecture}")

    # Denormalize if norm_params provided
    if norm_params is not None:
        from ml_benchmark_spategan.utils.normalize import denormalize_predictions

        y = denormalize_predictions(y, norm_params)
        y_pred = denormalize_predictions(y_pred, norm_params)

    # Function to convert flattened images to (B,1,H,W)
    def ensure_image_tensor(t):
        if t.dim() == 2:  # (B, H*W)
            B, HW = t.shape
            side = int(math.sqrt(HW))
            if side * side == HW:
                return t.view(B, 1, side, side)
            else:
                return t.view(B, 1, 128, 128)  # fallback
        if t.dim() == 3:  # (B, H, W)
            return t.unsqueeze(1)
        return t

    x = ensure_image_tensor(x) if x.dim() in (2, 3) else x
    y = ensure_image_tensor(y)
    y_pred = ensure_image_tensor(y_pred)

    # numpy versions - detach first to avoid gradient errors
    x_np = x.detach().cpu().numpy()[0]  # (C, H, W)
    y_np = y.detach().cpu().numpy()[0, 0]  # (H, W)
    y_pred_np = y_pred.detach().cpu().numpy()[0, 0]

    # Calculate common vmin/vmax for true and predicted fine-scale images
    # For precipitation, set vmin slightly above 0 so 0 values use the white color
    if cf.data.var_target == "pr":
        vmin = 0.1  # Very small positive value
        vmax = max(y_np.max(), y_pred_np.max())
    else:
        vmin = min(y_np.min(), y_pred_np.min())
        vmax = max(y_np.max(), y_pred_np.max())

    num_ch = x_np.shape[0]
    cols = input_cols
    rows = math.ceil(num_ch / cols)

    # Setup cartopy projection
    domain = cf.data.domain
    central_longitude = 180 if domain == "NZ" else 0
    projection = ccrs.PlateCarree(central_longitude=central_longitude)

    # Get coordinates from norm_params if available
    coords = None
    if norm_params is not None and "y_test_coords" in norm_params:
        coords = norm_params["y_test_coords"]
        spatial_dims = norm_params.get("spatial_dims", ("lat", "lon"))

        # Extract lat/lon arrays
        if domain == "ALPS":
            lon = coords["lon"].values
            lat = coords["lat"].values
        else:  # NZ or SA
            lon = coords[spatial_dims[1]].values
            lat = coords[spatial_dims[0]].values

        # Create 2D coordinate grids
        if lon.ndim == 1 and lat.ndim == 1:
            lon_2d, lat_2d = torch.meshgrid(
                torch.from_numpy(lon), torch.from_numpy(lat), indexing="xy"
            )
            lon_2d = lon_2d.numpy()
            lat_2d = lat_2d.numpy()
        else:
            lon_2d = lon
            lat_2d = lat

    # Figure size
    fig_height = max(2.7 * rows, 6)
    fig = plt.figure(figsize=(16, fig_height))

    # Main GridSpec: left (input grid), right (true/pred)
    outer = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.2], wspace=0.1)

    # LEFT: input channel grid
    left_gs = gridspec.GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=outer[0], hspace=0.25, wspace=0.1
    )

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(left_gs[r, c])
        if i < num_ch:
            im = ax.imshow(x_np[i], cmap=input_cmap)
            ax.set_title(f"Ch {i}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.01)
        else:
            ax.axis("off")

    # RIGHT: True and Pred stacked vertically with cartopy maps
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=[1, 1], hspace=0.3
    )

    # True map
    ax_true = fig.add_subplot(right_gs[0], projection=projection)
    if coords is not None:
        if domain == "ALPS":
            im_t = ax_true.pcolormesh(
                lon_2d,
                lat_2d,
                y_np,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        else:  # NZ or SA
            im_t = ax_true.pcolormesh(
                lon_2d,
                lat_2d,
                y_np,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
    else:
        im_t = ax_true.imshow(y_np, cmap=cmap, vmin=vmin, vmax=vmax)

    ax_true.coastlines(resolution="10m", linewidth=2)
    ax_true.add_feature(cfeature.BORDERS, linestyle=":", linewidth=2)
    ax_true.set_title("True Fine-Scale", fontsize=11)
    plt.colorbar(im_t, ax=ax_true, fraction=0.04, pad=0.02, orientation="horizontal")

    # Predicted map
    ax_pred = fig.add_subplot(right_gs[1], projection=projection)
    if coords is not None:
        if domain == "ALPS":
            im_p = ax_pred.pcolormesh(
                lon_2d,
                lat_2d,
                y_pred_np,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
        else:  # NZ or SA
            im_p = ax_pred.pcolormesh(
                lon_2d,
                lat_2d,
                y_pred_np,
                transform=ccrs.PlateCarree(),
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
    else:
        im_p = ax_pred.imshow(y_pred_np, cmap=cmap, vmin=vmin, vmax=vmax)

    ax_pred.coastlines(resolution="10m", linewidth=2)
    ax_pred.add_feature(cfeature.BORDERS, linestyle=":", linewidth=2)
    ax_pred.set_title("Predicted Fine-Scale", fontsize=11)
    plt.colorbar(im_p, ax=ax_pred, fraction=0.04, pad=0.02, orientation="horizontal")

    # Title + layout
    plt.suptitle(f"Epoch {epoch} — Sample {sample_idx}", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = (
        f"{cf.logging.run_dir}/sample_plots/predictions_epoch_{epoch}_s{sample_idx}.png"
    )
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    generator.train()


def plot_predictions_only(
    generator,
    x_batch,
    y_batch,
    cf,
    epoch,
    device,
    num_samples=3,
    norm_params=None,
):
    """
    Plot multiple prediction-target pairs in a grid layout.
    Each column shows one sample with True (top) and Prediction (bottom).

    Parameters
    ----------
    generator : nn.Module
        Generator model
    x_batch : torch.Tensor
        Input batch
    y_batch : torch.Tensor
        Target batch
    cf : Config
        Configuration object
    epoch : int
        Current epoch number
    device : torch.device
        Device to run model on
    num_samples : int, optional
        Number of samples to plot (default: 3)
    norm_params : dict, optional
        Normalization parameters for denormalizing predictions before plotting
    """
    generator.eval()

    # Select colormap based on target variable
    if cf.data.var_target == "tasmax":
        cmap = "RdYlBu_r"
    elif cf.data.var_target == "pr":
        # Create custom colormap: white for 0 and below, jet for positive values
        jet = plt.cm.get_cmap("jet")
        colors = ["white"] + [jet(i) for i in np.linspace(0, 1, 255)]
        cmap = mcolors.LinearSegmentedColormap.from_list("jet_white", colors, N=256)
        cmap.set_under("white")
    else:
        cmap = "viridis"

    # Limit num_samples to batch size
    num_samples = min(num_samples, x_batch.shape[0])

    # Select samples
    x = x_batch[:num_samples]
    y = y_batch[:num_samples]

    with torch.no_grad():
        match cf.model.architecture:
            case "spategan":
                y_pred = generator(x)
            case "diffusion_unet":
                timesteps = torch.zeros([num_samples]).to(device)
                y_pred = generator(x, timesteps)
            case "deepesd":
                y_pred = generator(x)
            case _:
                raise ValueError(f"Invalid option: {cf.model.architecture}")

    # Denormalize if norm_params provided
    if norm_params is not None:
        from ml_benchmark_spategan.utils.normalize import denormalize_predictions

        y = denormalize_predictions(y, norm_params)
        y_pred = denormalize_predictions(y_pred, norm_params)

    # Function to convert flattened images to (B,1,H,W)
    def ensure_image_tensor(t):
        if t.dim() == 2:  # (B, H*W)
            B, HW = t.shape
            side = int(math.sqrt(HW))
            if side * side == HW:
                return t.view(B, 1, side, side)
            else:
                return t.view(B, 1, 128, 128)  # fallback
        if t.dim() == 3:  # (B, H, W)
            return t.unsqueeze(1)
        return t

    y = ensure_image_tensor(y)
    y_pred = ensure_image_tensor(y_pred)

    # Setup cartopy projection
    domain = cf.data.domain
    central_longitude = 180 if domain == "NZ" else 0
    projection = ccrs.PlateCarree(central_longitude=central_longitude)

    # Get coordinates from norm_params if available
    coords = None
    if norm_params is not None and "y_test_coords" in norm_params:
        coords = norm_params["y_test_coords"]
        spatial_dims = norm_params.get("spatial_dims", ("lat", "lon"))

        # Extract lat/lon arrays
        if domain == "ALPS":
            lon = coords["lon"].values
            lat = coords["lat"].values
        else:  # NZ or SA
            lon = coords[spatial_dims[1]].values
            lat = coords[spatial_dims[0]].values

        # Create 2D coordinate grids
        if lon.ndim == 1 and lat.ndim == 1:
            lon_2d, lat_2d = torch.meshgrid(
                torch.from_numpy(lon), torch.from_numpy(lat), indexing="xy"
            )
            lon_2d = lon_2d.numpy()
            lat_2d = lat_2d.numpy()
        else:
            lon_2d = lon
            lat_2d = lat

    # Create figure with grid layout: num_samples columns, 2 rows (True, Pred)
    fig_width = 6 * num_samples
    fig = plt.figure(figsize=(fig_width, 10))

    outer = gridspec.GridSpec(
        2, num_samples, height_ratios=[1, 1], hspace=0.25, wspace=0.15
    )

    # Calculate common vmin/vmax across all samples
    y_np_all = y.detach().cpu().numpy()[:, 0]  # (num_samples, H, W)
    y_pred_np_all = y_pred.detach().cpu().numpy()[:, 0]

    if cf.data.var_target == "pr":
        vmin = 0.1  # Very small positive value
        vmax = max(y_np_all.max(), y_pred_np_all.max())
    else:
        vmin = min(y_np_all.min(), y_pred_np_all.min())
        vmax = max(y_np_all.max(), y_pred_np_all.max())

    # Plot each sample
    for sample_idx in range(num_samples):
        y_np = y_np_all[sample_idx]
        y_pred_np = y_pred_np_all[sample_idx]

        # True map (top row)
        ax_true = fig.add_subplot(outer[0, sample_idx], projection=projection)
        if coords is not None:
            if domain == "ALPS":
                im_t = ax_true.pcolormesh(
                    lon_2d,
                    lat_2d,
                    y_np,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
            else:  # NZ or SA
                im_t = ax_true.pcolormesh(
                    lon_2d,
                    lat_2d,
                    y_np,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
        else:
            im_t = ax_true.imshow(y_np, cmap=cmap, vmin=vmin, vmax=vmax)

        ax_true.coastlines(resolution="10m", linewidth=2)
        ax_true.add_feature(cfeature.BORDERS, linestyle=":", linewidth=2)
        ax_true.set_title(f"True - Sample {sample_idx}", fontsize=11)
        plt.colorbar(
            im_t, ax=ax_true, fraction=0.04, pad=0.02, orientation="horizontal"
        )

        # Predicted map (bottom row)
        ax_pred = fig.add_subplot(outer[1, sample_idx], projection=projection)
        if coords is not None:
            if domain == "ALPS":
                im_p = ax_pred.pcolormesh(
                    lon_2d,
                    lat_2d,
                    y_pred_np,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
            else:  # NZ or SA
                im_p = ax_pred.pcolormesh(
                    lon_2d,
                    lat_2d,
                    y_pred_np,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
        else:
            im_p = ax_pred.imshow(y_pred_np, cmap=cmap, vmin=vmin, vmax=vmax)

        ax_pred.coastlines(resolution="10m", linewidth=2)
        ax_pred.add_feature(cfeature.BORDERS, linestyle=":", linewidth=2)
        ax_pred.set_title(f"Predicted - Sample {sample_idx}", fontsize=11)
        plt.colorbar(
            im_p, ax=ax_pred, fraction=0.04, pad=0.02, orientation="horizontal"
        )

    # Title + layout
    plt.suptitle(f"Epoch {epoch} — Multiple Predictions", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = f"{cf.logging.run_dir}/sample_plots/predictions_only_epoch_{epoch}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    generator.train()
