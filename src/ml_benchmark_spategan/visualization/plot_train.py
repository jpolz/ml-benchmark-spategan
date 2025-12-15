import math

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
import matplotlib.colors as mcolors
import numpy as np


def plot_losses(loss_train, loss_test, cf):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(loss_train, label="Train Loss")
    axs.plot(loss_test, label="Test Loss")
    axs.legend()
    axs.set_yscale("log")
    axs.grid(True)
    plt.savefig(cf.logging.run_dir + "/losses.png", dpi=150)
    plt.close()


def plot_adversarial_losses(loss_gen_train, loss_disc_train, loss_gen_test, cf):
    """
    Plot GAN training losses including generator, discriminator, and validation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Training losses
    ax1.plot(loss_gen_train, label="Generator")
    ax1.plot(loss_disc_train, label="Discriminator")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Losses")
    ax1.grid(True)

    # Validation loss
    ax2.plot(loss_gen_test, label="Test MSE")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{cf.logging.run_dir}/losses.png", dpi=150)
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
    cf : Config
        Configuration object with logging settings
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot RMSE over epochs
    ax1.plot(
        diagnostic_history["epochs"],
        diagnostic_history["rmse"],
        "o-",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("RMSE (spatial mean)", fontsize=12)
    ax1.set_title("RMSE Evolution", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Plot Bias Mean over epochs
    ax2.plot(
        diagnostic_history["epochs"],
        diagnostic_history["bias_mean"],
        "o-",
        linewidth=2,
        markersize=6,
        color="orange",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Bias Mean (spatial mean)", fontsize=12)
    ax2.set_title("Bias Mean Evolution", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)

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
        match cf.model.generator_architecture:
            case "spategan":
                y_pred = generator(x)
            case "diffusion_unet":
                y_pred = generator(x, torch.zeros([1]).to(device))
            case _:
                raise ValueError(f"Invalid option: {cf.model.generator_architecture}")

    # Denormalize if norm_params provided
    if norm_params is not None:
        from ml_benchmark_spategan.utils.denormalize import denormalize_predictions

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

    # numpy versions
    x_np = x.cpu().numpy()[0]  # (C, H, W)
    y_np = y.cpu().numpy()[0, 0]  # (H, W)
    y_pred_np = y_pred.cpu().numpy()[0, 0]

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

    ax_pred.coastlines(resolution="10m", linewidth=0.5)
    ax_pred.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
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
