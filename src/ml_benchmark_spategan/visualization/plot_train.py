import math

import matplotlib.pyplot as plt
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


def plot_predictions(
    generator, x_batch, y_batch, cf, epoch, device, sample_idx=0, input_cols=5
):
    """
    Plot a single sample with grid layout:
      - Left: grid of all input channels
      - Right: True (top) and Prediction (bottom) with adjusted colormap and shared vmin/vmax
    """
    generator.eval()

    # Select colormap based on target variable
    cmap = "RdYlBu_r" if cf.data.var_target == "tasmax" else "jet"
    input_cmap = "viridis"

    # Select sample
    x = x_batch[sample_idx : sample_idx + 1]
    y = y_batch[sample_idx : sample_idx + 1]

    with torch.no_grad():
        match cf.model.architecture:
            case "spategan":
                y_pred = generator(x)
            case "diffusion_unet":
                y_pred = generator(x, torch.zeros([1]).to(device)).sample
            case _:
                raise ValueError(f"Invalid option: {cf.model.architecture}")

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
    vmin = min(y_np.min(), y_pred_np.min())
    vmax = max(y_np.max(), y_pred_np.max())

    num_ch = x_np.shape[0]
    cols = input_cols
    rows = math.ceil(num_ch / cols)

    # Figure size
    fig_height = max(2.7 * rows, 6)
    fig = plt.figure(figsize=(14, fig_height))

    # Main GridSpec: left (input grid), right (true/pred)
    outer = gridspec.GridSpec(1, 2, width_ratios=[2.0, 1.2], wspace=0.05)

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

    # RIGHT: True and Pred stacked vertically with shared vmin/vmax
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=[1, 1], hspace=0.25
    )

    ax_true = fig.add_subplot(right_gs[0])
    im_t = ax_true.imshow(y_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_true.set_title("True Fine-Scale", fontsize=11)
    ax_true.axis("off")
    plt.colorbar(im_t, ax=ax_true, fraction=0.04, pad=0.02)

    ax_pred = fig.add_subplot(right_gs[1])
    im_p = ax_pred.imshow(y_pred_np, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_pred.set_title("Predicted Fine-Scale", fontsize=11)
    ax_pred.axis("off")
    plt.colorbar(im_p, ax=ax_pred, fraction=0.04, pad=0.02)

    # Title + layout
    plt.suptitle(f"Epoch {epoch} — Sample {sample_idx}", fontsize=15, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = f"{cf.logging.run_dir}/predictions_epoch_{epoch}_s{sample_idx}.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    generator.train()
