import torch
import math
import numpy as np
import matplotlib.pyplot as plt
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


def plot_predictions(generator, x_batch, y_batch, cf, epoch, num_samples=3):
    """
    Visualize coarse input, true fine-scale target, and predicted fine-scale output.
    """
    generator.eval()

    # Use minimum of requested samples and available samples
    num_samples = min(num_samples, x_batch.shape[0])

    with torch.no_grad():
        # Generate predictions
        y_pred = generator(x_batch[:num_samples])  # Reshape if needed
        if y_pred.dim() == 2:  # (batch, H*W)
            y_pred = y_pred.view(-1, 1, 128, 128)
        if y_batch.dim() == 2:  # (batch, H*W)
            y_batch = y_batch.view(-1, 1, 128, 128)

        # Convert to numpy
        x_np = x_batch[:num_samples].cpu().numpy()
        y_true_np = y_batch[:num_samples].cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Plot coarse input (first channel)
            im0 = axes[i, 0].imshow(x_np[i, 0], cmap="viridis")
            axes[i, 0].set_title(f"Sample {i + 1}: Coarse Input (Ch 0)")
            axes[i, 0].axis("off")
            plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

            # Plot true fine-scale
            im1 = axes[i, 1].imshow(y_true_np[i, 0], cmap="viridis")
            axes[i, 1].set_title(f"Sample {i + 1}: True Fine-Scale")
            axes[i, 1].axis("off")
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            # Plot predicted fine-scale
            im2 = axes[i, 2].imshow(y_pred_np[i, 0], cmap="viridis")
            axes[i, 2].set_title(f"Sample {i + 1}: Predicted Fine-Scale")
            axes[i, 2].axis("off")
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        plt.suptitle(f"Epoch {epoch}: Predictions", fontsize=16, y=1.0)
        plt.tight_layout()
        plt.savefig(
            f"{cf.logging.run_dir}/predictions_epoch_{epoch}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    generator.train()


def plot_predictions_grid(generator, x_batch, y_batch, cf, epoch, device,
                          sample_idx=0, input_cols=5, cmap="viridis"):
    """
    Plot a single sample:
      - Left: grid of all input channels (C channels)
      - Right: True (top) and Prediction (bottom), but not oversized.
    Balanced layout using GridSpec.
    """
    generator.eval()

    # Select sample
    x = x_batch[sample_idx:sample_idx+1]
    y = y_batch[sample_idx:sample_idx+1]

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
    x_np = x.cpu().numpy()[0]         # (C, H, W)
    y_np = y.cpu().numpy()[0, 0]      # (H, W)
    y_pred_np = y_pred.cpu().numpy()[0, 0]

    num_ch = x_np.shape[0]
    cols = input_cols
    rows = math.ceil(num_ch / cols)

    # --- FIGURE SIZE ---
    # Height depends on num rows; width is fixed nicely
    fig_height = max(2.7 * rows, 6)
    fig = plt.figure(figsize=(14, fig_height))

    # Main GridSpec: left (input grid), right (true/pred)
    # Make right column only *slightly* wider → balanced
    outer = gridspec.GridSpec(
        1, 2,
        width_ratios=[2.0, 1.2],
        wspace=0.05
    )

    # LEFT: input channel grid
    left_gs = gridspec.GridSpecFromSubplotSpec(
        rows, cols, subplot_spec=outer[0],
        hspace=0.25, wspace=0.1
    )

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(left_gs[r, c])
        if i < num_ch:
            im = ax.imshow(x_np[i], cmap=cmap)
            ax.set_title(f"Ch {i}", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.01)
        else:
            ax.axis("off")

    # RIGHT: True and Pred stacked vertically
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1],
        height_ratios=[1, 1],   # equal height, not oversized
        hspace=0.25
    )

    ax_true = fig.add_subplot(right_gs[0])
    im_t = ax_true.imshow(y_np, cmap=cmap)
    ax_true.set_title("True Fine-Scale", fontsize=11)
    ax_true.axis("off")
    plt.colorbar(im_t, ax=ax_true, fraction=0.04, pad=0.02)

    ax_pred = fig.add_subplot(right_gs[1])
    im_p = ax_pred.imshow(y_pred_np, cmap=cmap)
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
