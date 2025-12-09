import matplotlib.pyplot as plt
import torch


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


def plot_predictions(generator, x_batch, y_batch, cf, epoch, num_samples=1):
    """
    Visualize all 15 coarse input channels, true fine-scale target, and predicted fine-scale output.
    """
    generator.eval()

    # Use minimum of requested samples and available samples
    num_samples = min(num_samples, x_batch.shape[0])

    # Select colormap based on target variable
    cmap = "RdYlBu_r" if cf.data.var_target == "tasmax" else "jet"

    with torch.no_grad():
        # Generate predictions
        y_pred = generator(x_batch[:num_samples])

        # Reshape if needed
        if y_pred.dim() == 2:  # (batch, H*W)
            y_pred = y_pred.view(-1, 1, 128, 128)
        if y_batch.dim() == 2:  # (batch, H*W)
            y_batch = y_batch.view(-1, 1, 128, 128)

        # Convert to numpy
        x_np = x_batch[:num_samples].cpu().numpy()
        y_true_np = y_batch[:num_samples].cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        for sample_idx in range(num_samples):
            # Create figure with gridspec for custom layout
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(
                5,
                5,
                hspace=0.3,
                wspace=0.3,
                left=0.05,
                right=0.75,
                top=0.95,
                bottom=0.05,
            )

            # Plot all 15 coarse input channels in a 5x3 grid
            n_channels = x_np.shape[1]
            for ch in range(min(15, n_channels)):
                row = ch // 3
                col = ch % 3
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(x_np[sample_idx, ch], cmap="viridis")
                ax.set_title(f"Coarse Ch {ch}", fontsize=10)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Create larger subplots for high-res images on the right
            gs_right = fig.add_gridspec(
                2,
                1,
                hspace=0.3,
                wspace=0.1,
                left=0.78,
                right=0.98,
                top=0.95,
                bottom=0.05,
            )

            # Calculate common vmin/vmax for true and predicted fine-scale images
            vmin = min(y_true_np[sample_idx, 0].min(), y_pred_np[sample_idx, 0].min())
            vmax = max(y_true_np[sample_idx, 0].max(), y_pred_np[sample_idx, 0].max())

            # Plot true fine-scale (larger)
            ax_true = fig.add_subplot(gs_right[0])
            im_true = ax_true.imshow(
                y_true_np[sample_idx, 0], cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax_true.set_title(
                f"Sample {sample_idx + 1}: True Fine-Scale",
                fontsize=12,
                fontweight="bold",
            )
            ax_true.axis("off")
            plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)

            # Plot predicted fine-scale (larger)
            ax_pred = fig.add_subplot(gs_right[1])
            im_pred = ax_pred.imshow(
                y_pred_np[sample_idx, 0], cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax_pred.set_title(
                f"Sample {sample_idx + 1}: Predicted Fine-Scale",
                fontsize=12,
                fontweight="bold",
            )
            ax_pred.axis("off")
            plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

            plt.suptitle(
                f"Epoch {epoch} - Sample {sample_idx + 1}",
                fontsize=16,
                fontweight="bold",
            )
            plt.savefig(
                f"{cf.logging.run_dir}/predictions_epoch_{epoch}_sample_{sample_idx + 1}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()
            plt.close()

    generator.train()
