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
