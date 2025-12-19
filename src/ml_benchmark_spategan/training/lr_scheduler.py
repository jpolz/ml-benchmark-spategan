"""Learning rate scheduler with warmup and exponential decay."""

import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from ml_benchmark_spategan.config import config


def create_warmup_scheduler(
    optimizer,
    warmup_epochs,
    total_epochs,
    warmup_start_lr=1e-6,
    plateau_epochs=0,
    transition_epochs=0,
    gamma=0.95,
):
    """
    Create a learning rate scheduler with warmup, plateau, smooth transition, and exponential decay phases.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup phase
        total_epochs: Total number of training epochs
        warmup_start_lr: Starting learning rate for warmup (default: 1e-6)
        plateau_epochs: Number of epochs to maintain target LR before transition (default: 0)
        transition_epochs: Number of epochs for smooth cosine transition to decay (default: 0)
        gamma: Exponential decay factor after transition (default: 0.95)

    Returns:
        LambdaLR scheduler
    """
    target_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Phase 1: Linear warmup from warmup_start_lr to target_lr
            return (
                warmup_start_lr + (target_lr - warmup_start_lr) * epoch / warmup_epochs
            ) / target_lr
        elif epoch < warmup_epochs + plateau_epochs:
            # Phase 2: Plateau - maintain target learning rate
            return 1.0
        elif epoch < warmup_epochs + plateau_epochs + transition_epochs:
            # Phase 3: Smooth cosine transition from 1.0 to gamma^0
            transition_progress = (
                epoch - warmup_epochs - plateau_epochs
            ) / transition_epochs
            # Cosine annealing from 1.0 to gamma^0 (which is 1.0, so we go to first decay step)
            # This creates a smooth S-curve transition
            cosine_decay = 0.5 * (1 + np.cos(np.pi * transition_progress))
            # Interpolate between no decay (1.0) and first decay step (gamma^0 = 1.0)
            # Actually, let's transition to gamma^transition_epochs to make it smooth
            start_val = 1.0
            end_val = gamma**transition_epochs
            return start_val * cosine_decay + end_val * (1 - cosine_decay)
        else:
            # Phase 4: Exponential decay after transition
            decay_epoch = epoch - warmup_epochs - plateau_epochs - transition_epochs
            return (gamma**transition_epochs) * (gamma**decay_epoch)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

###################################################################################
# ONLY FOR VISUALIZATION AND DEBUGGING BELOW
###################################################################################

def plot_lr_schedule(
    base_lr,
    warmup_epochs,
    total_epochs,
    warmup_start_lr=1e-6,
    plateau_epochs=0,
    transition_epochs=0,
    gamma=0.95,
    title="Learning Rate Schedule",
    save_path=None,
):
    """
    Plot the learning rate schedule over epochs.

    Args:
        base_lr: Base/target learning rate
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        warmup_start_lr: Starting LR for warmup
        plateau_epochs: Number of epochs at target LR before transition
        transition_epochs: Number of epochs for smooth transition to decay
        gamma: Exponential decay factor
        title: Plot title
        save_path: Optional path to save the plot
    """
    # Create a dummy optimizer to get the scheduler
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=base_lr)
    scheduler = create_warmup_scheduler(
        optimizer,
        warmup_epochs,
        total_epochs,
        warmup_start_lr,
        plateau_epochs,
        transition_epochs,
        gamma,
    )

    # Simulate epochs and collect learning rates
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = np.arange(1, total_epochs + 1)

    ax.plot(epochs, lrs, linewidth=2, color="tab:blue")

    # Add phase markers
    if warmup_epochs > 0:
        ax.axvline(
            x=warmup_epochs,
            color="green",
            linestyle="--",
            alpha=0.6,
            label="End of Warmup",
        )
    if plateau_epochs > 0:
        ax.axvline(
            x=warmup_epochs + plateau_epochs,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Start of Decay",
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add annotations
    ax.text(
        0.80,
        0.5,
        f"Base LR: {base_lr:.2e}\n"
        f"Warmup Start: {warmup_start_lr:.2e}\n"
        f"Warmup Epochs: {warmup_epochs}\n"
        f"Plateau Epochs: {plateau_epochs}\n"
        f"Decay γ: {gamma}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
    return lrs


def plot_lr_with_current_config(config_path="config.yml"):
    """
    Plot learning rate schedules using parameters from config file.

    Args:
        config_path: Path to configuration YAML file
    """
    # Find project base directory
    project_base = pathlib.Path(os.getcwd())

    # Load configuration
    full_config_path = os.path.join(project_base, config_path)
    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"Configuration file not found: {full_config_path}")

    cf = config.load_config_from_yaml(full_config_path)

    # Extract scheduler parameters
    warmup_epochs = getattr(cf.training, "warmup_epochs", 5)
    plateau_epochs = getattr(cf.training, "plateau_epochs", 0)
    transition_epochs = getattr(cf.training, "transition_epochs", 0)
    lr_decay_gamma = getattr(cf.training, "lr_decay_gamma", 0.95)
    warmup_start_lr = getattr(cf.training, "warmup_start_lr", 1e-6)
    total_epochs = cf.training.epochs

    gen_lr = cf.training.generator.learning_rate
    disc_lr = cf.training.discriminator.learning_rate

    print("=" * 60)
    print("Learning Rate Schedule Configuration")
    print("=" * 60)
    print(f"Total Epochs:            {total_epochs}")
    print(f"Warmup Epochs:           {warmup_epochs}")
    print(f"Plateau Epochs:          {plateau_epochs}")
    print(f"Transition Epochs:       {transition_epochs}")
    print(
        f"Decay Starts at Epoch:   {warmup_epochs + plateau_epochs + transition_epochs + 1}"
    )
    print(f"Warmup Start LR:         {warmup_start_lr:.2e}")
    print(f"Decay Gamma:             {lr_decay_gamma}")
    print(f"\nGenerator Base LR:       {gen_lr:.2e}")
    print(f"Discriminator Base LR:   {disc_lr:.2e}")
    print("=" * 60)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot generator schedule
    dummy_model = torch.nn.Linear(1, 1)
    gen_opt = torch.optim.Adam(dummy_model.parameters(), lr=gen_lr)
    gen_scheduler = create_warmup_scheduler(
        gen_opt,
        warmup_epochs,
        total_epochs,
        warmup_start_lr,
        plateau_epochs,
        transition_epochs,
        lr_decay_gamma,
    )

    gen_lrs = []
    for epoch in range(total_epochs):
        gen_lrs.append(gen_opt.param_groups[0]["lr"])
        gen_scheduler.step()

    epochs = np.arange(1, total_epochs + 1)
    ax1.plot(epochs, gen_lrs, linewidth=2, color="tab:blue")

    if warmup_epochs > 0:
        ax1.axvline(
            x=warmup_epochs,
            color="green",
            linestyle="--",
            alpha=0.6,
            label="End of Warmup",
        )
    if plateau_epochs > 0:
        ax1.axvline(
            x=warmup_epochs + plateau_epochs,
            color="orange",
            linestyle="--",
            alpha=0.6,
            label="Start of Transition",
        )
    if transition_epochs > 0:
        ax1.axvline(
            x=warmup_epochs + plateau_epochs + transition_epochs,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Start of Decay",
        )

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Learning Rate", fontsize=12)
    ax1.set_title("Generator Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale("log")
    ax1.text(
        0.70,
        0.5,
        f"Base LR: {gen_lr:.2e}\n"
        f"Start: {warmup_start_lr:.2e}\n"
        f"Warmup: {warmup_epochs} epochs\n"
        f"Plateau: {plateau_epochs} epochs\n"
        f"Transition: {transition_epochs} epochs\n"
        f"Decay γ: {lr_decay_gamma}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
        fontsize=9,
    )  # Plot discriminator schedule
    disc_opt = torch.optim.Adam(dummy_model.parameters(), lr=disc_lr)
    disc_scheduler = create_warmup_scheduler(
        disc_opt,
        warmup_epochs,
        total_epochs,
        warmup_start_lr,
        plateau_epochs,
        transition_epochs,
        lr_decay_gamma,
    )

    disc_lrs = []
    for epoch in range(total_epochs):
        disc_lrs.append(disc_opt.param_groups[0]["lr"])
        disc_scheduler.step()

    ax2.plot(epochs, disc_lrs, linewidth=2, color="tab:orange")

    if warmup_epochs > 0:
        ax2.axvline(
            x=warmup_epochs,
            color="green",
            linestyle="--",
            alpha=0.6,
            label="End of Warmup",
        )
    if plateau_epochs > 0:
        ax2.axvline(
            x=warmup_epochs + plateau_epochs,
            color="orange",
            linestyle="--",
            alpha=0.6,
            label="Start of Transition",
        )
    if transition_epochs > 0:
        ax2.axvline(
            x=warmup_epochs + plateau_epochs + transition_epochs,
            color="red",
            linestyle="--",
            alpha=0.6,
            label="Start of Decay",
        )

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Learning Rate", fontsize=12)
    ax2.set_title(
        "Discriminator Learning Rate Schedule", fontsize=13, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_yscale("log")
    ax2.text(
        0.70,
        0.5,
        f"Base LR: {disc_lr:.2e}\n"
        f"Start: {warmup_start_lr:.2e}\n"
        f"Warmup: {warmup_epochs} epochs\n"
        f"Plateau: {plateau_epochs} epochs\n"
        f"Transition: {transition_epochs} epochs\n"
        f"Decay γ: {lr_decay_gamma}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
        fontsize=9,
    )

    plt.tight_layout()

    # Save plot
    save_path = project_base / "lr_schedule_preview.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    plt.show()

    # Print some key values
    print("\nKey Learning Rate Values:")
    print("-" * 60)
    print(
        f"Generator - Epoch 1:   {gen_lrs[0]:.6e} | Epoch {warmup_epochs}: {gen_lrs[warmup_epochs - 1]:.6e} | Final: {gen_lrs[-1]:.6e}"
    )
    print(
        f"Discriminator - Epoch 1: {disc_lrs[0]:.6e} | Epoch {warmup_epochs}: {disc_lrs[warmup_epochs - 1]:.6e} | Final: {disc_lrs[-1]:.6e}"
    )
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize learning rate schedule from config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration YAML file (default: config.yml)",
    )
    args = parser.parse_args()

    plot_lr_with_current_config(args.config)
