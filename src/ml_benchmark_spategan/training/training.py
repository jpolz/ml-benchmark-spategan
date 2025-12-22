"""
Training module for SpatialGAN and Diffusion UNet models.

This module implements the main training pipeline for deep learning emulators
of Regional Climate Models (RCMs) using the CORDEX Benchmark dataset. It supports
two architectures:
- SpatialGAN: Custom GAN architecture for spatial downscaling
- Diffusion UNet: U-Net based conditional generation model

The training workflow includes:
1. Configuration loading and experiment setup
2. Data loading with normalization (supports multiple normalization methods)
3. Model initialization (generator and discriminator)
4. Adversarial training with mixed precision
5. Validation and diagnostic computation (RMSE, bias)
6. Checkpointing and visualization

Key features:
- Command-line configuration via --config argument
- Multiple normalization methods (standardization, minmax, log transforms, etc.)
- FSS (Fractions Skill Score) loss for spatial pattern matching
- Learnable or fixed bilinear upsampling
- Comprehensive logging and visualization during training
- Integration with CORDEX Benchmark diagnostics

Usage:
    python -m ml_benchmark_spategan.training --config config.yml

The module saves:
- Model checkpoints (generator, discriminator, optimizers)
- Normalization parameters for denormalization during inference
- Training diagnostics and loss history
- Sample prediction visualizations
"""

############
# Imports
############

import argparse
import logging
import os
import pathlib

# Import diagnostics
import sys

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from IPython.display import clear_output
from tqdm import tqdm

from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.dataloader import dataloader
from ml_benchmark_spategan.model.registry import create_discriminator, create_generator
from ml_benchmark_spategan.training.gan_training import test_gan_step, train_gan_step
from ml_benchmark_spategan.training.gan_training.losses import FSSLoss
from ml_benchmark_spategan.training.lr_scheduler import create_warmup_scheduler
from ml_benchmark_spategan.utils.interpolate import LearnableUpsampler
from ml_benchmark_spategan.utils.normalize import save_normalization_params, predictions_to_xarray
from ml_benchmark_spategan.visualization.plot_train import (
    plot_adversarial_losses,
    plot_diagnostic_history,
    plot_predictions_only,
)

# Add evaluation directory to path to import diagnostics
sys.path.insert(
    0, str(pathlib.Path(__file__).parent.parent.parent.parent / "evaluation")
)
import diagnostics


def compute_diagnostics(y_pred_all, y_true_all, norm_params, cf, mean_fss_test, epoch):
    """
    Compute diagnostic metrics from predictions and ground truth.

    Args:
        y_pred_all: Concatenated predictions tensor (all test batches)
        y_true_all: Concatenated ground truth tensor (all test batches)
        norm_params: Normalization parameters dictionary
        cf: Configuration object
        mean_fss_test: Mean FSS test loss for this epoch
        epoch: Current epoch number

    Returns:
        dict: Dictionary of diagnostic metrics
    """
    logger = logging.getLogger(__name__)

    # Convert to xarray with denormalization
    pred_ds, true_ds = predictions_to_xarray(
        y_pred_all, y_true_all, norm_params, var_name=cf.data.var_target
    )

    # Compute diagnostics
    rmse = diagnostics.rmse(true_ds, pred_ds, var=cf.data.var_target, dim="time")
    bias_mean = diagnostics.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[cf.data.var_target].mean("time"),
    )
    bias_q95 = diagnostics.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[cf.data.var_target].quantile(0.95, dim="time"),
    )
    bias_q98 = diagnostics.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[cf.data.var_target].quantile(0.98, dim="time"),
    )
    std_ratio = diagnostics.ratio_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[cf.data.var_target].std("time"),
    )

    # Mean Absolute Error
    mae = np.abs(pred_ds[cf.data.var_target] - true_ds[cf.data.var_target]).mean("time")

    # Pearson correlation
    correlation = xr.corr(
        pred_ds[cf.data.var_target],
        true_ds[cf.data.var_target],
        dim="time",
    )

    spatial_dims = norm_params["spatial_dims"]
    # Anomaly correlation (after removing climatology)
    pred_anomaly = pred_ds[cf.data.var_target] - pred_ds[cf.data.var_target].mean(
        "time"
    )
    true_anomaly = true_ds[cf.data.var_target] - true_ds[cf.data.var_target].mean(
        "time"
    )
    anomaly_correlation = xr.corr(pred_anomaly, true_anomaly, dim=spatial_dims)

    # Build diagnostics dictionary with spatially-averaged values
    diagnostics_dict = {
        "rmse": rmse[cf.data.var_target].mean().values.item(),
        "bias_mean": bias_mean.mean().values.item(),
        "bias_q95": bias_q95.mean().values.item(),
        "bias_q98": bias_q98.mean().values.item(),
        "std_ratio": std_ratio.mean().values.item(),
        "mae": mae.mean().values.item(),
        "correlation": correlation.mean().values.item(),
        "anomaly_correlation": anomaly_correlation.mean().values.item(),
        "fss": mean_fss_test,
        "epoch": epoch,
    }

    # Log key diagnostics
    logger.info(f"  RMSE (spatial mean): {diagnostics_dict['rmse']:.4f}")
    logger.info(f"  Bias Mean (spatial mean): {diagnostics_dict['bias_mean']:.4f}")
    logger.info(f"  Bias Q95 (spatial mean): {diagnostics_dict['bias_q95']:.4f}")
    logger.info(f"  Std Ratio (spatial mean): {diagnostics_dict['std_ratio']:.4f}")
    logger.info(f"  Correlation (spatial mean): {diagnostics_dict['correlation']:.4f}")

    return diagnostics_dict


def setup_optimizers(cf, generator, discriminator, upsampler, device):
    """Set up optimizers for generator and discriminator based on config."""
    # Optimizers
    if upsampler is not None:
        # Include upsampler parameters with generator
        gen_params = list(generator.parameters()) + list(upsampler.parameters())
    else:
        gen_params = generator.parameters()

    if cf.training.generator.optimizer == "AdamW":
        gen_opt = torch.optim.AdamW(
            gen_params,
            lr=cf.training.generator.learning_rate,
            betas=(
                cf.training.generator.beta1,
                cf.training.generator.beta2,
            ),
            weight_decay=cf.training.generator.weight_decay,
        )
    elif cf.training.generator.optimizer == "Adam":
        gen_opt = torch.optim.Adam(
            gen_params,
            lr=cf.training.generator.learning_rate,
            betas=(
                cf.training.generator.beta1,
                cf.training.generator.beta2,
            ),
            weight_decay=cf.training.generator.weight_decay,
        )

    if cf.training.discriminator.optimizer == "AdamW":
        disc_opt = torch.optim.AdamW(
            discriminator.parameters(),
            lr=cf.training.discriminator.learning_rate,
            betas=(
                cf.training.discriminator.beta1,
                cf.training.discriminator.beta2,
            ),
            weight_decay=cf.training.discriminator.weight_decay,
        )
    elif cf.training.discriminator.optimizer == "Adam":
        disc_opt = torch.optim.Adam(
            discriminator.parameters(),
            lr=cf.training.discriminator.learning_rate,
            betas=(
                cf.training.discriminator.beta1,
                cf.training.discriminator.beta2,
            ),
            weight_decay=cf.training.discriminator.weight_decay,
        )

    # Create learning rate schedulers
    warmup_epochs = getattr(cf.training, "warmup_epochs", 5)
    plateau_epochs = getattr(cf.training, "plateau_epochs", 0)
    transition_epochs = getattr(cf.training, "transition_epochs", 0)
    lr_decay_gamma = getattr(cf.training, "lr_decay_gamma", 0.95)
    warmup_start_lr = getattr(cf.training, "warmup_start_lr", 1e-6)

    gen_scheduler = create_warmup_scheduler(
        gen_opt,
        warmup_epochs=warmup_epochs,
        total_epochs=cf.training.epochs,
        warmup_start_lr=warmup_start_lr,
        plateau_epochs=plateau_epochs,
        transition_epochs=transition_epochs,
        gamma=lr_decay_gamma,
    )

    disc_scheduler = create_warmup_scheduler(
        disc_opt,
        warmup_epochs=warmup_epochs,
        total_epochs=cf.training.epochs,
        warmup_start_lr=warmup_start_lr,
        plateau_epochs=plateau_epochs,
        transition_epochs=transition_epochs,
        gamma=lr_decay_gamma,
    )

    return gen_opt, disc_opt, gen_scheduler, disc_scheduler


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SpatialGAN model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration YAML file (default: config.yml)",
    )
    args = parser.parse_args()

    # find project base directory
    project_base = pathlib.Path(os.getcwd())

    # load configuration from specified file
    config_path = os.path.join(project_base, args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cf = config.load_config_from_yaml(config_path)

    # Set up run directory
    run_id = config.generate_run_id()
    run_dir = config.setup_experiment_directory(project_base, run_id)
    cf.logging.run_id = run_id
    cf.logging.run_dir = run_dir
    cf.config_path = os.path.join(run_dir, "config.yaml")
    cf.save()

    # Set up logging to file in run directory
    log_file = os.path.join(cf.logging.run_dir, "training.log")

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Keep console output as well
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Using configuration file: {config_path}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Run directory: {run_dir}")

    dataloader_train, test_dataloader, cf, norm_params = dataloader.build_dataloaders(
        cf
    )
    # dataloader_train, test_dataloader = dataloader.build_dummy_dataloaders()
    # update cf in run directory
    cf.save()

    # Save normalization parameters
    save_normalization_params(norm_params, cf.logging.run_dir)

    # describe shapes of data
    logger.info("Training data shapes:")
    x_shape, y_shape = dataloader_train.dataset._get_shapes()
    logger.info(f"  x: {x_shape}")
    logger.info(f"  y: {y_shape}")

    ##################
    # Model setup
    ##################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize upsampler based on config
    use_learnable_upsampler = cf.model.get("use_learnable_upsampler", False)
    if use_learnable_upsampler:
        logger.info("Using learnable upsampler")
        upsampler = LearnableUpsampler(in_channels=15).to(device)
    else:
        logger.info("Using fixed bilinear upsampler")
        upsampler = None

    # Create generator and discriminator using registry
    architecture = cf.model.get("architecture") or cf.model.get(
        "generator_architecture", "spategan"
    )
    logger.info(f"Using {architecture} architecture")

    generator = create_generator(cf, device)
    discriminator = create_discriminator(cf, device)

    # Set conditioning flag based on discriminator architecture
    condition_separate_channels = cf.model.discriminator_architecture != "unet"

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # FSS criterion
    fss_criterion = FSSLoss(
        thresholds=[
            0.1,
            0.2,
            0.4,
            0.8,
            1.6,
            2.4,
            4,
            6,
            10,
            25,
        ],  # not normalized thresholds
        scales=[2, 8, 16],
        device="cuda",
        sharpness=3.0,
        batch_size=10,
        config=cf,
        norm_params=norm_params,
    )

    # Optimizers and schedulers
    gen_opt, disc_opt, gen_scheduler, disc_scheduler = setup_optimizers(
        cf, generator, discriminator, upsampler, device
    )

    # For mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    ##################
    # Training loop
    ##################

    # GAN Training loop
    loss_gen_train = []
    loss_disc_train = []

    # Test loss tracking - now with individual components
    loss_test_history = {
        "gen_total": [],
        "disc_total": [],
        "l1": [],
        "mse": [],
        "gan": [],
        "fss": [],
        "disc_real": [],
        "disc_fake": [],
    }

    # Store diagnostics
    diagnostic_history = {
        "rmse": [],
        "bias_mean": [],
        "bias_q95": [],
        "bias_q98": [],
        "std_ratio": [],
        "mae": [],
        "correlation": [],
        "anomaly_correlation": [],
        "fss": [],
        "epochs": [],
    }

    # Track best validation loss
    best_val_loss = float("inf")
    best_val_epoch = 0

    logger.info(f"Starting GAN training for {cf.training.epochs} epochs...")

    # Get a fixed batch for visualization
    val_iter = iter(test_dataloader)
    x_vis, y_vis = next(val_iter)
    x_vis, y_vis = x_vis.to(device), y_vis.to(device)

    for epoch in range(cf.training.epochs):
        # Training phase
        epoch_gen_losses = []
        epoch_disc_losses = []

        ##################
        # Train
        ##################

        for batch_idx, (x_batch, y_batch) in tqdm(
            enumerate(dataloader_train), total=len(dataloader_train)
        ):
            x_batch = x_batch.to(device)
            if upsampler is not None:
                x_batch_hr = upsampler(x_batch)
            else:
                x_batch_hr = dataloader.upscale_nn(x_batch)
            # during training, noise channel is added during train step
            y_batch_2d = y_batch.to(device)

            # zero timestep, for diffusion UNET.
            timesteps = torch.zeros([x_batch.shape[0]]).to(
                device
            )  # only for diffusion unet

            # Train discriminator n_critic times
            n_critic = getattr(cf.training, "n_critic", 1)
            disc_losses_batch = []

            for _ in range(n_critic):
                # Train discriminator only
                _, disc_loss = train_gan_step(
                    config=cf,
                    input_image=x_batch,
                    input_image_hr=x_batch_hr,
                    target=y_batch_2d,
                    step=epoch * len(dataloader_train) + batch_idx,
                    discriminator=discriminator,
                    generator=generator,
                    gen_opt=None,  # Don't update generator
                    disc_opt=disc_opt,
                    scaler=scaler,
                    criterion=criterion,
                    timesteps=timesteps,
                    loss_weights=cf.training.loss_weights,
                    condition_separate_channels=condition_separate_channels,
                    fss_criterion=fss_criterion,
                )
                disc_losses_batch.append(disc_loss)

            # Train generator once
            gen_loss, _ = train_gan_step(
                config=cf,
                input_image=x_batch,
                input_image_hr=x_batch_hr,
                target=y_batch_2d,
                step=epoch * len(dataloader_train) + batch_idx,
                discriminator=discriminator,
                generator=generator,
                gen_opt=gen_opt,
                disc_opt=None,  # Don't update discriminator
                scaler=scaler,
                criterion=criterion,
                timesteps=timesteps,
                loss_weights=cf.training.loss_weights,
                condition_separate_channels=condition_separate_channels,
                fss_criterion=fss_criterion,
            )

            epoch_gen_losses.append(gen_loss)
            epoch_disc_losses.append(np.mean(disc_losses_batch))

        # Calculate average training losses
        train_gen_loss = np.mean(epoch_gen_losses)
        train_disc_loss = np.mean(epoch_disc_losses)
        loss_gen_train.append(train_gen_loss)
        loss_disc_train.append(train_disc_loss)

        ##################
        # Validation
        ##################

        # Validation phase using test_gan_step
        generator.eval()
        discriminator.eval()

        batch_loss_dicts = []

        for batch_idx, (x_batch, y_batch) in enumerate(test_dataloader):
            if batch_idx >= cf.training.batches_per_validation:
                break

            x_batch = x_batch.to(device)
            if upsampler is not None:
                x_batch_hr = upsampler(x_batch)
            else:
                x_batch_hr = dataloader.upscale_nn(x_batch)
            # Note: test_gan_step doesn't add noise channel
            y_batch_2d = y_batch.to(device)

            # zero timestep, for diffusion UNET.
            timesteps = torch.zeros([x_batch.shape[0]]).to(device)

            # Use test_gan_step to get all loss components
            loss_dict = test_gan_step(
                config=cf,
                input_image=x_batch,
                input_image_hr=x_batch_hr,
                target=y_batch_2d,
                discriminator=discriminator,
                generator=generator,
                criterion=criterion,
                fss_criterion=fss_criterion,
                timesteps=timesteps,
                loss_weights=cf.training.loss_weights,
                condition_separate_channels=condition_separate_channels,
            )
            batch_loss_dicts.append(loss_dict)

        # Average losses across batches
        for key in loss_test_history.keys():
            values = [d.get(key, 0.0) for d in batch_loss_dicts]
            epoch_mean = np.mean(values)
            loss_test_history[key].append(epoch_mean)

        # For backward compatibility with diagnostic computation
        test_loss = loss_test_history["gen_total"][-1]
        mean_fss_test = loss_test_history["fss"][-1]

        # Step learning rate schedulers
        gen_scheduler.step()
        disc_scheduler.step()

        # Get current learning rates
        current_gen_lr = gen_opt.param_groups[0]["lr"]
        current_disc_lr = disc_opt.param_groups[0]["lr"]

        # Track best validation loss
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            best_val_epoch = epoch + 1

        # Print progress and plot
        if (epoch + 1) % cf.logging.log_frequency == 0 or epoch == 0:
            clear_output(wait=True)
            logger.info(f"Epoch {epoch + 1}/{cf.training.epochs}")
            logger.info(
                f"  Generator Loss:     {train_gen_loss:.6f} (LR: {current_gen_lr:.2e})"
            )
            logger.info(
                f"  Discriminator Loss: {train_disc_loss:.6f} (LR: {current_disc_lr:.2e})"
            )
            logger.info(f"  Test Loss (Gen Total): {test_loss:.6f}")
            if loss_test_history["l1"][-1] > 0:
                logger.info(f"  Test L1:            {loss_test_history['l1'][-1]:.6f}")
            if loss_test_history["mse"][-1] > 0:
                logger.info(f"  Test MSE:           {loss_test_history['mse'][-1]:.6f}")
            if loss_test_history["fss"][-1] > 0:
                logger.info(f"  Test FSS:           {loss_test_history['fss'][-1]:.6f}")
            if loss_test_history["gan"][-1] > 0:
                logger.info(f"  Test GAN:           {loss_test_history['gan'][-1]:.6f}")
            logger.info(
                f"  Best Val Loss:      {best_val_loss:.6f} (epoch {best_val_epoch})"
            )

            # Plot losses with individual components
            plot_adversarial_losses(
                loss_gen_train, loss_disc_train, loss_test_history, cf
            )

        ##################
        # Diagnostics
        ##################

        # Compute diagnostics
        if (epoch + 1) % cf.logging.diagnostic_frequency == 0:
            logger.info("  Computing diagnostics...")
            generator.eval()

            # Collect all test predictions
            all_preds = []
            all_targets = []

            # to do change y to 2D
            with torch.no_grad():
                for x_batch, y_batch in test_dataloader:
                    x_batch = x_batch.to(device)
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    else:
                        x_batch_hr = dataloader.upscale_nn(x_batch)
                    x_batch_hr = dataloader.add_noise_channel(x_batch_hr)

                    y_batch = torch.flatten(y_batch, start_dim=1)

                    timesteps = torch.zeros([x_batch.shape[0]]).to(device)

                    with torch.amp.autocast("cuda"):
                        match architecture:
                            case "spategan":
                                y_pred = generator(x_batch)
                                y_pred = torch.flatten(y_pred, start_dim=1)
                            case "diffusion_unet":
                                y_pred = generator(x_batch_hr, timesteps)
                                y_pred = torch.flatten(y_pred, start_dim=1)
                            case "deepesd":
                                y_pred = generator(x_batch)
                                y_pred = torch.flatten(y_pred, start_dim=1)

                    all_preds.append(y_pred.cpu())
                    all_targets.append(y_batch.cpu())

            # Concatenate all batches
            y_pred_all = torch.cat(all_preds, dim=0)
            y_true_all = torch.cat(all_targets, dim=0)

            # Compute diagnostics using helper function
            diag_results = compute_diagnostics(
                y_pred_all, y_true_all, norm_params, cf, mean_fss_test, epoch + 1
            )

            # Store in history
            for key, value in diag_results.items():
                if key == "epoch":
                    diagnostic_history["epochs"].append(value)
                else:
                    diagnostic_history[key].append(value)

            plot_diagnostic_history(diagnostic_history, cf)

        # Save checkpoint
        if (epoch + 1) % cf.logging.checkpoint_frequency == 0:
            checkpoint_dict = {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "gen_optimizer_state_dict": gen_opt.state_dict(),
                "disc_optimizer_state_dict": disc_opt.state_dict(),
                "gen_scheduler_state_dict": gen_scheduler.state_dict(),
                "disc_scheduler_state_dict": disc_scheduler.state_dict(),
                "train_gen_loss": train_gen_loss,
                "train_disc_loss": train_disc_loss,
                "test_loss": test_loss,
                "diagnostic_history": diagnostic_history,
            }
            if upsampler is not None:
                checkpoint_dict["upsampler_state_dict"] = upsampler.state_dict()
            torch.save(
                checkpoint_dict,
                f"{cf.logging.run_dir}/checkpoints/checkpoint_epoch_{epoch + 1}.pt",
            )
            logger.info("  Checkpoint saved")

        if (epoch + 1) % cf.logging.map_frequency == 0:
            # Visualize predictions with denormalization
            g = torch.Generator(device="cpu")
            g.seed()  # uses system entropy
            idx = torch.randint(0, x_vis.size(0), (1,), generator=g).item()
            if architecture == "diffusion_unet":
                if upsampler is not None:
                    x_vis_up = upsampler(x_vis)
                else:
                    x_vis_up = dataloader.upscale_nn(x_vis)
                x_vis_up = dataloader.add_noise_channel(
                    x_vis_up
                )  # add noise to HR or LR?
            else:
                x_vis_up = x_vis
            # logger.info(f"  Plotting sample {idx}")
            # plot_predictions(
            #     generator,
            #     x_vis_up,
            #     y_vis,
            #     cf,
            #     epoch + 1,
            #     device,
            #     sample_idx=idx,
            #     norm_params=norm_p
            logger.info("Plotting samples 0-2")
            plot_predictions_only(
                generator,
                x_vis_up,
                y_vis,
                cf,
                epoch + 1,
                device,
                num_samples=3,
                norm_params=norm_params,
            )

    # Save final models
    checkpoint_dict = {
        "epoch": cf.training.epochs,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "gen_optimizer_state_dict": gen_opt.state_dict(),
        "disc_optimizer_state_dict": disc_opt.state_dict(),
        "gen_scheduler_state_dict": gen_scheduler.state_dict(),
        "disc_scheduler_state_dict": disc_scheduler.state_dict(),
        "diagnostic_history": diagnostic_history,
        "best_val_loss": best_val_loss,
        "best_val_epoch": best_val_epoch,
    }
    if upsampler is not None:
        checkpoint_dict["upsampler_state_dict"] = upsampler.state_dict()
    torch.save(
        checkpoint_dict,
        f"{cf.logging.run_dir}/checkpoints/final_models.pt",
    )

    logger.info("\nTraining complete!")
    logger.info(
        f"Best validation L1 loss: {best_val_loss:.6f} at epoch {best_val_epoch}"
    )

    logger.info(f"\nGAN training complete! Models saved to {cf.logging.run_dir}")


if __name__ == "__main__":
    main()
