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
import json
import logging
import os
import pathlib

# Import diagnostics
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.evaluate.diagnostics import (
    compute_diagnostics,
    compute_model_selection_score,
)
from ml_benchmark_spategan.evaluate.visualization.plot_train import (
    plot_adversarial_losses,
    plot_diagnostic_history,
    plot_predictions_only,
)
from ml_benchmark_spategan.train.dataloader import dataloader
from ml_benchmark_spategan.train.gan_training import test_gan_step, train_gan_step
from ml_benchmark_spategan.train.gan_training.fss import FSSLoss
from ml_benchmark_spategan.train.gan_training.train_gan_step import (
    _generate_ensemble,
)
from ml_benchmark_spategan.train.lr_scheduler import setup_optimizers
from ml_benchmark_spategan.train.model.registry import (
    create_discriminator,
    create_generator,
)
from ml_benchmark_spategan.train.normalize import (
    save_normalization_params,
)
from ml_benchmark_spategan.utils.interpolate import (
    add_noise_channel,
    upscale_bilinear,
)


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

    # Set up run directory - use RUN_ID from environment if provided (for grid searches)
    run_id = os.environ.get("RUN_ID")
    if run_id:
        logger_msg = f"Using pre-generated RUN_ID from environment: {run_id}"
        print(logger_msg)  # Print before logging is set up
    else:
        run_id = config.generate_run_id()
        logger_msg = f"Generated new RUN_ID: {run_id}"
        print(logger_msg)

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
    # update cf in run directory
    cf.save()

    # Save normalization parameters
    save_normalization_params(norm_params, cf.logging.run_dir)

    # describe shapes of data
    logger.info("Training data shapes:")
    x_shape, y_shape = dataloader_train.dataset._get_shapes()
    logger.info(f"  x: {x_shape}")
    logger.info(f"  y: {y_shape}")
    logger.info(f"  Training batches: {len(dataloader_train)}")
    logger.info(f"  Test batches: {len(test_dataloader)}")
    logger.info(f"  Batches per validation: {cf.training.batches_per_validation}")

    ##################
    # Model setup
    ##################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # FSS criterion with variable-specific thresholds
    var_target = cf.data.var_target
    if var_target == "pr":
        # Precipitation thresholds (mm/day)
        fss_thresholds = [0.1, 0.2, 0.4, 0.8, 1.6, 2.4, 4, 6, 10, 25]
    elif var_target == "tasmax":
        # Temperature thresholds (Kelvin) - relative to typical range
        # These cover ~270K to ~310K with finer resolution in middle
        fss_thresholds = [270, 275, 280, 285, 290, 295, 300, 305, 310]
    else:
        # Default fallback
        logger.warning(
            f"No FSS thresholds defined for variable {var_target}, using precipitation defaults"
        )
        fss_thresholds = [0.1, 0.2, 0.4, 0.8, 1.6, 2.4, 4, 6, 10, 25]

    fss_criterion = FSSLoss(
        thresholds=fss_thresholds,
        scales=[2, 8, 16],
        device="cuda",
        sharpness=3.0,
        batch_size=10,
        config=cf,
        norm_params=norm_params,
    )

    # Optimizers and schedulers
    gen_opt, disc_opt, gen_scheduler, disc_scheduler = setup_optimizers(
        cf, generator, discriminator, upsampler
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
        "psd_distance": [],
        "fss": [],
        "lag1_corr_bias": [],
        "interannual_var_bias": [],
        # Variable-specific metrics (will be empty if not applicable)
        "su_bias": [],
        "txx_bias": [],
        "txn_bias": [],
        "rx1day_bias": [],
        "sdii_bias": [],
        "cdd_bias": [],
        "cwd_bias": [],
        "epochs": [],
        # Model selection score (composite weighted score)
        "model_score": [],
    }

    # Track best validation loss
    best_val_loss = float("inf")
    best_val_epoch = 0

    # Track best model selection score (lower is better)
    best_model_score = float("inf")
    best_model_epoch = 0

    logger.info(f"Starting GAN training for {cf.training.epochs} epochs...")

    # Get a fixed batch for visualization
    val_iter = iter(test_dataloader)
    vis_batch = next(val_iter)
    if len(vis_batch) == 3:
        x_vis, y_vis, doy_vis = vis_batch
    else:
        x_vis, y_vis = vis_batch
    x_vis, y_vis = x_vis.to(device), y_vis.to(device)

    if cf.data.use_orography:
        orography = dataloader_train.dataset.orography.to(device)

    for epoch in range(cf.training.epochs):
        # Training phase
        epoch_gen_losses = []
        epoch_disc_losses = []

        # Initialize dataloader iterator for n_critic > 1 support
        dataloader_train_iter = iter(dataloader_train)

        ##################
        # Train
        ##################

        for batch_idx, batch_data in tqdm(
            enumerate(dataloader_train), total=len(dataloader_train)
        ):
            # Unpack batch data (may include doy)
            if len(batch_data) == 3:
                x_batch, y_batch, doy_batch = batch_data
                # doy_batch is already sinusoidally encoded by the dataloader
                timesteps = doy_batch.to(device)
            else:
                x_batch, y_batch = batch_data
                # Fallback to zero timestep if no doy available
                timesteps = torch.zeros([x_batch.shape[0]]).to(device)

            x_batch = x_batch.to(device)
            if upsampler is not None:
                x_batch_hr = upsampler(x_batch)
            else:
                x_batch_hr = upscale_bilinear(x_batch)
            # during training, noise channel is added during train step
            y_batch_2d = y_batch.to(device)

            # Train discriminator n_critic times
            n_critic = getattr(cf.training, "n_critic", 1)
            disc_losses_batch = []

            # repeat orography for each sample in batch if used
            if cf.data.use_orography:
                orography_batch = orography.repeat(x_batch.shape[0], 1, 1).unsqueeze(1)
            else:
                orography_batch = None

            # Train discriminator n_critic times with DIFFERENT batches
            for critic_step in range(n_critic):
                # Get a fresh batch for discriminator training (prevents overfitting)
                if critic_step > 0:
                    try:
                        batch_data = next(dataloader_train_iter)
                        if len(batch_data) == 3:
                            x_batch, y_batch, doy_batch = batch_data
                            # doy_batch is already sinusoidally encoded by the dataloader
                            timesteps = doy_batch.to(device)
                        else:
                            x_batch, y_batch = batch_data
                            timesteps = torch.zeros([x_batch.shape[0]]).to(device)
                    except StopIteration:
                        # If we run out of batches, reset iterator
                        dataloader_train_iter = iter(dataloader_train)
                        batch_data = next(dataloader_train_iter)
                        if len(batch_data) == 3:
                            x_batch, y_batch, doy_batch = batch_data
                            # doy_batch is already sinusoidally encoded by the dataloader
                            timesteps = doy_batch.to(device)
                        else:
                            x_batch, y_batch = batch_data
                            timesteps = torch.zeros([x_batch.shape[0]]).to(device)

                    x_batch = x_batch.to(device)
                    y_batch_2d = y_batch.to(device)

                    # Recompute upsampled version for new batch
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    else:
                        x_batch_hr = upscale_bilinear(x_batch)

                    if cf.data.use_orography:
                        orography_batch = orography.repeat(
                            x_batch.shape[0], 1, 1
                        ).unsqueeze(1)
                    else:
                        orography_batch = None

                # Train discriminator only
                _, disc_loss = train_gan_step(
                    config=cf,
                    input_image=x_batch,
                    input_image_hr=x_batch_hr,
                    orography=orography_batch,
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
                orography=orography_batch,
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

        # Only run validation if batches_per_validation is set and > 0
        if (
            cf.training.batches_per_validation is not None
            and cf.training.batches_per_validation > 0
        ):
            batch_count = 0
            for batch_idx, batch_data in enumerate(test_dataloader):
                if batch_idx >= cf.training.batches_per_validation:
                    break

                batch_count += 1

                # Unpack batch data (may include doy)
                if len(batch_data) == 3:
                    x_batch, y_batch, doy_batch = batch_data
                    # doy_batch is already sinusoidally encoded by the dataloader
                    timesteps = doy_batch.to(device)
                else:
                    x_batch, y_batch = batch_data
                    timesteps = torch.zeros([x_batch.shape[0]]).to(device)

                x_batch = x_batch.to(device)
                if upsampler is not None:
                    x_batch_hr = upsampler(x_batch)
                else:
                    x_batch_hr = upscale_bilinear(x_batch)
                y_batch = y_batch.to(device)
                y_batch_2d = y_batch.view(-1, 1, 128, 128)

                if cf.data.use_orography:
                    orography_batch = orography.repeat(
                        x_batch.shape[0], 1, 1
                    ).unsqueeze(1)
                else:
                    orography_batch = None

                # Use test_gan_step to get all loss components
                loss_dict = test_gan_step(
                    config=cf,
                    input_image=x_batch,
                    input_image_hr=x_batch_hr,
                    orography=orography_batch,
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

            # Log warning if no validation batches were processed
            if batch_count == 0:
                logger.warning(
                    "Validation requested but test_dataloader is empty or has 0 batches!"
                )

        # Average losses across batches (only if validation ran)
        if batch_loss_dicts:
            for key in loss_test_history.keys():
                values = [d.get(key, 0.0) for d in batch_loss_dicts]
                epoch_mean = np.mean(values)
                loss_test_history[key].append(epoch_mean)

            # For backward compatibility with diagnostic computation
            test_loss = loss_test_history["gen_total"][-1]
            mean_fss_test = loss_test_history["fss"][-1]
        else:
            # No validation was run this epoch
            test_loss = None
            mean_fss_test = None

        # Step learning rate schedulers
        gen_scheduler.step()
        disc_scheduler.step()

        # Get current learning rates
        current_gen_lr = gen_opt.param_groups[0]["lr"]
        current_disc_lr = disc_opt.param_groups[0]["lr"]

        # Track best validation loss (only if validation ran)
        if test_loss is not None and test_loss < best_val_loss:
            best_val_loss = test_loss
            best_val_epoch = epoch + 1

        # Print progress and plot
        if (epoch + 1) % cf.logging.log_frequency == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{cf.training.epochs}")
            logger.info(
                f"  Generator Loss:     {train_gen_loss:.6f} (LR: {current_gen_lr:.2e})"
            )
            logger.info(
                f"  Discriminator Loss: {train_disc_loss:.6f} (LR: {current_disc_lr:.2e})"
            )

            # Only log test losses if validation ran
            if test_loss is not None:
                logger.info(f"  Test Loss (Gen Total): {test_loss:.6f}")
                logger.info(
                    f"  Test Disc Total:    {loss_test_history['disc_total'][-1]:.6f}"
                )
                logger.info(
                    f"  Test Disc Real:     {loss_test_history['disc_real'][-1]:.6f}"
                )
                logger.info(
                    f"  Test Disc Fake:     {loss_test_history['disc_fake'][-1]:.6f}"
                )
                if loss_test_history["l1"][-1] > 0:
                    logger.info(
                        f"  Test L1:            {loss_test_history['l1'][-1]:.6f}"
                    )
                if loss_test_history["mse"][-1] > 0:
                    logger.info(
                        f"  Test MSE:           {loss_test_history['mse'][-1]:.6f}"
                    )
                if loss_test_history["fss"][-1] > 0:
                    logger.info(
                        f"  Test FSS:           {loss_test_history['fss'][-1]:.6f}"
                    )
                if loss_test_history["gan"][-1] > 0:
                    logger.info(
                        f"  Test GAN:           {loss_test_history['gan'][-1]:.6f}"
                    )
            logger.info(
                f"  Best Val Loss:      {best_val_loss:.6f} (epoch {best_val_epoch})"
            )

            # Plot losses with individual components
            plot_adversarial_losses(
                loss_gen_train, loss_disc_train, loss_test_history, cf
            )

            # Save loss history to JSON file
            # Pad test history to match training history length (fill with None for epochs without validation)
            loss_test_history_padded = {}
            for key, values in loss_test_history.items():
                padded_values = []
                test_idx = 0
                for epoch_idx in range(len(loss_gen_train)):
                    if test_idx < len(values):
                        padded_values.append(values[test_idx])
                        test_idx += 1
                    else:
                        padded_values.append(None)
                loss_test_history_padded[key] = padded_values

            loss_history = {
                "loss_gen_train": loss_gen_train,
                "loss_disc_train": loss_disc_train,
                "loss_test_history": loss_test_history_padded,
            }
            loss_history_path = os.path.join(cf.logging.run_dir, "loss_history.json")
            with open(loss_history_path, "w") as f:
                json.dump(loss_history, f, indent=2)

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
            all_ensemble_preds = []

            # to do change y to 2D
            with torch.no_grad():
                for batch_data in test_dataloader:
                    # Unpack batch data (may include doy)
                    if len(batch_data) == 3:
                        x_batch, y_batch, doy_batch = batch_data
                        # doy_batch is already sinusoidally encoded by the dataloader
                        timesteps = doy_batch.to(device)
                    else:
                        x_batch, y_batch = batch_data
                        timesteps = torch.zeros([x_batch.shape[0]]).to(device)

                    x_batch = x_batch.to(device)
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    x_batch = x_batch.to(device)
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    else:
                        x_batch_hr = upscale_bilinear(x_batch)
                    # during training, noise channel is added during train step
                    y_batch_2d = y_batch.to(device).view(-1, 1, 128, 128)

                    # Concatenate orography if available (before adding noise)
                    if cf.data.use_orography:
                        orography_batch_diag = orography.repeat(
                            x_batch.shape[0], 1, 1
                        ).unsqueeze(1)
                        x_batch_hr_with_oro = torch.cat(
                            [x_batch_hr, orography_batch_diag], dim=1
                        )
                    else:
                        x_batch_hr_with_oro = x_batch_hr

                    x_batch_hr_with_oro = add_noise_channel(x_batch_hr_with_oro)

                    y_batch = torch.flatten(y_batch, start_dim=1)

                    timesteps = torch.zeros([x_batch.shape[0]]).to(device)

                    with torch.amp.autocast("cuda"):
                        # Generate ensemble predictions for variability metric
                        ensemble_size = getattr(cf.training, "ensemble_size", 10)
                        gen_ensemble = _generate_ensemble(
                            generator=generator,
                            architecture=architecture,
                            input_image=x_batch,
                            input_image_hr=x_batch_hr,
                            orography=orography_batch_diag
                            if cf.data.use_orography
                            else None,
                            timesteps=timesteps,
                            ensemble_size=ensemble_size,
                            noise_std=cf.training.get("noise_std_gen", 0.0),
                        )

                        # Use ensemble mean as the single prediction
                        y_pred = gen_ensemble.mean(dim=1).flatten(start_dim=1)

                        # Store ensemble for variability computation (flatten spatial dims)
                        # gen_ensemble shape: (B, N_ensemble, H, W) -> flatten to (B, N_ensemble, H*W)
                        gen_ensemble_flat = gen_ensemble.flatten(start_dim=2)
                        all_ensemble_preds.append(gen_ensemble_flat.cpu())

                    all_preds.append(y_pred.cpu())
                    all_targets.append(y_batch.cpu())

            # Concatenate all batches
            y_pred_all = torch.cat(all_preds, dim=0)
            y_true_all = torch.cat(all_targets, dim=0)
            ensemble_preds_all = torch.cat(
                all_ensemble_preds, dim=0
            )  # (B, N_ensemble, H*W)

            # Compute diagnostics using helper function
            diag_results = compute_diagnostics(
                y_pred_all,
                y_true_all,
                norm_params,
                cf,
                mean_fss_test,
                epoch + 1,
                ensemble_preds_all=ensemble_preds_all,
            )

            # Store in history
            for key, value in diag_results.items():
                if key == "epoch":
                    diagnostic_history["epochs"].append(value)
                else:
                    # Initialize key if it doesn't exist (for variable-specific metrics)
                    if key not in diagnostic_history:
                        diagnostic_history[key] = []
                    diagnostic_history[key].append(value)

            # Compute model selection score
            model_score = compute_model_selection_score(diag_results)
            diagnostic_history["model_score"].append(model_score)
            logger.info(f"  Model Selection Score: {model_score:.4f} (lower is better)")

            # Save best model checkpoint if score improved
            if model_score < best_model_score:
                best_model_score = model_score
                best_model_epoch = epoch + 1
                logger.info(
                    f"  New best model! Score: {model_score:.4f} at epoch {best_model_epoch}"
                )

                # Save best model checkpoint (overwrite previous)
                best_checkpoint_dict = {
                    "epoch": epoch + 1,
                    "model_score": model_score,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "gen_optimizer_state_dict": gen_opt.state_dict(),
                    "disc_optimizer_state_dict": disc_opt.state_dict(),
                    "gen_scheduler_state_dict": gen_scheduler.state_dict(),
                    "disc_scheduler_state_dict": disc_scheduler.state_dict(),
                    "diagnostic_history": diagnostic_history,
                    "diagnostics": diag_results,
                }
                if upsampler is not None:
                    best_checkpoint_dict["upsampler_state_dict"] = (
                        upsampler.state_dict()
                    )
                torch.save(
                    best_checkpoint_dict,
                    f"{cf.logging.run_dir}/checkpoints/best_model.pt",
                )
                logger.info("  Best model checkpoint saved")

            # Save diagnostic history to JSON file (independent of checkpoints)
            diagnostic_history_path = os.path.join(
                cf.logging.run_dir, "diagnostic_history.json"
            )
            with open(diagnostic_history_path, "w") as f:
                json.dump(diagnostic_history, f, indent=2)

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
                    x_vis_up = upscale_bilinear(x_vis)
                # Concatenate orography if available (before adding noise)
                if cf.data.use_orography:
                    orography_batch_vis = orography.repeat(
                        x_vis.shape[0], 1, 1
                    ).unsqueeze(1)
                    x_vis_up = torch.cat([x_vis_up, orography_batch_vis], dim=1)
                x_vis_up = add_noise_channel(x_vis_up)  # add noise to HR or LR?
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
        "best_model_score": best_model_score,
        "best_model_epoch": best_model_epoch,
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
    logger.info(
        f"Best model selection score: {best_model_score:.4f} at epoch {best_model_epoch}"
    )
    logger.info(
        f"Best model checkpoint saved to: {cf.logging.run_dir}/checkpoints/best_model.pt"
    )

    logger.info(f"\nGAN training complete! Models saved to {cf.logging.run_dir}")


if __name__ == "__main__":
    main()
