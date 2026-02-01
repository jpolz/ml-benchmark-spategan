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
import math
import os
import pathlib

# Import diagnostics
import sys

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from tqdm import tqdm

from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.dataloader import dataloader
from ml_benchmark_spategan.model.learnable_noise import (
    LearnableNoiseScale,
    SpatialNoiseScale,
)
from ml_benchmark_spategan.model.registry import create_discriminator, create_generator
from ml_benchmark_spategan.training.gan_training import test_gan_step, train_gan_step
from ml_benchmark_spategan.training.gan_training.losses import FSSLoss
from ml_benchmark_spategan.training.gan_training.train_gan_step import (
    _generate_ensemble,
)
from ml_benchmark_spategan.training.lr_scheduler import setup_optimizers
from ml_benchmark_spategan.utils.interpolate import LearnableUpsampler
from ml_benchmark_spategan.utils.normalize import (
    predictions_to_xarray,
    save_normalization_params,
)
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
import indices
from model_selection_score import DEFAULT_WEIGHTS as SCORE_WEIGHTS


def compute_model_selection_score(
    diagnostics_dict: dict, weights: dict = None
) -> float:
    """
    Compute the weighted composite model selection score from diagnostic metrics.

    Lower scores indicate better models. This score is used to determine the
    best model checkpoint during training.

    Args:
        diagnostics_dict: Dictionary of diagnostic metrics (single epoch, not history)
        weights: Optional custom weights dict. Uses DEFAULT_WEIGHTS if None.

    Returns:
        Composite score (lower is better)
    """
    if weights is None:
        weights = SCORE_WEIGHTS

    total_weight = 0.0
    weighted_sum = 0.0

    for metric_name, weight in weights.items():
        if metric_name not in diagnostics_dict:
            continue

        value = diagnostics_dict[metric_name]

        # Skip NaN values or non-numeric
        if not isinstance(value, (int, float)) or np.isnan(value):
            continue

        # Metrics where ideal value is 1.0
        if metric_name in ["std_ratio", "correlation", "anomaly_correlation"]:
            value = abs(value - 1.0)
        else:
            # For metrics where higher is better (negative weight), negate first
            if weight < 0:
                value = -value
            # Then take absolute value so all contributions are positive
            value = abs(value)

        # Apply weight
        weighted_value = abs(weight) * value
        weighted_sum += weighted_value
        total_weight += abs(weight)

    if total_weight > 0:
        return weighted_sum / total_weight
    return float("inf")


def sinusoidal_encoding_doy(doy: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Apply sinusoidal encoding to day of year values.

    For models that expect values between 0 and 1, this creates a smooth
    cyclic representation where day 1 and day 365/366 are close together.

    This is used when cf.data.use_doy is True to replace the zero timestep
    with seasonal conditioning information.

    Args:
        doy: Day of year tensor (values 1-366)
        normalize: If True, normalize to [0, 1] range. If False, keep raw encoding.

    Returns:
        Encoded day of year tensor
    """
    # Convert to angle (0 to 2*pi)
    angle = (doy - 1) / 365.25 * 2 * math.pi

    if normalize:
        # Use sine encoding normalized to [0, 1]
        # sin ranges from [-1, 1], so (sin + 1) / 2 gives [0, 1]
        encoded = (torch.sin(angle) + 1.0) / 2.0
    else:
        # Use raw sine encoding [-1, 1]
        encoded = torch.sin(angle)

    return encoded


def compute_diagnostics(
    y_pred_all,
    y_true_all,
    norm_params,
    cf,
    mean_fss_test,
    epoch,
    ensemble_preds_all=None,
):
    """
    Compute diagnostic metrics from predictions and ground truth.

    Args:
        y_pred_all: Concatenated predictions tensor (all test batches)
        y_true_all: Concatenated ground truth tensor (all test batches)
        norm_params: Normalization parameters dictionary
        cf: Configuration object
        mean_fss_test: Mean FSS test loss for this epoch
        epoch: Current epoch number
        ensemble_preds_all: Optional ensemble predictions (B, N_ensemble, H, W) for variability computation

    Returns:
        dict: Dictionary of diagnostic metrics
    """
    logger = logging.getLogger(__name__)

    # Convert to xarray with denormalization
    pred_ds, true_ds = predictions_to_xarray(
        y_pred_all, y_true_all, norm_params, var_name=cf.data.var_target
    )

    var_target = cf.data.var_target

    # Compute standard diagnostics
    rmse = diagnostics.rmse(true_ds, pred_ds, var=var_target, dim="time")
    bias_mean = diagnostics.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].mean("time"),
    )
    bias_q95 = diagnostics.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].quantile(0.95, dim="time"),
    )
    bias_q98 = diagnostics.bias_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].quantile(0.98, dim="time"),
    )
    std_ratio = diagnostics.ratio_index(
        true_ds,
        pred_ds,
        index_fn=lambda x, **kw: x[var_target].std("time"),
    )

    # Mean Absolute Error
    mae = np.abs(pred_ds[var_target] - true_ds[var_target]).mean("time")

    # Pearson correlation
    correlation = xr.corr(
        pred_ds[var_target],
        true_ds[var_target],
        dim="time",
    )

    spatial_dims = norm_params["spatial_dims"]
    # Anomaly correlation (after removing climatology)
    pred_anomaly = pred_ds[var_target] - pred_ds[var_target].mean("time")
    true_anomaly = true_ds[var_target] - true_ds[var_target].mean("time")
    anomaly_correlation = xr.corr(pred_anomaly, true_anomaly, dim=spatial_dims)

    # Power Spectral Density and distance metric
    psd_true, psd_pred = diagnostics.psd(x0=true_ds, x1=pred_ds, var=var_target)

    # Compute PSD distance (RMSE in log space)
    wavenumber_min = 1
    wavenumber_max = min(60, len(psd_true) - 1)
    wavenumber = psd_true["wavenumber"].values
    mask = (wavenumber >= wavenumber_min) & (wavenumber <= wavenumber_max)
    eps = 1e-10
    log_psd_true = np.log10(psd_true.values[mask] + eps)
    log_psd_pred = np.log10(psd_pred.values[mask] + eps)
    psd_distance = float(np.sqrt(np.mean((log_psd_true - log_psd_pred) ** 2)))

    # Build diagnostics dictionary with spatially-averaged values
    diagnostics_dict = {
        "rmse": rmse[var_target].mean().values.item(),
        "bias_mean": bias_mean.mean().values.item(),
        "bias_q95": bias_q95.mean().values.item(),
        "bias_q98": bias_q98.mean().values.item(),
        "std_ratio": std_ratio.mean().values.item(),
        "mae": mae.mean().values.item(),
        "correlation": correlation.mean().values.item(),
        "anomaly_correlation": anomaly_correlation.mean().values.item(),
        "psd_distance": psd_distance,
        "fss": mean_fss_test,
        "epoch": epoch,
    }

    # Ensemble variability (if ensemble predictions provided)
    if ensemble_preds_all is not None:
        # ensemble_preds_all shape: (B, N_ensemble, H*W)
        # Compute std along ensemble dimension, then spatial mean
        ensemble_std = ensemble_preds_all.std(dim=1).mean().item()
        diagnostics_dict["ensemble_std"] = float(ensemble_std)
        logger.info(f"  Ensemble Variability (std): {ensemble_std:.4f}")

    # Variable-specific climate indices
    if var_target == "tasmax":
        # Temperature-specific indices
        # Summer days (days > 25°C, threshold=298.15K for data in Kelvin)
        su_true = indices.su(true_ds, var_target, threshold=298.15)
        su_pred = indices.su(pred_ds, var_target, threshold=298.15)
        su_bias = (su_pred[var_target] - su_true[var_target]).mean().values.item()
        diagnostics_dict["su_bias"] = float(su_bias)

        # Mean annual maximum temperature
        txx_true = indices.txx(true_ds, var_target)
        txx_pred = indices.txx(pred_ds, var_target)
        txx_bias = (txx_pred[var_target] - txx_true[var_target]).mean().values.item()
        diagnostics_dict["txx_bias"] = float(txx_bias)

        # Mean annual minimum temperature
        txn_true = indices.txn(true_ds, var_target)
        txn_pred = indices.txn(pred_ds, var_target)
        txn_bias = (txn_pred[var_target] - txn_true[var_target]).mean().values.item()
        diagnostics_dict["txn_bias"] = float(txn_bias)

        logger.info(f"  Summer Days Bias: {su_bias:.4f}")
        logger.info(f"  TXx (Annual Max) Bias: {txx_bias:.4f}")
        logger.info(f"  TXn (Annual Min) Bias: {txn_bias:.4f}")

    elif var_target == "pr":
        # Precipitation-specific indices
        # Maximum 1-day precipitation
        rx1day_true = indices.rx1day(true_ds, var_target)
        rx1day_pred = indices.rx1day(pred_ds, var_target)
        rx1day_bias = (
            (rx1day_pred[var_target] - rx1day_true[var_target]).mean().values.item()
        )
        diagnostics_dict["rx1day_bias"] = float(rx1day_bias)

        # Simple precipitation intensity (mean precip on wet days)
        sdii_true = indices.sdii(true_ds, var_target, wet_threshold=1.0)
        sdii_pred = indices.sdii(pred_ds, var_target, wet_threshold=1.0)
        sdii_bias = (sdii_pred[var_target] - sdii_true[var_target]).mean().values.item()
        diagnostics_dict["sdii_bias"] = float(sdii_bias)

        # Consecutive dry days
        cdd_true = indices.cdd(true_ds, var_target, dry_threshold=1.0)
        cdd_pred = indices.cdd(pred_ds, var_target, dry_threshold=1.0)
        cdd_bias = (cdd_pred[var_target] - cdd_true[var_target]).mean().values.item()
        diagnostics_dict["cdd_bias"] = float(cdd_bias)

        # Consecutive wet days
        cwd_true = indices.cwd(true_ds, var_target, wet_threshold=1.0)
        cwd_pred = indices.cwd(pred_ds, var_target, wet_threshold=1.0)
        cwd_bias = (cwd_pred[var_target] - cwd_true[var_target]).mean().values.item()
        diagnostics_dict["cwd_bias"] = float(cwd_bias)

        logger.info(f"  Rx1day (Max 1-day Precip) Bias: {rx1day_bias:.4f}")
        logger.info(f"  SDII (Precip Intensity) Bias: {sdii_bias:.4f}")
        logger.info(f"  CDD (Max Dry Spell) Bias: {cdd_bias:.4f}")
        logger.info(f"  CWD (Max Wet Spell) Bias: {cwd_bias:.4f}")

    # Universal indices (applicable to both variables)
    # Lag-1 autocorrelation
    lag1_true = indices.lag1_corr(true_ds, var_target)
    lag1_pred = indices.lag1_corr(pred_ds, var_target)
    lag1_bias = (lag1_pred[var_target] - lag1_true[var_target]).mean().values.item()
    diagnostics_dict["lag1_corr_bias"] = float(lag1_bias)

    # Interannual variability
    interann_true = indices.interannual_var(true_ds, var_target)
    interann_pred = indices.interannual_var(pred_ds, var_target)
    interann_bias = (
        (interann_pred[var_target] - interann_true[var_target]).mean().values.item()
    )
    diagnostics_dict["interannual_var_bias"] = float(interann_bias)

    # Log key diagnostics
    logger.info(f"  RMSE (spatial mean): {diagnostics_dict['rmse']:.4f}")
    logger.info(f"  Bias Mean (spatial mean): {diagnostics_dict['bias_mean']:.4f}")
    logger.info(f"  Bias Q95 (spatial mean): {diagnostics_dict['bias_q95']:.4f}")
    logger.info(f"  Std Ratio (spatial mean): {diagnostics_dict['std_ratio']:.4f}")
    logger.info(f"  Correlation (spatial mean): {diagnostics_dict['correlation']:.4f}")
    logger.info(f"  PSD Distance (log RMSE): {diagnostics_dict['psd_distance']:.4f}")
    logger.info(f"  Lag-1 Autocorr Bias: {lag1_bias:.4f}")
    logger.info(f"  Interannual Var Bias: {interann_bias:.4f}")

    return diagnostics_dict


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
    logger.info(f"  Training batches: {len(dataloader_train)}")
    logger.info(f"  Test batches: {len(test_dataloader)}")
    logger.info(f"  Batches per validation: {cf.training.batches_per_validation}")

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

    # Create learnable noise module if enabled
    learnable_noise_module = None
    if cf.training.get("noise_learnable", False):
        noise_type = cf.training.get("noise_type", "global")  # "global" or "spatial"
        noise_init = cf.training.get(
            "noise_init", cf.training.get("noise_std_gen", 0.05)
        )
        noise_min = cf.training.get("noise_min", 0.0)
        noise_max = cf.training.get("noise_max", 1.0)

        if noise_type == "spatial":
            learnable_noise_module = SpatialNoiseScale(
                height=128,
                width=128,
                init_value=noise_init,
                min_value=noise_min,
                max_value=noise_max,
            ).to(device)
            logger.info(
                f"Using learnable spatial noise (init={noise_init}, range=[{noise_min}, {noise_max}])"
            )
        else:
            learnable_noise_module = LearnableNoiseScale(
                init_value=noise_init, min_value=noise_min, max_value=noise_max
            ).to(device)
            logger.info(
                f"Using learnable global noise (init={noise_init}, range=[{noise_min}, {noise_max}])"
            )
    else:
        logger.info(f"Using fixed noise scale: {cf.training.get('noise_std_gen', 0.0)}")

    # Set conditioning flag based on discriminator architecture
    condition_separate_channels = cf.model.discriminator_architecture != "unet"

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # FSS criterion with variable-specific thresholds
    var_target = cf.data.var_target
    if var_target == "pr":
        # Precipitation thresholds (mm/day)
        fss_thresholds = [0.1, 0.2, 0.4, 0.8, 1.6, 2.4, 4, 6, 10, 25]
    elif var_target in ["tasmax", "tas", "tasmin"]:
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
        cf, generator, discriminator, upsampler, learnable_noise_module
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
                doy_batch = doy_batch.to(device)
            else:
                x_batch, y_batch = batch_data
                doy_batch = None

            x_batch = x_batch.to(device)
            if upsampler is not None:
                x_batch_hr = upsampler(x_batch)
            else:
                x_batch_hr = dataloader.upscale_nn(x_batch)
            # during training, noise channel is added during train step
            y_batch_2d = y_batch.to(device)

            # Use day of year as timestep for diffusion UNET (with sinusoidal encoding)
            if doy_batch is not None:
                timesteps = sinusoidal_encoding_doy(doy_batch, normalize=True)
            else:
                # Fallback to zero timestep if no doy available
                timesteps = torch.zeros([x_batch.shape[0]]).to(device)

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
                            doy_batch = doy_batch.to(device)
                        else:
                            x_batch, y_batch = batch_data
                            doy_batch = None
                    except StopIteration:
                        # If we run out of batches, reset iterator
                        dataloader_train_iter = iter(dataloader_train)
                        batch_data = next(dataloader_train_iter)
                        if len(batch_data) == 3:
                            x_batch, y_batch, doy_batch = batch_data
                            doy_batch = doy_batch.to(device)
                        else:
                            x_batch, y_batch = batch_data
                            doy_batch = None

                    x_batch = x_batch.to(device)
                    y_batch_2d = y_batch.to(device)

                    # Recompute upsampled version for new batch
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    else:
                        x_batch_hr = dataloader.upscale_nn(x_batch)

                    # Use day of year as timestep for diffusion UNET
                    if doy_batch is not None:
                        timesteps = sinusoidal_encoding_doy(doy_batch, normalize=True)
                    else:
                        timesteps = torch.zeros([x_batch.shape[0]]).to(device)

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
                    learnable_noise_module=learnable_noise_module,
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
                learnable_noise_module=learnable_noise_module,
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
                    doy_batch = doy_batch.to(device)
                else:
                    x_batch, y_batch = batch_data
                    doy_batch = None

                x_batch = x_batch.to(device)
                if upsampler is not None:
                    x_batch_hr = upsampler(x_batch)
                else:
                    x_batch_hr = dataloader.upscale_nn(x_batch)
                y_batch = y_batch.to(device)
                y_batch_2d = y_batch.view(-1, 1, 128, 128)

                # Use day of year as timestep for diffusion UNET
                if doy_batch is not None:
                    timesteps = sinusoidal_encoding_doy(doy_batch, normalize=True)
                else:
                    timesteps = torch.zeros([x_batch.shape[0]]).to(device)

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
                    learnable_noise_module=learnable_noise_module,
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
                        doy_batch = doy_batch.to(device)
                    else:
                        x_batch, y_batch = batch_data
                        doy_batch = None

                    x_batch = x_batch.to(device)
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    x_batch = x_batch.to(device)
                    if upsampler is not None:
                        x_batch_hr = upsampler(x_batch)
                    else:
                        x_batch_hr = dataloader.upscale_nn(x_batch)
                    # during training, noise channel is added during train step
                    y_batch_2d = y_batch.to(device).view(-1, 1, 128, 128)

                    # Use day of year as timestep for diffusion UNET
                    if doy_batch is not None:
                        timesteps = sinusoidal_encoding_doy(doy_batch, normalize=True)
                    else:
                        timesteps = torch.zeros([x_batch.shape[0]]).to(device)

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

                    x_batch_hr_with_oro = dataloader.add_noise_channel(
                        x_batch_hr_with_oro
                    )

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
                            learnable_noise_module=learnable_noise_module,
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
                if learnable_noise_module is not None:
                    best_checkpoint_dict["learnable_noise_state_dict"] = (
                        learnable_noise_module.state_dict()
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
                    x_vis_up = dataloader.upscale_nn(x_vis)
                # Concatenate orography if available (before adding noise)
                if cf.data.use_orography:
                    orography_batch_vis = orography.repeat(
                        x_vis.shape[0], 1, 1
                    ).unsqueeze(1)
                    x_vis_up = torch.cat([x_vis_up, orography_batch_vis], dim=1)
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
