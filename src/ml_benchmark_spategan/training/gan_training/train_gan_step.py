"""GAN training step function."""

import torch
import torch.nn as nn
from torch import amp

from ml_benchmark_spategan.training.gan_training.losses import GANLossManager
from ml_benchmark_spategan.utils.interpolate import add_noise_channel


def _generate_ensemble(
    generator: nn.Module,
    architecture: str,
    input_image: torch.Tensor,
    input_image_hr: torch.Tensor,
    timesteps: torch.Tensor,
    ensemble_size: int,
) -> torch.Tensor:
    """
    Generate ensemble predictions efficiently.

    Args:
        generator: Generator model
        architecture: Model architecture name
        input_image: Low-resolution input (B, C, 16, 16)
        input_image_hr: High-resolution input (B, C, 128, 128)
        timesteps: Timesteps for diffusion models
        ensemble_size: Number of ensemble members

    Returns:
        Ensemble predictions (B, ensemble_size, 128, 128)
    """
    batch_size = input_image.shape[0]

    # Pre-allocate output tensor for efficiency
    gen_ensemble = torch.empty(
        batch_size,
        ensemble_size,
        128,
        128,
        device=input_image.device,
        dtype=input_image.dtype,
    )

    if architecture == "spategan":
        for i in range(ensemble_size):
            gen_ensemble[:, i] = generator(input_image).view(-1, 128, 128)
    elif architecture == "diffusion_unet":
        # Pre-compute noise channel addition outside loop if possible
        input_with_noise = add_noise_channel(input_image_hr)
        for i in range(ensemble_size):
            gen_ensemble[:, i] = generator(input_with_noise, timesteps).view(
                -1, 128, 128
            )
    elif architecture == "deepesd":
        for i in range(ensemble_size):
            gen_ensemble[:, i] = generator(input_image).view(-1, 128, 128)
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    return gen_ensemble


def train_gan_step(
    config,
    input_image,
    input_image_hr,
    target,
    step,
    discriminator,
    generator,
    gen_opt,
    disc_opt,
    scaler,
    criterion,
    fss_criterion,
    timesteps,
    loss_weights={"l1": 1.0, "gan": 1.0},
    condition_separate_channels: bool = False,
):
    """
    Performs a single training step for the GAN.

    Parameters
    ----------
    config : Config
        Configuration object containing model and training parameters.
    input_image : torch.Tensor
        Input tensor to the generator, shape (batch, C, H, W).
    input_image_hr : torch.Tensor
        High-resolution input tensor for conditioning the discriminator, shape (batch, C, H, W).
    target : torch.Tensor
        Ground truth tensor, shape (batch, 1, H, W).
    step : int
        Current training step.
    discriminator : nn.Module
        Discriminator model.
    generator : nn.Module
        Generator model.
    gen_opt : torch.optim.Optimizer or None
        Optimizer for the generator. If None, generator is not updated.
    disc_opt : torch.optim.Optimizer or None
        Optimizer for the discriminator. If None, discriminator is not updated.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    criterion : nn.Module
        Loss function (e.g., BCEWithLogitsLoss).
    fss_criterion:
        FSS loss function, if used for pixel-wise loss.
    timesteps : torch.Tensor
        Timesteps for diffusion models, shape (batch,).
    loss_weights : dict, optional
        Weights for different loss components, by default {'l1': 1.0, 'gan': 1.0}.
    condition_separate_channels : bool, optional
        If True, condition the discriminator with separate channels, by default False.

    Returns
    -------
    tuple[float, float]
        Generator loss and discriminator loss
    """
    generator.train()
    discriminator.train()

    # Initialize loss manager
    loss_manager = GANLossManager(
        loss_weights=loss_weights,
        gan_criterion=criterion,
        fss_criterion=fss_criterion,
        use_fss=config.training.fss_loss,
    )

    gen_loss = 0.0
    disc_loss = 0.0
    pred_log = None

    ##################
    ### Generator: ###
    ##################
    if gen_opt is not None:
        gen_opt.zero_grad(set_to_none=True)

        with amp.autocast("cuda"):
            # Generate ensemble predictions efficiently
            gen_ensemble = _generate_ensemble(
                generator=generator,
                architecture=config.model.architecture,
                input_image=input_image,
                input_image_hr=input_image_hr,
                timesteps=timesteps,
                ensemble_size=config.training.ensemble_size,
            )

            # Add channel dimension for consistency (B, N, H, W) -> (B, N, 1, H, W)
            gen_ensemble = gen_ensemble.unsqueeze(2)
            pred_log = gen_ensemble[:, 0]  # First ensemble member for discriminator

            # Get discriminator output if using GAN loss
            disc_fake_output = None
            if loss_weights.get("gan", 0.0) > 0.0:
                if condition_separate_channels:
                    disc_fake_output = discriminator(pred_log, input_image)
                else:
                    disc_fake_output = discriminator(
                        torch.cat((pred_log, input_image_hr), dim=1), timesteps
                    )

            # Compute combined loss using loss manager
            loss, loss_components = loss_manager.compute_generator_loss(
                gen_ensemble=gen_ensemble.squeeze(2),  # Remove channel dim for loss
                target=target,
                disc_fake_output=disc_fake_output,
            )

        scaler.scale(loss).backward()
        scaler.step(gen_opt)
        scaler.update()

        gen_loss = loss.item()
    else:
        # Generate prediction for discriminator training without updating generator
        with torch.no_grad():
            if config.model.architecture == "spategan":
                pred_log = generator(input_image).view(-1, 1, 128, 128)
            elif config.model.architecture == "diffusion_unet":
                pred_log = generator(add_noise_channel(input_image_hr), timesteps).view(
                    -1, 1, 128, 128
                )
            elif config.model.architecture == "deepesd":
                pred_log = generator(input_image).view(-1, 1, 128, 128)
            else:
                raise ValueError(f"Invalid architecture: {config.model.architecture}")

    ####################
    ## Discriminator: ##
    ####################
    if disc_opt is not None:
        disc_opt.zero_grad(set_to_none=True)

        # Ensure pred_log is detached for discriminator training
        if gen_opt is not None:
            pred_log = pred_log.detach()

        with amp.autocast("cuda"):
            # Get discriminator outputs for real and fake samples
            disc_real_output = discriminator(target, input_image)
            disc_fake_output = discriminator(pred_log, input_image)

            # Compute discriminator loss using loss manager
            loss, loss_components = loss_manager.compute_discriminator_loss(
                disc_real_output=disc_real_output,
                disc_fake_output=disc_fake_output,
                use_label_smoothing=True,
            )

        scaler.scale(loss).backward()
        scaler.step(disc_opt)
        scaler.update()

        disc_loss = loss.item()

    return gen_loss, disc_loss
