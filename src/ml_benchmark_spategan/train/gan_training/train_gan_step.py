"""GAN training step function."""

import torch
import torch.nn as nn
from torch import amp

from ml_benchmark_spategan.train.gan_training.losses import GANLossManager
from ml_benchmark_spategan.utils.interpolate import add_noise_channel


def compute_gradient_penalty_r1(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    """
    Compute R1 gradient penalty (regularization on real data gradients).

    This prevents discriminator from becoming too confident and encourages
    smoother decision boundaries.

    Args:
        discriminator: Discriminator model
        real_data: Real samples (B, C, H, W)
        condition: Conditioning input (B, C, h, w)

    Returns:
        Gradient penalty scalar
    """
    real_data.requires_grad_(True)

    disc_real = discriminator(real_data, condition)

    # Compute gradients w.r.t. real data
    gradients = torch.autograd.grad(
        outputs=disc_real.sum(),
        inputs=real_data,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # R1 penalty: ||∇D(x)||²
    penalty = gradients.pow(2).reshape(gradients.shape[0], -1).sum(1).mean()

    return penalty


def _generate_ensemble(
    generator: nn.Module,
    architecture: str,
    input_image: torch.Tensor,
    input_image_hr: torch.Tensor,
    orography: torch.Tensor,
    timesteps: torch.Tensor,
    ensemble_size: int,
    noise_std: float = 0.2,
) -> torch.Tensor:
    """
    Generate ensemble predictions efficiently.

    Args:
        generator: Generator model
        architecture: Model architecture name
        input_image: Low-resolution input (B, C, 16, 16)
        input_image_hr: High-resolution input (B, C, 128, 128)
        orography: Orography input (B, 1, 128, 128)
        timesteps: Timesteps for diffusion models
        ensemble_size: Number of ensemble members
        noise_std: Standard deviation of noise channel for diffusion models

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
        # Concatenate orography if available
        if orography is not None:
            input_with_oro = torch.cat([input_image_hr, orography], dim=1)
        else:
            input_with_oro = input_image_hr
        # Add DIFFERENT noise for each ensemble member
        for i in range(ensemble_size):
            input_with_noise = add_noise_channel(input_with_oro, noise_std=noise_std)

            gen_ensemble[:, i] = generator(input_with_noise, timesteps).view(
                -1, 128, 128
            )
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    return gen_ensemble


def train_gan_step(
    config,
    input_image,
    input_image_hr,
    orography,
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
    orography : torch.Tensor
        Orography input tensor, shape (batch, 1, H, W).
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
                orography=orography,
                timesteps=timesteps,
                ensemble_size=config.training.ensemble_size,
                noise_std=config.training.get("noise_std_gen", 0.0),
            )

            # Add channel dimension for consistency (B, N, H, W) -> (B, N, 1, H, W)
            gen_ensemble = gen_ensemble.unsqueeze(2)
            pred_log = gen_ensemble[:, 0]  # First ensemble member for discriminator

            # Get discriminator output if using GAN loss
            disc_fake_output = None
            if loss_weights.get("gan", 0.0) > 0.0:
                # Apply consistent noise (same std as discriminator training)
                noise_std = config.training.get("noise_std", 0.0)
                if noise_std > 0.0:
                    noise_fake = torch.randn_like(pred_log) * noise_std
                    pred_log_noisy = pred_log + noise_fake
                else:
                    pred_log_noisy = pred_log
                if condition_separate_channels:
                    disc_fake_output = discriminator(pred_log_noisy, input_image)
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
                # Concatenate orography if available
                if orography is not None:
                    input_with_oro = torch.cat([input_image_hr, orography], dim=1)
                else:
                    input_with_oro = input_image_hr
                pred_log = generator(
                    add_noise_channel(
                        input_with_oro,
                        noise_std=config.training.get("noise_std_gen", 0.0),
                    ),
                    timesteps,
                ).view(-1, 1, 128, 128)
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

        # Apply consistent noise to discriminator inputs
        # This helps stabilize training and prevents mode collapse
        noise_std = config.training.get("noise_std", 0.0)
        if noise_std > 0.0:
            # Use fixed noise std (not random like before)
            noise_real = torch.randn_like(target) * noise_std
            noise_fake = torch.randn_like(pred_log) * noise_std
            target_noisy = target + noise_real
            pred_log_noisy = pred_log + noise_fake
        else:
            target_noisy = target
            pred_log_noisy = pred_log

        with amp.autocast("cuda"):
            # Get discriminator outputs for real and fake samples
            disc_real_output = discriminator(target_noisy, input_image)
            disc_fake_output = discriminator(pred_log_noisy, input_image)

            # Compute gradient penalty for regularization
            gp_weight = getattr(config.training, "gradient_penalty_weight", 10.0)
            if gp_weight > 0.0:
                # Compute R1 penalty on real data
                gradient_penalty = compute_gradient_penalty_r1(
                    discriminator=discriminator,
                    real_data=target_noisy,
                    condition=input_image,
                )
            else:
                gradient_penalty = None

            # Compute discriminator loss using loss manager
            loss, loss_components = loss_manager.compute_discriminator_loss(
                disc_real_output=disc_real_output,
                disc_fake_output=disc_fake_output,
                use_label_smoothing=True,
                gradient_penalty=gradient_penalty,
                gp_weight=gp_weight,
            )

        scaler.scale(loss).backward()
        scaler.step(disc_opt)
        scaler.update()

        disc_loss = loss.item()

    return gen_loss, disc_loss


def test_gan_step(
    config,
    input_image,
    input_image_hr,
    orography,
    target,
    discriminator,
    generator,
    criterion,
    fss_criterion,
    timesteps,
    loss_weights={"l1": 1.0, "gan": 1.0},
    condition_separate_channels: bool = False,
):
    """
    Performs a single evaluation step for the GAN, computing all loss components.

    Parameters
    ----------
    config : Config
        Configuration object containing model and training parameters.
    input_image : torch.Tensor
        Input tensor to the generator, shape (batch, C, H, W).
    input_image_hr : torch.Tensor
        High-resolution input tensor for conditioning the discriminator, shape (batch, C, H, W).
    orography : torch.Tensor
        Orography input tensor, shape (batch, 1, H, W).
    target : torch.Tensor
        Ground truth tensor, shape (batch, 1, H, W).
    discriminator : nn.Module
        Discriminator model.
    generator : nn.Module
        Generator model.
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
    dict
        Dictionary containing all individual loss components:
        - 'gen_total': Total generator loss
        - 'disc_total': Total discriminator loss
        - 'l1': L1 loss (if computed)
        - 'mse': MSE loss (if computed)
        - 'gan': GAN loss (if computed)
        - 'fss': FSS loss (if computed)
        - 'disc_real': Discriminator loss on real samples
        - 'disc_fake': Discriminator loss on fake samples
    """
    generator.eval()
    discriminator.eval()

    # Initialize loss manager
    loss_manager = GANLossManager(
        loss_weights=loss_weights,
        gan_criterion=criterion,
        fss_criterion=fss_criterion,
        use_fss=config.training.fss_loss,
    )

    loss_dict = {}

    with torch.no_grad():
        # Generate ensemble predictions efficiently
        gen_ensemble = _generate_ensemble(
            generator=generator,
            architecture=config.model.architecture,
            input_image=input_image,
            input_image_hr=input_image_hr,
            orography=orography,
            timesteps=timesteps,
            ensemble_size=config.training.ensemble_size,
            noise_std=config.training.get("noise_std_gen", 0.0),
        )

        # Add channel dimension for consistency (B, N, H, W) -> (B, N, 1, H, W)
        gen_ensemble = gen_ensemble.unsqueeze(2)
        pred_log = gen_ensemble[:, 0]  # First ensemble member for discriminator

        # Get discriminator outputs
        disc_fake_output = None
        disc_real_output = None

        if loss_weights.get("gan", 0.0) > 0.0:
            if condition_separate_channels:
                disc_fake_output = discriminator(pred_log, input_image)
                disc_real_output = discriminator(target, input_image)
            else:
                disc_fake_output = discriminator(
                    torch.cat((pred_log, input_image_hr), dim=1), timesteps
                )
                disc_real_output = discriminator(
                    torch.cat((target, input_image_hr), dim=1), timesteps
                )

        # Compute generator losses
        gen_total_loss, gen_loss_components = loss_manager.compute_generator_loss(
            gen_ensemble=gen_ensemble.squeeze(2),  # Remove channel dim for loss
            target=target,
            disc_fake_output=disc_fake_output,
        )

        loss_dict["gen_total"] = gen_total_loss.item()
        loss_dict.update(gen_loss_components)

        # Compute discriminator losses if GAN loss is used
        if loss_weights.get("gan", 0.0) > 0.0 and disc_real_output is not None:
            disc_total_loss, disc_loss_components = (
                loss_manager.compute_discriminator_loss(
                    disc_real_output=disc_real_output,
                    disc_fake_output=disc_fake_output,
                    use_label_smoothing=False,  # No label smoothing during evaluation
                )
            )

            loss_dict["disc_total"] = disc_total_loss.item()
            loss_dict.update(disc_loss_components)
        else:
            loss_dict["disc_total"] = 0.0

    return loss_dict
