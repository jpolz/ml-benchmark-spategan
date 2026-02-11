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

    Handles both 4D (spatial) and 5D (temporal) inputs:
    - 4D: diffusion_unet with (B, C, H, W)
    - 5D: diffusion_unet_3d with (B, C, T, H, W)

    Args:
        generator: Generator model
        architecture: Model architecture name
        input_image: Low-resolution input (B, C, 16, 16) or (B, C*T, 16, 16)
        input_image_hr: High-resolution input (B, C, 128, 128) or (B, C, T, 128, 128)
        orography: Orography input (B, 1, 128, 128) or (B, 1, T, 128, 128)
        timesteps: Timesteps for diffusion models
        ensemble_size: Number of ensemble members
        noise_std: Standard deviation of noise channel for diffusion models

    Returns:
        Ensemble predictions:
        - 4D case: (B, ensemble_size, 128, 128)
        - 5D case with n_pred_steps=1: (B, ensemble_size, 128, 128)
        - 5D case with n_pred_steps>1: (B, ensemble_size, n_pred_steps, 128, 128)
    """
    batch_size = input_image.shape[0]

    # Detect if we're dealing with temporal (5D) data
    is_temporal = input_image_hr.ndim == 5

    if architecture == "spategan":
        # Pre-allocate output tensor for efficiency
        gen_ensemble = torch.empty(
            batch_size,
            ensemble_size,
            128,
            128,
            device=input_image.device,
            dtype=input_image.dtype,
        )
        for i in range(ensemble_size):
            gen_ensemble[:, i] = generator(input_image).view(-1, 128, 128)

    elif architecture in ["diffusion_unet", "diffusion_unet_3d"]:
        # Concatenate orography if available
        if orography is not None:
            if is_temporal:
                # 5D: (B, C, T, H, W) + (B, 1, T, H, W)
                input_with_oro = torch.cat([input_image_hr, orography], dim=1)
            else:
                # 4D: (B, C, H, W) + (B, 1, H, W)
                input_with_oro = torch.cat([input_image_hr, orography], dim=1)
        else:
            input_with_oro = input_image_hr

        # Generate first sample to determine output shape (handles n_pred_steps)
        input_with_noise = add_noise_channel(input_with_oro, noise_std=noise_std)
        first_output = generator(input_with_noise, timesteps)

        # Determine output shape
        if first_output.ndim == 3:
            # (B, H, W) - single timestep output
            output_shape = (batch_size, ensemble_size, 128, 128)
        elif first_output.ndim == 4:
            if first_output.shape[1] == 1:
                # (B, 1, H, W) - squeeze channel dim
                output_shape = (batch_size, ensemble_size, 128, 128)
                first_output = first_output.squeeze(1)
            else:
                # (B, n_pred_steps, H, W) - multi-step prediction
                n_pred_steps = first_output.shape[1]
                output_shape = (batch_size, ensemble_size, n_pred_steps, 128, 128)
        elif first_output.ndim == 5:
            # (B, C, T, H, W) from diffusion_unet_3d
            if first_output.shape[1] == 1 and first_output.shape[2] == 1:
                # (B, 1, 1, H, W) - squeeze both channel and temporal dims
                output_shape = (batch_size, ensemble_size, 128, 128)
                first_output = first_output.squeeze(1).squeeze(1)
            elif first_output.shape[1] == 1:
                # (B, 1, T, H, W) - squeeze channel dim only
                n_pred_steps = first_output.shape[2]
                output_shape = (batch_size, ensemble_size, n_pred_steps, 128, 128)
                first_output = first_output.squeeze(1)
            else:
                raise ValueError(f"Unexpected 5D output with C={first_output.shape[1]}")
        else:
            raise ValueError(f"Unexpected generator output shape: {first_output.shape}")

        # Pre-allocate output tensor
        gen_ensemble = torch.empty(
            *output_shape,
            device=input_image.device,
            dtype=input_image.dtype,
        )

        # Store first sample
        gen_ensemble[:, 0] = first_output

        # Generate remaining ensemble members with DIFFERENT noise
        for i in range(1, ensemble_size):
            input_with_noise = add_noise_channel(input_with_oro, noise_std=noise_std)
            output = generator(input_with_noise, timesteps)

            # Handle output shape consistently
            if output.ndim == 5:
                # (B, C, T, H, W) from diffusion_unet_3d
                if output.shape[1] == 1 and output.shape[2] == 1:
                    # (B, 1, 1, H, W) - squeeze both dims
                    output = output.squeeze(1).squeeze(1)
                elif output.shape[1] == 1:
                    # (B, 1, T, H, W) - squeeze channel dim
                    output = output.squeeze(1)
            elif (
                output.ndim == 4
                and output.shape[1] == 1
                and output_shape[-2:] == (128, 128)
            ):
                # (B, 1, H, W) - squeeze channel dim
                output = output.squeeze(1)

            gen_ensemble[:, i] = output

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

    # Debug: Print input shapes
    print(f"[DEBUG train_gan_step] input_image shape: {input_image.shape}")
    print(f"[DEBUG train_gan_step] input_image_hr shape: {input_image_hr.shape}")
    print(f"[DEBUG train_gan_step] target shape: {target.shape}")
    print(
        f"[DEBUG train_gan_step] orography shape: {orography.shape if orography is not None else None}"
    )

    # For temporal models (5D input_image_hr), extract center timestep for discriminator
    # Generator uses full 5D, discriminator needs 4D
    # Note: temporal data is (B, T, C, H, W) not (B, C, T, H, W)
    input_image_hr_for_disc = input_image_hr
    if input_image_hr.ndim == 5 and config.data.get("t_past", 0) > 0:
        t_past = config.data.t_past
        print(
            f"[DEBUG] Extracting center timestep (t_past={t_past}) from 5D input_image_hr"
        )
        input_image_hr_for_disc = input_image_hr[
            :, t_past, :, :, :
        ]  # Extract center: (B, T, C, H, W) -> (B, C, H, W)
        print(
            f"[DEBUG] input_image_hr_for_disc shape after extraction: {input_image_hr_for_disc.shape}"
        )
    else:
        print(f"[DEBUG] Using input_image_hr as-is (ndim={input_image_hr.ndim})")

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

            # Extract first ensemble member for discriminator
            # gen_ensemble is either (B, N, H, W) or (B, N, T, H, W)
            pred_log = gen_ensemble[:, 0]  # (B, H, W) or (B, T, H, W)

            # For temporal models, ensure we have 4D output (B, C, H, W) for discriminator
            # If pred_log is (B, T, H, W), extract center timestep to get (B, 1, H, W)
            if pred_log.ndim == 3:
                pred_log = pred_log.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            elif pred_log.ndim == 4 and config.data.get("t_past", 0) > 0:
                # Temporal model: (B, T, H, W) -> extract center timestep
                t_past = config.data.t_past
                pred_log = pred_log[
                    :, t_past : t_past + 1, :, :
                ]  # (B, T, H, W) -> (B, 1, H, W)

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
                    disc_fake_output = discriminator(
                        pred_log_noisy, input_image_hr_for_disc
                    )
                else:
                    disc_fake_output = discriminator(
                        torch.cat((pred_log, input_image_hr_for_disc), dim=1), timesteps
                    )

            # Compute combined loss using loss manager
            loss, loss_components = loss_manager.compute_generator_loss(
                gen_ensemble=gen_ensemble,
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
            elif config.model.architecture in ["diffusion_unet", "diffusion_unet_3d"]:
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

        print("[DEBUG] Before discriminator call:")
        print(f"  target_noisy shape: {target_noisy.shape}")
        print(f"  pred_log_noisy shape: {pred_log_noisy.shape}")
        print(f"  input_image_hr_for_disc shape: {input_image_hr_for_disc.shape}")

        with amp.autocast("cuda"):
            # Get discriminator outputs for real and fake samples
            disc_real_output = discriminator(target_noisy, input_image_hr_for_disc)
            disc_fake_output = discriminator(pred_log_noisy, input_image_hr_for_disc)

            # Compute gradient penalty for regularization
            gp_weight = getattr(config.training, "gradient_penalty_weight", 10.0)
            if gp_weight > 0.0:
                # Compute R1 penalty on real data
                gradient_penalty = compute_gradient_penalty_r1(
                    discriminator=discriminator,
                    real_data=target_noisy,
                    condition=input_image_hr_for_disc,
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

    # For temporal models (5D input_image_hr), extract center timestep for discriminator
    input_image_hr_for_disc = input_image_hr
    if input_image_hr.ndim == 5 and config.data.get("t_past", 0) > 0:
        t_past = config.data.t_past
        input_image_hr_for_disc = input_image_hr[
            :, t_past, :, :, :
        ]  # (B, T, C, H, W) -> (B, C, H, W)

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

        # Extract first ensemble member for discriminator
        # gen_ensemble is either (B, N, H, W) or (B, N, T, H, W)
        pred_log = gen_ensemble[:, 0]  # (B, H, W) or (B, T, H, W)

        # For temporal models, ensure we have 4D output (B, C, H, W) for discriminator
        if pred_log.ndim == 3:
            pred_log = pred_log.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        elif pred_log.ndim == 4 and config.data.get("t_past", 0) > 0:
            # Temporal model: (B, T, H, W) -> extract center timestep
            t_past = config.data.t_past
            pred_log = pred_log[
                :, t_past : t_past + 1, :, :
            ]  # (B, T, H, W) -> (B, 1, H, W)

        # Get discriminator outputs
        disc_fake_output = None
        disc_real_output = None

        if loss_weights.get("gan", 0.0) > 0.0:
            if condition_separate_channels:
                disc_fake_output = discriminator(pred_log, input_image_hr_for_disc)
                disc_real_output = discriminator(target, input_image_hr_for_disc)
            else:
                disc_fake_output = discriminator(
                    torch.cat((pred_log, input_image_hr_for_disc), dim=1), timesteps
                )
                disc_real_output = discriminator(
                    torch.cat((target, input_image_hr_for_disc), dim=1), timesteps
                )

        # Compute generator losses
        gen_total_loss, gen_loss_components = loss_manager.compute_generator_loss(
            gen_ensemble=gen_ensemble,
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
