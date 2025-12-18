"""GAN training step function."""

import torch
import torch.nn as nn
from torch import amp

from ml_benchmark_spategan.utils.interpolate import add_noise_channel


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

    gen_loss = 0.0
    disc_loss = 0.0

    ##################
    ### Generator: ###
    ##################
    if gen_opt is not None:
        gen_opt.zero_grad(set_to_none=True)

        # mixed precission
        with amp.autocast("cuda"):
            ## generate multiple ensemble prediction-
            # Generator outputs flattened predictions, reshape to 2D
            match config.model.architecture:
                case "spategan":
                    gen_outputs = [
                        generator(input_image).view(-1, 1, 128, 128)
                        for _ in range(config.training.ensemble_size)
                    ]
                case "diffusion_unet":
                    gen_outputs = [
                        generator(add_noise_channel(input_image_hr), timesteps).view(
                            -1, 1, 128, 128
                        )
                        for _ in range(config.training.ensemble_size)
                    ]
                case "deepesd":
                    gen_outputs = [
                        generator(input_image).view(-1, 1, 128, 128)
                        for _ in range(config.training.ensemble_size)
                    ]
                case _:
                    raise ValueError(f"Invalid option: {config.model.architecture}")

            gen_ensemble = torch.cat(gen_outputs, dim=1)
            pred_log = gen_ensemble[:, 0:1]

            # calculate ensemble mean
            gen_ensemble_mean = torch.mean(gen_ensemble, dim=1, keepdim=True)

            if loss_weights["gan"] > 0.0:
                # Classify all fake batch with D
                if condition_separate_channels:
                    disc_fake_output = discriminator(pred_log, input_image)
                else:
                    disc_fake_output = discriminator(
                        torch.cat((pred_log, input_image_hr), dim=1), timesteps
                    )

                # BCE Loss:
                gen_gan_loss = criterion(
                    disc_fake_output, torch.ones_like(disc_fake_output)
                )

                # Hinge Loss:
                # gen_gan_loss = -torch.mean(disc_fake_output)
            else:
                gen_gan_loss = 0.0

            l1loss = nn.L1Loss()(gen_ensemble_mean, target)

            if config.training.fss_loss:
                fss_loss = fss_criterion(gen_ensemble, target)
                loss = (
                    loss_weights["l1"] * l1loss
                    + loss_weights["gan"] * gen_gan_loss
                    + loss_weights["fss"] * fss_loss
                )

            else:
                loss = loss_weights["l1"] * l1loss + loss_weights["gan"] * gen_gan_loss

        scaler.scale(loss).backward()
        scaler.step(gen_opt)
        scaler.update()

        gen_loss = loss.item()
    else:
        # Generate prediction for discriminator training without updating generator
        with torch.no_grad():
            match config.model.architecture:
                case "spategan":
                    pred_log = generator(input_image).view(-1, 1, 128, 128)
                case "diffusion_unet":
                    pred_log = generator(
                        add_noise_channel(input_image_hr), timesteps
                    ).view(-1, 1, 128, 128)
                case "deepesd":
                    pred_log = generator(input_image).view(-1, 1, 128, 128)
                case _:
                    raise ValueError(f"Invalid option: {config.model.architecture}")

    ####################
    ## Discriminator: ##
    ####################
    if disc_opt is not None:
        disc_opt.zero_grad(set_to_none=True)

        # Ensure pred_log is detached for discriminator training
        if gen_opt is not None:
            pred_log = pred_log.detach()

        with amp.autocast("cuda"):
            # discriminator prediction
            disc_real_output = discriminator(target, input_image)

            # label smoothing
            real_labels = 0.8 + 0.2 * torch.rand_like(disc_real_output)

            # BCE loss real:
            disc_real = criterion(disc_real_output, real_labels)

            # Classify all fake batch with D
            disc_fake_output = discriminator(pred_log, input_image)

            # Calculate D's loss on the all-fake batch BCE:
            disc_fake = criterion(disc_fake_output, torch.zeros_like(disc_fake_output))

        # Calculate the gradients for this batch
        scaler.scale(disc_fake + disc_real).backward()
        scaler.step(disc_opt)
        scaler.update()

        disc_loss = (disc_fake + disc_real).item()

    return gen_loss, disc_loss
