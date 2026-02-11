"""Registry for initializing models for training."""

import torch
from torchinfo import summary


def create_generator(config, device: torch.device = None):
    """
    Create and initialize a generator model for training.

    Args:
        config: Model configuration object
        device: Device to place model on

    Returns:
        Initialized generator model
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architecture = config.model.get("architecture") or config.model.get(
        "generator_architecture", "spategan"
    )

    if architecture == "spategan":
        from ml_benchmark_spategan.train.model.generators.spategan import Generator

        generator = Generator(config.model).to(device)

        print("Generator architecture:")
        print(summary(generator, input_size=(1, 15, 16, 16), verbose=0))

        return generator

    elif architecture == "diffusion_unet":
        # Check if temporal dimensions are enabled
        t_past = getattr(config.data, "t_past", 0)
        t_future = getattr(config.data, "t_future", 0)
        is_temporal = (t_past > 0) or (t_future > 0)

        if is_temporal:
            raise NotImplementedError("Temporal UNet (3D) architecture is not implemented in diffusion_unet. "
                                      "Use the diffusion_unet_3d model in your config for loading temporal UNet models.")
        else:
            from ml_benchmark_spategan.train.model.generators.unet2d import (
                create_unet_generator,
            )

            unet_cfg = config.model.generator.diffusion_unet
            generator = create_unet_generator(
                unet_cfg, normalization=config.data.normalization
            ).to(device)

            print(
                summary(
                    generator,
                    input_size=[
                        (
                            1,
                            unet_cfg.in_channels,
                            unet_cfg.sample_size[0],
                            unet_cfg.sample_size[1],
                        ),
                        (1,),
                    ],
                    dtypes=[torch.float32, torch.long],
                    verbose=0,
                )
            )

        return generator

    elif architecture == "diffusion_unet_3d":
        # UNet3D with true temporal convolutions (requires t_past or t_future > 0)
        t_past = getattr(config.data, "t_past", 0)
        t_future = getattr(config.data, "t_future", 0)

        if t_past == 0 and t_future == 0:
            raise ValueError(
                "diffusion_unet_3d requires temporal context (t_past > 0 or t_future > 0). "
                "For single timestep use 'diffusion_unet' architecture instead."
            )

        from ml_benchmark_spategan.train.model.generators.unet3d import (
            create_unet3d_temporal_generator,
        )

        unet_cfg = config.model.generator.diffusion_unet

        # Optional multi-step prediction parameters
        n_pred_steps = getattr(config.model.generator, "n_pred_steps", 1)
        n_context_steps = getattr(config.model.generator, "n_context_steps", None)

        generator = create_unet3d_temporal_generator(
            unet_cfg,
            normalization=config.data.normalization,
            t_past=t_past,
            t_future=t_future,
            n_pred_steps=n_pred_steps,
            n_context_steps=n_context_steps,
        ).to(device)

        # Calculate input shape for summary (5D: B, C, T, H, W)
        num_timesteps = t_past + 1 + t_future
        print(
            summary(
                generator,
                input_size=[
                    (
                        1,
                        unet_cfg.in_channels,  # per-timestep channels (with noise)
                        num_timesteps,
                        unet_cfg.sample_size[0],
                        unet_cfg.sample_size[1],
                    ),
                    (1,),
                ],
                dtypes=[torch.float32, torch.long],
                verbose=0,
            )
        )

        return generator

    else:
        raise ValueError(f"Unknown generator architecture: {architecture}")


def create_discriminator(config, device: torch.device = None):
    """
    Create and initialize a discriminator model for training.

    Args:
        config: Model configuration object
        device: Device to place model on

    Returns:
        Initialized discriminator model
    """
    from diffusers import UNet2DModel

    from ml_benchmark_spategan.train.model.discriminators.spategan_disc import (
        Discriminator,
    )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disc_arch = config.model.discriminator_architecture

    if disc_arch == "unet":
        print("Using UNet2DModel as discriminator")
        discriminator = UNet2DModel(
            sample_size=(128, 128),
            in_channels=2,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(32, 64, 128),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        ).to(device)

        print(
            summary(
                discriminator,
                input_size=[(1, 2, 128, 128), (1,)],
                dtypes=[torch.float32, torch.long],
                verbose=0,
            )
        )

        return discriminator

    elif disc_arch == "spategan":
        print("Using spategan Discriminator")
        discriminator = Discriminator(config).to(device)

        print("\nDiscriminator architecture:")
        print(
            summary(
                discriminator,
                input_size=[(1, 1, 128, 128), (1, 15, 16, 16)],
                verbose=0,
            )
        )

        return discriminator

    else:
        raise ValueError(f"Unknown discriminator architecture: {disc_arch}")
