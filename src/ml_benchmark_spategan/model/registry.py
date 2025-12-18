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
        from ml_benchmark_spategan.model.generators.spategan import Generator

        generator = Generator(config.model).to(device)

        print("Generator architecture:")
        print(summary(generator, input_size=(1, 15, 16, 16), verbose=0))

        return generator

    elif architecture == "diffusion_unet":
        from ml_benchmark_spategan.model.generators.unet2d import create_unet_generator

        unet_cfg = config.model.generator.diffusion_unet
        generator = create_unet_generator(
            unet_cfg, normalization=config.data.normalization
        ).to(device)

        print(
            summary(
                generator,
                input_size=[(1, 16, 128, 128), (1,)],
                dtypes=[torch.float32, torch.long],
                verbose=0,
            )
        )

        return generator

    elif architecture == "deepesd":
        from ml_benchmark_spategan.model.generators.deepesd import DeepESD

        deepesd_cfg = config.model.generator.deepesd
        generator = DeepESD(
            x_shape=deepesd_cfg.x_shape,
            y_shape=deepesd_cfg.y_shape,
            filters_last_conv=deepesd_cfg.filters_last_conv,
        ).to(device)

        print("Generator architecture:")
        print(summary(generator, input_size=(1, 15, 16, 16), verbose=0))

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

    from ml_benchmark_spategan.model.discriminators.spategan_disc import Discriminator

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
