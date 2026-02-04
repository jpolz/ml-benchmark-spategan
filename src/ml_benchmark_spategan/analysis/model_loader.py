"""Simple registry for loading different model types for inference."""

import torch


def load_model(model_type: str, **kwargs):
    """
    Factory function to load models for inference.

    Args:
        model_type: Type of model ('deepesd', 'gan')
        **kwargs: Model-specific arguments:
            - For 'gan': run_dir, config, checkpoint_epoch, device, orography

    Returns:
        Model wrapper instance with predict() method

    Note:
        For 'gan' model_type, the appropriate wrapper (SpaGANWrapper or UNetWrapper)
        is automatically selected based on the config architecture.

    Example:
        >>> # Load GAN model (auto-detects SpaGAN vs UNet)
        >>> model = load_model('gan',
        ...                    run_dir='runs/20251218_0211_zjh10zws',
        ...                    config=config_obj)
    """
    device = kwargs.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if model_type.lower() == "gan":
        config = kwargs["config"]

        # Determine architecture to select appropriate wrapper
        arch = config.model.get("architecture") or config.model.get(
            "generator_architecture", "spategan"
        )

        if arch == "spategan":
            from ml_benchmark_spategan.train.model.generators.spategan import (
                SpaGANWrapper,
            )

            return SpaGANWrapper(
                run_dir=kwargs["run_dir"],
                config=config,
                checkpoint_epoch=kwargs.get("checkpoint_epoch", None),
                device=device,
            )

        elif arch == "diffusion_unet":
            from ml_benchmark_spategan.train.model.generators.unet2d import UNetWrapper

            return UNetWrapper(
                run_dir=kwargs["run_dir"],
                config=config,
                checkpoint_epoch=kwargs.get("checkpoint_epoch", None),
                device=device,
                orography=kwargs.get("orography", None),
            )

        else:
            raise ValueError(
                f"Unknown GAN architecture: {arch}. "
                f"Supported: 'spategan', 'diffusion_unet'"
            )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported types: 'gan'")
