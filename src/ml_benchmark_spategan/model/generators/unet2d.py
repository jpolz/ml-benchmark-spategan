"""UNet2D model wrapper with configurable activation functions."""

import torch
import torch.nn as nn
from diffusers import UNet2DModel

from ml_benchmark_spategan.model.base import BaseModel


class UNetWithActivation(BaseModel):
    """
    Wrapper for UNet2DModel with configurable final activation.

    This wrapper adds a final activation function to the UNet2D model output,
    which is useful for enforcing specific output ranges or distributions.

    Args:
        base_model: UNet2DModel instance
        activation: Activation function to apply to output (e.g., nn.Softplus(), nn.Identity())
    """

    def __init__(self, base_model, activation):
        super().__init__()
        self.model = base_model
        self.activation = activation

    def forward(self, sample, timestep):
        """
        Forward pass through UNet with activation.

        Args:
            sample: Input tensor (B, C, H, W)
            timestep: Timestep tensor for diffusion models

        Returns:
            Output tensor with activation applied
        """
        output = self.model(sample, timestep).sample
        return self.activation(output)

    def train_step(
        self,
        batch,
        optimizers,
        criterion,
        scaler,
        config,
        **kwargs,
    ) -> dict:
        """
        UNet training is handled by the GAN training step.
        This method is not used directly for diffusion_unet architecture.

        Args:
            batch: Tuple of (input, target) tensors
            optimizers: Dict of optimizers
            criterion: Loss function
            scaler: Gradient scaler
            config: Configuration object
            **kwargs: Additional arguments

        Returns:
            Empty dict (training handled by train_gan_step)
        """
        raise NotImplementedError(
            "UNet training is handled by train_gan_step in spagan2d module"
        )

    def predict_step(
        self, x: torch.Tensor, timesteps: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """
        Perform prediction without denormalization.

        Args:
            x: Input tensor
            timesteps: Timestep tensor (defaults to zeros if not provided)
            **kwargs: Unused

        Returns:
            Raw model output
        """
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], device=x.device)

        with torch.no_grad():
            return self(x, timesteps)


def create_unet_generator(unet_cfg, normalization: str = "minus1_to_plus1"):
    """
    Create a UNet2D generator with appropriate activation function.

    Args:
        unet_cfg: Configuration object with UNet parameters
        normalization: Normalization method ('m1p1_log_target' uses Softplus, others use Identity)

    Returns:
        UNetWithActivation instance
    """
    base_generator = UNet2DModel(
        sample_size=tuple(unet_cfg.sample_size),
        in_channels=unet_cfg.in_channels,
        out_channels=unet_cfg.out_channels,
        layers_per_block=unet_cfg.layers_per_block,
        block_out_channels=tuple(unet_cfg.block_out_channels),
        down_block_types=tuple(unet_cfg.down_block_types),
        up_block_types=tuple(unet_cfg.up_block_types),
    )

    # Choose activation based on normalization method
    if normalization == "m1p1_log_target":
        activation = nn.Softplus()
    else:
        activation = nn.Identity()

    return UNetWithActivation(base_generator, activation)
