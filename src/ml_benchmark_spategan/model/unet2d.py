"""UNet2D model wrapper with configurable activation functions."""

import torch.nn as nn
from diffusers import UNet2DModel


class UNetWithActivation(nn.Module):
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
