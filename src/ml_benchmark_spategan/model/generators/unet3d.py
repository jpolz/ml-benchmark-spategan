"""UNet3D model wrapper for handling temporal dimensions."""

import torch
import torch.nn as nn
from diffusers import UNet2DModel

from ml_benchmark_spategan.model.base import BaseModel


class UNet3DWithActivation(BaseModel):
    """
    Wrapper for UNet2DModel that handles temporal dimensions.

    When temporal dimensions (t_past or t_future > 0) are present, this model
    treats time as additional channels. The input shape changes from (B, C, H, W)
    to (B, C*T, H, W) where T = t_past + t_future + 1.

    This wrapper adds a final activation function to the UNet model output,
    which is useful for enforcing specific output ranges or distributions.

    Args:
        base_model: UNet2DModel instance
        activation: Activation function to apply to output (e.g., nn.Softplus(), nn.Identity())
        is_temporal: Whether the model handles temporal dimensions (t_past or t_future > 0)
    """

    def __init__(self, base_model, activation, is_temporal=False):
        super().__init__()
        self.model = base_model
        self.activation = activation
        self.is_temporal = is_temporal

    def forward(self, sample, timestep):
        """
        Forward pass through UNet with activation.

        Args:
            sample: Input tensor
                - For temporal: (B, C*T, H, W) where T is number of time steps
                - For non-temporal: (B, C, H, W)
            timestep: Timestep tensor for diffusion models

        Returns:
            Output tensor with activation applied
        """
        # Process through base UNet model (handles temporal as channels)
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
            x: Input tensor (with temporal dimension as channels if applicable)
            timesteps: Timestep tensor (defaults to zeros if not provided)
            **kwargs: Unused

        Returns:
            Raw model output
        """
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], device=x.device)

        with torch.no_grad():
            return self(x, timesteps)


def create_unet3d_generator(
    unet_cfg,
    normalization: str = "minus1_to_plus1",
    t_past: int = 0,
    t_future: int = 0,
    base_channels: int = 15,
):
    """
    Create a UNet generator with support for temporal dimensions.

    When t_past or t_future > 0, temporal frames are stacked as channels.
    The input channels are automatically adjusted: in_channels = base_channels * (t_past + t_future + 1)

    Args:
        unet_cfg: Configuration object with UNet parameters
        normalization: Normalization method ('m1p1_log_target' uses Softplus, others use Identity)
        t_past: Number of past time steps to include
        t_future: Number of future time steps to include
        base_channels: Base number of input channels per time step (default: 15)

    Returns:
        UNet3DWithActivation instance
    """
    # Determine if temporal dimension is active
    is_temporal = (t_past > 0) or (t_future > 0)

    # Calculate total input channels for temporal stacking
    # Total time steps = past + current + future
    num_time_steps = t_past + t_future + 1

    # For temporal models, we need to account for the noise channel too
    # The config in_channels includes 1 noise channel, so we adjust
    if is_temporal:
        # Assuming config specifies base channels + 1 noise
        # We multiply base channels by time steps and add noise
        total_in_channels = (unet_cfg.in_channels - 1) * num_time_steps + 1
    else:
        total_in_channels = unet_cfg.in_channels

    print(
        f"Creating UNet generator: temporal={is_temporal}, t_past={t_past}, "
        f"t_future={t_future}, num_time_steps={num_time_steps}"
    )
    print(
        f"Input channels: {total_in_channels} (base={unet_cfg.in_channels}, "
        f"per_timestep={unet_cfg.in_channels - 1})"
    )

    base_generator = UNet2DModel(
        sample_size=tuple(unet_cfg.sample_size),
        in_channels=total_in_channels,
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

    return UNet3DWithActivation(base_generator, activation, is_temporal=is_temporal)
