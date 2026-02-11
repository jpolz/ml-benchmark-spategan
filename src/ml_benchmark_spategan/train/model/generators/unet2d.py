"""UNet2D model wrapper with configurable activation functions."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from diffusers import UNet2DModel

from ml_benchmark_spategan.train.model.base import BaseModel, BaseWrapper


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


###########################################################################
### UNET INFERENCE WRAPPER
###########################################################################


class UNetWrapper(BaseWrapper):
    """
    Inference wrapper for UNet2D (diffusion_unet architecture) models.

    Handles model loading, checkpoint management, normalization, upsampling,
    and prediction with proper preprocessing and denormalization.

    This wrapper is specifically for UNet-based generators and handles:
    - Bilinear or learnable upsampling from 16x16 to 128x128
    - Noise channel addition for diffusion conditioning
    - Optional orography concatenation
    - Timestep conditioning for diffusion models

    Args:
        run_dir: Directory containing the trained model
        config: Model configuration object
        checkpoint_epoch: Specific epoch to load (None for final model)
        device: Device to run model on
        orography: Optional orography tensor for conditioning
    """

    def __init__(
        self,
        run_dir: str,
        config,
        checkpoint_epoch: Optional[int | str] = None,
        device: Optional[torch.device] = None,
        orography: Optional[torch.Tensor] = None,
    ):
        # Determine checkpoint name based on epoch
        if checkpoint_epoch == "best":
            checkpoint_name = "checkpoints/best_model.pt"
        elif checkpoint_epoch is not None:
            checkpoint_name = f"checkpoints/checkpoint_epoch_{checkpoint_epoch}.pt"
        else:
            checkpoint_name = "checkpoints/final_models.pt"

        # Initialize base class
        super().__init__(
            run_dir=Path(run_dir),
            checkpoint_name=checkpoint_name,
            device=device,
        )

        self.config = config
        self.orography = orography
        self.upsampler = None
        self.checkpoint_epoch = checkpoint_epoch

        # Load model and weights
        self._load_model()
        self._load_normalization()

    def _load_model(self):
        """Load UNet generator architecture and weights."""

        # Check if this is temporal (3D) or standard (2D) UNet
        t_past = getattr(self.config.data, "t_past", 0)
        t_future = getattr(self.config.data, "t_future", 0)
        is_temporal = (t_past > 0) or (t_future > 0)

        if is_temporal:
            raise NotImplementedError("Temporal UNet (3D) architecture is not implemented in this wrapper.")
        else:
            # Standard 2D UNet
            unet_cfg = self.config.model.generator.diffusion_unet
            normalization = self.config.data.get("normalization", "minus1_to_plus1")
            self.model = create_unet_generator(unet_cfg, normalization=normalization)

        # Load checkpoint
        checkpoint_path = self.run_dir / self.checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["generator_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if self.checkpoint_epoch is None:
            self.checkpoint_epoch = checkpoint.get("epoch", None)

    def predict(self, x: torch.Tensor, doy: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from input with UNet-specific preprocessing.

        Performs the following steps:
        1. Upscale input from 16x16 to 128x128 (learnable or bilinear)
        2. Concatenate orography if configured
        3. Add noise channel for diffusion conditioning
        4. Generate with timestep=0
        5. Denormalize output

        Args:
            x: Input tensor (B, C, 16, 16)
            doy: Day of year tensor (B,)
        Returns:
            Denormalized predictions (B, 1, 128, 128)
        """
        from ml_benchmark_spategan.utils.interpolate import (
            add_noise_channel,
            upscale_bilinear,
        )
        from ml_benchmark_spategan.train.normalize import denormalize_predictions

        x = x.to(self.device)

        with torch.no_grad():
            # Upscale - use learnable upsampler if available
            if self.upsampler is not None:
                x_hr = self.upsampler(x)
            else:
                x_hr = upscale_bilinear(x, target_size=(128, 128))

            # Concatenate orography if available and configured
            if (
                self.config.data.get("use_orography", False)
                and self.orography is not None
            ):
                # Repeat orography for each sample in batch
                orography_batch = (
                    self.orography.repeat(x.shape[0], 1, 1).unsqueeze(1).to(self.device)
                )
                x_hr = torch.cat([x_hr, orography_batch], dim=1)

            # Add noise channel for diffusion conditioning
            x_with_noise = add_noise_channel(x_hr, noise_std=self.config.training.noise_std_gen)

            # Generate with timestep conditioning (timestep=0 for inference)
            # timesteps = torch.zeros(x.shape[0], device=self.device)
            output = self.model(x_with_noise, doy)

            # Denormalize predictions
            norm_params = self._build_norm_params()
            output = denormalize_predictions(output, norm_params)

            return output

    def to(self, device: torch.device):
        """Move model, upsampler, and orography to specified device."""
        super().to(device)
        if self.upsampler is not None:
            self.upsampler = self.upsampler.to(device)
        if self.orography is not None:
            self.orography = self.orography.to(device)
        return self
