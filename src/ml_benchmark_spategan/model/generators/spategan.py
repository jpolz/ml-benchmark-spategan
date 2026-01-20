"""Spatial GAN generator architecture."""

from typing import Optional

import torch
import torch.nn as nn

from ml_benchmark_spategan.model.base import BaseModel, BaseWrapper
from ml_benchmark_spategan.model.layers import CustomDropout, ResidualBlock2D

###########################################################################
### SUPPORTING LAYERS
###########################################################################


class Interpolate(nn.Module):
    """Bilinear interpolation layer for upsampling."""

    def __init__(self, scale_factor: tuple):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(x, scale_factor=self.scale_factor, mode="bilinear")
        return x


class Constraint(nn.Module):
    """Identity constraint layer (placeholder for future constraints)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


###########################################################################
### GENERATOR
###########################################################################


class Generator(BaseModel):
    """
    Spatial GAN generator model.

    CNN-based generator with residual blocks that directly generates
    high-resolution output from low-resolution input.

    Args:
        cf: Configuration object with model parameters
    """

    def __init__(self, cf):
        super().__init__()

        self.filter_size = cf.filter_size
        self.n_input_channels = cf.n_input_channels
        self.n_output_channels = cf.n_output_channels
        self.dropout_seed = cf.dropout_seed
        self.dropout_ratio = cf.dropout_ratio
        self._initialize_layers()

    def _initialize_layers(self):
        f = self.filter_size

        self.res1 = ResidualBlock2D(
            self.n_input_channels, f, use_layer_norm=False, padding_type=True
        )
        self.res2 = ResidualBlock2D(f, f, use_layer_norm=False, padding_type=True)
        self.res3 = ResidualBlock2D(f, f, use_layer_norm=True, padding_type=True)

        self.down0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(f, f, kernel_size=(3, 3), stride=(2, 2), padding=0),
            nn.ReLU(inplace=True),
        )

        self.upu1 = Interpolate((2, 2))
        self.res3b = ResidualBlock2D(f, f, padding_type=True)

        self.up0 = Interpolate((2, 2))
        self.res4 = ResidualBlock2D(f, f, padding_type=True)
        self.up1 = Interpolate((2, 2))
        self.res5 = ResidualBlock2D(f, f, padding_type=True)

        self.up2 = Interpolate((1, 1))
        self.res6 = ResidualBlock2D(f, f, padding_type=True)

        self.up3 = Interpolate((3, 3))
        self.res7 = ResidualBlock2D(f, f, padding_type=True)

        self.up4 = Interpolate((2, 2))
        self.res8 = ResidualBlock2D(f, f, padding_type=True)
        self.res9 = ResidualBlock2D(f, f, use_layer_norm=False, padding_type=True)

        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(f, self.n_output_channels, kernel_size=(3, 3), padding=0),
        )

        self.constraint_layer = Constraint()

    def forward(self, x: torch.Tensor, dropout_seed: int = None) -> torch.Tensor:
        if dropout_seed is None:
            dropout_seed = self.dropout_seed

        # 16x16
        x1 = self.res1(x)
        x1 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x1)
        x2_stay = self.res2(x1)

        # 8x8
        x2 = self.down0(x2_stay)
        x2 = self.res3b(x2)
        x2 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x2)
        # 16x16
        x2 = self.upu1(x2)

        x2 = x2_stay + x2
        x2 = self.res3(x2)
        x2 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x2)

        # 32x32
        x2 = self.up0(x2)
        x2 = self.res4(x2)

        # 64x64
        x2 = self.up1(x2)
        x2 = self.res5(x2)
        x2 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x2)

        # 128x128
        x2 = self.up4(x2)
        x2 = self.res8(x2)
        x2 = self.res9(x2)

        output = self.output_conv(x2)

        return output

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
        Generator training is handled by train_gan_step function.
        This method delegates to that function.

        Args:
            batch: Tuple of (input, target) tensors
            optimizers: Dict with 'generator' and 'discriminator' keys
            criterion: Loss function
            scaler: Gradient scaler
            config: Configuration object
            **kwargs: Additional arguments (discriminator, fss_criterion, timesteps, etc.)

        Returns:
            Dict with 'gen_loss' and 'disc_loss' keys
        """
        # GAN training is complex and handled by train_gan_step
        # This method exists for interface compliance
        raise NotImplementedError(
            "Generator training is handled by train_gan_step function"
        )

    def predict_step(
        self, x: torch.Tensor, dropout_seed: int = None, **kwargs
    ) -> torch.Tensor:
        """
        Perform prediction without denormalization.

        Args:
            x: Input tensor
            dropout_seed: Random seed for dropout (optional)
            **kwargs: Unused

        Returns:
            Raw model output
        """
        with torch.no_grad():
            return self(x, dropout_seed)


class SpaGANWrapper(BaseWrapper):
    """
    Inference wrapper for SpaGAN and UNet2D generator models.

    Handles model loading, checkpoint management, normalization parameter loading,
    and prediction with proper preprocessing and denormalization.

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
        checkpoint_epoch: Optional[int] = None,
        device: Optional[torch.device] = None,
        orography: Optional[torch.Tensor] = None,
    ):
        from pathlib import Path

        # Determine checkpoint name based on epoch
        if checkpoint_epoch is not None:
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
        """Load generator architecture and weights."""
        from ml_benchmark_spategan.utils.interpolate import LearnableUpsampler

        # Initialize generator based on architecture
        arch = self.config.model.get("architecture") or self.config.model.get(
            "generator_architecture"
        )

        if arch == "spategan":
            self.model = Generator(self.config.model)

        elif arch == "diffusion_unet":
            from ml_benchmark_spategan.model.generators.unet2d import (
                create_unet_generator,
            )

            # Get UNet config from new structure
            unet_cfg = self.config.model.generator.diffusion_unet
            normalization = self.config.data.get("normalization", "minus1_to_plus1")
            self.model = create_unet_generator(unet_cfg, normalization=normalization)
        else:
            raise ValueError(f"Unknown architecture: {arch}")

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

        # Load upsampler if it exists in checkpoint
        if "upsampler_state_dict" in checkpoint:
            # Recreate the upsampler architecture with correct number of input channels
            n_input_channels = self.config.model.get("n_input_channels", 15)
            self.upsampler = LearnableUpsampler(in_channels=n_input_channels).to(
                self.device
            )
            self.upsampler.load_state_dict(checkpoint["upsampler_state_dict"])
            self.upsampler.eval()

        if self.checkpoint_epoch is None:
            self.checkpoint_epoch = checkpoint.get("epoch", None)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from input with proper preprocessing and denormalization.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Denormalized predictions (B, 1, H_out, W_out)
        """
        from ml_benchmark_spategan.utils.interpolate import (
            add_noise_channel,
            upscale_bilinear,
        )
        from ml_benchmark_spategan.utils.normalize import denormalize_predictions

        x = x.to(self.device)

        with torch.no_grad():
            arch = self.config.model.get("architecture") or self.config.model.get(
                "generator_architecture"
            )

            if arch == "diffusion_unet":
                # Diffusion UNet needs upscaled input with noise channel
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
                        self.orography.repeat(x.shape[0], 1, 1)
                        .unsqueeze(1)
                        .to(self.device)
                    )
                    x_hr = torch.cat([x_hr, orography_batch], dim=1)

                x_with_noise = add_noise_channel(x_hr, noise_std=0.2)

                # Generate with timestep conditioning
                timesteps = torch.zeros(x.shape[0], device=self.device)
                output = self.model(x_with_noise, timesteps)

            elif arch == "spategan":
                # SpatGAN works directly on 16x16 input, no upscaling or noise
                output = self.model(x)

            else:
                raise ValueError(f"Unknown architecture: {arch}")

            # Denormalize predictions
            norm_params = self._build_norm_params()
            output = denormalize_predictions(output, norm_params)

            return output

    def to(self, device: torch.device):
        """Move model and upsampler to specified device."""
        super().to(device)
        if self.upsampler is not None:
            self.upsampler = self.upsampler.to(device)
        if self.orography is not None:
            self.orography = self.orography.to(device)
        return self
