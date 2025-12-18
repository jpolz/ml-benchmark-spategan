"""Spatial GAN generator architecture."""

from typing import Optional

import torch
import torch.nn as nn

from ml_benchmark_spategan.model.base import BaseModel


class CustomDropout(nn.Module):
    def __init__(self, p: float, d_seed: int):
        super().__init__()
        self.p = p
        torch.manual_seed(d_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch, channels, height, width = x.shape

        mask_shape = (batch, channels, height, width)
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * (1 - self.p))
        mask = mask.repeat(1, 1, 1, 1) / (1 - self.p)

        return x * mask


class ResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layer_norm: bool = True,
        stride: int = 1,
        padding_type: Optional[bool] = None,
    ):
        super().__init__()

        padding = 0 if padding_type else 1
        self.use_layer_norm = use_layer_norm
        self.padding_type = padding_type

        self.padding_layer = nn.ReflectionPad2d((1, 1, 1, 1)) if padding_type else None

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding
        )

        # Shortcut connection with 1x1 convolution if input/output channels differ
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=stride
            )
        else:
            self.shortcut = None

        self.layer_norm1 = (
            nn.GroupNorm(num_channels=out_channels, num_groups=32)
            if use_layer_norm
            else None
        )
        self.layer_norm2 = (
            nn.GroupNorm(num_channels=out_channels, num_groups=32)
            if use_layer_norm
            else None
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.padding_layer:
            out = self.padding_layer(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = self.activation(out)

        if self.padding_layer:
            out = self.padding_layer(out)
            out = self.conv2(out)
        else:
            out = self.conv2(out)

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.activation(out)

        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor: tuple):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interp(x, scale_factor=self.scale_factor, mode="bilinear")
        return x


class Constraint(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Generator(BaseModel):
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


class SpaGANWrapper:
    """
    Inference wrapper for SpaGAN and UNet2D generator models.

    Handles model loading, checkpoint management, normalization parameter loading,
    and prediction with proper preprocessing and denormalization.

    Args:
        run_dir: Directory containing the trained model
        config: Model configuration object
        checkpoint_epoch: Specific epoch to load (None for final model)
        device: Device to run model on
    """

    def __init__(
        self,
        run_dir: str,
        config,
        checkpoint_epoch: int = None,
        device: torch.device = None,
    ):
        from pathlib import Path

        from ml_benchmark_spategan.utils.interpolate import LearnableUpsampler

        self.device = device or torch.device("cpu")
        self.run_dir = Path(run_dir)
        self.config = config

        # Initialize generator based on architecture
        self._load_generator()

        # Load checkpoint - either specific epoch or final model
        if checkpoint_epoch is not None:
            checkpoint_path = (
                self.run_dir / "checkpoints" / f"checkpoint_epoch_{checkpoint_epoch}.pt"
            )
        else:
            checkpoint_path = self.run_dir / "checkpoints" / "final_models.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["generator_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load upsampler if it exists in checkpoint
        self.upsampler = None
        if "upsampler_state_dict" in checkpoint:
            # Recreate the upsampler architecture with correct number of input channels
            n_input_channels = self.config.model.get("n_input_channels", 15)
            self.upsampler = LearnableUpsampler(in_channels=n_input_channels).to(
                self.device
            )
            self.upsampler.load_state_dict(checkpoint["upsampler_state_dict"])
            self.upsampler.eval()

        self.checkpoint_epoch = checkpoint.get("epoch", checkpoint_epoch)

        # Load normalization parameters if they exist
        self.y_min = None
        self.y_max = None
        self.y_min_log = None
        self.y_max_log = None
        self._load_normalization()

    def _load_generator(self):
        """Load generator architecture."""
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

    def _load_normalization(self):
        """Load normalization parameters."""
        import xarray as xr

        ymin_path = self.run_dir / "y_min.nc"
        ymax_path = self.run_dir / "y_max.nc"
        ymin_log_path = self.run_dir / "y_min_log.nc"
        ymax_log_path = self.run_dir / "y_max_log.nc"

        if ymin_path.exists() and ymax_path.exists():
            self.y_min = xr.open_dataarray(ymin_path)
            self.y_max = xr.open_dataarray(ymax_path)

        if ymin_log_path.exists() and ymax_log_path.exists():
            self.y_min_log = xr.open_dataarray(ymin_log_path)
            self.y_max_log = xr.open_dataarray(ymax_log_path)

    def _build_norm_params(self) -> dict:
        """Build norm_params dict for denormalization."""
        norm_params = {
            "normalization": self.config.data.get("normalization", "minus1_to_plus1")
        }

        if self.y_min is not None:
            norm_params["y_min"] = self.y_min
        if self.y_max is not None:
            norm_params["y_max"] = self.y_max
        if self.y_min_log is not None:
            norm_params["y_min_log"] = self.y_min_log
        if self.y_max_log is not None:
            norm_params["y_max_log"] = self.y_max_log

        return norm_params

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from input.

        Args:
            x: Input tensor

        Returns:
            Denormalized predictions
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
                x_with_noise = add_noise_channel(x_hr, noise_std=0.2)

                # Generate
                timesteps = torch.zeros(x.shape[0], device=self.device)
                output = self.model(x_with_noise, timesteps)

            elif arch == "spategan":
                # SpatGAN works directly on 16x16 input, no upscaling or noise
                output = self.model(x)

            else:
                raise ValueError(f"Unknown architecture: {arch}")

            # Denormalize if needed
            norm_params = self._build_norm_params()
            output = denormalize_predictions(output, norm_params)

            return output

    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        if self.upsampler is not None:
            self.upsampler = self.upsampler.to(device)
        return self
