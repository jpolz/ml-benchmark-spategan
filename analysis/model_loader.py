"""Module for loading and wrapping different model types."""

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

# Add src to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ml_benchmark_spategan.utils.interpolate import LearnableUpsampler


class ModelWrapper:
    """Base wrapper class for all models."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions from input."""
        raise NotImplementedError

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self


class DeepESDWrapper(ModelWrapper):
    """Wrapper for DeepESD baseline model."""

    def __init__(
        self,
        model_path: str,
        x_shape: tuple,
        y_shape: tuple,
        filters_last_conv: int = 1,
        device: torch.device = None,
    ):
        super().__init__(device or torch.device("cpu"))

        # Define DeepESD architecture
        class DeepESD(nn.Module):
            def __init__(self, x_shape: tuple, y_shape: tuple, filters_last_conv: int):
                super(DeepESD, self).__init__()
                self.x_shape = x_shape
                self.y_shape = y_shape
                self.filters_last_conv = filters_last_conv

                self.conv_1 = nn.Conv2d(
                    in_channels=self.x_shape[1],
                    out_channels=50,
                    kernel_size=3,
                    padding=1,
                )
                self.conv_2 = nn.Conv2d(
                    in_channels=50, out_channels=25, kernel_size=3, padding=1
                )
                self.conv_3 = nn.Conv2d(
                    in_channels=25,
                    out_channels=self.filters_last_conv,
                    kernel_size=3,
                    padding=1,
                )
                self.out = nn.Linear(
                    in_features=self.x_shape[2]
                    * self.x_shape[3]
                    * self.filters_last_conv,
                    out_features=self.y_shape[1],
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.relu(self.conv_1(x))
                x = torch.relu(self.conv_2(x))
                x = torch.relu(self.conv_3(x))
                x = torch.flatten(x, start_dim=1)
                return self.out(x)

        # Initialize and load model
        self.model = DeepESD(x_shape, y_shape, filters_last_conv)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using DeepESD model."""
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)


class GANWrapper(ModelWrapper):
    """Wrapper for GAN generator models."""

    def __init__(
        self,
        run_dir: str,
        config: Any,
        checkpoint_epoch: int = None,
        device: torch.device = None,
    ):
        super().__init__(device or torch.device("cpu"))

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
        self._load_normalization()

    def _load_generator(self):
        """Load generator architecture."""
        arch = self.config.model.get("architecture") or self.config.model.get(
            "generator_architecture"
        )

        if arch == "spategan":
            from ml_benchmark_spategan.model.spagan2d import Generator

            self.model = Generator(self.config.model)

        elif arch == "diffusion_unet":
            # Match the training script's UNetWithActivation wrapper
            import torch.nn as nn

            class UNetWithActivation(nn.Module):
                def __init__(self, base_model, activation):
                    super().__init__()
                    self.model = base_model
                    self.activation = activation

                def forward(self, sample, timestep):
                    output = self.model(sample, timestep).sample
                    return self.activation(output)

            # Get UNet config from new structure
            unet_cfg = self.config.model.generator.diffusion_unet
            base_generator = UNet2DModel(
                sample_size=tuple(unet_cfg.sample_size),
                in_channels=unet_cfg.in_channels,
                out_channels=unet_cfg.out_channels,
                layers_per_block=unet_cfg.layers_per_block,
                block_out_channels=tuple(unet_cfg.block_out_channels),
                down_block_types=tuple(unet_cfg.down_block_types),
                up_block_types=tuple(unet_cfg.up_block_types),
            )

            # Use same activation logic as training script
            normalization = self.config.data.get("normalization", "minus1_to_plus1")
            if normalization == "m1p1_log_target":
                self.model = UNetWithActivation(base_generator, nn.Softplus())
            else:
                self.model = UNetWithActivation(base_generator, nn.Identity())
        else:
            raise ValueError(f"Unknown architecture: {arch}")

    def _load_normalization(self):
        """Load normalization parameters."""
        import xarray as xr

        ymin_path = self.run_dir / "y_min.nc"
        ymax_path = self.run_dir / "y_max.nc"

        if ymin_path.exists() and ymax_path.exists():
            self.y_min = xr.open_dataarray(ymin_path)
            self.y_max = xr.open_dataarray(ymax_path)

    def _upscale_nn(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale input from 16x16 to 128x128."""
        return F.interpolate(x, size=(128, 128), mode="bilinear")

    def _add_noise_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise channel to input."""
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return torch.cat([x, noise], dim=1)

    def _denormalize(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions from normalized space back to original scale."""
        if self.y_min is None or self.y_max is None:
            return y_pred

        y_min = torch.tensor(self.y_min.values, device=y_pred.device)
        y_max = torch.tensor(self.y_max.values, device=y_pred.device)

        if y_pred.dim() == 4:  # (batch, 1, H, W)
            y_min = y_min.unsqueeze(0).unsqueeze(0)
            y_max = y_max.unsqueeze(0).unsqueeze(0)

        # Check normalization method from config
        normalization = self.config.data.get("normalization", "minus1_to_plus1")

        if normalization == "mp1p1_input_m1p1log_target":
            # For mp1p1_input_m1p1log_target:
            # Forward: log1p(y + 1e-6) - log1p(1e-6), then min-max to [-1, 1]
            # Reverse:
            # 1. Reverse min-max: y_log = ((y_norm + 1) / 2) * (y_max - y_min) + y_min
            y_log = ((y_pred + 1) / 2) * (y_max - y_min) + y_min
            # 2. Reverse log: y = expm1(y_log + log1p(1e-6)) - 1e-6
            y_denorm = (
                torch.expm1(
                    y_log
                    + torch.tensor(
                        float(torch.log1p(torch.tensor(1e-6))), device=y_pred.device
                    )
                )
                - 1e-6
            )
            # Clamp to avoid negative values for precipitation
            y_denorm = torch.clamp(y_denorm, min=0.0)
        elif normalization == "m1p1_log_target":
            # For m1p1_log_target:
            # Model outputs softplus activation (already positive, log-space)
            # Reverse log transform: expm1
            y_denorm = torch.expm1(y_pred) - 1e-6
            # Clamp to avoid negative values
            y_denorm = torch.clamp(y_denorm, min=0.0)
        else:
            # For minus1_to_plus1:
            # Reverse normalization: y_norm = (y - y_min) / (y_max - y_min) * 2 - 1
            # So: y = ((y_norm + 1) / 2) * (y_max - y_min) + y_min
            y_denorm = ((y_pred + 1) / 2) * (y_max - y_min) + y_min

        return y_denorm

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using GAN generator."""
        x = x.to(self.device)

        with torch.no_grad():
            # Upscale - use learnable upsampler if available
            if self.upsampler is not None:
                x_hr = self.upsampler(x)
            else:
                x_hr = self._upscale_nn(x)
            x_with_noise = self._add_noise_channel(x_hr)

            # Generate
            timesteps = torch.zeros(x.shape[0], device=self.device)

            arch = self.config.model.get("architecture") or self.config.model.get(
                "generator_architecture"
            )
            if arch == "diffusion_unet":
                # UNetWithActivation wrapper returns the output directly
                output = self.model(x_with_noise, timesteps)
            else:
                output = self.model(x_with_noise)

            # Denormalize if needed
            output = self._denormalize(output)

            return output


def load_model(model_type: str, **kwargs) -> ModelWrapper:
    """
    Factory function to load models.

    Args:
        model_type: Type of model ('deepesd', 'gan')
        **kwargs: Model-specific arguments

    Returns:
        ModelWrapper instance
    """
    device = kwargs.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if model_type.lower() == "deepesd":
        return DeepESDWrapper(
            model_path=kwargs["model_path"],
            x_shape=kwargs["x_shape"],
            y_shape=kwargs["y_shape"],
            filters_last_conv=kwargs.get("filters_last_conv", 1),
            device=device,
        )
    elif model_type.lower() == "gan":
        return GANWrapper(
            run_dir=kwargs["run_dir"],
            config=kwargs["config"],
            checkpoint_epoch=kwargs.get("checkpoint_epoch", None),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
