"""Module for loading and wrapping different model types."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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
        self.y_min_log = None
        self.y_max_log = None
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
            from ml_benchmark_spategan.model.unet2d import create_unet_generator

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
        """Predict using GAN generator."""
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
