"""DeepESD baseline model architecture and inference wrapper."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import amp

from ml_benchmark_spategan.model.base import BaseModel, BaseWrapper


class DeepESD(BaseModel):
    """
    DeepESD baseline model for statistical downscaling.

    Simple CNN architecture with 3 convolutional layers and a final linear layer
    that flattens spatial dimensions to predict all output grid points.

    Args:
        x_shape: Shape of input tensor (batch, channels, height, width)
        y_shape: Shape of output tensor (batch, n_gridpoints)
        filters_last_conv: Number of filters in the final convolutional layer
    """

    def __init__(self, x_shape: tuple, y_shape: tuple, filters_last_conv: int = 1):
        super(DeepESD, self).__init__()
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv
        
        # Backward compatibility: handle both old (batch, height*width) and new (batch, height, width) formats
        if len(y_shape) == 2:
            # Old format: (batch, height*width) - assume square output
            import math
            n_gridpoints = y_shape[1]
            side = int(math.sqrt(n_gridpoints))
            if side * side != n_gridpoints:
                raise ValueError(f"Cannot infer spatial dimensions from flattened shape {y_shape}")
            self.output_height = side
            self.output_width = side
            self.output_features = n_gridpoints
        else:
            # New format: (batch, height, width)
            self.output_height = y_shape[1]
            self.output_width = y_shape[2]
            self.output_features = y_shape[1] * y_shape[2]

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
            in_features=self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
            out_features=self.output_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        # reshape from (batch, n_gridpoints) to (batch, height, width)
        x = x.view(-1, self.output_height, self.output_width)
        return x

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
        Perform a single training step for DeepESD.

        Args:
            batch: Tuple of (input, target) tensors
            optimizers: Dict with 'model' optimizer
            criterion: Loss function (e.g., MSELoss)
            scaler: Gradient scaler for mixed precision
            config: Configuration object
            **kwargs: Unused

        Returns:
            Dict with 'loss' key
        """
        x, y = batch
        optimizer = optimizers["model"]

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast("cuda"):
            y_pred = self(x)
            loss = criterion(y_pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return {"loss": loss.item()}

    def predict_step(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform prediction without denormalization.

        Args:
            x: Input tensor
            **kwargs: Unused

        Returns:
            Raw model output
        """
        with torch.no_grad():
            return self(x)


class DeepESDWrapper(BaseWrapper):
    """
    Inference wrapper for DeepESD model.

    Handles model loading, device management, normalization, and prediction.

    Args:
        model_path: Path to saved model checkpoint (or run directory)
        x_shape: Shape of input tensor (batch, channels, height, width)
        y_shape: Shape of output tensor (batch, n_gridpoints)
        filters_last_conv: Number of filters in final conv layer
        device: Device to run model on (CPU or CUDA)
    """

    def __init__(
        self,
        model_path: str,
        x_shape: tuple,
        y_shape: tuple,
        filters_last_conv: int = 1,
        device: Optional[torch.device] = None,
    ):
        # If model_path is a .pth or .pt file, extract the directory
        model_path = Path(model_path)
        if model_path.suffix in [".pth", ".pt"]:
            run_dir = model_path.parent
            checkpoint_name = model_path.name
        else:
            run_dir = model_path
            checkpoint_name = "final_model.pth"

        super().__init__(run_dir, checkpoint_name, device)

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv

        self._load_model()

    def _load_model(self):
        """Initialize and load DeepESD model."""
        self.model = DeepESD(self.x_shape, self.y_shape, self.filters_last_conv)
        checkpoint_path = self.run_dir / self.checkpoint_name
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from input.

        Args:
            x: Input tensor

        Returns:
            Predictions tensor (DeepESD doesn't use normalization, so returns raw output)
        """
        x = x.to(self.device)
        return self.model.predict_step(x)
