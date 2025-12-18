"""DeepESD baseline model architecture and inference wrapper."""

import torch
import torch.nn as nn


class DeepESD(nn.Module):
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
            out_features=self.y_shape[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1)
        return self.out(x)


class DeepESDWrapper:
    """
    Inference wrapper for DeepESD model.

    Handles model loading, device management, and prediction.

    Args:
        model_path: Path to saved model checkpoint
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
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")

        # Initialize and load model
        self.model = DeepESD(x_shape, y_shape, filters_last_conv)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=False)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions from input.

        Args:
            x: Input tensor

        Returns:
            Predictions tensor
        """
        x = x.to(self.device)
        with torch.no_grad():
            return self.model(x)

    def to(self, device: torch.device):
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
