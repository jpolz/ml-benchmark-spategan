"""Common neural network layers used across models."""

from typing import Optional

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    """
    2D Residual block with optional normalization, padding, and dropout.

    Used in both generator and discriminator architectures for SpatialGAN.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        use_layer_norm: Whether to use GroupNorm
        stride: Stride for first convolution
        padding_type: If True, use ReflectionPad2d; otherwise use standard padding
        dropout: Dropout probability (0.0 = no dropout)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layer_norm: bool = True,
        stride: int = 1,
        padding_type: Optional[bool] = None,
        dropout: float = 0.0,
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

        # Adaptive group count for GroupNorm to handle small channel counts
        # Use min(num_groups, out_channels) to ensure divisibility
        if use_layer_norm:
            num_groups = min(32, out_channels) if out_channels >= 32 else out_channels
            # Ensure num_groups divides out_channels
            while out_channels % num_groups != 0:
                num_groups //= 2

            self.layer_norm1 = nn.GroupNorm(
                num_channels=out_channels, num_groups=num_groups
            )
            self.layer_norm2 = nn.GroupNorm(
                num_channels=out_channels, num_groups=num_groups
            )
        else:
            self.layer_norm1 = None
            self.layer_norm2 = None

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor (B, C_out, H', W')
        """
        residual = x

        if self.padding_layer:
            out = self.padding_layer(x)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        if self.layer_norm1 is not None:
            out = self.layer_norm1(out)

        out = self.activation(out)

        if self.dropout is not None:
            out = self.dropout(out)

        if self.padding_layer:
            out = self.padding_layer(out)
            out = self.conv2(out)
        else:
            out = self.conv2(out)

        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = out + residual
        out = self.activation(out)

        return out


class CustomDropout(nn.Module):
    """
    Custom dropout with fixed seed for reproducibility.

    Used in SpatialGAN generator for deterministic dropout patterns.

    Args:
        p: Dropout probability
        d_seed: Random seed for reproducibility
    """

    def __init__(self, p: float, d_seed: int):
        super().__init__()
        self.p = p
        torch.manual_seed(d_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout with fixed mask pattern.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Output tensor with dropout applied
        """
        device = x.device
        batch, channels, height, width = x.shape

        mask_shape = (batch, channels, height, width)
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * (1 - self.p))
        mask = mask.repeat(1, 1, 1, 1) / (1 - self.p)

        return x * mask
