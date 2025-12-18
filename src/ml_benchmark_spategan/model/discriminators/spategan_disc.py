"""Spatial GAN discriminator architecture."""

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    """Residual block for discriminator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layer_norm: bool = True,
        stride: int = 1,
        padding_type: bool = False,
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


class Discriminator(nn.Module):
    """
    Spatial GAN discriminator with dual-path architecture.

    Processes high-resolution (fine) and low-resolution (coarse) inputs separately,
    then combines them for classification.
    """

    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.apply(self._init_weights)
        self.n_coarse_channels = config.model.n_input_channels
        self.n_fine_channels = config.model.n_output_channels

        # Get discriminator config with defaults
        disc_config = getattr(config.model, "discriminator", None)
        if disc_config is None:
            # Default channel progression
            self.hr_channels = [128, 128, 128, 64, 64]
            self.lr_channels = [64, 32]
            self.combined_channels = [64]
            self.output_channels = [64, 1]
        else:
            self.hr_channels = getattr(
                disc_config, "hr_channels", [128, 128, 128, 64, 64]
            )
            self.lr_channels = getattr(disc_config, "lr_channels", [64, 32])
            self.combined_channels = getattr(disc_config, "combined_channels", [64])
            self.output_channels = getattr(disc_config, "output_channels", [64, 1])

        # HIGH RESOLUTION path
        hr_layers = []
        in_ch = self.n_fine_channels
        for i, out_ch in enumerate(self.hr_channels):
            hr_layers.append(
                ResidualBlock2D(
                    in_ch,
                    out_ch,
                    use_layer_norm=(i > 0),  # No norm on first layer
                    stride=(1, 1) if i == 0 else (2, 2),
                )
            )
            in_ch = out_ch

        self.hr_path = nn.ModuleList(hr_layers)

        # LOW RESOLUTION path
        lr_layers = []
        in_ch = self.n_coarse_channels
        for i, out_ch in enumerate(self.lr_channels):
            lr_layers.append(
                ResidualBlock2D(
                    in_ch,
                    out_ch,
                    use_layer_norm=(i > 0),
                    stride=(1, 1) if i == 0 else (2, 2),
                )
            )
            in_ch = out_ch

        self.lr_path = nn.ModuleList(lr_layers)

        # Combined path
        combined_layers = []
        in_ch = self.hr_channels[-1] + self.lr_channels[-1]
        for out_ch in self.combined_channels:
            combined_layers.append(
                ResidualBlock2D(
                    in_ch,
                    out_ch,
                    use_layer_norm=True,
                    stride=(2, 2),
                )
            )
            in_ch = out_ch

        self.combined_path = nn.ModuleList(combined_layers)

        # Output head
        output_layers = []
        in_ch = (
            self.combined_channels[-1]
            if self.combined_channels
            else (self.hr_channels[-1] + self.lr_channels[-1])
        )
        for i, out_ch in enumerate(self.output_channels):
            if i < len(self.output_channels) - 1:
                output_layers.extend(
                    [
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            else:
                output_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            in_ch = out_ch

        self.output_conv = nn.Sequential(*output_layers)

    def _init_weights(self, m):
        if (
            isinstance(m, nn.Linear)
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Conv3d)
        ):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        """
        Forward pass through discriminator.

        Args:
            x: High-resolution (fine) input, shape (batch, n_fine_channels, H, W)
            y: Low-resolution (coarse) input, shape (batch, n_coarse_channels, h, w)

        Returns:
            Discriminator output (logits), shape (batch, 1, H', W')
        """
        # Add small random (0-0.05) gaussian noise to discriminator input
        sigma_x = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.05
        sigma_y = torch.rand(y.size(0), 1, 1, 1, device=y.device) * 0.05

        x = x + torch.randn_like(x) * sigma_x
        y = y + torch.randn_like(y) * sigma_y

        # High resolution path
        hr_out = x
        for layer in self.hr_path:
            hr_out = layer(hr_out)

        # Low resolution path
        lr_out = y
        for layer in self.lr_path:
            lr_out = layer(lr_out)

        # Concatenate
        combined = torch.cat((hr_out, lr_out), dim=1)

        # Combined path
        for layer in self.combined_path:
            combined = layer(combined)

        # Output
        out = self.output_conv(combined)

        return out
