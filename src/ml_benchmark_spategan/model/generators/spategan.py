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
