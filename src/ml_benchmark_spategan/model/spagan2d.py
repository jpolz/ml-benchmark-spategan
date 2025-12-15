from typing import Optional

import torch
import torch.nn as nn
from torch import amp
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            bias=False,
        )
        # Use GroupNorm instead of InstanceNorm (works with 1x1 spatial)
        num_groups = min(32, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        self.norm1 = nn.GroupNorm(num_groups, out_channels, affine=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels, affine=True)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.adjust_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.adjust_norm = nn.GroupNorm(num_groups, out_channels, affine=True)
        else:
            self.adjust_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.padding_layer:
            x = self.padding_layer(x)

        out = self.conv1(x)
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.relu(out)

        if self.padding_layer:
            out = self.padding_layer(out)

        out = self.conv2(out)
        if self.use_layer_norm:
            out = self.norm2(out)

        if self.adjust_conv:
            residual = self.adjust_conv(residual)
            residual = self.adjust_norm(residual)

        out += residual
        return self.relu(out)


class Interpolate(nn.Module):
    def __init__(self, scale_factor: tuple, mode: str = "bilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )


class Constraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, constraint):
        constraint = constraint[:, :, 5:-5, 8:-8, 8:-8].sum(dim=1, keepdim=True)
        scale = (constraint[:, 0].mean(dim=(1, 2, 3)) / 6).view(-1, 1, 1, 1, 1)
        pred_mean = prediction[:, 0].mean(dim=(1, 2, 3)).view(-1, 1, 1, 1, 1)
        return prediction * (scale / pred_mean)


class Generator(nn.Module):
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
        # self.linout = nn.Linear(128*128, 128*128)

        self.constraint_layer = Constraint()

    def forward(self, x: torch.Tensor, dropout_seed: int = None) -> torch.Tensor:
        if dropout_seed is None:
            dropout_seed = self.dropout_seed

        # 16x16
        x1 = self.res1(x)
        x1 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x1)
        x2_stay = self.res2(x1)
        # x2_stay = x1

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
        # output = torch.flatten(output, start_dim=1)
        # output = self.linout(output)

        # Avoid in-place operation to prevent gradient computation error
        # output = output[:, 0, :, :]
        # output_constrained[:, :, :, 32:-32, 32:-32] = self.constraint_layer(
        #     output[:, :, :, 32:-32, 32:-32], x
        # )

        return output


class Discriminator(nn.Module):
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
            # nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
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


# move to utils
def add_noise_channel(x):
    """
    Adds one Gaussian noise channel to a BCHW tensor.
    x: (batch, C, H, W)
    returns: (batch, C+1, H, W)
    """
    noise = torch.randn(
        x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
    )
    return torch.cat([x, noise], dim=1)


def train_gan_step(
    config,
    input_image,
    input_image_hr,
    target,
    step,
    discriminator,
    generator,
    gen_opt,
    disc_opt,
    scaler,
    criterion,
    fss_criterion,
    timesteps,
    loss_weights={"l1": 1.0, "gan": 1.0},
    condition_separate_channels: bool = False,
):
    """
    Performs a single training step for the GAN.

    Parameters
    ----------
    config : Config
        Configuration object containing model and training parameters.
    input_image : torch.Tensor
        Input tensor to the generator, shape (batch, C, H, W).
    input_image_hr : torch.Tensor
        High-resolution input tensor for conditioning the discriminator, shape (batch, C, H, W).
    target : torch.Tensor
        Ground truth tensor, shape (batch, 1, H, W).
    step : int
        Current training step.
    discriminator : nn.Module
        Discriminator model.
    generator : nn.Module
        Generator model.
    gen_opt : torch.optim.Optimizer or None
        Optimizer for the generator. If None, generator is not updated.
    disc_opt : torch.optim.Optimizer or None
        Optimizer for the discriminator. If None, discriminator is not updated.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    criterion : nn.Module
        Loss function (e.g., BCEWithLogitsLoss).
    fss_criterion:
        FSS loss function, if used for pixel-wise loss.
    timesteps : torch.Tensor
        Timesteps for diffusion models, shape (batch,).
    loss_weights : dict, optional
        Weights for different loss components, by default {'l1': 1.0, 'gan': 1.0}.
    condition_separate_channels : bool, optional
        If True, condition the discriminator with separate channels, by default False.
    """
    generator.train()
    discriminator.train()

    gen_loss = 0.0
    disc_loss = 0.0

    ##################
    ### Generator: ###
    ##################
    if gen_opt is not None:
        gen_opt.zero_grad(set_to_none=True)

        # mixed precission
        with amp.autocast("cuda"):
            ## generate multiple ensemble prediction-
            # Generator outputs flattened predictions, reshape to 2D
            match config.model.generator_architecture:
                case "spategan":
                    gen_outputs = [
                        generator(input_image).view(-1, 1, 128, 128) for _ in range(3)
                    ]
                case "diffusion_unet":
                    gen_outputs = [
                        generator(add_noise_channel(input_image_hr), timesteps).view(
                            -1, 1, 128, 128
                        )
                        for _ in range(3)
                    ]
                case _:
                    raise ValueError(
                        f"Invalid option: {config.model.generator_architecture}"
                    )

            gen_ensemble = torch.cat(gen_outputs, dim=1)
            pred_log = gen_ensemble[:, 0:1]

            # calculate ensemble mean
            gen_ensemble_mean = (
                gen_ensemble[:, 0:1] + gen_ensemble[:, 1:2] + gen_ensemble[:, 2:3]
            ) / 3

            # Classify all fake batch with D
            if condition_separate_channels:
                disc_fake_output = discriminator(pred_log, input_image)
            else:
                disc_fake_output = discriminator(
                    torch.cat((pred_log, input_image_hr), dim=1), timesteps
                )

            # BCE Loss:
            gen_gan_loss = criterion(
                disc_fake_output, torch.ones_like(disc_fake_output)
            )

            # Hinge Loss:
            # gen_gan_loss = -torch.mean(disc_fake_output)
            
            l1loss = nn.L1Loss()(gen_ensemble_mean, target)
            
            if config.training.fss_loss == True:
                fss_loss = fss_criterion(
                    gen_ensemble, target
                )
                loss = (
                    loss_weights["l1"] * l1loss
                    + loss_weights["gan"] * gen_gan_loss
                    + loss_weights["fss"] * fss_loss
                )
                
                # print(f"Step {step}: L1 Loss: {l1loss.item():.4f}, GAN Loss: {gen_gan_loss.item():.4f}, FSS Loss: {fss_loss.item():.4f}")
                
            else:
                loss = loss_weights["l1"] * l1loss + loss_weights["gan"] * gen_gan_loss 
                
                # print(f"Step {step}: L1 Loss: {l1loss.item():.4f}, GAN Loss: {gen_gan_loss.item():.4f}")


        scaler.scale(loss).backward()
        # Gradient Norm Clipping
        # nn.utils.clip_grad_norm_(generator.parameters(), max_norm=2.0, norm_type=2)

        scaler.step(gen_opt)
        scaler.update()
        # Unscale gradients to prevent underflow -
        # scaler.unscale_(gen_opt)

        gen_loss = loss.item()
    else:
        # Generate prediction for discriminator training without updating generator
        with torch.no_grad():
            match config.model.generator_architecture:
                case "spategan":
                    pred_log = generator(input_image).view(-1, 1, 128, 128)
                case "diffusion_unet":
                    pred_log = generator(
                        add_noise_channel(input_image_hr), timesteps
                    ).view(-1, 1, 128, 128)
                case _:
                    raise ValueError(
                        f"Invalid option: {config.model.generator_architecture}"
                    )

    ####################
    ## Discriminator: ##
    ####################
    if disc_opt is not None:
        disc_opt.zero_grad(set_to_none=True)

        # Ensure pred_log is detached for discriminator training
        if gen_opt is not None:
            pred_log = pred_log.detach()

        with amp.autocast("cuda"):
            # discriminator prediction
            disc_real_output = discriminator(target, input_image)

            # label smoothing not needed for hinge loss.
            real_labels = 0.8 + 0.2 * torch.rand_like(disc_real_output)

            # BCE loss real:
            disc_real = criterion(disc_real_output, real_labels)
            # disc_real = criterion(disc_real_output, torch.ones_like(disc_real_output))

            # Classify all fake batch with D
            disc_fake_output = discriminator(pred_log, input_image)

            # Calculate D's loss on the all-fake batch BCE:
            disc_fake = criterion(disc_fake_output, torch.zeros_like(disc_fake_output))

        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        scaler.scale(disc_fake + disc_real).backward()

        # Gradient Norm Clipping
        # nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=2.0, norm_type=2)
        scaler.step(disc_opt)
        scaler.update()
        # Unscale gradients to prevent underflow - dont know if really necessary
        # scaler.unscale_(gen_opt)

        disc_loss = (disc_fake + disc_real).item()

    return gen_loss, disc_loss


if __name__ == "__main__":
    model = Generator().to(device)
    # summary(model, input_size=(1, 15, 16, 16))
