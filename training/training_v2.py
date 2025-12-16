############
# Imports
############


import logging
import os
import pathlib

# Import diagnostics
import sys

import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DModel
from IPython.display import clear_output
from torchinfo import summary
from tqdm import tqdm

from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.dataloader import dataloader
from ml_benchmark_spategan.model.spagan2d import (
    Discriminator,
    Generator,
    train_gan_step,
)
from ml_benchmark_spategan.utils.denormalize import predictions_to_xarray
from ml_benchmark_spategan.utils.losses import FSSLoss
from ml_benchmark_spategan.visualization.plot_train import (
    plot_adversarial_losses,
    plot_diagnostic_history,
    plot_predictions,
)

sys.path.append("evaluation")
import diagnostics

# find project base directory
project_base = pathlib.Path(os.getcwd())
# load configuration
cf = config.set_up_run(project_base)

# Set up logging to file in run directory
log_file = os.path.join(cf.logging.run_dir, "training.log")

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # Keep console output as well
    ],
    force=True,
)
logger = logging.getLogger(__name__)

dataloader_train, test_dataloader, cf, norm_params = dataloader.build_dataloaders(cf)
# dataloader_train, test_dataloader = dataloader.build_dummy_dataloaders()
# update cf in run directory
cf.save()

# Save normalization parameters if they exist
if "y_min" in norm_params and "y_max" in norm_params:
    norm_params["y_min"].to_netcdf(f"{cf.logging.run_dir}/y_min.nc")
    norm_params["y_max"].to_netcdf(f"{cf.logging.run_dir}/y_max.nc")
    logger.info(
        f"Target normalization parameters (y_min, y_max) saved to {cf.logging.run_dir}"
    )

# Save x normalization parameters (mean/std or min/max)
if "x_mean" in norm_params and "x_std" in norm_params:
    norm_params["x_mean"].to_netcdf(f"{cf.logging.run_dir}/x_mean.nc")
    norm_params["x_std"].to_netcdf(f"{cf.logging.run_dir}/x_std.nc")
    logger.info(
        f"Input normalization parameters (x_mean, x_std) saved to {cf.logging.run_dir}"
    )
elif "x_min" in norm_params and "x_max" in norm_params:
    norm_params["x_min"].to_netcdf(f"{cf.logging.run_dir}/x_min.nc")
    norm_params["x_max"].to_netcdf(f"{cf.logging.run_dir}/x_max.nc")
    logger.info(
        f"Input normalization parameters (x_min, x_max) saved to {cf.logging.run_dir}"
    )
# describe shapes of data
logger.info("Training data shapes:")
x_shape, y_shape = dataloader_train.dataset._get_shapes()
logger.info(f"  x: {x_shape}")
logger.info(f"  y: {y_shape}")

##################
# Model setup
##################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

match cf.model.generator_architecture:
    case "spategan":
        logger.info("Using spategan architecture")
        # Initialize models
        generator = Generator(cf.model).to(device)

        # Print model summaries
        logger.info("Generator architecture:")
        summary(generator, input_size=(1, 15, 16, 16))
    case "diffusion_unet":
        logger.info("Using Diffusion UNet architecture")

        # Wrapper class for adding final activation to UNet2DModel
        class UNetWithActivation(nn.Module):
            def __init__(self, base_model, activation):
                super().__init__()
                self.model = base_model
                self.activation = activation

            def forward(self, sample, timestep):
                output = self.model(sample, timestep).sample
                return self.activation(output)

        base_generator = UNet2DModel(
            sample_size=(128, 128),
            in_channels=15 + 1,  # +1 == noise
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512, 1024),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        # if predictions are not [-1, 1], change final activation
        if cf.data.normalization == "m1p1_log_target":
            generator = UNetWithActivation(base_generator, nn.Softplus()).to(device)
        else:  # use linear activation by default
            generator = UNetWithActivation(base_generator, nn.Identity()).to(device)

        logger.info(
            str(
                summary(
                    generator,
                    input_size=[(1, 16, 128, 128), (1,)],
                    dtypes=[torch.float32, torch.long],
                )
            )
        )

    case _:
        raise ValueError(f"Invalid option: {cf.model.generator_architecture}")

if cf.model.discriminator_architecture == "unet":
    logger.info("Using UNet2DModel as discriminator")
    discriminator = UNet2DModel(
        sample_size=(128, 128),
        in_channels=2,  # +1 == noise
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    ).to(device)
    condition_separate_channels = True
    logger.info(
        str(
            summary(
                discriminator,
                input_size=[(1, 2, 128, 128), (1,)],
                dtypes=[torch.float32, torch.long],
            )
        )
    )
elif cf.model.discriminator_architecture == "spategan":
    logger.info("Using spategan Discriminator")
    discriminator = Discriminator(cf).to(device)
    logger.info("\nDiscriminator architecture:")
    # Note: Discriminator takes (high_res_target, low_res_input)
    logger.info(
        str(summary(discriminator, input_size=[(1, 1, 128, 128), (1, 15, 16, 16)]))
    )
    condition_separate_channels = True
else:
    raise ValueError(f"Invalid option: {cf.model.discriminator_architecture}")

# Move models to device
generator = generator.to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# FSS criterion
fss_criterion = FSSLoss(
    thresholds=[
        0.1,
        0.2,
        0.4,
        0.8,
        1.6,
        2.4,
        4,
        6,
        10,
        25,
    ],  # not normalized thresholds
    scales=[2, 8, 16],
    device="cuda",
    sharpness=3.0,
    batch_size=10,
    config=cf,
    norm_params=norm_params,
)


# Optimizers
gen_opt = torch.optim.AdamW(
    generator.parameters(),
    lr=cf.training.generator.learning_rate,
    betas=(
        cf.training.generator.beta1,
        cf.training.generator.beta2,
    ),  # todo: try with momentum
    weight_decay=cf.training.generator.weight_decay,
)

disc_opt = torch.optim.AdamW(
    discriminator.parameters(),
    lr=cf.training.discriminator.learning_rate,
    betas=(
        cf.training.discriminator.beta1,
        cf.training.discriminator.beta2,
    ),  # not using momentum on the disc since it can lead to instability
    weight_decay=cf.training.discriminator.weight_decay,
)

# For mixed precision training
scaler = torch.amp.GradScaler("cuda")


##################
# Training loop
##################

# GAN Training loop
loss_gen_train = []
loss_disc_train = []
loss_gen_test = []
loss_fss_test = []

# Store diagnostics
diagnostic_history = {"rmse": [], "bias_mean": [], "epochs": []}

logger.info(f"Starting GAN training for {cf.training.epochs} epochs...")

# Get a fixed batch for visualization
val_iter = iter(test_dataloader)
x_vis, y_vis = next(val_iter)
x_vis, y_vis = x_vis.to(device), y_vis.to(device)
if cf.model.generator_architecture == "diffusion_unet":
    x_vis = dataloader.upscale_nn(x_vis)
    x_vis = dataloader.add_noise_channel(x_vis)  # add noise to HR or LR?


for epoch in range(cf.training.epochs):
    # Training phase
    epoch_gen_losses = []
    epoch_disc_losses = []

    for batch_idx, (x_batch, y_batch) in tqdm(
        enumerate(dataloader_train), total=len(dataloader_train)
    ):
        x_batch = x_batch.to(device)
        x_batch_hr = dataloader.upscale_nn(
            x_batch
        )  # move to where it is needed, if needed
        # during training, noise channel is added during train step
        y_batch_2d = y_batch.to(device)

        # zero timestep, for diffusion UNET.
        timesteps = torch.zeros([x_batch.shape[0]]).to(
            device
        )  # only for diffusion unet

        # Train discriminator n_critic times
        n_critic = getattr(cf.training, "n_critic", 1)
        disc_losses_batch = []

        for _ in range(n_critic):
            # Train discriminator only
            _, disc_loss = train_gan_step(
                config=cf,
                input_image=x_batch,
                input_image_hr=x_batch_hr,
                target=y_batch_2d,
                step=epoch * len(dataloader_train) + batch_idx,
                discriminator=discriminator,
                generator=generator,
                gen_opt=None,  # Don't update generator
                disc_opt=disc_opt,
                scaler=scaler,
                criterion=criterion,
                timesteps=timesteps,
                loss_weights=cf.training.loss_weights,
                condition_separate_channels=condition_separate_channels,
                fss_criterion=fss_criterion,
            )
            disc_losses_batch.append(disc_loss)

        # Train generator once
        gen_loss, _ = train_gan_step(
            config=cf,
            input_image=x_batch,
            input_image_hr=x_batch_hr,
            target=y_batch_2d,
            step=epoch * len(dataloader_train) + batch_idx,
            discriminator=discriminator,
            generator=generator,
            gen_opt=gen_opt,
            disc_opt=None,  # Don't update discriminator
            scaler=scaler,
            criterion=criterion,
            timesteps=timesteps,
            loss_weights=cf.training.loss_weights,
            condition_separate_channels=condition_separate_channels,
            fss_criterion=fss_criterion,
        )

        epoch_gen_losses.append(gen_loss)
        epoch_disc_losses.append(np.mean(disc_losses_batch))

    # Calculate average training losses
    train_gen_loss = np.mean(epoch_gen_losses)
    train_disc_loss = np.mean(epoch_disc_losses)
    loss_gen_train.append(train_gen_loss)
    loss_disc_train.append(train_disc_loss)

    # Validation phase
    generator.eval()
    test_losses = []
    fss_test_losses = []

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_dataloader):
            if batch_idx >= cf.training.batches_per_validation:
                break

            x_batch = x_batch.to(device)
            x_batch_hr = dataloader.upscale_nn(x_batch)
            x_batch_hr = dataloader.add_noise_channel(
                x_batch_hr
            )  # add noise to HR or LR?
            y_batch_2d = y_batch.to(device)

            # zero timestep, for diffusion UNET.
            timesteps = torch.zeros([x_batch.shape[0]]).to(device)

            with torch.amp.autocast("cuda"):
                match cf.model.generator_architecture:
                    case "spategan":
                        y_pred = generator(x_batch)
                    case "diffusion_unet":
                        y_pred = generator(x_batch_hr, timesteps)

                loss = nn.L1Loss()(y_pred, y_batch_2d)

                fss_test_loss = fss_criterion(y_pred, y_batch_2d)

            test_losses.append(loss.item())
            fss_test_losses.append(fss_test_loss.item())

    test_loss = np.mean(test_losses)
    loss_gen_test.append(test_loss)
    mean_fss_test = np.mean(fss_test_losses)
    loss_fss_test.append(mean_fss_test)

    # Print progress and plot
    if (epoch + 1) % cf.logging.log_frequency == 0 or epoch == 0:
        clear_output(wait=True)
        logger.info(f"Epoch {epoch + 1}/{cf.training.epochs}")
        logger.info(f"  Generator Loss:     {train_gen_loss:.6f}")
        logger.info(f"  Discriminator Loss: {train_disc_loss:.6f}")
        logger.info(f"  Test Loss (L1):     {test_loss:.6f}")
        logger.info(f"  Test Loss (FSS):     {mean_fss_test:.6f}")

        # Plot losses
        plot_adversarial_losses(
            loss_gen_train, loss_disc_train, loss_gen_test, loss_fss_test, cf
        )

    # Compute diagnostics
    if (epoch + 1) % cf.logging.diagnostic_frequency == 0:
        logger.info("  Computing diagnostics...")
        generator.eval()

        # Collect all test predictions
        all_preds = []
        all_targets = []

        # to do change y to 2D
        with torch.no_grad():
            for x_batch, y_batch in test_dataloader:
                x_batch = x_batch.to(device)
                x_batch_hr = dataloader.upscale_nn(x_batch)
                x_batch_hr = dataloader.add_noise_channel(x_batch_hr)

                y_batch = torch.flatten(y_batch, start_dim=1)

                timesteps = torch.zeros([x_batch.shape[0]]).to(device)

                with torch.amp.autocast("cuda"):
                    match cf.model.generator_architecture:
                        case "spategan":
                            y_pred = generator(x_batch)
                            y_pred = torch.flatten(y_pred, start_dim=1)
                        case "diffusion_unet":
                            y_pred = generator(x_batch_hr, timesteps)
                            y_pred = torch.flatten(y_pred, start_dim=1)

                all_preds.append(y_pred.cpu())
                all_targets.append(y_batch.cpu())

        # Concatenate all batches
        y_pred_all = torch.cat(all_preds, dim=0)
        y_true_all = torch.cat(all_targets, dim=0)

        # Convert to xarray with denormalization
        pred_ds, true_ds = predictions_to_xarray(
            y_pred_all, y_true_all, norm_params, var_name=cf.data.var_target
        )

        # Compute diagnostics
        rmse = diagnostics.rmse(true_ds, pred_ds, var=cf.data.var_target, dim="time")
        bias_mean = diagnostics.bias_index(
            true_ds,
            pred_ds,
            index_fn=lambda x, **kw: x[cf.data.var_target].mean("time"),
        )
        diagnostic_history["rmse"].append(rmse[cf.data.var_target].mean().values.item())
        diagnostic_history["bias_mean"].append(bias_mean.mean().values.item())
        diagnostic_history["epochs"].append(epoch + 1)

        logger.info(f"  RMSE (spatial mean): {diagnostic_history['rmse'][-1]:.4f}")
        logger.info(
            f"  Bias Mean (spatial mean): {diagnostic_history['bias_mean'][-1]:.4f}"
        )
        plot_diagnostic_history(diagnostic_history, cf)

    # Save checkpoint
    if (epoch + 1) % cf.logging.checkpoint_frequency == 0:
        torch.save(
            {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "gen_optimizer_state_dict": gen_opt.state_dict(),
                "disc_optimizer_state_dict": disc_opt.state_dict(),
                "train_gen_loss": train_gen_loss,
                "train_disc_loss": train_disc_loss,
                "test_loss": test_loss,
                "diagnostic_history": diagnostic_history,
            },
            f"{cf.logging.run_dir}/checkpoints/checkpoint_epoch_{epoch + 1}.pt",
        )
        logger.info("  Checkpoint saved")

    if (epoch + 1) % cf.logging.map_frequency == 0:
        # Visualize predictions with denormalization
        g = torch.Generator(device="cpu")
        g.seed()  # uses system entropy
        idx = torch.randint(0, x_vis.size(0), (1,), generator=g).item()
        logger.info(f"  Plotting sample {idx}")
        plot_predictions(
            generator,
            x_vis,
            y_vis,
            cf,
            epoch + 1,
            device,
            sample_idx=idx,
            norm_params=norm_params,
        )

# Save final models
torch.save(
    {
        "epoch": cf.training.epochs,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "gen_optimizer_state_dict": gen_opt.state_dict(),
        "disc_optimizer_state_dict": disc_opt.state_dict(),
        "diagnostic_history": diagnostic_history,
    },
    f"{cf.logging.run_dir}/checkpoints/final_models.pt",
)

logger.info(f"\nGAN training complete! Models saved to {cf.logging.run_dir}")
