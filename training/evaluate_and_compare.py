import os
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import sys
sys.path.append('./evaluation')
import indices, diagnostics

from ml_benchmark_spategan.utils.denormalize import predictions_to_xarray

from einops import rearrange
import torch.nn.functional as F

def denorm(y_pred, y_min, y_max):
        y_denorm = (y_pred - y_min)/(y_max - y_min)
        y_denorm = y_denorm*2 - 1

        return y_denorm

def upscale_nn(x):
    # x: (15, 16, 16)
    return F.interpolate(
        x, 
        size=(128, 128),
        mode="bilinear"
    )

    
def add_noise_channel(x):
    # x: (B, 15, 128, 128)
    noise = torch.randn(
        x.size(0),      # batch
        1,              # 1 noise channel
        x.size(2),      # height = 128
        x.size(3),      # width = 128
        device=x.device # put noise on same device (GPU or CPU)
    )
    return torch.cat([x, noise], dim=1)

def plot_data_map(data, var_name, domain, vmin, vmax,
                  fig_title='', figsize=(8,8), cmap='viridis'):
    
    central_longitude = 180 if domain == 'NZ' else 0 if domain == 'ALPS' else None
    
    plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))

    if (domain == 'NZ') or (domain == 'SA'):
        data[var_name].plot(ax=ax, transform=ccrs.PlateCarree(),
                            vmin=vmin, vmax=vmax,
                            cmap=cmap)
    elif domain == 'ALPS':
        cs = ax.pcolormesh(data[var_name]['lon'], data[var_name]['lat'],
                           data[var_name],
                           transform=ccrs.PlateCarree(),
                           vmin=vmin, vmax=vmax,
                           cmap=cmap)
    
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    plt.title(fig_title)
    plt.show()

def plot_psd(psd_target, psd_pred):
    plt.loglog(psd_target.wavenumber, psd_target, label="Target")
    plt.loglog(psd_pred.wavenumber, psd_pred, label="Prediction")
    plt.xlabel("Wavenumber")
    plt.title("Power Spectral Density")
    plt.legend()
    plt.show()

DATA_PATH = '/bg/fast/aihydromet/cordexbench/'

MODELS_PATH = './training/models'
os.makedirs(MODELS_PATH, exist_ok=True)

domain = 'SA'
training_experiment = 'ESD_pseudo_reality'

# Set the period
if training_experiment == 'ESD_pseudo_reality':
    period_training = '1961-1980'
elif training_experiment == 'Emulator_hist_future':
    period_training = '1961-1980_2080-2099'
else:
    raise ValueError('Provide a valid date')

# Set the GCM
if domain == 'ALPS':
    gcm_name = 'CNRM-CM5'
elif (domain == 'NZ') or (domain == 'SA'):
    gcm_name = 'ACCESS-CM2'

predictor_filename = f'{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/predictors/{gcm_name}_{period_training}.nc'
predictor = xr.open_dataset(predictor_filename)

if domain == 'SA':
    predictor = predictor.drop_vars('time_bnds')

var_target = 'tasmax'

predictand_filename = f'{DATA_PATH}/{domain}/{domain}_domain/train/{training_experiment}/target/pr_tasmax_{gcm_name}_{period_training}.nc'
predictand = xr.open_dataset(predictand_filename)
predictand = predictand[[var_target]]

if training_experiment == 'ESD_pseudo_reality':
    years_train = list(range(1961, 1975))
    years_test = list(range(1975, 1980+1))
elif training_experiment == 'Emulator_hist_fut':
    years_train = list(range(1961, 1980+1)) + list(range(2080, 2090))
    years_test = list(range(2090, 2099+1))

x_train = predictor.sel(time=np.isin(predictor['time'].dt.year, years_train))
y_train = predictand.sel(time=np.isin(predictand['time'].dt.year, years_train))

x_test = predictor.sel(time=np.isin(predictor['time'].dt.year, years_test))
y_test = predictand.sel(time=np.isin(predictand['time'].dt.year, years_test))

mean_train = x_train.mean('time')
std_train = x_train.std('time')

x_train_stand = (x_train - mean_train) / std_train
x_test_stand = (x_test - mean_train) / std_train

if domain == 'ALPS':
    spatial_dims = ('x', 'y')
elif (domain == 'NZ') or (domain == 'SA'):
    spatial_dims = ('lat', 'lon')

y_train_stack = y_train.stack(gridpoint=spatial_dims)
y_test_stack = y_test.stack(gridpoint=spatial_dims)

x_train_stand_array = torch.from_numpy(x_train_stand.to_array().transpose("time", "variable", "lat", "lon").values)
y_train_stack_array = torch.from_numpy(y_train_stack.to_array()[0, :].values)

x_test_stand_array = torch.from_numpy(x_test_stand.to_array().transpose("time", "variable", "lat", "lon").values)

class EmulationTrainingDataset(Dataset):
    def __init__(self, x_data, y_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        if not isinstance(y_data, torch.Tensor):
            y_data = torch.tensor(y_data)
        self.x_data, self.y_data = x_data, y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x_sample, y_sample = self.x_data[idx, :], self.y_data[idx, :]
        return x_sample, y_sample

class EmulationTestDataset(Dataset):
    def __init__(self, x_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx, :]


#########################################################
######## Construct DeepESD model and load weights #######
#########################################################

class DeepESD(torch.nn.Module):

    def __init__(self, x_shape: tuple, y_shape: tuple,
                 filters_last_conv: int):

        super(DeepESD, self).__init__()

        self.x_shape = x_shape
        self.y_shape = y_shape
        self.filters_last_conv = filters_last_conv

        self.conv_1 = torch.nn.Conv2d(in_channels=self.x_shape[1],
                                      out_channels=50,
                                      kernel_size=3,
                                      padding=1)

        self.conv_2 = torch.nn.Conv2d(in_channels=50,
                                      out_channels=25,
                                      kernel_size=3,
                                      padding=1)

        self.conv_3 = torch.nn.Conv2d(in_channels=25,
                                      out_channels=self.filters_last_conv,
                                      kernel_size=3,
                                      padding=1)

        self.out = torch.nn.Linear(in_features=\
                                       self.x_shape[2] * self.x_shape[3] * self.filters_last_conv,
                                       out_features=self.y_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_1(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        x = torch.relu(x)

        x = self.conv_3(x)
        x = torch.relu(x)

        x = torch.flatten(x, start_dim=1)

        out = self.out(x)

        return out


model = DeepESD(x_shape=x_train_stand_array.shape,
                y_shape=y_train_stack_array.shape,
                filters_last_conv=1)

# load weights as saved using torch.save(model.state_dict(), f'{MODELS_PATH}/{model_name}')
model_name = 'model.pt'  # or whatever your model filename is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}', map_location=device))
model.to(device)
model.eval()



#####################################################
######## Construct GAN model and load weights #######
#####################################################

from diffusers import UNet2DModel
from ml_benchmark_spategan.model.spagan2d import Generator, Discriminator, train_gan_step
from torchinfo import summary
from ml_benchmark_spategan.config import config

run_id = "20251210_0427_gcvutf9u"
cf = config.load_config_from_yaml(f'./runs/{run_id}/config.yaml')
print(cf)

match cf.model.architecture:
    case "spategan":
        print("Using SpaGAN architecture")
        # Initialize models
        generator = Generator(cf.model).to(device)

        # Print model summaries
    case "diffusion_unet":
        print("Using Diffusion UNet architecture")
        generator = UNet2DModel(
                sample_size=(128, 128),
                in_channels=15+1,  # +1 == noise
                out_channels=1,
                layers_per_block=2,
                block_out_channels=(64, 128, 128, 256),
                down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            ).to(device)

    case _:
        raise ValueError(f"Invalid option: {cf.model.architecture}")

# Load the checkpoint
checkpoint = torch.load(f'./runs/{run_id}/final_models.pt', map_location=device)

# Load the generator state dict
generator.load_state_dict(checkpoint['generator_state_dict'])
generator = generator.to(device)
generator.eval()


###############
### compute ###
###############


dataset_test = EmulationTestDataset(x_data=x_test_stand_array)
test_dataloader = DataLoader(dataset=dataset_test,
                             batch_size=32, shuffle=False)

# Compute predictions
model.eval()

predictions = []
predictions_generator = []
with torch.no_grad():
    for batch_x in test_dataloader:
        batch_x = batch_x.to(next(model.parameters()).device)
        outputs = model(batch_x)
        x_batch_hr = upscale_nn(batch_x)
        timesteps = torch.zeros([x_batch_hr.shape[0]]).to(device)
        outputs_generator = generator(add_noise_channel(x_batch_hr) , timesteps).sample
        predictions.append(outputs.cpu().numpy())
        predictions_generator.append(outputs_generator.cpu().numpy())

# Concatenate all batches into one array
predictions = np.concatenate(predictions, axis=0)
predictions_generator = rearrange(np.concatenate(predictions_generator, axis=0), 'b 1 h w -> b (h w)')
# Denormalize predictions
y_min = xr.open_dataarray(f'./runs/{run_id}/ymin.nc')
y_max = xr.open_dataarray(f'./runs/{run_id}/ymax.nc')
predictions_generator = denorm(torch.from_numpy(predictions_generator), y_min, y_max).numpy()
print(f'Predictions shape (DeepESD): {predictions.shape}')
print(f'Predictions shape (Generator): {predictions_generator.shape}')

y_pred_stack = y_test_stack.copy(deep=True)
y_pred_stack[var_target].values = predictions
y_pred = y_pred_stack.unstack()

y_pred_generator_stack = y_test_stack.copy(deep=True)
y_pred_generator_stack[var_target].values = predictions_generator
y_pred_generator = y_pred_generator_stack.unstack()

rmse = diagnostics.rmse(x0=y_test, x1=y_pred,
                        var=var_target, dim='time')
rmse_generator = diagnostics.rmse(x0=y_test, x1=y_pred_generator,
                        var=var_target, dim='time')
print(f'Mean RMSE (DeepESD): {rmse[var_target].mean().values.item()}')
print(f'Mean RMSE (Generator): {rmse_generator[var_target].mean().values.item()}')

# plot_data_map(data=rmse, var_name=var_target, domain=domain, vmin=0, vmax=5,
#               fig_title='RMSE', cmap='Reds')