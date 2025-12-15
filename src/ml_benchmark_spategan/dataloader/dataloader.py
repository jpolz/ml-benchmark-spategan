"""
DataLoader module for RCM (Regional Climate Model) emulation training.

This module provides PyTorch Dataset classes and data loading utilities for training
deep learning emulators on the CORDEX Benchmark dataset. The workflow handles:
- Loading predictor (GCM) and predictand (RCM target) data from NetCDF files
- Splitting data into training and test sets based on year ranges
- Normalizing/standardizing input features
- Creating PyTorch DataLoaders for training and evaluation

For more details on the CORDEX Benchmark dataset properties, see the data notebooks.
"""

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset


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


class EmulationTrainingDataset(Dataset):
    """
    PyTorch Dataset for RCM emulation training.

    This dataset outputs both inputs (GCM predictors) and targets (RCM outputs)
    for supervised training of the emulator.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
        y_data: Target predictand data (RCM output). Can be numpy array or torch tensor.
    """

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

    def _get_shapes(self):
        return self.x_data.shape[1:], self.y_data.shape[1:]


class EmulationTrainingDatasetSpate(Dataset):
    """
    PyTorch Dataset for RCM emulation training with spatiotemporal samples.

    This dataset outputs both inputs (GCM predictors) and targets (RCM outputs)
    for supervised training of the emulator.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
        y_data: Target predictand data (RCM output). Can be numpy array or torch tensor.
        t_future: Number of future time steps to use as input
        t_past: Number of past time steps to use as input
    """

    def __init__(self, x_data, y_data, times, t_future=1, t_past=1):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        if not isinstance(y_data, torch.Tensor):
            y_data = torch.tensor(y_data)
        self.x_data, self.y_data = x_data, y_data
        self.times = times
        self.t_future = t_future
        self.t_past = t_past

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.t_future == 0 and self.t_past == 0:
            x_sample, y_sample = self.x_data[idx, :], self.y_data[idx, :]
            return x_sample, y_sample
        else:
            # get time of index
            t = self.times[idx]
            print(f"t: {t}")
            # get indices for past and future time steps
            idxs = []
            for tp in range(-self.t_past, self.t_future+1):
                print(f"tp: {tp}")
                print(f"t + tp: {t + np.timedelta64(tp, 'D')}")
                idx_t = np.where(self.times == t + np.timedelta64(tp, 'D'))[0]
                print(f"idx_t: {idx_t}, idx: {idx}")
                if len(idx_t) > 0:
                    idxs.append(idx_t[0])
            x_sample, y_sample = self.x_data[idxs, :], self.y_data[idxs, :]
            return x_sample, y_sample

    def _get_shapes(self):
        return self.x_data.shape[1:], self.y_data.shape[1:]


class EmulationTestDataset(Dataset):
    """
    PyTorch Dataset for making predictions with trained RCM emulator.

    This dataset outputs only inputs (GCM predictors) and is used for inference
    when ground-truth target data is not available.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
    """

    def __init__(self, x_data):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx, :]

    def _get_shapes(self):
        return self.x_data.shape[1:]


def build_dataloaders(cf):
    """
    Build training and test DataLoaders from CORDEX Benchmark dataset.

    This function handles the complete data loading pipeline:
    1. Loads predictor (GCM) and predictand (RCM target) NetCDF files
    2. Splits data into train/test sets based on year ranges
    3. Applies normalization (standardization, minmax, or none)
    4. Flattens spatial dimensions for models with fully connected output layers
    5. Creates PyTorch DataLoaders for training and evaluation

    The benchmark provides two training experiments:
    - 'ESD_pseudo_reality': Train on 1961-1974, test on 1975-1980
    - 'Emulator_hist_future': Train on 1961-1980 + 2080-2089, test on 2090-2099

    For predictands, both daily maximum temperature ('tasmax') and daily accumulated
    precipitation ('pr') are available.

    Note: The benchmark does not yet provide ground-truth RCM data for evaluation
    experiments, so a test set is generated from the training data to provide an
    initial indication of emulator performance.

    Args:
        cf: Configuration object containing data settings:
            - cf.data.training_experiment: 'ESD_pseudo_reality' or 'Emulator_hist_future'
            - cf.data.domain: 'ALPS', 'NZ', or 'SA'
            - cf.data.var_target: Target variable ('tasmax' or 'pr')
            - cf.data.data_path: Path to CORDEX benchmark data
            - cf.data.normalization: 'standardization', 'minmax', 'log', or None, or 'minus1_to_plus1'
            - cf.data.num_workers: Number of workers for DataLoader
            - cf.training.batch_size: Batch size for training

    Returns:
        tuple: (dataloader_train, test_dataloader, cf)
            - dataloader_train: DataLoader for training data
            - test_dataloader: DataLoader for test data
            - cf: Updated configuration with gcm_name added
    """
    if cf.data.training_experiment == "ESD_pseudo_reality":
        period_training = "1961-1980"
    elif cf.data.training_experiment == "Emulator_hist_future":
        period_training = "1961-1980_2080-2099"
    else:
        raise ValueError("Provide a valid date")

    # Set the GCM
    if cf.data.domain == "ALPS":
        cf.data.gcm_name = "CNRM-CM5"
    elif (cf.data.domain == "NZ") or (cf.data.domain == "SA"):
        cf.data.gcm_name = "ACCESS-CM2"

    predictor_filename = f"{cf.data.data_path}/{cf.data.domain}/{cf.data.domain}_domain/train/{cf.data.training_experiment}/predictors/{cf.data.gcm_name}_{period_training}.nc"
    predictor = xr.open_dataset(predictor_filename)
    if cf.data.domain == "SA":
        predictor = predictor.drop_vars("time_bnds")

    predictand_filename = f"{cf.data.data_path}/{cf.data.domain}/{cf.data.domain}_domain/train/{cf.data.training_experiment}/target/pr_tasmax_{cf.data.gcm_name}_{period_training}.nc"
    predictand = xr.open_dataset(predictand_filename)
    print(cf.data.var_target)
    predictand = predictand[[cf.data.var_target]].astype("float32")
    print(predictand)

    if cf.data.training_experiment == "ESD_pseudo_reality":
        years_train = list(range(1961, 1975))
        years_test = list(range(1975, 1980 + 1))
    elif cf.data.training_experiment == "Emulator_hist_future":
        years_train = list(range(1961, 1980 + 1)) + list(range(2080, 2090))
        years_test = list(range(2090, 2099 + 1))

    x_train = predictor.sel(time=np.isin(predictor["time"].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand["time"].dt.year, years_train))
    times_train = x_train["time"].values

    x_test = predictor.sel(time=np.isin(predictor["time"].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand["time"].dt.year, years_test))
    times_test = x_test["time"].values

    if cf.data.normalization == "standardization":
        mean_train = x_train.mean("time")
        std_train = x_train.std("time")

        x_train_stand = (x_train - mean_train) / std_train
        x_test_stand = (x_test - mean_train) / std_train
    elif cf.data.normalization == "minmax":
        min_train = x_train.min("time")
        max_train = x_train.max("time")

        x_train_stand = (x_train - min_train) / (max_train - min_train)
        x_test_stand = (x_test - min_train) / (max_train - min_train)
    elif cf.data.normalization == "std_log_target":
        mean_train = x_train.mean("time")
        std_train = x_train.std("time")

        x_train_stand = (x_train - mean_train) / std_train
        x_test_stand = (x_test - mean_train) / std_train

        # log transform y
        y_train = np.log1p(y_train+1e-6)
        y_test = np.log1p(y_test+1e-6)

    elif cf.data.normalization == "mp1p1_input_m1p1log_target":
        
        # x sample normalization
        min_train = x_train.min("time")
        max_train = x_train.max("time")
        

        x_train_stand = (x_train - min_train) / (max_train - min_train)
        x_train_stand = x_train_stand * 2 - 1
        x_test_stand = (x_test - min_train) / (max_train - min_train)
        x_test_stand = x_test_stand * 2 - 1

        # y sample normalization
        # 1. log transform y
        y_train = np.log1p(y_train+1e-6) - np.log1p(1e-6)
        y_test = np.log1p(y_test+1e-6) - np.log1p(1e-6)
        
        # 2. min-max to [-1, 1]
        y_min_train = y_train.min("time")
        y_max_train = y_train.max("time")
        
        y_train = (y_train - y_min_train) / (y_max_train - y_min_train)
        y_train = y_train * 2 - 1
        
        y_test = (y_test - y_min_train) / (y_max_train - y_min_train)
        y_test = y_test * 2 - 1
        
        # to float32
        y_train = y_train.astype(np.float32)
        y_test  = y_test.astype(np.float32)
        
        
    elif cf.data.normalization == "m1p1_log_target":
        min_train = x_train.min("time")
        max_train = x_train.max("time")

        x_train_stand = (x_train - min_train) / (max_train - min_train)
        x_train_stand = x_train_stand * 2 - 1
        x_test_stand = (x_test - min_train) / (max_train - min_train)
        x_test_stand = x_test_stand * 2 - 1

        # log transform y
        y_train = np.log1p(y_train+1e-6)
        y_test = np.log1p(y_test+1e-6)
        
        

    elif cf.data.normalization == "minus1_to_plus1":
        min_train = x_train.min("time")
        max_train = x_train.max("time")
        # min_train = x_train.min()
        # max_train = x_train.max()

        x_train_stand = (x_train - min_train) / (max_train - min_train)
        x_train_stand = x_train_stand * 2 - 1
        x_test_stand = (x_test - min_train) / (max_train - min_train)
        x_test_stand = x_test_stand * 2 - 1

        # normalize y as well (?) --> inverse missing
        # y_min_train = y_train[cf.data.var_target].min().values
        # y_max_train = y_train[cf.data.var_target].max().values
        y_min_train = y_train.min("time")
        y_max_train = y_train.max("time")
        # print(f"y_min_train: {y_min_train}")
        # print(f"y_max_train: {y_max_train}")
        # cf.data.y_min = float(y_min_train)
        # cf.data.y_max = float(y_max_train)

        y_train = (y_train - y_min_train) / (y_max_train - y_min_train)
        y_train = y_train * 2 - 1
        y_test = (y_test - y_min_train) / (y_max_train - y_min_train)
        y_test = y_test * 2 - 1

    else:
        x_train_stand = x_train
        x_test_stand = x_test

    if cf.data.domain == "ALPS":
        spatial_dims = ("x", "y")
    elif (cf.data.domain == "NZ") or (cf.data.domain == "SA"):
        spatial_dims = ("lat", "lon")

    y_train_stack = y_train.stack(gridpoint=spatial_dims)
    y_test_stack = y_test.stack(gridpoint=spatial_dims)

    x_train_stand_array = torch.from_numpy(
        x_train_stand.to_array().transpose("time", "variable", "lat", "lon").values
    )
    y_train_stack_array = torch.from_numpy(y_train_stack.to_array()[0, :].values)

    x_test_stand_array = torch.from_numpy(
        x_test_stand.to_array().transpose("time", "variable", "lat", "lon").values
    )
    y_test_stack_array = torch.from_numpy(y_test_stack.to_array()[0, :].values)

    # 2D y_test
    y_train_stack_array = y_train_stack_array.view(-1, 1, 128, 128)

    dataset_training = EmulationTrainingDatasetSpate(
        x_data=x_train_stand_array, 
        y_data=y_train_stack_array, 
        times=times_train, 
        t_future=cf.data.t_future, 
        t_past=cf.data.t_past,
    )

    if cf.training.batches_per_epoch is not None:
        num_samples = cf.training.batches_per_epoch * cf.training.batch_size
    else:
        num_samples = int(len(dataset_training)/cf.training.batch_size)*cf.training.batch_size
    sampler = torch.utils.data.RandomSampler(
        dataset_training, 
        replacement=False,
        num_samples=num_samples
    )
    dataloader_train = DataLoader(
        dataset=dataset_training,
        batch_size=cf.training.batch_size,
        sampler=sampler,
        num_workers=cf.data.num_workers,
    )

    # 2D y_test
    y_test_stack_array = y_test_stack_array.view(-1, 1, 128, 128)

    dataset_test = EmulationTrainingDataset(
        x_data=x_test_stand_array, y_data=y_test_stack_array
    )
    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=cf.training.batch_size,
        shuffle=False,
        num_workers=cf.data.num_workers,
    )

    # Store normalization parameters for denormalization
    norm_params = {
        "normalization": cf.data.normalization,
        "spatial_dims": spatial_dims,
        "y_test_coords": y_test.coords,  # Store original unstacked coords
        "spatial_shape": (
            len(y_test[spatial_dims[0]]),
            len(y_test[spatial_dims[1]]),
        ),  # (H, W)
    }

    if cf.data.normalization == "minus1_to_plus1":
        norm_params["y_min"] = y_min_train
        norm_params["y_max"] = y_max_train
    
    # store log min and max values:
    if cf.data.normalization == "mp1p1_input_m1p1log_target":
        norm_params["y_min"] = y_min_train
        norm_params["y_max"] = y_max_train
        
        

    return dataloader_train, test_dataloader, cf, norm_params


def build_dummy_dataloaders(
    batch_size=2, num_samples_train=100, num_samples_test=20, num_workers=0
):
    """
    Build dummy DataLoaders with random data for debugging and testing.
    Creates synthetic data with the expected shapes for RCM emulation:
    - Input (x): (batch, 15, 16, 16)
    - Target (y): (batch, 128, 128) flattened to (batch, 16384)
    """
    # Generate random training data
    x_train = torch.randn(num_samples_train, 15, 16, 16, dtype=torch.float32)
    y_train = torch.randn(num_samples_train, 128 * 128, dtype=torch.float32)

    # Generate random test data
    x_test = torch.randn(num_samples_test, 15, 16, 16, dtype=torch.float32)
    y_test = torch.randn(num_samples_test, 128 * 128, dtype=torch.float32)

    # Create datasets
    dataset_train = EmulationTrainingDataset(x_data=x_train, y_data=y_train)
    dataset_test = EmulationTrainingDataset(x_data=x_test, y_data=y_test)

    # Create dataloaders
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloader_train, test_dataloader
