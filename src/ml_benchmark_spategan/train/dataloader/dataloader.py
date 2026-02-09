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

import math
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from ml_benchmark_spategan.utils.normalize import normalize_predictors


def sinusoidal_encoding_doy(doy: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Apply sinusoidal encoding to day of year values.

    For models that expect values between 0 and 1, this creates a smooth
    cyclic representation where day 1 and day 365/366 are close together.

    This is used when cf.data.use_doy is True to replace the zero timestep
    with seasonal conditioning information.

    Args:
        doy: Day of year tensor (values 1-366)
        normalize: If True, normalize to [0, 1] range. If False, keep raw encoding.

    Returns:
        Encoded day of year tensor
    """
    # Convert to angle (0 to 2*pi)
    angle = (doy - 1) / 365.25 * 2 * math.pi

    if normalize:
        # Use sine encoding normalized to [0, 1]
        # sin ranges from [-1, 1], so (sin + 1) / 2 gives [0, 1]
        encoded = (torch.sin(angle) + 1.0) / 2.0
    else:
        # Use raw sine encoding [-1, 1]
        encoded = torch.sin(angle)

    return encoded


def temporal_collate_fn(batch):
    """
    Custom collate function to handle temporal dimensions.

    Reshapes temporal data from (T, C, H, W) to (C*T, H, W) by stacking
    temporal frames as channels for 2D UNet processing.

    Args:
        batch: List of samples from Dataset.__getitem__
               Each sample is either (x, y) or (x, y, doy)
               where x can be (C, H, W) or (T, C, H, W)

    Returns:
        Batched tensors with temporal dimension flattened into channels
    """
    # Check if batch contains doy
    has_doy = len(batch[0]) == 3

    if has_doy:
        x_list, y_list, doy_list = zip(*batch)
    else:
        x_list, y_list = zip(*batch)
        doy_list = None

    # Check if x has temporal dimension (4D: T, C, H, W vs 3D: C, H, W)
    if x_list[0].ndim == 4:
        # Temporal case: reshape (T, C, H, W) -> (C*T, H, W)
        x_reshaped = []
        for x in x_list:
            T, C, H, W = x.shape
            # Reshape to (C*T, H, W) by flattening temporal and channel dims
            x_flat = x.permute(1, 0, 2, 3).reshape(C * T, H, W)
            x_reshaped.append(x_flat)
        x_batch = torch.stack(x_reshaped)
    else:
        # Non-temporal case: just stack normally
        x_batch = torch.stack(x_list)

    y_batch = torch.stack(y_list)

    if has_doy:
        doy_batch = torch.stack(doy_list)
        return x_batch, y_batch, doy_batch
    else:
        return x_batch, y_batch


def load_cordex_data(
    domain: str,
    training_experiment: str,
    var_target: str = "tasmax",
    data_path: str = "/bg/fast/aihydromet/cordexbench/",
    predictand_dtype: str = "float32",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Load CORDEX benchmark data.

    Args:
        domain: Domain name ('SA', 'NZ', 'ALPS')
        training_experiment: Experiment name ('ESD_pseudo_reality' or 'Emulator_hist_future')
        var_target: Target variable ('tasmax' or 'pr')
        data_path: Path to data directory

    Returns:
        Tuple of (predictor, predictand) datasets
    """
    # Determine period and GCM
    if training_experiment == "ESD_pseudo_reality":
        period_training = "1961-1980"
    elif training_experiment == "Emulator_hist_future":
        period_training = "1961-1980_2080-2099"
    else:
        raise ValueError(f"Invalid experiment: {training_experiment}")

    if domain == "ALPS":
        gcm_name = "CNRM-CM5"
    elif domain in ["NZ", "SA"]:
        gcm_name = "ACCESS-CM2"
    else:
        raise ValueError(f"Invalid domain: {domain}")

    # Load predictor
    predictor_filename = (
        f"{data_path}/{domain}/{domain}_domain/train/{training_experiment}/"
        f"predictors/{gcm_name}_{period_training}.nc"
    )
    predictor = xr.open_dataset(predictor_filename)

    if domain == "SA":
        predictor = predictor.drop_vars("time_bnds", errors="ignore")

    # Load predictand
    predictand_filename = (
        f"{data_path}/{domain}/{domain}_domain/train/{training_experiment}/"
        f"target/pr_tasmax_{gcm_name}_{period_training}.nc"
    )
    predictand = xr.open_dataset(predictand_filename)
    predictand = predictand[[var_target]].astype(predictand_dtype)

    return predictor, predictand


def load_orography(
    domain: str,
    training_experiment: str,
    data_path: str = "/bg/fast/aihydromet/cordexbench/",
) -> xr.DataArray:
    """
    Load orography (static elevation field) from CORDEX benchmark data.

    Args:
        domain: Domain name ('SA', 'NZ', 'ALPS')
        training_experiment: Experiment name ('ESD_pseudo_reality' or 'Emulator_hist_future')
        data_path: Path to data directory

    Returns:
        Orography DataArray
    """
    orography_filename = (
        f"{data_path}/{domain}/{domain}_domain/train/{training_experiment}/"
        f"predictors/Static_fields.nc"
    )
    static_fields = xr.open_dataset(orography_filename)
    orography = static_fields["orog"]
    return orography


def split_train_test(
    predictor: xr.Dataset,
    predictand: xr.Dataset,
    training_experiment: str,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Split data into train and test sets based on experiment.

    Args:
        predictor: Predictor dataset
        predictand: Predictand dataset
        training_experiment: Experiment name ('ESD_pseudo_reality' or 'Emulator_hist_future')

    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    if training_experiment == "ESD_pseudo_reality":
        years_train = list(range(1961, 1975))
        years_test = list(range(1975, 1981))
    elif training_experiment == "Emulator_hist_future":
        years_train = list(range(1961, 1981)) + list(range(2080, 2090))
        years_test = list(range(2090, 2100))
    else:
        raise ValueError(f"Invalid experiment: {training_experiment}")

    x_train = predictor.sel(time=np.isin(predictor["time"].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand["time"].dt.year, years_train))
    x_test = predictor.sel(time=np.isin(predictor["time"].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand["time"].dt.year, years_test))

    return x_train, y_train, x_test, y_test


class EmulationTrainingDataset(Dataset):
    """
    PyTorch Dataset for RCM emulation training with spatiotemporal samples.

    This dataset outputs both inputs (GCM predictors) and targets (RCM outputs)
    for supervised training of the emulator.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
        y_data: Target predictand data (RCM output). Can be numpy array or torch tensor.
        times: Time values for each sample
        t_future: Number of future time steps to use as input
        t_past: Number of past time steps to use as input
        orography: Static orography field. Can be numpy array or torch tensor. Optional.
        doy: Day of year (1-366) for each sample. Can be numpy array or torch tensor. Optional.
    """

    def __init__(
        self, x_data, y_data, times, t_future=1, t_past=1, orography=None, doy=None
    ):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        if not isinstance(y_data, torch.Tensor):
            y_data = torch.tensor(y_data)
        self.x_data, self.y_data = x_data, y_data
        self.times = times
        self.t_future = t_future
        self.t_past = t_past

        # Cache orography as internal variable
        if orography is not None:
            if not isinstance(orography, torch.Tensor):
                orography = torch.tensor(orography)
            self.orography = orography
        else:
            self.orography = None

        # Cache day of year as internal variable
        if doy is not None:
            if not isinstance(doy, torch.Tensor):
                doy = torch.tensor(doy, dtype=torch.float32)
            self.doy = doy
        else:
            self.doy = None

        # Pre-compute time-to-index mapping for efficient temporal lookups
        if times is not None and (t_future > 0 or t_past > 0):
            self.time_to_idx = {t: i for i, t in enumerate(times)}
        else:
            self.time_to_idx = None

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.t_future == 0 and self.t_past == 0:
            x_sample, y_sample = self.x_data[idx, :], self.y_data[idx, :]
            if self.doy is not None:
                doy_sample = self.doy[idx]
                # Apply sinusoidal encoding to doy
                doy_sample = sinusoidal_encoding_doy(doy_sample, normalize=True)
                return x_sample, y_sample, doy_sample
            return x_sample, y_sample
        else:
            # Get time of index and build temporal window
            t = self.times[idx]
            idxs = []
            for tp in range(-self.t_past, self.t_future + 1):
                t_offset = t + np.timedelta64(tp, "D")
                idx_t = self.time_to_idx.get(t_offset)
                if idx_t is not None:
                    idxs.append(idx_t)
            x_sample, y_sample = self.x_data[idxs, :], self.y_data[idxs, :]
            if self.doy is not None:
                doy_sample = self.doy[idx]
                # Apply sinusoidal encoding to doy
                doy_sample = sinusoidal_encoding_doy(doy_sample, normalize=True)
                return x_sample, y_sample, doy_sample
            return x_sample, y_sample

    def _get_shapes(self):
        return self.x_data.shape[1:], self.y_data.shape[1:]


class EmulationTestDataset(Dataset):
    """
    PyTorch Dataset for making predictions with trained RCM emulator.

    This dataset outputs only inputs (GCM predictors) and is used for inference
    when ground-truth target data is not available. Supports temporal samples
    for spatiotemporal models.

    Args:
        x_data: Input predictor data (GCM variables). Can be numpy array or torch tensor.
        times: Time values for each sample. Required for temporal support.
        t_future: Number of future time steps to use as input. Default 0.
        t_past: Number of past time steps to use as input. Default 0.
        orography: Static orography field. Can be numpy array or torch tensor. Optional.
        doy: Day of year (1-366) for each sample. Can be numpy array or torch tensor. Optional.
    """

    def __init__(
        self, x_data, times=None, t_future=0, t_past=0, orography=None, doy=None
    ):
        if not isinstance(x_data, torch.Tensor):
            x_data = torch.tensor(x_data)
        self.x_data = x_data
        self.times = times
        self.t_future = t_future
        self.t_past = t_past

        # Cache orography as internal variable
        if orography is not None:
            if not isinstance(orography, torch.Tensor):
                orography = torch.tensor(orography)
            self.orography = orography
        else:
            self.orography = None

        # Cache day of year as internal variable
        if doy is not None:
            if not isinstance(doy, torch.Tensor):
                doy = torch.tensor(doy, dtype=torch.float32)
            self.doy = doy
        else:
            self.doy = None

        # Pre-compute time-to-index mapping for efficient temporal lookups
        if times is not None and (t_future > 0 or t_past > 0):
            self.time_to_idx = {t: i for i, t in enumerate(times)}
        else:
            self.time_to_idx = None

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if self.t_future == 0 and self.t_past == 0:
            x_sample = self.x_data[idx, :]
            if self.doy is not None:
                doy_sample = self.doy[idx]
                # Apply sinusoidal encoding to doy
                doy_sample = sinusoidal_encoding_doy(doy_sample, normalize=True)
                return x_sample, doy_sample
            return x_sample
        else:
            # Get time of index and build temporal window
            t = self.times[idx]
            idxs = []
            for tp in range(-self.t_past, self.t_future + 1):
                t_offset = t + np.timedelta64(tp, "D")
                idx_t = self.time_to_idx.get(t_offset)
                if idx_t is not None:
                    idxs.append(idx_t)
            x_sample = self.x_data[idxs, :]
            if self.doy is not None:
                doy_sample = self.doy[idx]
                # Apply sinusoidal encoding to doy
                doy_sample = sinusoidal_encoding_doy(doy_sample, normalize=True)
                return x_sample, doy_sample
            return x_sample

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
    # Load data using the new load_cordex_data function
    predictor, predictand = load_cordex_data(
        domain=cf.data.domain,
        training_experiment=cf.data.training_experiment,
        var_target=cf.data.var_target,
        data_path=cf.data.data_path,
    )
    if cf.data.use_orography:
        # Load orography (static field)
        orography = load_orography(
            domain=cf.data.domain,
            training_experiment=cf.data.training_experiment,
            data_path=cf.data.data_path,
        )
    else:
        orography = None

    # Set GCM name in config for backward compatibility
    if cf.data.domain == "ALPS":
        cf.data.gcm_name = "CNRM-CM5"
    elif cf.data.domain in ["NZ", "SA"]:
        cf.data.gcm_name = "ACCESS-CM2"

    # Split into train and test sets
    x_train, y_train, x_test, y_test = split_train_test(
        predictor=predictor,
        predictand=predictand,
        training_experiment=cf.data.training_experiment,
    )

    # Extract times for temporal dataset
    times_train = x_train["time"].values
    times_test = x_test["time"].values

    # Extract day of year (1-366) from time coordinate if enabled
    if getattr(cf.data, "use_doy", False):
        doy_train = x_train["time"].dt.dayofyear.values
        doy_test = x_test["time"].dt.dayofyear.values
    else:
        doy_train = None
        doy_test = None

    # Normalize predictors and predictands
    log_base = getattr(cf.data, "log_base", None)
    x_train_stand, x_test_stand, y_train, y_test, norm_params_partial = (
        normalize_predictors(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            normalization=cf.data.normalization,
            orography=orography,
            log_base=log_base,
        )
    )

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

    # Convert normalized orography to tensor
    orography_norm = norm_params_partial.get("orography_norm")
    if orography_norm is not None:
        orography_array = torch.from_numpy(orography_norm.values).float()
    else:
        orography_array = None

    # 2D y_test
    y_train_stack_array = y_train_stack_array.view(-1, 1, 128, 128)

    dataset_training = EmulationTrainingDataset(
        x_data=x_train_stand_array,
        y_data=y_train_stack_array,
        times=times_train,
        t_future=cf.data.t_future,
        t_past=cf.data.t_past,
        orography=orography_array,
        doy=doy_train if getattr(cf.data, "use_doy", False) else None,
    )

    if cf.training.batches_per_epoch is not None:
        num_samples = cf.training.batches_per_epoch * cf.training.batch_size
    else:
        num_samples = (
            int(len(dataset_training) / cf.training.batch_size) * cf.training.batch_size
        )
    sampler = torch.utils.data.RandomSampler(
        dataset_training, replacement=False, num_samples=num_samples
    )

    # Use temporal collate function if temporal dimensions are active
    is_temporal = (cf.data.t_past > 0) or (cf.data.t_future > 0)
    collate_fn = temporal_collate_fn if is_temporal else None

    dataloader_train = DataLoader(
        dataset=dataset_training,
        batch_size=cf.training.batch_size,
        sampler=sampler,
        num_workers=cf.data.num_workers,
        collate_fn=collate_fn,
    )

    # 2D y_test
    y_test_stack_array = y_test_stack_array.view(-1, 1, 128, 128)

    dataset_test = EmulationTrainingDataset(
        x_data=x_test_stand_array,
        y_data=y_test_stack_array,
        times=times_test,
        t_future=cf.data.t_future,
        t_past=cf.data.t_past,
        orography=orography_array,
        doy=doy_test if getattr(cf.data, "use_doy", False) else None,
    )

    # Use temporal collate function if temporal dimensions are active
    test_dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=cf.training.batch_size,
        shuffle=False,
        num_workers=cf.data.num_workers,
        collate_fn=collate_fn,  # Use same collate_fn as training for temporal support
    )

    # Store normalization parameters for denormalization
    norm_params = norm_params_partial.copy()
    norm_params["spatial_dims"] = spatial_dims
    norm_params["y_test_coords"] = y_test.coords  # Store original unstacked coords
    norm_params["spatial_shape"] = (
        len(y_test[spatial_dims[0]]),
        len(y_test[spatial_dims[1]]),
    )  # (H, W)

    return dataloader_train, test_dataloader, cf, norm_params
