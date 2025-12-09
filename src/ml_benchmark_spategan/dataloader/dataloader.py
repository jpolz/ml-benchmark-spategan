import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np

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


def build_dataloaders(cf):
    """
    """
    if cf.data.training_experiment == 'ESD_pseudo_reality':
        period_training = '1961-1980'
    elif cf.data.training_experiment == 'Emulator_hist_future':
        period_training = '1961-1980_2080-2099'
    else:
        raise ValueError('Provide a valid date')

    # Set the GCM
    if cf.data.domain == 'ALPS':
        cf.data.gcm_name = 'CNRM-CM5'
    elif (cf.data.domain == 'NZ') or (cf.data.domain == 'SA'):
        cf.data.gcm_name = 'ACCESS-CM2'

    predictor_filename = f'{cf.data.data_path}/{cf.data.domain}/{cf.data.domain}_domain/train/{cf.data.training_experiment}/predictors/{cf.data.gcm_name}_{period_training}.nc'
    predictor = xr.open_dataset(predictor_filename)
    if cf.data.domain == 'SA':
        predictor = predictor.drop_vars('time_bnds')

    predictand_filename = f'{cf.data.data_path}/{cf.data.domain}/{cf.data.domain}_domain/train/{cf.data.training_experiment}/target/pr_tasmax_{cf.data.gcm_name}_{period_training}.nc'
    predictand = xr.open_dataset(predictand_filename)
    predictand = predictand[[cf.data.var_target]]

    if cf.data.training_experiment == 'ESD_pseudo_reality':
        years_train = list(range(1961, 1975))
        years_test = list(range(1975, 1980+1))
    elif cf.data.training_experiment == 'Emulator_hist_future':
        years_train = list(range(1961, 1980+1)) + list(range(2080, 2090))
        years_test = list(range(2090, 2099+1))

    x_train = predictor.sel(time=np.isin(predictor['time'].dt.year, years_train))
    y_train = predictand.sel(time=np.isin(predictand['time'].dt.year, years_train))

    x_test = predictor.sel(time=np.isin(predictor['time'].dt.year, years_test))
    y_test = predictand.sel(time=np.isin(predictand['time'].dt.year, years_test))

    if cf.data.normalization == 'standardization':
        mean_train = x_train.mean('time')
        std_train = x_train.std('time')

        x_train_stand = (x_train - mean_train) / std_train
        x_test_stand = (x_test - mean_train) / std_train
    elif cf.data.normalization == 'minmax':
        min_train = x_train.min('time')
        max_train = x_train.max('time')

        x_train_stand = (x_train - min_train) / (max_train - min_train)
        x_test_stand = (x_test - min_train) / (max_train - min_train)
    elif cf.data.normalization == 'log':
        raise NotImplementedError('Log normalization not implemented yet')
    else:
        x_train_stand = x_train
        x_test_stand = x_test

    if cf.data.domain == 'ALPS':
        spatial_dims = ('x', 'y')
    elif (cf.data.domain == 'NZ') or (cf.data.domain == 'SA'):
        spatial_dims = ('lat', 'lon')

    y_train_stack = y_train.stack(gridpoint=spatial_dims)
    y_test_stack = y_test.stack(gridpoint=spatial_dims)

    x_train_stand_array = torch.from_numpy(x_train_stand.to_array().transpose("time", "variable", "lat", "lon").values)
    y_train_stack_array = torch.from_numpy(y_train_stack.to_array()[0, :].values)

    x_test_stand_array = torch.from_numpy(x_test_stand.to_array().transpose("time", "variable", "lat", "lon").values)
    y_test_stack_array = torch.from_numpy(y_test_stack.to_array()[0, :].values)
    
    dataset_training = EmulationTrainingDataset(x_data=x_train_stand_array, y_data=y_train_stack_array)
    dataloader_train = DataLoader(dataset=dataset_training,
                                batch_size=cf.training.batch_size, shuffle=True, num_workers=cf.data.num_workers)
    
    dataset_test = EmulationTrainingDataset(x_data=x_test_stand_array, y_data=y_test_stack_array)
    test_dataloader = DataLoader(dataset=dataset_test,
                                batch_size=cf.training.batch_size, shuffle=False, num_workers=cf.data.num_workers)
    
    return dataloader_train, test_dataloader, cf
    