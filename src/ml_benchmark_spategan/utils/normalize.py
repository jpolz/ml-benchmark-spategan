import xarray as xr
import numpy as np
import torch

def apply_transforms(data: xr.DataArray,
                     data_ref: xr.DataArray,
                     config) -> xr.DataArray:
    """ Apply a sequence of transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants

    Returns:
        The transformed data
    
    """
    
    if 'log' in config.transforms and "total_precipitation" in data_ref.name:
        data = log_transform(data, config.epsilon)



    if 'normalize_minus1_to_plus1' in config.transforms:
        data = norm_minus1_to_plus1_transform(data, data_ref, config)
        
    return data   



def apply_inverse_transforms(data: xr.DataArray,
                            data_ref: xr.DataArray,
                            config) -> xr.DataArray:
    """ Apply a sequence of inverse transformations given a training set reference

    Args:
        data: Data to be transformed
        data_ref: Reference data from the training set
        config: Conifguration dataclass of transforms and constants
    
    Returns:
        The data tranformed back to the physical space
    """
    
    if "normalize_minus1_to_plus1" in config.transforms:
        data = inv_norm_minus1_to_plus1_transform(data, data_ref, config)


    if "log" in config.transforms and "total_precipitation" in data_ref.name:
        data = inv_log_transform(data, config.epsilon)

    return data


def log_transform(x, epsilon):
    if isinstance(x, torch.Tensor):
        # make sure epsilon matches dtype/device
        epsilon = torch.as_tensor(epsilon, dtype=x.dtype, device=x.device)
        return torch.log(x + epsilon) - torch.log(epsilon)
    else:
        return np.log(x + epsilon) - np.log(epsilon)


def inv_log_transform(x, epsilon):
    if isinstance(x, torch.Tensor):
        epsilon = torch.as_tensor(epsilon, dtype=x.dtype, device=x.device)
        return torch.exp(x + torch.log(epsilon)) - epsilon
    else:
        return np.exp(x + np.log(epsilon)) - epsilon


def norm_minus1_to_plus1_transform(x, x_ref, config):
    global_min = x_ref.attrs['global_min'][0] if isinstance(x_ref.attrs['global_min'], (list, tuple)) else x_ref.attrs['global_min']
    global_max = x_ref.attrs['global_max'][0] if isinstance(x_ref.attrs['global_max'], (list, tuple)) else x_ref.attrs['global_max']

    if "log" in config.transforms and "total_precipitation" in x_ref.name:
        global_min = log_transform(global_min, config.epsilon)
        global_max = log_transform(global_max, config.epsilon)
    x = (x - global_min)/(global_max - global_min)
    x = x*2 - 1
    return x 


def inv_norm_minus1_to_plus1_transform(x, x_ref, config):
    global_min = x_ref.attrs['global_min'][0] if isinstance(x_ref.attrs['global_min'], (list, tuple)) else x_ref.attrs['global_min']
    global_max = x_ref.attrs['global_max'][0] if isinstance(x_ref.attrs['global_max'], (list, tuple)) else x_ref.attrs['global_max']

    if "log" in config.transforms and "total_precipitation" in x_ref.name:   
        global_min = log_transform(global_min, config.epsilon)
        global_max = log_transform(global_max, config.epsilon)
    
    x = (x + 1)/2
    x = x * (global_max - global_min) + global_min

    
    return x