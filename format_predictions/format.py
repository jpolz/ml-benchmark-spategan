import xarray as xr

def set_cordex_ml_benchmark_attributes(ds, attributes_dict):

    """
    Set CORDEX ML-Benchmark-specific global attributes on an xarray Dataset.
    This function removes all existing global attributes and replaces them with
    only the CORDEX ML-Benchmark-compliant metadata.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to modify
    attributes_dict : dict
        Dictionary containing the attributes to set. Required keys:
        - 'project_id': Should be 'CORDEX'
        - 'activity_id': Should be 'ML-Benchmark'
        - 'product': Should be 'emulator-output'
        - 'benchmark_id': Identifier for the benchmark version
        - 'institution_id': Short acronym for the institution responsible for predictions
        - 'institution': Full name of institution responsible for the emulated data sets
        - 'contact': Contact information (name, email optional) of the institution or researcher that is
           responsible for the predictions 
        - 'creation_date': Date of creation (e.g., 2025-03-20)
        - 'emulator_id': Short identifier for the emulation method family
        - 'emulator': Description of the emulation method (architecture, key features, etc.)
        - 'training_id': Short code for the specific training configuration
        - 'training': Full description of training setup (predictors, predictands, domains, time periods, pre-processing, loss, etc.)
        - 'stochastic_output': Does the emulator generate stochastic/probabilistic realizations? (yes or no)
        - 'version_realization': Description of the realization(s) used to obtain the final prediction. If the model is deterministic,
           this field can be left empty. When there are M realizations and we refer to a specific run vX, we denote it as vX-rM.
           If we aggregate over M realizations (for example, their mean) instead of the individual realizations, we write vX-aggM
        - 'version_realization_info': Optional additional information about version_realization field. For example: the realizations
           were aggregated using the standard mean

        - 'reference_url': Reference information about the model used (e.g., DOI or publication URL)
        - 'reproducibility_url': Reproducibility information about the model used (e.g., GitHub repository)

    Returns:
    --------
    xarray.Dataset
        Dataset with updated attributes
    """

    # Create a copy to avoid modifying the original
    ds_copy = ds.copy()

    # Clear all existing global attributes
    ds_copy.attrs.clear()

    # Set all provided attributes
    for key, value in attributes_dict.items():
        ds_copy.attrs[key] = value

    return ds_copy

# Example usage
if __name__ == "__main__":

    # Load the dataset
    data = xr.open_dataset('./example.nc')

    # Define the CORDEX ML-Benchmark attributes
    cordex_attrs = {
        # General attributes
        'project_id': 'CORDEX',
        'activity_id': 'ML-Benchmark',
        'product': 'emulator-output',
        'benchmark_id': 'v1.0',

        # Institution information
        'institution_id': 'IFCA',
        'institution': 'Instituto de Física de Cantabria (IFCA), CSIC-Universidad de Cantabria',
        'contact': 'Contact person, email@example.com',
        'creation_date': '2025-03-20',

        # Emulator information
        'emulator_id': 'DeepESD',
        'emulator': 'Deep convolutional neural network including 3 convolution and one dense layer, with ReLU activation functions.',

        # Training configuration
        'training_id': 'm1',
        'training': (
            'Standardized input data at gridbox level using mean/std of reanalysis in training period. '
            'No bias adjustment performed. Training on historical and future experiments.'
        ),

        # Output characteristics
        'stochastic_output': 'no',
        'version_realization': '',
        'version_realization_info': '',

        # Reference and reproducibility
        'reference_url': 'https://doi.org/10.5194/gmd-15-6747-2022',
        'reproducibility_url': 'https://zenodo.org/records/6828304'
    }

    # Create the attributes
    data_with_attrs = set_cordex_ml_benchmark_attributes(data, cordex_attrs)