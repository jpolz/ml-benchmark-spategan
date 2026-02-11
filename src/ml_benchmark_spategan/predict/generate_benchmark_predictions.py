#!/usr/bin/env python3
"""
Generate benchmark predictions in CORDEX ML-Benchmark format.

This script generates predictions for trained models and stores them in the
format required by the CORDEX ML-Benchmark submission process.

Features:
- Supports both single predictions and ensemble generation (10 members)
- Uses templates for correct spatial coordinates and metadata
- Generates predictions for all test files in the benchmark dataset
- Outputs follow the required directory structure
- Supports YAML config file for batch prediction with multiple models

Usage:
    # Generate predictions for a single model run
    python generate_benchmark_predictions.py \\
        --run-dir runs/20260120_0546_b3kx9dyg \\
        --checkpoint best_model \\
        --output-dir ./predictions/my_submission

    # Generate ensemble predictions (10 members) for generative models
    python generate_benchmark_predictions.py \\
        --run-dir runs/20260120_0546_b3kx9dyg \\
        --checkpoint best_model \\
        --ensemble-size 10 \\
        --output-dir ./predictions/my_submission

    # Generate predictions for specific domain/variable
    python generate_benchmark_predictions.py \\
        --run-dir runs/20260120_0546_b3kx9dyg \\
        --domain NZ \\
        --var-target pr \\
        --output-dir ./predictions/my_submission

    # Use YAML config for batch prediction with multiple models
    python generate_benchmark_predictions.py \\
        --config benchmark_config.yml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import xarray as xr
import yaml
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_benchmark_spategan.analysis.model_loader import load_model
from ml_benchmark_spategan.config import config
from ml_benchmark_spategan.train.dataloader.dataloader import (
    load_cordex_data,
    load_orography,
    split_train_test,
)
from ml_benchmark_spategan.train.normalize import normalize_predictors

# Domain configuration
DOMAIN_INFO = {
    "ALPS": {"train_gcm": "CNRM-CM5", "spatial_dims": ("x", "y")},
    "NZ": {"train_gcm": "ACCESS-CM2", "spatial_dims": ("lat", "lon")},
    "SA": {"train_gcm": "ACCESS-CM2", "spatial_dims": ("lat", "lon")},
}

# Experiments
EXPERIMENTS = ["ESD_pseudo_reality", "Emulator_hist_future"]


def load_template(var_target: str, domain: str) -> xr.Dataset:
    """Load prediction template for correct coordinates and attributes."""
    template_path = (
        PROJECT_ROOT / f"format_predictions/templates/{var_target}_{domain}.nc"
    )
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return xr.open_dataset(template_path)


def get_test_predictor_files(data_path: str, domain: str) -> list:
    """Find all test predictor files for a domain."""
    test_dir = Path(data_path) / domain / f"{domain}_domain" / "test"
    predictor_files = list(test_dir.glob("**/predictors/*/*.nc"))
    return sorted(predictor_files)


def get_training_test_predictor(data_path: str, domain: str, experiment: str) -> Path:
    """
    Get the predictor file used for training-time test split.

    For ESD_pseudo_reality: Uses the training GCM predictor from the training folder
    For Emulator_hist_future: Uses historical test data

    Returns path to the predictor file.
    """
    base_path = Path(data_path) / domain / f"{domain}_domain"

    # The training predictor is what we use for train/test split during training
    train_dir = base_path / "train" / experiment / "predictors"

    # Find the main predictor file (not Static_fields.nc)
    predictor_files = [f for f in train_dir.glob("*.nc") if "Static" not in f.name]

    if predictor_files:
        return predictor_files[0]

    # Fallback to historical test if training predictor not found
    return (
        base_path
        / "test"
        / "historical"
        / "predictors"
        / "perfect"
        / f"{DOMAIN_INFO[domain]['train_gcm']}_1981-2000.nc"
    )


def parse_predictor_path(pred_path: Path) -> dict:
    """Parse predictor file path to extract metadata."""
    parts = pred_path.parts

    # Find 'test' index to navigate relative structure
    try:
        test_idx = parts.index("test")
    except ValueError:
        raise ValueError(f"Could not find 'test' in path: {pred_path}")

    # Extract components: test/{period}/predictors/{condition}/{filename}.nc
    period_folder = parts[
        test_idx + 1
    ]  # e.g., 'historical', 'mid_century', 'end_century'
    condition = parts[test_idx + 3]  # e.g., 'perfect', 'imperfect'
    filename = pred_path.name

    # Parse GCM and period from filename (e.g., 'CNRM-CM5_1981-2000.nc')
    name_parts = filename.replace(".nc", "").split("_")
    gcm = name_parts[0]
    time_period = "_".join(name_parts[1:])

    return {
        "period_folder": period_folder,
        "condition": condition,
        "filename": filename,
        "gcm": gcm,
        "time_period": time_period,
    }


def generate_single_prediction(
    model_wrapper,
    x_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Generate a single prediction (ensemble mean if applicable)."""
    x_tensor = x_tensor.to(device)

    with torch.no_grad():
        output = model_wrapper.predict(x_tensor)

    # Handle different output shapes
    if output.ndim == 4:  # (B, 1, H, W)
        output = output.squeeze(1)  # (B, H, W)

    return output.cpu().numpy()


def generate_ensemble_prediction(
    model_wrapper,
    x_tensor: torch.Tensor,
    device: torch.device,
    ensemble_size: int = 10,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Generate ensemble predictions for stochastic/generative models.

    Args:
        model_wrapper: Model wrapper with predict method
        x_tensor: Input tensor (B, C, H, W)
        device: Torch device
        ensemble_size: Number of ensemble members (default: 10 per benchmark requirement)
        noise_std: Standard deviation of noise for ensemble generation

    Returns:
        Ensemble predictions array (B, ensemble_size, H, W)
    """
    x_tensor = x_tensor.to(device)
    batch_size = x_tensor.shape[0]

    ensemble_preds = []

    with torch.no_grad():
        for _ in range(ensemble_size):
            # For generative models, each call with different noise produces different output
            # The noise is typically added inside the model's predict method
            output = model_wrapper.predict(x_tensor)

            if output.ndim == 4:  # (B, 1, H, W)
                output = output.squeeze(1)  # (B, H, W)

            ensemble_preds.append(output.cpu().numpy())

    # Stack along new ensemble dimension: (ensemble_size, B, H, W) -> (B, ensemble_size, H, W)
    ensemble_array = np.stack(ensemble_preds, axis=0)
    ensemble_array = np.transpose(ensemble_array, (1, 0, 2, 3))

    return ensemble_array


def predictions_to_xarray(
    predictions: np.ndarray,
    template: xr.Dataset,
    time_coords: xr.DataArray,
    var_target: str,
    domain: str,
    is_ensemble: bool = False,
) -> xr.Dataset:
    """
    Convert numpy predictions to xarray Dataset with proper coordinates.

    Args:
        predictions: Predictions array (time, H, W) or (time, member, H, W) for ensemble
        template: Template dataset for coordinates
        time_coords: Time coordinate values
        var_target: Variable name ('pr' or 'tasmax')
        domain: Domain name for determining spatial dimension order
        is_ensemble: Whether predictions include ensemble dimension

    Returns:
        xr.Dataset with predictions in correct format
    """
    spatial_dims = DOMAIN_INFO[domain]["spatial_dims"]

    # Get spatial coordinates from template
    coords = {
        "time": time_coords,
        spatial_dims[0]: template[spatial_dims[0]],
        spatial_dims[1]: template[spatial_dims[1]],
    }

    if is_ensemble:
        # Add member dimension
        n_members = predictions.shape[1]
        coords["member"] = np.arange(1, n_members + 1)
        dims = ("time", "member") + spatial_dims
    else:
        dims = ("time",) + spatial_dims

    # Create DataArray with template attributes
    da = xr.DataArray(
        predictions,
        coords=coords,
        dims=dims,
        name=var_target,
        attrs=template[var_target].attrs if var_target in template else {},
    )

    ds = xr.Dataset({var_target: da})

    return ds


def run_predictions_for_file(
    model_wrapper,
    predictor_path: Path,
    template: xr.Dataset,
    domain: str,
    var_target: str,
    normalization: str,
    norm_params: dict,
    device: torch.device,
    ensemble_size: Optional[int] = None,
    batch_size: int = 64,
) -> xr.Dataset:
    """
    Generate predictions for a single predictor file.

    Args:
        model_wrapper: Loaded model wrapper
        predictor_path: Path to predictor NetCDF file
        template: Template dataset for coordinates
        domain: Domain name
        var_target: Target variable
        normalization: Normalization method
        norm_params: Pre-computed normalization parameters
        device: Torch device
        ensemble_size: If provided, generate ensemble predictions
        batch_size: Batch size for inference

    Returns:
        xr.Dataset with predictions
    """
    # Load and preprocess predictor data
    ds_test = xr.open_dataset(predictor_path)
    # if domain == "SA":
    #     print(ds_test)
    #     ds_test = ds_test.drop_vars("time_bnds", errors="ignore") # Drop time_bnds if present, as it can cause issues with alignment and is not needed for prediction
    #     print(ds_test)
    # Apply normalization using pre-computed parameters
    print(ds_test.t_850.mean(["lat", "lon"]))
    # Standardize: (x - mean) / std
    if normalization == "standardization":
        ds_normalized = (ds_test - norm_params["mean"]) / norm_params["std"]
    elif normalization in ["minus1_to_plus1", "mp1p1_input_m1p1log_target"]:
        # Min-max normalize to [-1, 1]
        ds_normalized = (
            2
            * (ds_test - norm_params["min"])
            / (norm_params["max"] - norm_params["min"])
            - 1
        )
    else:
        # Default: standardization
        ds_normalized = (ds_test - norm_params["mean"]) / norm_params["std"]

    print(ds_normalized.t_850.mean(["lat", "lon"]))
    # Convert to tensor
    x_arr = ds_normalized.to_array().transpose("time", "variable", "lat", "lon")
    print(x_arr)
    x_arr = x_arr.values
    x_tensor = torch.from_numpy(x_arr).float()

    # Generate predictions in batches
    n_samples = x_tensor.shape[0]
    all_preds = []

    for i in tqdm(range(0, n_samples, batch_size), desc="Predicting", leave=False):
        batch = x_tensor[i : i + batch_size]

        if ensemble_size is not None:
            preds = generate_ensemble_prediction(
                model_wrapper, batch, device, ensemble_size
            )
        else:
            preds = generate_single_prediction(model_wrapper, batch, device)

        all_preds.append(preds)

    # Concatenate all predictions
    predictions = np.concatenate(all_preds, axis=0)

    # Convert to xarray
    ds_out = predictions_to_xarray(
        predictions,
        template,
        ds_test.time,
        var_target,
        domain,
        is_ensemble=(ensemble_size is not None),
    )

    return ds_out


def load_model_and_config(
    run_dir: Path,
    checkpoint: str,
    domain: str,
    var_target: str,
    data_path: str,
    experiment: str,
    device: torch.device,
) -> Tuple:
    """
    Load model wrapper and prepare normalization parameters.

    Returns:
        Tuple of (model_wrapper, norm_params, config)
    """
    # Load config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cf = config.load_config_from_yaml(str(config_path))

    # Determine checkpoint file
    if checkpoint in ["best_model", "best"]:
        checkpoint_name = "checkpoints/best_model.pt"
        checkpoint_epoch = "best"  # Special marker for best model
    elif checkpoint == "final":
        checkpoint_name = "checkpoints/final_models.pt"
        checkpoint_epoch = None
    else:
        # Assume it's an epoch number
        checkpoint_epoch = int(checkpoint)
        checkpoint_name = f"checkpoints/checkpoint_epoch_{checkpoint_epoch}.pt"

    # Load training data for normalization parameters
    predictor, predictand = load_cordex_data(
        domain=domain,
        training_experiment=experiment,
        var_target=var_target,
        data_path=data_path,
    )
    x_train, y_train, x_test, y_test = split_train_test(
        predictor, predictand, experiment
    )

    # Load orography if configured
    orography_da = None
    orography = None
    if cf.data.get("use_orography", False):
        orography_da = load_orography(
            domain=domain,
            training_experiment=experiment,
            data_path=data_path,
        )

    # Get normalization method and compute parameters
    normalization = cf.data.get("normalization", "standardization")
    log_base = cf.data.get("log_base", None)
    _, _, _, _, full_norm_params = normalize_predictors(
        x_train,
        x_test,
        y_train,
        y_test,
        normalization,
        orography=orography_da,
        log_base=log_base,
    )

    # Extract normalization stats for predictors
    if normalization == "standardization":
        norm_params = {
            "mean": x_train.mean("time"),
            "std": x_train.std("time"),
        }
    elif normalization in [
        "minus1_to_plus1",
        "mp1p1_input_m1p1log_target",
        "mp1p1_input_minmaxlog_target",
    ]:
        norm_params = {
            "min": x_train.min("time"),
            "max": x_train.max("time"),
        }
    else:
        norm_params = {
            "mean": x_train.mean("time"),
            "std": x_train.std("time"),
        }

    # Get orography tensor if needed
    if orography_da is not None and "orography_norm" in full_norm_params:
        orography = torch.from_numpy(full_norm_params["orography_norm"].values).float()

    # Load model
    model_wrapper = load_model(
        "gan",
        run_dir=str(run_dir),
        config=cf,
        checkpoint_epoch=checkpoint_epoch,
        device=device,
        orography=orography,
    )

    return model_wrapper, norm_params, cf, normalization


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark predictions in CORDEX ML-Benchmark format"
    )
    # Config-based mode
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to benchmark config YAML file for batch prediction with multiple models",
    )
    # Single-run mode arguments
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to model run directory (for single-run mode)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model",
        help="Checkpoint to load: 'best_model', 'final', or epoch number (default: best_model)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["SA", "NZ", "ALPS"],
        help="Domain to generate predictions for (default: from config)",
    )
    parser.add_argument(
        "--var-target",
        type=str,
        default=None,
        choices=["pr", "tasmax"],
        help="Target variable (default: from config)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENTS,
        help="Training experiment (default: from config)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/bg/fast/aihydromet/cordexbench/",
        help="Path to CORDEX benchmark data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--validation-output",
        type=str,
        default=None,
        help="Output directory for validation predictions (default: validation_predictions)",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=None,
        help="Generate ensemble predictions with N members (required for stochastic models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Generate predictions for all domains",
    )
    parser.add_argument(
        "--all-variables",
        action="store_true",
        help="Generate predictions for both pr and tasmax",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Generate only validation predictions (skip benchmark test predictions)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Generate only benchmark test predictions (skip validation predictions)",
    )

    args = parser.parse_args()

    # Check if using config-based mode
    if args.config:
        main_from_config(args)
        return

    # Single-run mode requires run-dir
    if not args.run_dir:
        parser.error("Either --config or --run-dir is required")

    # Continue with single-run mode
    main_single_run(args)


def main_from_config(args):
    """Run prediction generation from a YAML config file."""
    config_path = args.config
    print(f"Loading benchmark config from: {config_path}")

    with open(config_path, "r") as f:
        bench_config = yaml.safe_load(f)

    settings = bench_config.get("settings", {})

    # Command-line arguments override config file settings
    data_path = args.data_path or settings.get(
        "data_path", "/bg/fast/aihydromet/cordexbench/"
    )
    output_base = (
        Path(args.output_dir)
        if args.output_dir != "./predictions"
        else Path(settings.get("output_base", "benchmark_predictions"))
    )
    ensemble_size = settings.get("ensemble_size", 5)
    batch_size = args.batch_size or settings.get("batch_size", 32)

    # Control what to generate
    if args.validation_only and args.benchmark_only:
        print("ERROR: Cannot specify both --validation-only and --benchmark-only")
        return

    if args.validation_only:
        generate_validation = True
        generate_benchmark = False
    elif args.benchmark_only:
        generate_validation = False
        generate_benchmark = True
    else:
        # Default: follow config or generate both
        generate_validation = settings.get("generate_validation", True)
        generate_benchmark = settings.get("generate_benchmark", True)

    validation_output = (
        Path(args.validation_output)
        if args.validation_output
        else Path(settings.get("validation_output", "validation_predictions"))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_config = bench_config.get("models", {})
    if not models_config:
        print("No models configured in benchmark config!")
        return

    output_base.mkdir(parents=True, exist_ok=True)
    if generate_validation:
        validation_output.mkdir(parents=True, exist_ok=True)

    # Track what we're processing
    total_models = 0
    processed_models = 0

    # Handle both list and dict formats for models_config
    if isinstance(models_config, list):
        # Flat list format (from collect_runs.py)
        models_list = models_config
        total_models = len(models_list)
    else:
        # Legacy nested dict format
        models_list = []
        for domain, domain_config in models_config.items():
            if domain_config is None:
                continue
            for var_target, var_config in domain_config.items():
                if var_config is None:
                    continue
                for experiment, exp_config in var_config.items():
                    if exp_config is not None:
                        models_list.append(exp_config)
                        total_models += 1

    print(f"\nFound {total_models} model configurations to process")
    if generate_benchmark:
        print(f"Benchmark output: {output_base}")
    if generate_validation:
        print(f"Validation output: {validation_output}")
    print(f"Default ensemble size: {ensemble_size} (may be overridden per model)")
    print()

    # Group models by (domain, experiment, use_orography) to process both variables together
    from collections import defaultdict
    grouped_models = defaultdict(dict)
    for model_config in models_list:
        domain = model_config.get("domain")
        experiment = model_config.get("training_experiment")
        use_orography = model_config.get("use_orography", False)
        var_target = model_config.get("var_target")
        
        key = (domain, experiment, use_orography)
        grouped_models[key][var_target] = model_config
    print(grouped_models)
    print(f"Grouped into {len(grouped_models)} domain/experiment/orography combinations")
    print()

    # Process each group (both variables together)
    processed_models = 0
    for (domain, experiment, use_orography), var_configs in grouped_models.items():
        print(f"\n{'=' * 70}")
        print(f"Processing: {domain} / {experiment}")
        print(f"  Variables: {', '.join(var_configs.keys())}")
        print(f"  Orography: {'yes' if use_orography else 'no'}")
        print(f"{'=' * 70}")

        # Check that we have both variables
        if 'pr' not in var_configs or 'tasmax' not in var_configs:
            print(f"  WARNING: Missing one or both variables for {domain}/{experiment}")
            print(f"  Available: {list(var_configs.keys())}")
            print("  Skipping this combination - need both pr and tasmax")
            continue

        # Load models and config for both variables
        models_and_params = {}
        templates = {}
        
        try:
            for var_target, model_config in var_configs.items():
                run_dir = Path(model_config.get("run_dir"))
                checkpoint = model_config.get("checkpoint", "best_model")
                model_ensemble_size = model_config.get("ensemble_size", ensemble_size)
                
                if not run_dir.exists():
                    print(f"  ERROR: Run directory not found for {var_target}: {run_dir}")
                    raise FileNotFoundError(f"Run directory not found: {run_dir}")
                
                print(f"  Loading {var_target} model from {run_dir.name}, checkpoint: {checkpoint}")
                
                # Load model
                model_wrapper, norm_params, model_cf, normalization = load_model_and_config(
                    run_dir=run_dir,
                    checkpoint=checkpoint,
                    domain=domain,
                    var_target=var_target,
                    data_path=data_path,
                    experiment=experiment,
                    device=device,
                )
                
                models_and_params[var_target] = {
                    'model_wrapper': model_wrapper,
                    'norm_params': norm_params,
                    'normalization': normalization,
                    'ensemble_size': model_ensemble_size,
                    'run_dir': run_dir,
                    'checkpoint': checkpoint,
                }
                
                # Load template
                templates[var_target] = load_template(var_target, domain)
        
        except Exception as e:
            print(f"  ERROR loading models: {e}")
            import traceback
            traceback.print_exc()
            continue

        # === Generate validation predictions (training-time test split) ===
        # Note: Validation predictions are still saved separately per variable for backward compatibility
        if generate_validation:
            print("\n  --- Generating validation predictions ---")
            try:
                train_pred_path = get_training_test_predictor(
                    data_path, domain, experiment
                )
                print(f"  Training predictor: {train_pred_path.name}")

                for var_target, params in models_and_params.items():
                    ds_val_preds = run_predictions_for_file(
                        model_wrapper=params['model_wrapper'],
                        predictor_path=train_pred_path,
                        template=templates[var_target],
                        domain=domain,
                        var_target=var_target,
                        normalization=params['normalization'],
                        norm_params=params['norm_params'],
                        device=device,
                        ensemble_size=params['ensemble_size'],
                        batch_size=batch_size,
                    )

                    # Save to flat validation folder: validation_predictions/{domain}_{var}_{experiment}_{runid}_{checkpoint}.nc
                    checkpoint_str = params['checkpoint'].replace(".", "").replace("/", "_")
                    val_filename = f"{domain}_{var_target}_{experiment}_{params['run_dir'].name}_ep{checkpoint_str}.nc"
                    val_path = validation_output / val_filename
                    ds_val_preds.to_netcdf(val_path)
                    print(f"  Saved validation predictions for {var_target}: {val_path}")

            except Exception as e:
                print(f"  Error generating validation predictions: {e}")

        # === Generate benchmark test predictions ===
        # Combine both variables into single files
        if generate_benchmark:
            print("\n  --- Generating benchmark test predictions (combined files) ---")
            test_files = get_test_predictor_files(data_path, domain)
            print(f"  Found {len(test_files)} test predictor files")
            for pred_path in tqdm(
                test_files, desc=f"{domain}/{experiment}"
            ):
                print(parse_predictor_path(pred_path))
            for pred_path in tqdm(
                test_files, desc=f"{domain}/{experiment}"
            ):
                try:
                    path_info = parse_predictor_path(pred_path)

                    # Generate predictions for BOTH variables
                    ds_preds_by_var = {}
                    for var_target, params in models_and_params.items():
                        ds_preds = run_predictions_for_file(
                            model_wrapper=params['model_wrapper'],
                            predictor_path=pred_path,
                            template=templates[var_target],
                            domain=domain,
                            var_target=var_target,
                            normalization=params['normalization'],
                            norm_params=params['norm_params'],
                            device=device,
                            ensemble_size=params['ensemble_size'],
                            batch_size=batch_size,
                        )
                        ds_preds_by_var[var_target] = ds_preds

                    # Merge both variables into a single dataset
                    ds_combined = xr.merge([
                        ds_preds_by_var['pr'],
                        ds_preds_by_var['tasmax']
                    ])

                    # Output path: <base>/<model_name>/<Domain>_Domain/<experiment>/<period>/<condition>/
                    # Model name: spaGAN_orog or spaGAN_no_orog
                    model_name = (
                        "spaGAN_orog" if use_orography else "spaGAN_no_orog"
                    )
                    out_dir = (
                        output_base
                        / model_name
                        / f"{domain}_Domain"
                        / experiment
                        / path_info["period_folder"]
                        / path_info["condition"]
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Combined filename with both variables
                    out_filename = f"Predictions_pr_tasmax_{path_info['gcm']}_{path_info['time_period']}.nc"
                    out_path = out_dir / out_filename

                    ds_combined.to_netcdf(out_path)

                except Exception as e:
                    print(f"  Error processing {pred_path}: {e}")
                    continue

        processed_models += 1

    print(f"\n{'=' * 70}")
    print(f"Completed: {processed_models}/{total_models} model configurations")
    if generate_benchmark:
        print(f"Benchmark predictions saved to: {output_base}")
    if generate_validation:
        print(f"Validation predictions saved to: {validation_output}")
    print(f"{'=' * 70}")


def main_single_run(args):
    """Original single-run mode."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load config for defaults
    config_path = run_dir / "config.yaml"
    cf = config.load_config_from_yaml(str(config_path))

    # Determine domains to process
    if args.all_domains:
        domains = list(DOMAIN_INFO.keys())
    elif args.domain:
        domains = [args.domain]
    else:
        domains = [cf.data.domain]

    # Determine variables to process
    if args.all_variables:
        variables = ["pr", "tasmax"]
    elif args.var_target:
        variables = [args.var_target]
    else:
        variables = [cf.data.var_target]

    # Determine experiment
    experiment = args.experiment or cf.data.training_experiment

    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create validation output directory
    validation_output = Path("validation_predictions")
    validation_output.mkdir(parents=True, exist_ok=True)

    # Save run info
    run_info = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "checkpoint": args.checkpoint,
        "ensemble_size": args.ensemble_size,
        "domains": domains,
        "variables": variables,
        "experiment": experiment,
    }
    with open(output_base / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print("\nGenerating predictions:")
    print(f"  Run: {run_dir.name}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Domains: {domains}")
    print(f"  Variables: {variables}")
    print(f"  Experiment: {experiment}")
    print(f"  Ensemble size: {args.ensemble_size or 'None (deterministic)'}")
    print(f"  Benchmark output: {output_base}")
    print(f"  Validation output: {validation_output}")
    print()

    # Process each domain and variable
    for domain in domains:
        print(f"\n{'=' * 60}")
        print(f"Processing Domain: {domain}")
        print(f"{'=' * 60}")

        for var_target in variables:
            print(f"\n--- Variable: {var_target} ---")

            # Load model and normalization parameters
            try:
                model_wrapper, norm_params, model_config, normalization = (
                    load_model_and_config(
                        run_dir=run_dir,
                        checkpoint=args.checkpoint,
                        domain=domain,
                        var_target=var_target,
                        data_path=args.data_path,
                        experiment=experiment,
                        device=device,
                    )
                )
            except Exception as e:
                print(f"Error loading model for {domain}/{var_target}: {e}")
                continue

            # Load template
            template = load_template(var_target, domain)

            # === Generate validation predictions (training-time test split) ===
            print("\n  --- Generating validation predictions ---")
            try:
                train_pred_path = get_training_test_predictor(
                    args.data_path, domain, experiment
                )
                print(f"  Training predictor: {train_pred_path.name}")

                ds_val_preds = run_predictions_for_file(
                    model_wrapper=model_wrapper,
                    predictor_path=train_pred_path,
                    template=template,
                    domain=domain,
                    var_target=var_target,
                    normalization=normalization,
                    norm_params=norm_params,
                    device=device,
                    ensemble_size=args.ensemble_size,
                    batch_size=args.batch_size,
                )

                # Save to flat validation folder
                checkpoint_str = args.checkpoint.replace(".", "").replace("/", "_")
                val_filename = f"{domain}_{var_target}_{experiment}_{run_dir.name}_ep{checkpoint_str}.nc"
                val_path = validation_output / val_filename
                ds_val_preds.to_netcdf(val_path)
                print(f"  Saved validation predictions: {val_path}")

            except Exception as e:
                print(f"  Error generating validation predictions: {e}")

            # === Generate benchmark test predictions ===
            print("\n  --- Generating benchmark test predictions ---")
            # Find all test predictor files
            test_files = get_test_predictor_files(args.data_path, domain)
            print(f"  Found {len(test_files)} test predictor files")

            for pred_path in tqdm(test_files, desc=f"{domain}/{var_target}"):
                try:
                    # Parse path metadata
                    path_info = parse_predictor_path(pred_path)
                    print(f"\nProcessing: {pred_path.name}")

                    # Generate predictions
                    ds_preds = run_predictions_for_file(
                        model_wrapper=model_wrapper,
                        predictor_path=pred_path,
                        template=template,
                        domain=domain,
                        var_target=var_target,
                        normalization=normalization,
                        norm_params=norm_params,
                        device=device,
                        ensemble_size=args.ensemble_size,
                        batch_size=args.batch_size,
                    )

                    # Build output path following benchmark structure
                    # <submission>/Domain_Domain/Experiment/period/condition/
                    out_dir = (
                        output_base
                        / f"{domain}_Domain"
                        / experiment
                        / path_info["period_folder"]
                        / path_info["condition"]
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)

                    # Output filename: Predictions_{var}_{GCM}_{period}.nc
                    out_filename = f"Predictions_{var_target}_{path_info['gcm']}_{path_info['time_period']}.nc"
                    out_path = out_dir / out_filename

                    # Save predictions
                    ds_preds.to_netcdf(out_path)

                except Exception as e:
                    print(f"Error processing {pred_path}: {e}")
                    continue

    print(f"\n{'=' * 60}")
    print("Prediction generation complete!")
    print(f"Benchmark predictions saved to: {output_base}")
    print(f"Validation predictions saved to: {validation_output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
