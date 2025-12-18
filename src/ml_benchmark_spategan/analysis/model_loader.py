"""Simple registry for loading different model types for inference."""

import torch


def load_model(model_type: str, **kwargs):
    """
    Factory function to load models for inference.

    Args:
        model_type: Type of model ('deepesd', 'gan')
        **kwargs: Model-specific arguments
            - For 'deepesd': model_path, x_shape, y_shape, filters_last_conv, device
            - For 'gan': run_dir, config, checkpoint_epoch, device

    Returns:
        Model wrapper instance with predict() method

    Example:
        >>> # Load DeepESD model
        >>> model = load_model('deepesd',
        ...                    model_path='path/to/model.pt',
        ...                    x_shape=(1, 15, 16, 16),
        ...                    y_shape=(1, 16384))
        >>>
        >>> # Load GAN model
        >>> model = load_model('gan',
        ...                    run_dir='runs/20251218_0211_zjh10zws',
        ...                    config=config_obj)
    """
    device = kwargs.get(
        "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if model_type.lower() == "deepesd":
        from ml_benchmark_spategan.model.deepesd import DeepESDWrapper

        return DeepESDWrapper(
            model_path=kwargs["model_path"],
            x_shape=kwargs["x_shape"],
            y_shape=kwargs["y_shape"],
            filters_last_conv=kwargs.get("filters_last_conv", 1),
            device=device,
        )

    elif model_type.lower() == "gan":
        from ml_benchmark_spategan.model.spagan2d import SpaGANWrapper

        return SpaGANWrapper(
            run_dir=kwargs["run_dir"],
            config=kwargs["config"],
            checkpoint_epoch=kwargs.get("checkpoint_epoch", None),
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Supported types: 'deepesd', 'gan'"
        )
