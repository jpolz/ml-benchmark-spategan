# Adding a New Model

Quick guide to implementing a new model in this codebase.

## Step 1: Create the Model Class

Create a new file in `src/ml_benchmark_spategan/model/` (e.g., `mymodel.py`):

```python
import torch
import torch.nn as nn
from .base import BaseModel

class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Define your architecture
        self.layers = nn.Sequential(
            nn.Conv2d(...),
            nn.ReLU(),
            # ... more layers
        )
    
    def forward(self, x):
        """Standard PyTorch forward pass."""
        return self.layers(x)
    
    def train_step(self, batch, optimizers, criterion, scaler, config, **kwargs):
        """Single training iteration."""
        x, y = batch
        optimizer = optimizers['model']
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            output = self(x)
            loss = criterion(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        return {'loss': loss.item()}
    
    def predict_step(self, x, **kwargs):
        """Inference without denormalization."""
        with torch.no_grad():
            return self(x)
```

## Step 2: Create the Wrapper Class

In the same file, add a wrapper for inference:

```python
from pathlib import Path
from .base import BaseWrapper

class MyModelWrapper(BaseWrapper):
    def __init__(self, run_dir, checkpoint_name='final_model.pth', device=None):
        super().__init__(run_dir, checkpoint_name, device)
        # Store any model-specific parameters
        self.config = self._load_config()
        self._load_model()
    
    def _load_model(self):
        """Load model architecture and weights."""
        self.model = MyModel(self.config)
        checkpoint_path = self.run_dir / self.checkpoint_name
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        )
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, x):
        """Generate predictions with denormalization."""
        # Run inference
        y_pred = self.model.predict_step(x)
        
        # Denormalize if needed
        from ..utils.normalize import denormalize_predictions
        y_pred = denormalize_predictions(y_pred, **self.norm_params)
        
        return y_pred
```

## Step 3: Add to Training Registry (Optional)

If you want to use your model in training, add it to `model/registry.py`:

```python
def create_generator(architecture, config):
    """Factory function for creating generator models."""
    if architecture == "mymodel":
        from .mymodel import MyModel
        return MyModel(config)
    # ... existing cases
```

## Step 4: Add to Model Loader (Optional)

If you want to use your model in evaluation scripts, add it to `analysis/model_loader.py`:

```python
def load_model(model_type, **kwargs):
    if model_type.lower() == "mymodel":
        from ml_benchmark_spategan.model.mymodel import MyModelWrapper
        return MyModelWrapper(
            run_dir=kwargs["run_dir"],
            checkpoint_name=kwargs.get("checkpoint_name", "final_model.pth"),
            device=kwargs.get("device"),
        )
    # ... existing cases
```

## That's It!

Your model now:
- ✅ Follows the standard interface
- ✅ Can be trained with existing training scripts
- ✅ Can be evaluated with existing evaluation scripts
- ✅ Works with all existing visualization and diagnostic tools

## Key Points

1. **Model class inherits from `BaseModel`**: Implement `forward()`, `train_step()`, `predict_step()`
2. **Wrapper class inherits from `BaseWrapper`**: Implement `_load_model()` and `predict()`
3. **`train_step()` returns dict of losses**: `{'loss': value}` or `{'gen_loss': v1, 'disc_loss': v2}`
4. **`predict_step()` is raw output**: Denormalization happens in wrapper's `predict()`
5. **Use mixed precision**: Wrap forward pass in `torch.amp.autocast('cuda')`

See `model/deepesd.py` for a complete simple example.
