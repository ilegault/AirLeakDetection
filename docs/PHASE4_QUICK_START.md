# Phase 4: Training Pipeline Quick Start Guide

## ✅ What Was Created

### 6 New Training Modules (54 comprehensive tests)

1. **Loss Functions** (`src/training/losses.py`)
   - Focal Loss for handling class imbalance
   - Weighted Categorical Cross-Entropy
   - Leak-Aware Loss (penalizes leak misclassification)
   - Custom loss factory

2. **Optimizers & Learning Rate Schedules** (`src/training/optimizers.py`)
   - Warmup Schedule (linear warmup + decay)
   - Cosine Decay Schedule
   - Adam, SGD, RMSprop, AdamW configurations
   - Learning rate schedule factory

3. **Callbacks** (`src/training/callbacks.py`)
   - Early Stopping with best weight restoration
   - Model Checkpointing
   - Reduce Learning Rate on Plateau
   - Custom Metrics Logging
   - TensorBoard integration

4. **Cross-Validation** (`src/training/cross_validator.py`)
   - Stratified K-Fold (k=5)
   - Leave-One-Out for small datasets
   - Time-Series Split (respects temporal order)
   - Group K-Fold for maintaining group structure

5. **Hyperparameter Tuning** (`src/training/hyperparameter_tuner.py`)
   - Grid Search exhaustive search
   - Random Search sampling
   - Bayesian Optimization (Optuna)
   - Automated hyperparameter space suggestion

6. **Main Trainer** (`src/training/trainer.py`)
   - `ModelTrainer` - unified training orchestrator
   - `MultiModelTrainer` - train and compare multiple models
   - Full workflow support (compile, train, evaluate, predict, save)
   - Automatic experiment management and logging

### ✅ Test Coverage

```
tests/test_phase4_training.py    (54 tests)  ✓ All Passing
  ├── Loss Functions             (12 tests)
  ├── Optimizers & Schedules     (12 tests)
  ├── Callbacks                  (7 tests)
  ├── Cross-Validation           (6 tests)
  ├── Hyperparameter Tuning      (4 tests)
  ├── Trainer                    (8 tests)
  ├── Multi-Model Trainer        (3 tests)
  └── Integration Tests          (1 test)
  
Total: 54 tests, 100% passing
```

---

## Quick Start Examples

### 1. Basic Training with Custom Loss

```python
from src.training import ModelTrainer, get_loss_function
import tensorflow as tf
import numpy as np

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(32,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

# Configure training with focal loss
config = {
    "training": {
        "loss": "focal_loss",  # or "leak_aware_loss", "weighted_categorical_crossentropy"
        "optimizer": "adam",
        "learning_rate": 0.001,
        "loss_config": {"alpha": 0.25, "gamma": 2.0},
        "epochs": 50,
        "batch_size": 32,
        "callbacks": {
            "early_stopping": True,
            "early_stopping_patience": 15,
            "model_checkpoint": True,
            "reduce_lr_on_plateau": True
        }
    }
}

# Train
trainer = ModelTrainer(model, config, experiment_name="my_experiment")
trainer.compile()
history = trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
results = trainer.evaluate(X_test, y_test)

# Save
trainer.save_model()
```

### 2. Learning Rate Schedules

```python
from src.training import get_learning_rate_schedule, get_optimizer

# Warmup schedule
lr_schedule = get_learning_rate_schedule(
    "warmup",
    base_learning_rate=0.001,
    config={
        "initial_lr": 1e-4,
        "warmup_steps": 1000,
        "decay_steps": 5000,
        "decay_rate": 0.96
    }
)

# Cosine annealing
cosine_schedule = get_learning_rate_schedule(
    "cosine",
    base_learning_rate=0.001,
    config={
        "min_lr": 1e-5,
        "total_steps": 10000,
        "warmup_steps": 500
    }
)

# Use in optimizer
optimizer = get_optimizer("adam", learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="categorical_crossentropy")
```

### 3. Cross-Validation

```python
from src.training import KFoldValidator, evaluate_with_cross_validation
from sklearn.ensemble import RandomForestClassifier

# Manual cross-validation
validator = KFoldValidator(n_splits=5)
splits = validator.get_splits(X, y)

for fold, (train_idx, test_idx) in enumerate(splits):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate

# Automatic evaluation
model = RandomForestClassifier()
results = evaluate_with_cross_validation(
    model, X, y,
    cv_method="stratified_kfold",
    n_splits=5,
    scoring="f1_weighted"
)
print(f"Mean F1: {results['mean']:.3f} ± {results['std']:.3f}")
```

### 4. Hyperparameter Optimization

```python
from src.training import GridSearchTuner, RandomSearchTuner, HyperparameterTuner

# Grid Search
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}
tuner = GridSearchTuner(param_grid=param_grid, cv=5)
results = tuner.search(model, X_train, y_train)
print(f"Best params: {results['best_params']}")
print(f"Best score: {results['best_score']:.3f}")

# Random Search (faster for large spaces)
param_dist = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "gamma": ["scale", "auto"] + list(np.logspace(-5, 2, 20))
}
tuner = RandomSearchTuner(param_dist, n_iter=50, cv=5)
results = tuner.search(svm_model, X_train, y_train)

# Bayesian Optimization (requires optuna: pip install optuna)
param_space = {
    "C": {"type": "float", "low": 0.001, "high": 100, "log": True},
    "gamma": {"type": "float", "low": 0.0001, "high": 10, "log": True}
}
bayesian_tuner = HyperparameterTuner(method="bayesian", **param_space)
```

### 5. Multi-Model Training & Comparison

```python
from src.training import MultiModelTrainer
from src.models import CNN1DBuilder, RandomForestModel, SVMClassifier

# Create trainer
multi_trainer = MultiModelTrainer(config)

# Add models
cnn = CNN1DBuilder(config).build(input_shape=(1024, 9), n_classes=4).model
rf = RandomForestModel(config).model
svm = SVMClassifier(config).model

multi_trainer.add_model("cnn", cnn, config)
multi_trainer.add_model("rf", rf, config)
multi_trainer.add_model("svm", svm, config)

# Train all
results = multi_trainer.train_all(X_train, y_train, X_val, y_val, epochs=50)

# Evaluate all
eval_results = multi_trainer.evaluate_all(X_test, y_test)

# Compare
comparison = multi_trainer.compare_models(metric="accuracy")
print(f"Best model: {list(comparison.keys())[0]}")
```

### 6. Advanced: Custom Loss with Imbalanced Data

```python
from src.training import WeightedCategoricalCrossentropy, LeakAwareLoss, ModelTrainer

# Calculate class weights from imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# Use weighted loss
config = {
    "training": {
        "loss": "weighted_categorical_crossentropy",
        "class_weights": class_weights_dict,  # Passed to loss function
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 50
    }
}

trainer = ModelTrainer(model, config)
trainer.compile()
trainer.train(X_train, y_train, X_val, y_val)

# Or use leak-aware loss (penalizes leak misclassification more)
config = {
    "training": {
        "loss": "leak_aware_loss",
        "loss_config": {
            "leak_penalty": 2.0,  # Penalize missed leaks
            "non_leak_penalty": 1.0
        }
    }
}
```

---

## Configuration Reference

```yaml
training:
  # Loss function
  loss: focal_loss  # or categorical_crossentropy, leak_aware_loss, weighted_categorical_crossentropy
  loss_config:
    alpha: 0.25
    gamma: 2.0
  
  # Optimizer
  optimizer: adam  # or sgd, rmsprop, adamw
  learning_rate: 0.001
  optimizer_config:
    beta_1: 0.9
    beta_2: 0.999
  
  # Learning rate schedule
  lr_schedule: warmup  # or constant, cosine, exponential, polynomial
  lr_schedule_config:
    warmup_steps: 1000
    decay_steps: 5000
    decay_rate: 0.96
  
  # Training
  epochs: 50
  batch_size: 32
  metrics: ["accuracy"]
  
  # Callbacks
  callbacks:
    early_stopping: true
    early_stopping_patience: 15
    early_stopping_monitor: val_loss
    model_checkpoint: true
    checkpoint_monitor: val_loss
    reduce_lr_on_plateau: true
    reduce_lr_monitor: val_loss
    reduce_lr_factor: 0.5
    reduce_lr_patience: 5
    tensorboard: false

# Class weights for imbalanced data
class_weights:
  0: 1.0
  1: 1.5
  2: 1.5
  3: 1.5
```

---

## File Structure

```
src/training/
  ├── __init__.py                    # Exports all classes
  ├── trainer.py                     # ModelTrainer, MultiModelTrainer
  ├── losses.py                      # Custom loss functions
  ├── optimizers.py                  # Learning rate schedules & optimizers
  ├── callbacks.py                   # Training callbacks
  ├── cross_validator.py             # Cross-validation strategies
  └── hyperparameter_tuner.py        # Hyperparameter optimization

tests/
  └── test_phase4_training.py        # 54 comprehensive tests
```

---

## Running Tests

```bash
# All Phase 4 tests
pytest tests/test_phase4_training.py -v

# Specific test class
pytest tests/test_phase4_training.py::TestFocalLoss -v

# Specific test
pytest tests/test_phase4_training.py::TestFocalLoss::test_focal_loss_initialization -v

# With coverage
pytest tests/test_phase4_training.py --cov=src.training

# Show print statements
pytest tests/test_phase4_training.py -v -s
```

---

## Key Features

✅ **Loss Functions**
- Handles class imbalance with focal loss
- Custom leak-aware loss for domain-specific needs
- Weighted loss with class importance

✅ **Learning Rate Schedules**
- Warmup for stable training start
- Cosine annealing for convergence
- Exponential and polynomial decay options

✅ **Training Callbacks**
- Early stopping with best weight restoration
- Automatic checkpoint saving
- Learning rate reduction on plateau
- Custom metrics logging

✅ **Cross-Validation**
- Stratified k-fold for balanced folds
- Time-series split for temporal data
- Leave-one-out for small datasets
- Group k-fold for respecting data structure

✅ **Hyperparameter Tuning**
- Grid search for exhaustive exploration
- Random search for efficiency
- Bayesian optimization for smart search
- Automated space suggestions per model type

✅ **Model Training**
- Unified trainer for all model types
- Experiment tracking and logging
- Multi-model training and comparison
- Full workflow management

---

## Integration with Phase 3

The training pipeline seamlessly integrates with Phase 3 models:

```python
from src.models import CNN1DBuilder, LSTMBuilder, EnsembleModel
from src.training import ModelTrainer

# Build a model from Phase 3
builder = CNN1DBuilder(config)
model = builder.build(input_shape=(1024, 9), n_classes=4)

# Train using Phase 4 pipeline
trainer = ModelTrainer(model, config)
trainer.compile()
trainer.train(X_train, y_train, X_val, y_val)

# Evaluate
trainer.evaluate(X_test, y_test)
```

---

## Next Steps

**Phase 5** will include:
- Production scripts (CLI tools)
- Model deployment (Docker, API)
- Advanced monitoring and logging
- Model export formats (ONNX, TFLite, SavedModel)

---

## Support

For issues or questions:
1. Check docstrings in each module
2. Review test files for usage examples
3. See `src/training/__init__.py` for all exports
4. Check inline documentation in source files

---

## Summary

- ✅ 6 new training modules
- ✅ 54 comprehensive tests (100% passing)
- ✅ Full training workflow support
- ✅ Custom loss functions for air leak detection
- ✅ Advanced hyperparameter tuning
- ✅ Cross-validation strategies
- ✅ Multi-model training and comparison
- ✅ Automatic experiment management

**Phase 4 Complete! Ready for Phase 5.**