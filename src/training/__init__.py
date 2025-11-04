"""
Training pipeline module for model training and optimization.

Provides unified training orchestration, custom loss functions, callbacks,
optimizers, cross-validation, and hyperparameter tuning.
"""

from .trainer import ModelTrainer, MultiModelTrainer
from .callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ReduceLROnPlateauCallback,
    MetricsLoggingCallback,
    TensorBoardCallback,
    get_callbacks
)
from .losses import (
    FocalLoss,
    WeightedCategoricalCrossentropy,
    LeakAwareLoss,
    get_loss_function
)
from .optimizers import (
    WarmupSchedule,
    CosineDecaySchedule,
    get_optimizer,
    get_learning_rate_schedule
)
from .cross_validator import (
    CrossValidator,
    KFoldValidator,
    TimeSeriesValidator,
    LeaveOneOutValidator,
    StratifiedGroupKFold,
    evaluate_with_cross_validation
)
from .hyperparameter_tuner import (
    GridSearchTuner,
    RandomSearchTuner,
    BayesianOptimizationTuner,
    HyperparameterTuner,
    suggest_hyperparameter_space
)

__all__ = [
    # Trainer
    "ModelTrainer",
    "MultiModelTrainer",
    # Callbacks
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "ReduceLROnPlateauCallback",
    "MetricsLoggingCallback",
    "TensorBoardCallback",
    "get_callbacks",
    # Losses
    "FocalLoss",
    "WeightedCategoricalCrossentropy",
    "LeakAwareLoss",
    "get_loss_function",
    # Optimizers
    "WarmupSchedule",
    "CosineDecaySchedule",
    "get_optimizer",
    "get_learning_rate_schedule",
    # Cross-validation
    "CrossValidator",
    "KFoldValidator",
    "TimeSeriesValidator",
    "LeaveOneOutValidator",
    "StratifiedGroupKFold",
    "evaluate_with_cross_validation",
    # Hyperparameter tuning
    "GridSearchTuner",
    "RandomSearchTuner",
    "BayesianOptimizationTuner",
    "HyperparameterTuner",
    "suggest_hyperparameter_space",
]