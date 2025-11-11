#!/usr/bin/env python3
"""
Train an air leak detection model.

This script trains a selected model type on prepared data with optional
experiment tracking via MLflow or WandB. Supports multiple model architectures
and hyperparameter configurations.

Usage:
    python scripts/train_model.py --model-type cnn_1d --data-path data/processed/
    python scripts/train_model.py --model-type random_forest --epochs 50 --batch-size 16
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, ConfigManager, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create and return argument parser for training script."""
    parser = argparse.ArgumentParser(
        description="Train an air leak detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train CNN with default settings
  python scripts/train_model.py --model-type cnn_1d

  # Train Random Forest with custom hyperparameters
  python scripts/train_model.py --model-type random_forest --epochs 100

  # Train with specific data and save to output directory
  python scripts/train_model.py \\
      --model-type cnn_1d \\
      --data-path data/processed/ \\
      --output-dir models/
        """
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_1d",
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost", "ensemble"],
        help="Type of model to train (default: cnn_1d)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/",
        help="Path to processed data directory (default: data/processed/)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Output directory for trained model (default: models/)"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    
    parser.add_argument(
        "--use-fft",
        action="store_true",
        help="Use FFT features instead of raw data"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for tracking (MLflow/WandB)"
    )
    
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        choices=["mlflow", "wandb"],
        help="Enable experiment tracking with MLflow or WandB"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging output"
    )
    
    return parser


def validate_inputs(args) -> bool:
    """Validate command-line arguments."""
    logger = get_logger(__name__)
    
    # Check data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {args.data_path}")
        return False
    
    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Config file not found, using defaults: {args.config}")
    
    # Validate numeric arguments
    if args.epochs <= 0:
        logger.error(f"Epochs must be positive: {args.epochs}")
        return False
    
    if args.batch_size <= 0:
        logger.error(f"Batch size must be positive: {args.batch_size}")
        return False
    
    if not (0 < args.learning_rate < 1):
        logger.error(f"Learning rate must be between 0 and 1: {args.learning_rate}")
        return False
    
    if not (0 < args.validation_split < 1):
        logger.error(f"Validation split must be between 0 and 1: {args.validation_split}")
        return False
    
    return True


def setup_output_directory(output_dir: str) -> Path:
    """Create and setup output directory for model."""
    output_path = Path(output_dir)
    FileUtils.ensure_directory(str(output_path))
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_path / f"model_{timestamp}"
    FileUtils.ensure_directory(str(model_dir))
    
    return model_dir


def setup_experiment_tracking(args):
    """Setup experiment tracking if requested."""
    logger = get_logger(__name__)
    
    if args.tracking_uri == "mlflow":
        try:
            import mlflow
            mlflow.set_experiment(args.experiment_name or "default")
            logger.info("MLflow tracking enabled")
            return mlflow
        except ImportError:
            logger.warning("MLflow not installed, tracking disabled")
            return None
    
    elif args.tracking_uri == "wandb":
        try:
            import wandb
            wandb.init(project="air-leak-detection", name=args.experiment_name or "default")
            logger.info("WandB tracking enabled")
            return wandb
        except ImportError:
            logger.warning("WandB not installed, tracking disabled")
            return None
    
    return None


def train_model(args):
    """Train the specified model."""
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config = ConfigManager(args.config) if Path(args.config).exists() else None
        
        # Setup output directory
        model_dir = setup_output_directory(args.output_dir)
        logger.info(f"Model output directory: {model_dir}")
        
        # Setup experiment tracking
        tracker = setup_experiment_tracking(args)
        
        # Log training parameters
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.learning_rate}")
        logger.info(f"Using FFT: {args.use_fft}")

        # Import necessary modules
        import numpy as np
        import json
        from src.training.trainer import ModelTrainer
        from src.training.callbacks import get_callbacks
        from src.models.cnn_1d import CNN1DBuilder
        from src.models.cnn_2d import CNN2DBuilder
        from src.models.lstm_model import LSTMBuilder
        from src.models.random_forest import RandomForestModel
        from src.models.svm_classifier import SVMClassifier

        # Load training data
        logger.info("Loading training data...")
        data_path_obj = Path(args.data_path)

        # Determine which features to use
        feature_file = "fft_features.npy" if args.use_fft else "signals.npy"

        X_train = np.load(data_path_obj / "train" / feature_file)
        y_train = np.load(data_path_obj / "train" / "labels.npy")
        X_val = np.load(data_path_obj / "val" / feature_file)
        y_val = np.load(data_path_obj / "val" / "labels.npy")

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        logger.info(f"Number of classes: {len(np.unique(y_train))}")

        num_classes = len(np.unique(y_train))

        # Build model based on type
        logger.info(f"Building {args.model_type} model...")

        # Build config for models
        model_config = {
            "training": {
                "learning_rate": args.learning_rate,
            },
            "model": {
                "conv_filters": [32, 64, 128],
                "kernel_sizes": [7, 5, 3],
                "dense_units": [256, 128],
                "dropout_rates": [0.3, 0.3, 0.4, 0.3, 0.3],  # 5 rates: 3 conv + 2 dense
                "cnn_2d": {
                    "conv_filters": [32, 64, 128],
                    "kernel_sizes": [[3, 3], [3, 3], [3, 3]],
                    "dense_units": [256, 128],
                    "dropout_rates": [0.3, 0.3, 0.4, 0.3, 0.3],  # 5 rates: 3 conv + 2 dense
                },
                "lstm": {
                    "lstm_units": [64, 32],
                    "dense_units": [128, 64],
                    "dropout_rates": [0.2, 0.3, 0.3, 0.3],  # 4 rates: 2 lstm + 2 dense
                    "bidirectional": True,
                },
                "random_forest": {
                    "n_estimators": 300,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "max_features": "sqrt",
                    "random_state": 42,
                    "n_jobs": -1,
                },
                "svm": {
                    "kernel": "rbf",
                    "C": 1.0,
                    "gamma": "scale",
                    "probability": True,
                    "random_state": 42,
                    "class_weight": None,
                },
            }
        }

        if args.model_type == "cnn_1d":
            builder = CNN1DBuilder(model_config)
            model = builder.build(
                input_shape=X_train.shape[1:],
                n_classes=num_classes
            )
            is_deep_learning = True

        elif args.model_type == "cnn_2d":
            builder = CNN2DBuilder(model_config)
            # Reshape for 2D CNN: (samples, timesteps, channels) -> (samples, timesteps, channels, 1)
            if len(X_train.shape) == 3:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
            elif len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, 1)
            model = builder.build(
                input_shape=X_train.shape[1:],
                n_classes=num_classes
            )
            is_deep_learning = True

        elif args.model_type == "lstm":
            builder = LSTMBuilder(model_config)
            # Ensure 3D shape for LSTM (samples, timesteps, features)
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            model = builder.build(
                input_shape=X_train.shape[1:],
                n_classes=num_classes
            )
            is_deep_learning = True

        elif args.model_type == "random_forest":
            model = RandomForestModel(model_config)
            is_deep_learning = False

        elif args.model_type == "svm":
            model = SVMClassifier(model_config)
            is_deep_learning = False

        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            return 1

        # Train model
        if is_deep_learning:
            # Use ModelTrainer for deep learning models
            trainer = ModelTrainer(model=model, config=model_config, experiment_name=args.model_type)

            # Compile model
            trainer.compile(
                loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            # Get callbacks
            callbacks_config = {
                "early_stopping": True,
                "early_stopping_patience": 10,
                "model_checkpoint": True,
                "reduce_lr_on_plateau": True,
                "reduce_lr_patience": 5,
            }
            callbacks = get_callbacks(
                config=callbacks_config,
                checkpoint_dir=str(model_dir / "checkpoints")
            )

            # Train
            history = trainer.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks
            )

            # Save model
            model_path = model_dir / f"{args.model_type}_model.h5"
            trainer.save_checkpoint(str(model_path))
            logger.info(f"Model saved to {model_path}")

        else:
            # Traditional ML models
            # Flatten features if needed
            if len(X_train.shape) > 2:
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
            else:
                X_train_flat = X_train
                X_val_flat = X_val

            logger.info("Training model...")
            
            # Train using the wrapper class's fit method
            model.fit(X_train_flat, y_train)
            
            # Evaluate on validation set
            if args.model_type == "random_forest":
                val_accuracy, _ = model.evaluate(X_val_flat, y_val)
            elif args.model_type == "svm":
                val_accuracy = model.model.score(X_val_flat, y_val)
            
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

            # Save model
            model_path = model_dir / f"{args.model_type}_model.pkl"
            model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")

        # Save training parameters to output directory
        params_file = model_dir / "training_params.yaml"
        training_params = {
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_split": args.validation_split,
            "use_fft": args.use_fft,
            "timestamp": datetime.now().isoformat()
        }
        
        import yaml
        with open(params_file, 'w') as f:
            yaml.dump(training_params, f)
        
        logger.info(f"Training parameters saved to {params_file}")
        logger.info("Training completed successfully")
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    # Validate inputs
    if not validate_inputs(args):
        logger.error("Input validation failed")
        return 1
    
    # Train model
    logger.info("=" * 60)
    logger.info("TRAINING SCRIPT - Train Air Leak Detection Model")
    logger.info("=" * 60)
    
    return train_model(args)


if __name__ == "__main__":
    sys.exit(main())