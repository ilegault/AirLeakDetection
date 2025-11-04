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
        
        # TODO: Implement actual training logic
        # This would call the appropriate training module
        logger.info("Training logic to be implemented with Phase 4 (Training Pipeline)")
        
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