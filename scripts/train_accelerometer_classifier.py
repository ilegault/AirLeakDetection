#!/usr/bin/env python3
"""
Train Stage 1 classifier for multi-accelerometer position identification.

This script trains models (Random Forest, SVM, etc.) to identify which accelerometer
position (0, 1, or 2) is closest to a leak source. This is Stage 1 of the two-stage
leak detection system.

Multi-Accelerometer Setup:
    - 3 accelerometers record simultaneously at different positions
    - Position 0: Closest to leak source (strongest signal)
    - Position 1: Middle distance
    - Position 2: Farthest from leak source

The trained classifier learns to identify which position has characteristics indicating
it's closest to the leak (e.g., highest signal amplitude, specific frequency patterns).

Usage:
    python scripts/train_accelerometer_classifier.py --data-path data/accelerometer_classifier/ --model-type random_forest
    python scripts/train_accelerometer_classifier.py --data-path data/accelerometer_classifier/ --model-type svm
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils
from src.models.random_forest import RandomForestModel
from src.models.svm_classifier import SVMClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Train accelerometer classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Random Forest
  python scripts/train_accelerometer_classifier.py --data-path data/accelerometer_classifier/ --model-type random_forest

  # Train SVM
  python scripts/train_accelerometer_classifier.py --data-path data/accelerometer_classifier/ --model-type svm

  # Train with custom output directory
  python scripts/train_accelerometer_classifier.py \\
      --data-path data/accelerometer_classifier/ \\
      --model-type random_forest \\
      --output-dir models/accelerometer_classifier/
        """
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to accelerometer classification data directory"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm"],
        help="Type of model to train (default: random_forest)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/accelerometer_classifier/",
        help="Output directory for trained model (default: models/accelerometer_classifier/)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    return parser


def train_accelerometer_classifier(args) -> int:
    """Train the accelerometer classifier."""
    try:
        data_path = Path(args.data_path)
        output_dir = Path(args.output_dir)

        if not data_path.exists():
            LOGGER.error(f"Data path not found: {data_path}")
            return 1

        # Create output directory
        FileUtils.ensure_directory(str(output_dir))

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = output_dir / f"model_{timestamp}"
        FileUtils.ensure_directory(str(model_dir))

        LOGGER.info("="*60)
        LOGGER.info("ACCELEROMETER CLASSIFIER TRAINING")
        LOGGER.info("="*60)
        LOGGER.info(f"Data path: {data_path}")
        LOGGER.info(f"Model type: {args.model_type}")
        LOGGER.info(f"Output directory: {model_dir}")

        # Load data
        LOGGER.info("\nLoading training data...")
        X_train = np.load(data_path / "train" / "features.npy")
        y_train = np.load(data_path / "train" / "labels.npy")
        X_val = np.load(data_path / "val" / "features.npy")
        y_val = np.load(data_path / "val" / "labels.npy")

        LOGGER.info(f"Training data shape: {X_train.shape}")
        LOGGER.info(f"Validation data shape: {X_val.shape}")
        LOGGER.info(f"Number of classes: {len(np.unique(y_train))}")

        # STEP 1 DIAGNOSTICS: Print unique labels
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("STEP 1: DATA LOADING DIAGNOSTICS")
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"Unique labels in training set: {np.unique(y_train)}")
        LOGGER.info(f"Unique labels in validation set: {np.unique(y_val)}")

        # Check for all zeros in features
        train_all_zeros = np.all(X_train == 0)
        val_all_zeros = np.all(X_val == 0)
        LOGGER.info(f"Training features all zeros: {train_all_zeros}")
        LOGGER.info(f"Validation features all zeros: {val_all_zeros}")

        # Print sample features for each accelerometer
        LOGGER.info(f"\nSample features for first 3 samples of each accelerometer:")
        for accel_id in range(3):
            mask = y_train == accel_id
            if np.any(mask):
                samples = X_train[mask][:3]
                LOGGER.info(f"\nAccelerometer {accel_id}:")
                LOGGER.info(f"  Shape: {samples.shape}")
                LOGGER.info(f"  Mean: {np.mean(samples):.6f}")
                LOGGER.info(f"  Std: {np.std(samples):.6f}")
                LOGGER.info(f"  Min: {np.min(samples):.6f}")
                LOGGER.info(f"  Max: {np.max(samples):.6f}")
                LOGGER.info(f"  First sample (first 10 values): {samples[0].flatten()[:10]}")

        # Check if features are identical across accelerometers
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("Checking for feature differences between accelerometers...")
        LOGGER.info(f"{'='*60}")
        for i in range(3):
            for j in range(i+1, 3):
                mask_i = y_train == i
                mask_j = y_train == j
                if np.any(mask_i) and np.any(mask_j):
                    mean_i = np.mean(X_train[mask_i])
                    mean_j = np.mean(X_train[mask_j])
                    std_i = np.std(X_train[mask_i])
                    std_j = np.std(X_train[mask_j])
                    LOGGER.info(f"Accel {i} vs Accel {j}:")
                    LOGGER.info(f"  Mean difference: {abs(mean_i - mean_j):.6f}")
                    LOGGER.info(f"  Std difference: {abs(std_i - std_j):.6f}")

        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        LOGGER.info("\nTraining set accelerometer distribution:")
        for accel_id, count in zip(unique, counts):
            LOGGER.info(f"  Accelerometer {accel_id}: {count} samples ({100*count/len(y_train):.2f}%)")

        # Load configuration
        config_file = Path(args.config)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            LOGGER.warning(f"Config file not found: {config_file}, using defaults")
            config = {}

        # Build model configuration
        model_config = {
            "model": {
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
                    "class_weight": "balanced",  # Handle potential class imbalance
                },
            }
        }

        # Merge with loaded config
        if "model" in config:
            for model_type in ["random_forest", "svm"]:
                if model_type in config["model"]:
                    model_config["model"][model_type].update(config["model"][model_type])

        # Build model
        LOGGER.info(f"\nBuilding {args.model_type} model...")

        if args.model_type == "random_forest":
            model = RandomForestModel(model_config)
        elif args.model_type == "svm":
            model = SVMClassifier(model_config)
        else:
            LOGGER.error(f"Unsupported model type: {args.model_type}")
            return 1

        # Flatten features if needed
        if len(X_train.shape) > 2:
            LOGGER.info(f"Flattening features from {X_train.shape} to 2D...")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val

        LOGGER.info(f"Final training shape: {X_train_flat.shape}")

        # Train model
        LOGGER.info("\nTraining model...")
        model.fit(X_train_flat, y_train)
        LOGGER.info("Training completed")

        # Evaluate on training set
        y_train_pred = model.predict(X_train_flat)
        train_accuracy = np.mean(y_train_pred == y_train)
        LOGGER.info(f"\nTraining accuracy: {train_accuracy:.4f} ({100*train_accuracy:.2f}%)")

        # Evaluate on validation set
        y_val_pred = model.predict(X_val_flat)
        val_accuracy = np.mean(y_val_pred == y_val)
        LOGGER.info(f"Validation accuracy: {val_accuracy:.4f} ({100*val_accuracy:.2f}%)")

        # Compute per-class accuracy
        LOGGER.info("\nPer-accelerometer validation accuracy:")
        for accel_id in np.unique(y_val):
            mask = y_val == accel_id
            accel_accuracy = np.mean(y_val_pred[mask] == y_val[mask])
            LOGGER.info(f"  Accelerometer {accel_id}: {accel_accuracy:.4f} ({100*accel_accuracy:.2f}%)")

        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_val, y_val_pred)
        LOGGER.info(f"\nConfusion Matrix:")
        LOGGER.info(f"{cm}")

        LOGGER.info(f"\nClassification Report:")
        report = classification_report(y_val, y_val_pred,
                                       target_names=['Accel 0', 'Accel 1', 'Accel 2'])
        LOGGER.info(f"\n{report}")

        # Save model
        model_path = model_dir / f"{args.model_type}_accelerometer.pkl"
        model.save(str(model_path))
        LOGGER.info(f"\nModel saved to {model_path}")

        # Save training metadata
        metadata = {
            "model_type": args.model_type,
            "timestamp": timestamp,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "feature_shape": list(X_train.shape[1:]),
            "n_classes": 3,
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "confusion_matrix": cm.tolist(),
            "model_path": str(model_path),
        }

        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        LOGGER.info(f"Metadata saved to {metadata_file}")

        LOGGER.info("\n" + "="*60)
        LOGGER.info("ACCELEROMETER CLASSIFIER TRAINING COMPLETED")
        LOGGER.info("="*60)
        LOGGER.info(f"Validation Accuracy: {100*val_accuracy:.2f}%")

        return 0

    except Exception as e:
        LOGGER.error(f"Training failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)

    return train_accelerometer_classifier(args)


if __name__ == "__main__":
    sys.exit(main())
