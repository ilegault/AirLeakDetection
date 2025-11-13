#!/usr/bin/env python3
"""
Train a two-stage classifier (v2) using amplitude-based features for multi-accelerometer arrays.

Multi-Accelerometer Setup:
    Each measurement contains simultaneous recordings from 3 accelerometers at different
    positions (closest, middle, farthest from leak sources).

Training Process:
    Stage 1: Train position classifier using amplitude features
             - Identifies which position (0, 1, 2) is closest to leak source
             - Uses pre-trained accelerometer classifier from train_accelerometer_classifier.py

    Stage 2: Train position-specific hole size classifiers
             - One classifier per position (0, 1, 2)
             - Each trained on data from that specific position
             - Predicts leak size: NOLEAK, 1/16", 3/32", 1/8"

This version uses amplitude-based features (mean, std, peak, RMS, etc.) for both stages.

Usage:
    python scripts/train_two_stage_classifier_v2.py \\
        --accelerometer-data data/accelerometer_classifier_v2/ \\
        --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \\
        --hole-size-data data/processed/ \\
        --output-dir models/two_stage_classifier_v2/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, FileUtils
from src.models.random_forest import RandomForestModel
from src.models.svm_classifier import SVMClassifier
from src.models.two_stage_classifier import TwoStageClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Train two-stage classifier (v2 with amplitude features)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--accelerometer-data",
        type=str,
        required=True,
        help="Path to accelerometer classification data (with amplitude features)"
    )

    parser.add_argument(
        "--accelerometer-classifier",
        type=str,
        required=True,
        help="Path to trained accelerometer classifier (.pkl)"
    )

    parser.add_argument(
        "--hole-size-data",
        type=str,
        required=True,
        help="Path to hole size classification data (processed NPZ files)"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm"],
        help="Type of model for hole size detection (default: random_forest)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/two_stage_classifier_v2/",
        help="Output directory for trained models"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    return parser


def extract_amplitude_features_from_npz(npz_file: Path, sample_rate: int = 10000) -> np.ndarray:
    """Extract amplitude features from NPZ file for all 3 accelerometers.

    Returns:
        Array of shape (3, n_features) - one row per accelerometer
    """
    from scripts.extract_amplitude_features import extract_amplitude_features

    try:
        data = np.load(npz_file, allow_pickle=True)

        if "signal" not in data:
            return None

        signal = data["signal"]
        if signal.shape[1] != 3:
            return None

        # Extract features for each accelerometer
        features_list = []
        for accel_id in range(3):
            accel_signal = signal[:, accel_id]
            features = extract_amplitude_features(accel_signal, sample_rate)
            features_list.append(features)

        return np.array(features_list, dtype=np.float32)

    except Exception as e:
        LOGGER.error(f"Failed to extract features from {npz_file}: {e}")
        return None


def load_hole_size_data_for_accelerometer(
    data_path: Path,
    split: str,
    accel_id: int
) -> tuple:
    """Load hole size training data for a specific accelerometer.

    Args:
        data_path: Path to processed data directory
        split: Split name ('train', 'val', 'test')
        accel_id: Accelerometer ID (0, 1, 2)

    Returns:
        (features, labels) for the specified accelerometer
    """
    split_dir = data_path / split
    npz_files = sorted(split_dir.glob("*.npz"))

    if not npz_files:
        LOGGER.warning(f"No NPZ files found in {split_dir}")
        return np.array([]), np.array([])

    features_list = []
    labels_list = []

    LOGGER.info(f"Extracting features for accelerometer {accel_id} from {len(npz_files)} files...")

    for i, npz_file in enumerate(npz_files, 1):
        if i % max(1, len(npz_files) // 10) == 0:
            LOGGER.info(f"  Progress: {i}/{len(npz_files)}")

        # Extract amplitude features for all accelerometers
        all_features = extract_amplitude_features_from_npz(npz_file)

        if all_features is None:
            continue

        # Get features for this specific accelerometer
        accel_features = all_features[accel_id, :]
        features_list.append(accel_features)

        # Get hole size label
        try:
            data = np.load(npz_file, allow_pickle=True)
            if "label" in data:
                labels_list.append(int(data["label"]))
        except Exception as e:
            LOGGER.error(f"Failed to load label from {npz_file}: {e}")
            features_list.pop()  # Remove the features we just added
            continue

    if not features_list:
        LOGGER.warning(f"No features extracted for accelerometer {accel_id} in {split}")
        return np.array([]), np.array([])

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    LOGGER.info(f"Extracted {len(features)} samples for accelerometer {accel_id}")
    LOGGER.info(f"  Feature shape: {features.shape}")
    LOGGER.info(f"  Class distribution: {np.unique(labels, return_counts=True)}")

    return features, labels


def train_hole_size_classifier_for_accelerometer(
    accel_id: int,
    data_path: Path,
    model_type: str,
    config: Dict
) -> object:
    """Train a hole size classifier for a specific accelerometer.

    Args:
        accel_id: Accelerometer ID (0, 1, 2)
        data_path: Path to processed data
        model_type: Model type ('random_forest' or 'svm')
        config: Configuration dictionary

    Returns:
        Trained model
    """
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Training Hole Size Classifier for Accelerometer {accel_id}")
    LOGGER.info(f"{'='*80}")

    # Load data
    X_train, y_train = load_hole_size_data_for_accelerometer(data_path, "train", accel_id)
    X_val, y_val = load_hole_size_data_for_accelerometer(data_path, "val", accel_id)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(f"No data found for accelerometer {accel_id}")

    LOGGER.info(f"\nTraining data: {X_train.shape}")
    LOGGER.info(f"Validation data: {X_val.shape}")

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    LOGGER.info(f"\nTraining set class distribution:")
    for class_id, count in zip(unique, counts):
        LOGGER.info(f"  Class {class_id}: {count} samples ({100*count/len(y_train):.2f}%)")

    # Build model
    if model_type == "random_forest":
        model = RandomForestModel(config)
    elif model_type == "svm":
        model = SVMClassifier(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train
    LOGGER.info("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train)

    y_val_pred = model.predict(X_val)
    val_accuracy = np.mean(y_val_pred == y_val)

    LOGGER.info(f"\nResults:")
    LOGGER.info(f"  Training accuracy: {train_accuracy:.4f} ({100*train_accuracy:.2f}%)")
    LOGGER.info(f"  Validation accuracy: {val_accuracy:.4f} ({100*val_accuracy:.2f}%)")

    # Per-class validation accuracy
    LOGGER.info(f"\nPer-class validation accuracy:")
    for class_id in np.unique(y_val):
        mask = y_val == class_id
        class_acc = np.mean(y_val_pred[mask] == y_val[mask])
        LOGGER.info(f"  Class {class_id}: {class_acc:.4f} ({100*class_acc:.2f}%)")

    return model


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    accel_data_path = Path(args.accelerometer_data)
    accel_classifier_path = Path(args.accelerometer_classifier)
    hole_size_data_path = Path(args.hole_size_data)
    output_dir = Path(args.output_dir)

    # Validate paths
    if not accel_data_path.exists():
        LOGGER.error(f"Accelerometer data not found: {accel_data_path}")
        return 1

    if not accel_classifier_path.exists():
        LOGGER.error(f"Accelerometer classifier not found: {accel_classifier_path}")
        return 1

    if not hole_size_data_path.exists():
        LOGGER.error(f"Hole size data not found: {hole_size_data_path}")
        return 1

    # Create output directory
    FileUtils.ensure_directory(str(output_dir))

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = output_dir / f"model_{timestamp}"
    FileUtils.ensure_directory(str(model_dir))

    LOGGER.info("="*80)
    LOGGER.info("TWO-STAGE CLASSIFIER TRAINING (V2 - Amplitude Features)")
    LOGGER.info("="*80)
    LOGGER.info(f"Accelerometer data: {accel_data_path}")
    LOGGER.info(f"Accelerometer classifier: {accel_classifier_path}")
    LOGGER.info(f"Hole size data: {hole_size_data_path}")
    LOGGER.info(f"Model type: {args.model_type}")
    LOGGER.info(f"Output directory: {model_dir}")

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
                "class_weight": "balanced",
            },
        }
    }

    # Train hole size classifiers for each accelerometer
    hole_size_classifiers = {}

    for accel_id in [0, 1, 2]:
        try:
            model = train_hole_size_classifier_for_accelerometer(
                accel_id=accel_id,
                data_path=hole_size_data_path,
                model_type=args.model_type,
                config=model_config
            )

            # Save model
            model_path = model_dir / f"accel_{accel_id}_hole_size_classifier.pkl"
            model.save(str(model_path))
            LOGGER.info(f"Saved model to {model_path}")

            hole_size_classifiers[accel_id] = str(model_path)

        except Exception as e:
            LOGGER.error(f"Failed to train classifier for accelerometer {accel_id}: {e}", exc_info=True)
            continue

    if not hole_size_classifiers:
        LOGGER.error("No hole size classifiers were trained successfully")
        return 1

    # Create two-stage classifier configuration
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("Creating two-stage classifier configuration...")
    LOGGER.info(f"{'='*80}")

    two_stage = TwoStageClassifier(
        accelerometer_classifier_path=str(accel_classifier_path),
        hole_size_classifier_paths=hole_size_classifiers,
        class_names={
            0: "NOLEAK",
            1: "1_16",
            2: "3_32",
            3: "1_8"
        }
    )

    # Save configuration
    config_path = model_dir / "two_stage_config.json"
    two_stage.save_config(str(config_path))
    LOGGER.info(f"Saved two-stage configuration to {config_path}")

    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "model_type": args.model_type,
        "accelerometer_classifier": str(accel_classifier_path),
        "hole_size_classifiers": hole_size_classifiers,
        "feature_type": "amplitude_based",
        "n_accelerometers": len(hole_size_classifiers)
    }

    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info(f"Saved metadata to {metadata_file}")

    LOGGER.info("\n" + "="*80)
    LOGGER.info("TWO-STAGE CLASSIFIER TRAINING COMPLETED")
    LOGGER.info("="*80)
    LOGGER.info(f"Models saved to: {model_dir}")
    LOGGER.info(f"  - Accelerometer classifier: {accel_classifier_path}")
    LOGGER.info(f"  - Hole size classifiers: {len(hole_size_classifiers)}")
    LOGGER.info(f"  - Configuration: {config_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
