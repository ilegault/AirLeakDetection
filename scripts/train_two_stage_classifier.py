#!/usr/bin/env python3
"""
Train a two-stage classifier combining accelerometer identification and hole size detection.

This script:
1. Trains separate hole size classifiers for each accelerometer
2. Uses a pre-trained accelerometer classifier
3. Combines them into a two-stage system
4. Evaluates end-to-end performance

Workflow:
    Input → [Stage 1: Identify Accelerometer] → [Stage 2: Detect Hole Size] → Output

Usage:
    python scripts/train_two_stage_classifier.py \\
        --hole-size-data data/processed/ \\
        --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \\
        --output-dir models/two_stage_classifier/
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

from src.utils import setup_logging, get_logger, FileUtils
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
        description="Train two-stage classifier (accelerometer + hole size)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_two_stage_classifier.py \\
      --hole-size-data data/processed/ \\
      --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl

  # Train with custom model type
  python scripts/train_two_stage_classifier.py \\
      --hole-size-data data/processed/ \\
      --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \\
      --model-type svm \\
      --output-dir models/two_stage_classifier/
        """
    )

    parser.add_argument(
        "--hole-size-data",
        type=str,
        required=True,
        help="Path to hole size classification data (output from prepare_data.py)"
    )

    parser.add_argument(
        "--accelerometer-classifier",
        type=str,
        required=True,
        help="Path to trained accelerometer classifier (.pkl)"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm"],
        help="Type of model to train for hole size detection (default: random_forest)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/two_stage_classifier/",
        help="Output directory for trained models (default: models/two_stage_classifier/)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )

    parser.add_argument(
        "--use-welch-bandpower",
        action="store_true",
        default=True,
        help="Use Welch band power features (default: True)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    return parser


def load_data_for_accelerometer(
    data_path: Path,
    split: str,
    accel_id: int,
    use_welch_bandpower: bool = True
) -> tuple:
    """Load data for a specific accelerometer from NPZ files.

    Args:
        data_path: Path to processed data directory
        split: Split name ('train', 'val', 'test')
        accel_id: Accelerometer ID (0, 1, 2)
        use_welch_bandpower: Use Welch band power features

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

    # STEP 2 DIAGNOSTICS: Check first NPZ file
    if npz_files:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"STEP 2: FEATURE EXTRACTION DIAGNOSTICS (Accelerometer {accel_id})")
        LOGGER.info(f"{'='*60}")
        first_file = npz_files[0]
        LOGGER.info(f"Inspecting first NPZ file: {first_file.name}")

        data = np.load(first_file, allow_pickle=True)
        LOGGER.info(f"Keys available in NPZ file: {list(data.keys())}")

        for key in data.keys():
            if key != "label":
                value = data[key]
                LOGGER.info(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, dtype={getattr(value, 'dtype', type(value))}")
                if hasattr(value, '__len__') and len(value) >= 3:
                    LOGGER.info(f"    Values for all 3 accelerometers: {value[:3] if len(value) == 3 else 'shape mismatch'}")
                    if accel_id < len(value):
                        LOGGER.info(f"    Value for accelerometer {accel_id}: {value[accel_id]}")

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)

            # Extract features for this accelerometer
            if use_welch_bandpower and "welch_bandpower" in data:
                # Use band power (single value per accelerometer)
                bandpower = data["welch_bandpower"]
                if len(bandpower) >= 3:
                    features = np.array([bandpower[accel_id]], dtype=np.float32)
                    features_list.append(features)
            elif "welch_psd" in data:
                # Use Welch PSD
                welch_psd = data["welch_psd"]
                if welch_psd.shape[1] >= 3:
                    features = welch_psd[:, accel_id]
                    features_list.append(features)
            elif "fft_magnitude" in data:
                # Use FFT magnitude
                fft_mag = data["fft_magnitude"]
                if fft_mag.shape[1] >= 3:
                    features = fft_mag[:, accel_id]
                    features_list.append(features)
            elif "signal" in data:
                # Use raw signal
                signal = data["signal"]
                if signal.shape[1] >= 3:
                    features = signal[:, accel_id]
                    features_list.append(features)
            else:
                continue

            # Get hole size label
            if "label" in data:
                labels_list.append(int(data["label"]))

        except Exception as e:
            LOGGER.error(f"Failed to load {npz_file}: {e}")
            continue

    if not features_list:
        LOGGER.warning(f"No features extracted for accelerometer {accel_id} in {split}")
        return np.array([]), np.array([])

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)

    # STEP 2 DIAGNOSTICS: Print extraction summary
    LOGGER.info(f"\nExtraction summary for accelerometer {accel_id}:")
    LOGGER.info(f"  Total files processed: {len(npz_files)}")
    LOGGER.info(f"  Features extracted: {len(features_list)}")
    LOGGER.info(f"  Final feature shape: {features.shape}")
    LOGGER.info(f"  Feature mean: {np.mean(features):.6f}")
    LOGGER.info(f"  Feature std: {np.std(features):.6f}")
    LOGGER.info(f"  Feature min: {np.min(features):.6f}")
    LOGGER.info(f"  Feature max: {np.max(features):.6f}")

    return features, labels


def train_hole_size_classifier_for_accelerometer(
    accel_id: int,
    data_path: Path,
    model_type: str,
    config: Dict,
    use_welch_bandpower: bool = True
) -> object:
    """Train a hole size classifier for a specific accelerometer.

    Args:
        accel_id: Accelerometer ID (0, 1, 2)
        data_path: Path to processed data
        model_type: Model type ('random_forest' or 'svm')
        config: Configuration dictionary
        use_welch_bandpower: Use Welch band power features

    Returns:
        Trained model
    """
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Training hole size classifier for Accelerometer {accel_id}")
    LOGGER.info(f"{'='*60}")

    # Load data for this accelerometer
    X_train, y_train = load_data_for_accelerometer(data_path, "train", accel_id, use_welch_bandpower)
    X_val, y_val = load_data_for_accelerometer(data_path, "val", accel_id, use_welch_bandpower)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(f"No data found for accelerometer {accel_id}")

    LOGGER.info(f"Training data shape: {X_train.shape}")
    LOGGER.info(f"Validation data shape: {X_val.shape}")

    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    LOGGER.info(f"Training set class distribution:")
    for class_id, count in zip(unique, counts):
        LOGGER.info(f"  Class {class_id}: {count} samples ({100*count/len(y_train):.2f}%)")

    # Flatten features if needed
    if len(X_train.shape) > 2:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
    else:
        X_train_flat = X_train
        X_val_flat = X_val

    # Build model
    if model_type == "random_forest":
        model = RandomForestModel(config)
    elif model_type == "svm":
        model = SVMClassifier(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train
    LOGGER.info("Training model...")
    model.fit(X_train_flat, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train_flat)
    train_accuracy = np.mean(y_train_pred == y_train)

    y_val_pred = model.predict(X_val_flat)
    val_accuracy = np.mean(y_val_pred == y_val)

    LOGGER.info(f"Training accuracy: {train_accuracy:.4f} ({100*train_accuracy:.2f}%)")
    LOGGER.info(f"Validation accuracy: {val_accuracy:.4f} ({100*val_accuracy:.2f}%)")

    return model


def train_two_stage_classifier(args) -> int:
    """Train the two-stage classifier."""
    try:
        data_path = Path(args.hole_size_data)
        accel_classifier_path = Path(args.accelerometer_classifier)
        output_dir = Path(args.output_dir)

        if not data_path.exists():
            LOGGER.error(f"Data path not found: {data_path}")
            return 1

        if not accel_classifier_path.exists():
            LOGGER.error(f"Accelerometer classifier not found: {accel_classifier_path}")
            return 1

        # Create output directory
        FileUtils.ensure_directory(str(output_dir))

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = output_dir / f"model_{timestamp}"
        FileUtils.ensure_directory(str(model_dir))

        LOGGER.info("="*60)
        LOGGER.info("TWO-STAGE CLASSIFIER TRAINING")
        LOGGER.info("="*60)
        LOGGER.info(f"Hole size data: {data_path}")
        LOGGER.info(f"Accelerometer classifier: {accel_classifier_path}")
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
                    data_path=data_path,
                    model_type=args.model_type,
                    config=model_config,
                    use_welch_bandpower=args.use_welch_bandpower
                )

                # Save model
                model_path = model_dir / f"accel_{accel_id}_classifier.pkl"
                model.save(str(model_path))
                LOGGER.info(f"Model saved to {model_path}")

                hole_size_classifiers[accel_id] = str(model_path)

            except Exception as e:
                LOGGER.error(f"Failed to train classifier for accelerometer {accel_id}: {e}")
                continue

        if not hole_size_classifiers:
            LOGGER.error("No hole size classifiers were trained successfully")
            return 1

        # Create two-stage classifier
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("Creating two-stage classifier...")
        LOGGER.info(f"{'='*60}")

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

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "model_type": args.model_type,
            "accelerometer_classifier": str(accel_classifier_path),
            "hole_size_classifiers": hole_size_classifiers,
            "use_welch_bandpower": args.use_welch_bandpower,
        }

        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        LOGGER.info(f"Metadata saved to {metadata_file}")

        LOGGER.info("\n" + "="*60)
        LOGGER.info("TWO-STAGE CLASSIFIER TRAINING COMPLETED")
        LOGGER.info("="*60)
        LOGGER.info(f"Models saved to: {model_dir}")
        LOGGER.info(f"  - Accelerometer classifier: {accel_classifier_path}")
        LOGGER.info(f"  - Hole size classifiers: {len(hole_size_classifiers)}")

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

    return train_two_stage_classifier(args)


if __name__ == "__main__":
    sys.exit(main())
