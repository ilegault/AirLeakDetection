#!/usr/bin/env python3
"""
Comprehensive evaluation for two-stage classifier (v2 with amplitude features).

This evaluates a two-stage leak detection system for multi-accelerometer arrays.
Each data sample contains simultaneous recordings from 3 accelerometers at different
positions (closest, middle, farthest from potential leak sources).

System Architecture:
    Multi-Accelerometer Array (3 sensors recording simultaneously)
        ↓
    [Stage 1: Identify which sensor is closest to the leak]
        ↓
    [Stage 2: Classify leak size using the closest sensor's data]
        ↓
    Leak Size Prediction (NOLEAK, 1/16", 3/32", 1/8")

Evaluation Metrics:
1. Stage 1 Accuracy: How well the system identifies the closest accelerometer
   (Ground truth = accelerometer with strongest signal amplitude)
2. Stage 2 Accuracy: How well the system predicts leak size
3. End-to-end Accuracy: Overall leak size prediction accuracy
4. Per-accelerometer Performance: Accuracy breakdown by sensor position

Usage:
    python scripts/evaluate_two_stage_classifier_v2.py \\
        --config models/two_stage_classifier_v2/model_*/two_stage_config.json \\
        --hole-size-data data/processed/ \\
        --output-dir results/two_stage_classifier/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, FileUtils
from src.models.two_stage_classifier import TwoStageClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage classifier (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to two-stage classifier config JSON"
    )

    parser.add_argument(
        "--hole-size-data",
        type=str,
        required=True,
        help="Path to hole size data (processed NPZ files)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/two_stage_classifier/",
        help="Output directory for results"
    )

    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on"
    )

    return parser


def extract_amplitude_features_from_npz(npz_file: Path, sample_rate: int = 10000) -> Tuple:
    """Extract amplitude features from NPZ file for all 3 accelerometers.

    The NPZ file contains a signal array with shape (timesteps, 3) where each column
    represents simultaneous recordings from one of the 3 accelerometers in the array.

    Args:
        npz_file: Path to NPZ file containing multi-accelerometer data
        sample_rate: Sampling rate in Hz (default: 10000)

    Returns:
        Tuple of:
        - features: Array shape (3, n_features) with features for each accelerometer
        - label: Hole size label from the NPZ file
        Returns (None, None) if extraction fails
    """
    from scripts.extract_amplitude_features import extract_amplitude_features

    try:
        data = np.load(npz_file, allow_pickle=True)

        if "signal" not in data:
            return None, None

        signal = data["signal"]
        if signal.shape[1] != 3:
            return None, None

        # Extract features for each accelerometer
        features_list = []
        for accel_id in range(3):
            accel_signal = signal[:, accel_id]
            features = extract_amplitude_features(accel_signal, sample_rate)
            features_list.append(features)

        features = np.array(features_list, dtype=np.float32)
        label = int(data.get("label", -1))

        return features, label

    except Exception as e:
        LOGGER.error(f"Failed to extract features from {npz_file}: {e}")
        return None, None


def load_test_data(hole_size_data_path: Path, split: str) -> Tuple:
    """Load test data from NPZ files.

    Each NPZ file contains simultaneous recordings from 3 accelerometers.
    Features are extracted for all 3 accelerometers from each file.

    Returns:
        Tuple of:
        - features_list: List of arrays, each shape (3, n_features) containing features
          from all 3 accelerometers for one test sample
        - hole_size_labels: Array of hole size labels (0=NOLEAK, 1=1/16", 2=3/32", 3=1/8")
        - npz_files: List of file paths for reference
    """
    split_dir = hole_size_data_path / split
    npz_files = sorted(split_dir.glob("*.npz"))

    LOGGER.info(f"Loading {split} data from {split_dir}")
    LOGGER.info(f"Found {len(npz_files)} NPZ files")

    features_list = []
    labels_list = []
    valid_files = []

    for i, npz_file in enumerate(npz_files, 1):
        if i % max(1, len(npz_files) // 10) == 0:
            LOGGER.info(f"  Progress: {i}/{len(npz_files)}")

        features, label = extract_amplitude_features_from_npz(npz_file)

        if features is not None:
            features_list.append(features)
            labels_list.append(label)
            valid_files.append(npz_file)

    LOGGER.info(f"Loaded {len(features_list)} valid samples")

    return features_list, np.array(labels_list), valid_files


def plot_stage1_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot confusion matrix for Stage 1 (accelerometer identification)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    class_names = ['Accel 0\n(Closest)', 'Accel 1\n(Middle)', 'Accel 2\n(Farthest)']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Accelerometer', fontsize=12)
    ax.set_ylabel('True Accelerometer', fontsize=12)
    ax.set_title('Stage 1: Accelerometer Identification\nConfusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info(f"Saved Stage 1 confusion matrix to {output_path}")


def plot_stage2_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot confusion matrix for Stage 2 (hole size classification)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    class_names = ['NOLEAK', '1/16"', '3/32"', '1/8"']

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Hole Size', fontsize=12)
    ax.set_ylabel('True Hole Size', fontsize=12)
    ax.set_title('Stage 2: Hole Size Classification\nConfusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info(f"Saved Stage 2 confusion matrix to {output_path}")


def plot_end_to_end_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot end-to-end confusion matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    class_names = ['NOLEAK', '1/16"', '3/32"', '1/8"']

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title('End-to-End Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title('End-to-End Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info(f"Saved end-to-end confusion matrix to {output_path}")


def evaluate_two_stage_classifier(
    two_stage: TwoStageClassifier,
    features_list: list,
    hole_size_labels: np.ndarray,
    output_dir: Path
) -> Dict:
    """Evaluate two-stage classifier and generate comprehensive report.

    Each sample in features_list contains data from all 3 accelerometers (shape: 3, n_features).
    The evaluation tests ALL 3 accelerometers from each sample to get balanced position coverage.

    Args:
        two_stage: Trained TwoStageClassifier instance
        features_list: List of feature arrays, each shape (3, n_features) for 3 accelerometers
        hole_size_labels: True hole size labels for each sample
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary containing evaluation metrics and confusion matrices
    """
    n_samples = len(features_list)

    # Arrays to store predictions - we'll test all 3 accelerometers per sample
    true_accel = []
    pred_accel = []
    pred_hole_size = []
    true_hole_size = []

    LOGGER.info(f"\nMaking predictions on {n_samples} samples (testing all 3 accelerometers per sample)...")
    LOGGER.info(f"Total predictions: {n_samples * 3}")

    for i, features_3accel in enumerate(features_list, 1):
        if i % max(1, n_samples // 10) == 0:
            LOGGER.info(f"  Progress: {i}/{n_samples}")

        # Test ALL 3 accelerometers from this sample
        for accel_id in range(3):
            # Get features for this specific accelerometer
            accel_features = features_3accel[accel_id, :]

            # Run through the two-stage classifier
            prediction = two_stage.predict_single(accel_features)

            # Store results
            true_accel.append(accel_id)  # Ground truth: this is position accel_id
            pred_accel.append(prediction['accelerometer_id'])
            pred_hole_size.append(prediction['hole_size_id'])
            true_hole_size.append(hole_size_labels[i - 1])

    true_accel = np.array(true_accel)
    pred_accel = np.array(pred_accel)
    pred_hole_size = np.array(pred_hole_size)
    true_hole_size = np.array(true_hole_size)

    # Stage 1 evaluation
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("STAGE 1: ACCELEROMETER IDENTIFICATION")
    LOGGER.info(f"{'='*80}")

    stage1_accuracy = accuracy_score(true_accel, pred_accel)
    LOGGER.info(f"Stage 1 Accuracy: {stage1_accuracy:.4f} ({100*stage1_accuracy:.2f}%)")

    stage1_cm = confusion_matrix(true_accel, pred_accel)
    LOGGER.info(f"\nStage 1 Confusion Matrix:")
    LOGGER.info(f"{stage1_cm}")

    plot_stage1_confusion_matrix(stage1_cm, output_dir / "stage1_confusion_matrix.png")

    # Stage 2 evaluation
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("STAGE 2: HOLE SIZE CLASSIFICATION")
    LOGGER.info(f"{'='*80}")

    stage2_accuracy = accuracy_score(true_hole_size, pred_hole_size)
    LOGGER.info(f"Stage 2 Accuracy: {stage2_accuracy:.4f} ({100*stage2_accuracy:.2f}%)")

    stage2_cm = confusion_matrix(true_hole_size, pred_hole_size)
    LOGGER.info(f"\nStage 2 Confusion Matrix:")
    LOGGER.info(f"{stage2_cm}")

    plot_stage2_confusion_matrix(stage2_cm, output_dir / "stage2_confusion_matrix.png")

    # End-to-end evaluation
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("END-TO-END PERFORMANCE")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"Overall Accuracy: {stage2_accuracy:.4f} ({100*stage2_accuracy:.2f}%)")

    plot_end_to_end_confusion_matrix(stage2_cm, output_dir / "end_to_end_confusion_matrix.png")

    # Classification report
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("CLASSIFICATION REPORT (End-to-End)")
    LOGGER.info(f"{'='*80}")
    class_names = ['NOLEAK', '1/16"', '3/32"', '1/8"']
    report = classification_report(true_hole_size, pred_hole_size, target_names=class_names)
    LOGGER.info(f"\n{report}")

    # Per-accelerometer performance
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("PER-ACCELEROMETER PERFORMANCE")
    LOGGER.info(f"{'='*80}")

    per_accel_stats = {}
    for accel_id in range(3):
        mask = pred_accel == accel_id
        if np.any(mask):
            accel_accuracy = accuracy_score(true_hole_size[mask], pred_hole_size[mask])
            n_samples_accel = np.sum(mask)

            per_accel_stats[f'accel_{accel_id}'] = {
                'accuracy': float(accel_accuracy),
                'n_samples': int(n_samples_accel)
            }

            LOGGER.info(f"\nAccelerometer {accel_id}:")
            LOGGER.info(f"  Samples routed: {n_samples_accel} ({100*n_samples_accel/len(pred_accel):.1f}%)")
            LOGGER.info(f"  Accuracy: {accel_accuracy:.4f} ({100*accel_accuracy:.2f}%)")

    # Create comprehensive report
    report_dict = {
        'stage1': {
            'accuracy': float(stage1_accuracy),
            'confusion_matrix': stage1_cm.tolist()
        },
        'stage2': {
            'accuracy': float(stage2_accuracy),
            'confusion_matrix': stage2_cm.tolist()
        },
        'end_to_end': {
            'accuracy': float(stage2_accuracy),
            'total_samples': int(len(pred_hole_size)),
            'test_npz_files': int(n_samples),
            'note': 'Each NPZ file contains 3 accelerometers, all tested separately'
        },
        'per_accelerometer': per_accel_stats
    }

    # Save report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report_dict, f, indent=2)

    LOGGER.info(f"\nSaved evaluation report to {report_file}")

    return report_dict


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    hole_size_data_path = Path(args.hole_size_data)
    output_dir = Path(args.output_dir)

    if not config_path.exists():
        LOGGER.error(f"Config file not found: {config_path}")
        return 1

    if not hole_size_data_path.exists():
        LOGGER.error(f"Hole size data not found: {hole_size_data_path}")
        return 1

    # Create output directory
    FileUtils.ensure_directory(str(output_dir))

    LOGGER.info("="*80)
    LOGGER.info("TWO-STAGE CLASSIFIER EVALUATION (V2)")
    LOGGER.info("="*80)
    LOGGER.info(f"Config: {config_path}")
    LOGGER.info(f"Data: {hole_size_data_path}")
    LOGGER.info(f"Split: {args.test_split}")
    LOGGER.info(f"Output: {output_dir}")

    # Load two-stage classifier
    LOGGER.info("\nLoading two-stage classifier...")
    two_stage = TwoStageClassifier.from_config(str(config_path))

    # Load test data
    features_list, hole_size_labels, npz_files = load_test_data(
        hole_size_data_path, args.test_split
    )

    if len(features_list) == 0:
        LOGGER.error("No test data loaded")
        return 1

    # Evaluate
    report = evaluate_two_stage_classifier(
        two_stage, features_list, hole_size_labels, output_dir
    )

    LOGGER.info("\n" + "="*80)
    LOGGER.info("EVALUATION COMPLETED")
    LOGGER.info("="*80)
    LOGGER.info(f"Results saved to: {output_dir}")
    LOGGER.info(f"  - stage1_confusion_matrix.png")
    LOGGER.info(f"  - stage2_confusion_matrix.png")
    LOGGER.info(f"  - end_to_end_confusion_matrix.png")
    LOGGER.info(f"  - evaluation_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
