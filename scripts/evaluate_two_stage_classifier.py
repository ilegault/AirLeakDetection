#!/usr/bin/env python3
"""
Evaluate the two-stage classifier on test data.

This script:
1. Loads the trained two-stage classifier
2. Evaluates on test data
3. Generates detailed performance metrics
4. Creates visualization plots

Usage:
    python scripts/evaluate_two_stage_classifier.py \\
        --model-dir models/two_stage_classifier/model_*/ \\
        --test-data data/processed/test/ \\
        --accelerometer-test-data data/accelerometer_classifier/test/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils
from src.models.two_stage_classifier import TwoStageClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate_two_stage_classifier.py \\
      --model-dir models/two_stage_classifier/model_*/ \\
      --test-data data/processed/test/ \\
      --accelerometer-test-data data/accelerometer_classifier/test/

  # With visualization
  python scripts/evaluate_two_stage_classifier.py \\
      --model-dir models/two_stage_classifier/model_*/ \\
      --test-data data/processed/test/ \\
      --accelerometer-test-data data/accelerometer_classifier/test/ \\
      --plot \\
      --output-dir results/two_stage_evaluation/
        """
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to trained two-stage classifier directory"
    )

    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data for hole size classification"
    )

    parser.add_argument(
        "--accelerometer-test-data",
        type=str,
        required=True,
        help="Path to test data for accelerometer classification"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/two_stage_evaluation/",
        help="Output directory for results (default: results/two_stage_evaluation/)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    return parser


def load_welch_bandpower_test_data(test_data_path: Path) -> tuple:
    """Load Welch band power features from test NPZ files for all accelerometers.
    
    This extracts the band power feature (single value per accelerometer) that was
    used to train the hole size classifiers.

    Args:
        test_data_path: Path to test data directory containing NPZ files

    Returns:
        (features, accel_ids, hole_size_ids) where:
        - features: shape (n_samples, 3) - band power for each accelerometer
        - accel_ids: shape (n_samples,) - which accelerometer the sample came from
        - hole_size_ids: shape (n_samples,) - true hole size class
    """
    npz_files = sorted(test_data_path.glob("*.npz"))
    
    if not npz_files:
        LOGGER.warning(f"No NPZ files found in {test_data_path}")
        return np.array([]), np.array([]), np.array([])
    
    features_list = []
    accel_ids_list = []
    hole_size_ids_list = []
    
    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract Welch band power (3 values - one per accelerometer)
            if "welch_bandpower" in data:
                bandpower = data["welch_bandpower"]
                if len(bandpower) >= 3:
                    features_list.append(bandpower[:3])  # Take all 3 accelerometers
                    
                    # Get labels
                    if "accelerometer_id" in data:
                        accel_ids_list.append(int(data["accelerometer_id"]))
                    if "label" in data:
                        hole_size_ids_list.append(int(data["label"]))
        except Exception as e:
            LOGGER.error(f"Failed to load {npz_file}: {e}")
            continue
    
    if not features_list:
        LOGGER.warning(f"No features extracted from {test_data_path}")
        return np.array([]), np.array([]), np.array([])
    
    features = np.array(features_list, dtype=np.float32)
    accel_ids = np.array(accel_ids_list, dtype=np.int32)
    hole_size_ids = np.array(hole_size_ids_list, dtype=np.int32)
    
    return features, accel_ids, hole_size_ids


def load_two_stage_classifier(model_dir: Path) -> TwoStageClassifier:
    """Load the two-stage classifier from directory.

    Args:
        model_dir: Path to model directory

    Returns:
        Loaded TwoStageClassifier
    """
    LOGGER.info(f"Loading two-stage classifier from {model_dir}")

    # Load metadata
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Get model paths
    accel_classifier_path = metadata["accelerometer_classifier"]
    hole_size_classifiers = metadata["hole_size_classifiers"]

    # Load classifier
    classifier = TwoStageClassifier(
        accelerometer_classifier_path=accel_classifier_path,
        hole_size_classifier_paths=hole_size_classifiers,
        class_names={
            0: "NOLEAK",
            1: "1_16",
            2: "3_32",
            3: "1_8"
        }
    )

    LOGGER.info("Two-stage classifier loaded successfully")
    return classifier


def evaluate_two_stage_classifier(args) -> int:
    """Evaluate the two-stage classifier."""
    try:
        model_dir = Path(args.model_dir)
        test_data_path = Path(args.test_data)
        accel_test_data_path = Path(args.accelerometer_test_data)
        output_dir = Path(args.output_dir)

        # Validate paths
        if not model_dir.exists():
            LOGGER.error(f"Model directory not found: {model_dir}")
            return 1

        if not test_data_path.exists():
            LOGGER.error(f"Test data not found: {test_data_path}")
            return 1

        if not accel_test_data_path.exists():
            LOGGER.error(f"Accelerometer test data not found: {accel_test_data_path}")
            return 1

        # Create output directory
        FileUtils.ensure_directory(str(output_dir))

        LOGGER.info("="*60)
        LOGGER.info("TWO-STAGE CLASSIFIER EVALUATION")
        LOGGER.info("="*60)
        LOGGER.info(f"Model directory: {model_dir}")
        LOGGER.info(f"Test data: {test_data_path}")
        LOGGER.info(f"Output directory: {output_dir}")

        # Load classifier
        classifier = load_two_stage_classifier(model_dir)

        # Load test data
        LOGGER.info("\nLoading test data...")
        
        # Load full features for accelerometer classifier (stage 1)
        X_test_full = np.load(accel_test_data_path / "features.npy")
        y_accel_test = np.load(accel_test_data_path / "labels.npy")
        y_hole_test = np.load(accel_test_data_path / "hole_size_labels.npy")
        
        # Flatten features if needed
        if len(X_test_full.shape) > 2:
            X_test_full = X_test_full.reshape(X_test_full.shape[0], -1)
        
        # Load band power features for hole size classifiers (stage 2)
        X_bandpower, _, _ = load_welch_bandpower_test_data(test_data_path / "test")
        
        if len(X_bandpower) == 0:
            LOGGER.error("Failed to load band power features for hole size classifiers")
            return 1
        
        if len(X_test_full) != len(X_bandpower):
            LOGGER.warning(f"Sample count mismatch: full features={len(X_test_full)}, band power={len(X_bandpower)}")
            # Use the minimum to ensure both have same length
            min_len = min(len(X_test_full), len(X_bandpower))
            X_test_full = X_test_full[:min_len]
            X_bandpower = X_bandpower[:min_len]
            y_accel_test = y_accel_test[:min_len]
            y_hole_test = y_hole_test[:min_len]
        
        LOGGER.info(f"Full features shape (for accelerometer ID): {X_test_full.shape}")
        LOGGER.info(f"Band power shape (for hole size): {X_bandpower.shape}")
        LOGGER.info(f"Number of test samples: {len(X_test_full)}")

        # Evaluate - need to combine both feature types
        # Stage 1: Use full features for accelerometer identification
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("Running evaluation...")
        LOGGER.info(f"{'='*60}")
        
        # Get accelerometer predictions using full features
        pred_accel_ids = classifier.accelerometer_classifier.predict(X_test_full)
        accel_proba = classifier.accelerometer_classifier.predict_proba(X_test_full)
        
        # Get hole size predictions using band power features
        pred_hole_ids = np.zeros_like(pred_accel_ids)
        confidences = np.zeros(len(pred_accel_ids))
        
        for i in range(len(pred_accel_ids)):
            accel_id = pred_accel_ids[i]
            
            if accel_id in classifier.hole_size_classifiers:
                hole_classifier = classifier.hole_size_classifiers[accel_id]
                # Extract band power for THIS accelerometer only (shape should be 1)
                accel_bandpower = X_bandpower[i:i+1, accel_id:accel_id+1]  # Shape (1, 1)
                pred_hole_ids[i] = hole_classifier.predict(accel_bandpower)[0]
                hole_proba = hole_classifier.predict_proba(accel_bandpower)[0]
                # Combined confidence: product of both stage confidences
                confidences[i] = float(accel_proba[i, accel_id] * hole_proba[int(pred_hole_ids[i])])
            else:
                pred_hole_ids[i] = -1
                confidences[i] = 0.0
        
        # Calculate metrics
        stage1_accuracy = np.mean(pred_accel_ids == y_accel_test)
        correct_accel_mask = pred_accel_ids == y_accel_test
        if np.sum(correct_accel_mask) > 0:
            stage2_accuracy = np.mean(
                pred_hole_ids[correct_accel_mask] == y_hole_test[correct_accel_mask]
            )
        else:
            stage2_accuracy = 0.0
        overall_accuracy = np.mean((pred_accel_ids == y_accel_test) & (pred_hole_ids == y_hole_test))
        
        per_accel_accuracy = {}
        for accel_id in np.unique(y_accel_test):
            mask = y_accel_test == accel_id
            if np.sum(mask) > 0:
                accel_correct = np.mean(
                    (pred_accel_ids[mask] == y_accel_test[mask]) &
                    (pred_hole_ids[mask] == y_hole_test[mask])
                )
                per_accel_accuracy[int(accel_id)] = float(accel_correct)
        
        results = {
            'stage1_accuracy': float(stage1_accuracy),
            'stage2_accuracy': float(stage2_accuracy),
            'overall_accuracy': float(overall_accuracy),
            'per_accelerometer_accuracy': per_accel_accuracy,
            'mean_confidence': float(np.mean(confidences)),
        }

        # Print results
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("EVALUATION RESULTS")
        LOGGER.info(f"{'='*60}")
        LOGGER.info(f"Stage 1 Accuracy (Accelerometer ID): {results['stage1_accuracy']:.4f} ({100*results['stage1_accuracy']:.2f}%)")
        LOGGER.info(f"Stage 2 Accuracy (Hole Size | Correct Accel): {results['stage2_accuracy']:.4f} ({100*results['stage2_accuracy']:.2f}%)")
        LOGGER.info(f"Overall Accuracy (End-to-End): {results['overall_accuracy']:.4f} ({100*results['overall_accuracy']:.2f}%)")
        LOGGER.info(f"Mean Confidence: {results['mean_confidence']:.4f}")

        LOGGER.info(f"\nPer-Accelerometer Overall Accuracy:")
        for accel_id, acc in results['per_accelerometer_accuracy'].items():
            LOGGER.info(f"  Accelerometer {accel_id}: {acc:.4f} ({100*acc:.2f}%)")

        # Predictions already calculated above from evaluate section

        # Generate confusion matrices
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("Confusion Matrices")
        LOGGER.info(f"{'='*60}")

        # Stage 1: Accelerometer confusion matrix
        cm_accel = confusion_matrix(y_accel_test, pred_accel_ids)
        LOGGER.info(f"\nStage 1 - Accelerometer Identification:")
        LOGGER.info(f"\n{cm_accel}")

        # Stage 2: Hole size confusion matrix (only for correctly identified accelerometers)
        correct_accel_mask = pred_accel_ids == y_accel_test
        if np.sum(correct_accel_mask) > 0:
            cm_hole = confusion_matrix(
                y_hole_test[correct_accel_mask],
                pred_hole_ids[correct_accel_mask]
            )
            LOGGER.info(f"\nStage 2 - Hole Size Detection (Correct Accelerometer):")
            LOGGER.info(f"\n{cm_hole}")

        # Overall confusion matrix
        # Create combined labels for visualization
        combined_true = y_accel_test * 10 + y_hole_test  # e.g., Accel 0 + NOLEAK = 0, Accel 1 + 1_16 = 11
        combined_pred = pred_accel_ids * 10 + pred_hole_ids

        # Classification reports
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("Classification Reports")
        LOGGER.info(f"{'='*60}")

        LOGGER.info(f"\nStage 1 - Accelerometer Identification:")
        report_accel = classification_report(
            y_accel_test, pred_accel_ids,
            target_names=['Accel 0', 'Accel 1', 'Accel 2']
        )
        LOGGER.info(f"\n{report_accel}")

        if np.sum(correct_accel_mask) > 0:
            LOGGER.info(f"\nStage 2 - Hole Size Detection (Correct Accelerometer):")
            report_hole = classification_report(
                y_hole_test[correct_accel_mask],
                pred_hole_ids[correct_accel_mask],
                target_names=['NOLEAK', '1_16', '3_32', '1_8']
            )
            LOGGER.info(f"\n{report_hole}")

        # Save results
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"\nResults saved to {results_file}")

        # Generate plots if requested
        if args.plot:
            LOGGER.info(f"\n{'='*60}")
            LOGGER.info("Generating visualization plots...")
            LOGGER.info(f"{'='*60}")

            # Plot 1: Stage 1 confusion matrix
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            sns.heatmap(cm_accel, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Accel 0', 'Accel 1', 'Accel 2'],
                       yticklabels=['Accel 0', 'Accel 1', 'Accel 2'])
            ax.set_title('Stage 1: Accelerometer Identification\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            plt.tight_layout()
            plot_path = output_dir / "stage1_confusion_matrix.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            LOGGER.info(f"Saved: {plot_path}")
            plt.close()

            # Plot 2: Stage 2 confusion matrix
            if np.sum(correct_accel_mask) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                sns.heatmap(cm_hole, annot=True, fmt='d', cmap='Greens', ax=ax,
                           xticklabels=['NOLEAK', '1_16', '3_32', '1_8'],
                           yticklabels=['NOLEAK', '1_16', '3_32', '1_8'])
                ax.set_title('Stage 2: Hole Size Detection (Correct Accelerometer)\nConfusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                plt.tight_layout()
                plot_path = output_dir / "stage2_confusion_matrix.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                LOGGER.info(f"Saved: {plot_path}")
                plt.close()

            # Plot 3: Confidence distribution
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Correct vs incorrect predictions
            correct_mask = (pred_accel_ids == y_accel_test) & (pred_hole_ids == y_hole_test)
            axes[0].hist(confidences[correct_mask], bins=20, alpha=0.7, label='Correct', color='green')
            axes[0].hist(confidences[~correct_mask], bins=20, alpha=0.7, label='Incorrect', color='red')
            axes[0].set_xlabel('Combined Confidence')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Confidence Distribution: Correct vs Incorrect')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Per accelerometer
            for accel_id in np.unique(y_accel_test):
                mask = y_accel_test == accel_id
                axes[1].hist(confidences[mask], bins=20, alpha=0.5, label=f'Accel {accel_id}')
            axes[1].set_xlabel('Combined Confidence')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Confidence Distribution by Accelerometer')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / "confidence_distributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            LOGGER.info(f"Saved: {plot_path}")
            plt.close()

            # Plot 4: Per-accelerometer accuracy
            accel_ids = list(results['per_accelerometer_accuracy'].keys())
            accuracies = [results['per_accelerometer_accuracy'][i] for i in accel_ids]

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            bars = ax.bar([f'Accel {i}' for i in accel_ids], accuracies, color=['blue', 'green', 'red'])
            ax.set_ylabel('Overall Accuracy')
            ax.set_title('Per-Accelerometer Overall Accuracy\n(Both Stages Correct)')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{100*acc:.1f}%',
                       ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plot_path = output_dir / "per_accelerometer_accuracy.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            LOGGER.info(f"Saved: {plot_path}")
            plt.close()

            LOGGER.info("Visualization plots generated")

        LOGGER.info("\n" + "="*60)
        LOGGER.info("EVALUATION COMPLETED")
        LOGGER.info("="*60)
        LOGGER.info(f"Overall Accuracy: {100*results['overall_accuracy']:.2f}%")

        return 0

    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)

    return evaluate_two_stage_classifier(args)


if __name__ == "__main__":
    sys.exit(main())
