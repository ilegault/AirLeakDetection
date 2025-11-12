#!/usr/bin/env python3
"""
Comprehensive evaluation for accelerometer classifier.

Generates detailed metrics, visualizations, and analysis:
- Confusion matrix (raw counts and percentages)
- Per-class precision, recall, F1-score
- Classification report
- ROC curves (one-vs-rest)
- Feature importance analysis
- Misclassification analysis

Usage:
    python scripts/evaluate_accelerometer_classifier.py \\
        --model-path models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \\
        --data-path data/accelerometer_classifier_v2/ \\
        --output-dir results/accelerometer_classifier/
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
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score,
    precision_recall_fscore_support
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, FileUtils
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
        description="Evaluate accelerometer classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.pkl file)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to accelerometer classification data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/accelerometer_classifier/",
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on (default: test)"
    )

    return parser


def load_model_and_data(model_path: Path, data_path: Path, split: str) -> Tuple:
    """Load trained model and test data.

    Returns:
        (model, X_test, y_test, feature_names)
    """
    LOGGER.info(f"Loading model from {model_path}")

    # Determine model type from path
    if "random_forest" in str(model_path):
        model = RandomForestModel.load(str(model_path))
    elif "svm" in str(model_path):
        model = SVMClassifier.load(str(model_path))
    else:
        raise ValueError(f"Cannot determine model type from path: {model_path}")

    # Load test data
    split_dir = data_path / split
    X_test = np.load(split_dir / "features.npy")
    y_test = np.load(split_dir / "labels.npy")

    LOGGER.info(f"Loaded {split} data: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Flatten if needed
    if len(X_test.shape) > 2:
        X_test = X_test.reshape(X_test.shape[0], -1)

    # Try to load feature names
    feature_names = None
    metadata_file = data_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            feature_names = metadata.get('feature_names', None)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]

    return model, X_test, y_test, feature_names


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot confusion matrix with both counts and percentages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    class_names = ['Accel 0\n(Closest)', 'Accel 1\n(Middle)', 'Accel 2\n(Farthest)']

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')

    # Percentages (normalized by row)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Percentages)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    LOGGER.info(f"Saved confusion matrix to {output_path}")


def plot_roc_curves(model, X_test: np.ndarray, y_test: np.ndarray, output_path: Path) -> Dict:
    """Plot ROC curves for each class (one-vs-rest).

    Returns:
        Dictionary with AUC scores
    """
    try:
        # Get probability predictions
        y_proba = model.predict_proba(X_test)

        fig, ax = plt.subplots(figsize=(10, 8))

        class_names = ['Accel 0 (Closest)', 'Accel 1 (Middle)', 'Accel 2 (Farthest)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        auc_scores = {}

        for i, (name, color) in enumerate(zip(class_names, colors)):
            # One-vs-rest
            y_true_binary = (y_test == i).astype(int)
            y_score = y_proba[:, i]

            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            auc_scores[f'class_{i}'] = float(roc_auc)

            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{name} (AUC = {roc_auc:.3f})')

        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        LOGGER.info(f"Saved ROC curves to {output_path}")

        return auc_scores

    except Exception as e:
        LOGGER.error(f"Failed to plot ROC curves: {e}")
        return {}


def plot_feature_importance(model, feature_names: list, output_path: Path, top_n: int = 15) -> None:
    """Plot feature importance (if model supports it)."""
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_

            # Get top N features
            indices = np.argsort(importances)[-top_n:][::-1]
            top_importances = importances[indices]
            top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}"
                          for i in indices]

            fig, ax = plt.subplots(figsize=(10, 8))

            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_importances, color='steelblue', edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.invert_yaxis()
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, v in enumerate(top_importances):
                ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            LOGGER.info(f"Saved feature importance to {output_path}")

        else:
            LOGGER.info("Model does not support feature importance")

    except Exception as e:
        LOGGER.error(f"Failed to plot feature importance: {e}")


def analyze_misclassifications(X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, feature_names: list,
                               output_path: Path) -> Dict:
    """Analyze misclassified samples.

    Returns:
        Dictionary with misclassification statistics
    """
    misclassified_mask = y_test != y_pred
    n_misclassified = np.sum(misclassified_mask)

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("MISCLASSIFICATION ANALYSIS")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"Total misclassified: {n_misclassified} / {len(y_test)} ({100*n_misclassified/len(y_test):.2f}%)")

    analysis = {
        'total_misclassified': int(n_misclassified),
        'misclassification_rate': float(n_misclassified / len(y_test)),
        'confusion_pairs': {}
    }

    if n_misclassified > 0:
        # Analyze confusion pairs
        for true_class in range(3):
            for pred_class in range(3):
                if true_class == pred_class:
                    continue

                mask = (y_test == true_class) & (y_pred == pred_class)
                count = np.sum(mask)

                if count > 0:
                    pair_key = f"{true_class}_as_{pred_class}"
                    analysis['confusion_pairs'][pair_key] = int(count)

                    LOGGER.info(f"\nAccel {true_class} misclassified as Accel {pred_class}: {count} samples")

                    # Get feature statistics for these misclassified samples
                    if count > 0:
                        misclass_samples = X_test[mask]
                        correct_samples = X_test[y_test == true_class]

                        LOGGER.info(f"  Average feature values:")
                        LOGGER.info(f"    Misclassified: mean={np.mean(misclass_samples):.6f}, std={np.std(misclass_samples):.6f}")
                        LOGGER.info(f"    Correct class: mean={np.mean(correct_samples):.6f}, std={np.std(correct_samples):.6f}")
    else:
        LOGGER.info("No misclassifications! Perfect classification.")

    # Save analysis
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    LOGGER.info(f"\nSaved misclassification analysis to {output_path}")

    return analysis


def generate_evaluation_report(model, X_test: np.ndarray, y_test: np.ndarray,
                               y_pred: np.ndarray, y_proba: np.ndarray,
                               auc_scores: Dict, output_path: Path) -> Dict:
    """Generate comprehensive evaluation report."""

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

    report = {
        'overall': {
            'accuracy': float(accuracy),
            'total_samples': int(len(y_test))
        },
        'per_class': {},
        'auc_scores': auc_scores
    }

    class_names = ['Accelerometer 0 (Closest)', 'Accelerometer 1 (Middle)', 'Accelerometer 2 (Farthest)']

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("EVALUATION REPORT")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"Overall Accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
    LOGGER.info(f"Total Samples: {len(y_test)}")
    LOGGER.info(f"\nPer-Class Metrics:")
    LOGGER.info(f"{'='*80}")

    for i, name in enumerate(class_names):
        report['per_class'][f'class_{i}'] = {
            'name': name,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }

        LOGGER.info(f"\n{name}:")
        LOGGER.info(f"  Precision: {precision[i]:.4f} ({100*precision[i]:.2f}%)")
        LOGGER.info(f"  Recall:    {recall[i]:.4f} ({100*recall[i]:.2f}%)")
        LOGGER.info(f"  F1-Score:  {f1[i]:.4f}")
        LOGGER.info(f"  Support:   {support[i]} samples")
        if f'class_{i}' in auc_scores:
            LOGGER.info(f"  AUC-ROC:   {auc_scores[f'class_{i}']:.4f}")

    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    LOGGER.info(f"\nSaved evaluation report to {output_path}")

    return report


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    if not model_path.exists():
        LOGGER.error(f"Model not found: {model_path}")
        return 1

    if not data_path.exists():
        LOGGER.error(f"Data path not found: {data_path}")
        return 1

    # Create output directory
    FileUtils.ensure_directory(str(output_dir))

    LOGGER.info("="*80)
    LOGGER.info("ACCELEROMETER CLASSIFIER EVALUATION")
    LOGGER.info("="*80)
    LOGGER.info(f"Model: {model_path}")
    LOGGER.info(f"Data: {data_path}")
    LOGGER.info(f"Split: {args.test_split}")
    LOGGER.info(f"Output: {output_dir}")

    # Load model and data
    model, X_test, y_test, feature_names = load_model_and_data(
        model_path, data_path, args.test_split
    )

    # Make predictions
    LOGGER.info("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Generate visualizations
    LOGGER.info("\nGenerating visualizations...")
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    auc_scores = plot_roc_curves(model, X_test, y_test, output_dir / "roc_curves.png")

    plot_feature_importance(model, feature_names, output_dir / "feature_importance.png")

    # Analyze misclassifications
    misclass_analysis = analyze_misclassifications(
        X_test, y_test, y_pred, feature_names,
        output_dir / "misclassification_analysis.json"
    )

    # Generate comprehensive report
    report = generate_evaluation_report(
        model, X_test, y_test, y_pred, y_proba, auc_scores,
        output_dir / "evaluation_report.json"
    )

    # Print classification report
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("SKLEARN CLASSIFICATION REPORT")
    LOGGER.info(f"{'='*80}")
    class_names = ['Accel 0', 'Accel 1', 'Accel 2']
    LOGGER.info(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    LOGGER.info("="*80)
    LOGGER.info("EVALUATION COMPLETED")
    LOGGER.info("="*80)
    LOGGER.info(f"Results saved to: {output_dir}")
    LOGGER.info(f"  - confusion_matrix.png")
    LOGGER.info(f"  - roc_curves.png")
    LOGGER.info(f"  - feature_importance.png")
    LOGGER.info(f"  - evaluation_report.json")
    LOGGER.info(f"  - misclassification_analysis.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
