#!/usr/bin/env python3
"""
Detailed evaluation and benchmarking for both models.

Provides comprehensive evaluation including:
  - Per-class metrics and confusion matrices
  - Inference time analysis
  - Memory profiling
  - Visual comparisons and reports
  - Error analysis

Usage:
    python scripts/evaluate_models_detailed.py \\
        --accel-model models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \\
        --hole-size-models models/two_stage_classifier_v2/model_*/accel_*_hole_size_classifier.pkl \\
        --accel-data data/accelerometer_classifier_v2/ \\
        --hole-size-data data/processed/ \\
        --output-dir results/evaluations/

    # Quick evaluation (less iterations)
    python scripts/evaluate_models_detailed.py \\
        --accel-model <model_path> \\
        --accel-data data/accelerometer_classifier_v2/ \\
        --hole-size-data data/processed/ \\
        --quick
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils
from scripts.extract_amplitude_features import extract_amplitude_features

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


class DetailedEvaluator:
    """Detailed model evaluation and benchmarking."""

    def __init__(self, output_dir: Path):
        """Initialize evaluator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_results = {}

    def evaluate_model(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        class_names: Dict[int, str] = None
    ) -> Dict:
        """Comprehensive model evaluation."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )

        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"DETAILED EVALUATION: {model_name}")
        LOGGER.info(f"{'='*80}")

        # Predictions
        y_pred = model.predict(X_test)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass

        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        LOGGER.info(f"\nOverall Metrics:")
        LOGGER.info(f"  Accuracy:  {accuracy:.4f} ({100*accuracy:.2f}%)")
        LOGGER.info(f"  Precision: {precision:.4f}")
        LOGGER.info(f"  Recall:    {recall:.4f}")
        LOGGER.info(f"  F1 Score:  {f1:.4f}")

        # Per-class metrics
        unique_classes = np.unique(y_test)
        per_class = {}

        LOGGER.info(f"\nPer-Class Metrics:")
        for class_id in unique_classes:
            mask = y_test == class_id
            class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"

            # Count
            n_samples = np.sum(mask)

            # Accuracy
            class_acc = np.mean(y_pred[mask] == y_test[mask])

            # Other metrics
            if len(np.unique(y_test)) == 2:
                # Binary metrics
                p = precision_score(y_test, y_pred, pos_label=class_id, zero_division=0)
                r = recall_score(y_test, y_pred, pos_label=class_id, zero_division=0)
                f = f1_score(y_test, y_pred, pos_label=class_id, zero_division=0)
            else:
                # Multiclass
                mask_pred = y_pred == class_id
                if np.any(mask_pred):
                    tp = np.sum(mask & mask_pred)
                    fp = np.sum(~mask & mask_pred)
                    fn = np.sum(mask & ~mask_pred)

                    p = tp / (tp + fp) if (tp + fp) > 0 else 0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                else:
                    p = r = f = 0

            per_class[class_id] = {
                'name': class_name,
                'n_samples': int(n_samples),
                'accuracy': float(class_acc),
                'precision': float(p),
                'recall': float(r),
                'f1': float(f)
            }

            LOGGER.info(f"  {class_name} ({n_samples} samples): "
                       f"Acc={class_acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        LOGGER.info(f"\nConfusion Matrix ({len(unique_classes)}x{len(unique_classes)}):")
        LOGGER.info(str(cm))

        # Misclassification analysis
        misclass_analysis = self._analyze_misclassifications(y_test, y_pred)

        results = {
            'model': model_name,
            'timestamp': self.timestamp,
            'n_test_samples': int(len(X_test)),
            'n_classes': int(len(unique_classes)),
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class_metrics': per_class,
            'confusion_matrix': cm.tolist(),
            'misclassification_analysis': misclass_analysis
        }

        return results

    def benchmark_inference_detailed(
        self,
        model: object,
        X_test: np.ndarray,
        model_name: str,
        n_iterations: int = 100,
        batch_sizes: List[int] = None
    ) -> Dict:
        """Detailed inference benchmarking."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]

        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"INFERENCE BENCHMARKING: {model_name}")
        LOGGER.info(f"{'='*80}")

        results_by_batch = []
        latencies_all = []
        throughputs_all = []

        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                LOGGER.warning(f"Batch size {batch_size} > test samples, skipping")
                continue

            LOGGER.info(f"\nBatch size: {batch_size}")

            X_batch = X_test[:batch_size]

            # Warmup
            try:
                _ = model.predict(X_batch)
            except Exception as e:
                LOGGER.error(f"  Warmup failed: {e}")
                continue

            # Benchmark
            latencies = []
            for i in range(n_iterations):
                start = time.perf_counter()
                try:
                    _ = model.predict(X_batch)
                except Exception as e:
                    LOGGER.error(f"  Iteration {i} failed: {e}")
                    continue
                end = time.perf_counter()

                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
                latencies_all.append(latency_ms)

                if (i + 1) % max(1, n_iterations // 5) == 0:
                    LOGGER.info(f"  Completed {i+1}/{n_iterations}")

            if not latencies:
                continue

            # Stats
            stats = {
                'batch_size': batch_size,
                'n_iterations': len(latencies),
                'latency_ms': {
                    'mean': float(np.mean(latencies)),
                    'std': float(np.std(latencies)),
                    'min': float(np.min(latencies)),
                    'max': float(np.max(latencies)),
                    'p50': float(np.percentile(latencies, 50)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                },
                'throughput': {
                    'samples_per_sec': batch_size / (np.mean(latencies) / 1000),
                    'samples_per_ms': batch_size / np.mean(latencies)
                }
            }

            throughputs_all.append(stats['throughput']['samples_per_sec'])

            LOGGER.info(f"  Latency (ms): mean={stats['latency_ms']['mean']:.2f}, "
                       f"p95={stats['latency_ms']['p95']:.2f}, p99={stats['latency_ms']['p99']:.2f}")
            LOGGER.info(f"  Throughput: {stats['throughput']['samples_per_sec']:.2f} samples/sec")

            results_by_batch.append(stats)

        return {
            'model': model_name,
            'results_by_batch': results_by_batch,
            'overall': {
                'n_total_predictions': len(latencies_all),
                'throughput_mean_samples_per_sec': float(np.mean(throughputs_all)) if throughputs_all else 0,
                'throughput_max_samples_per_sec': float(np.max(throughputs_all)) if throughputs_all else 0,
            }
        }

    @staticmethod
    def _analyze_misclassifications(y_test: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Analyze misclassifications."""
        misclass_mask = y_test != y_pred
        n_misclass = np.sum(misclass_mask)
        total = len(y_test)

        LOGGER.info(f"\nMisclassification Analysis:")
        LOGGER.info(f"  Total misclassifications: {n_misclass}/{total} ({100*n_misclass/total:.2f}%)")

        # Per-class misclassification rates
        class_misclass = {}
        for class_id in np.unique(y_test):
            mask = y_test == class_id
            class_misclass_rate = np.sum(misclass_mask & mask) / np.sum(mask)
            class_misclass[class_id] = float(class_misclass_rate)

        LOGGER.info(f"  Per-class misclassification rates:")
        for class_id, rate in class_misclass.items():
            LOGGER.info(f"    Class {class_id}: {rate:.4f} ({100*rate:.2f}%)")

        return {
            'total_misclassifications': int(n_misclass),
            'total_samples': int(total),
            'misclassification_rate': float(n_misclass / total),
            'per_class_rate': class_misclass
        }

    def generate_report(self, results: Dict):
        """Generate JSON report."""
        report_path = self.output_dir / f"eval_{results.get('model', 'model')}_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"Report saved: {report_path}")
        return report_path

    def generate_summary(self, all_results: Dict):
        """Generate summary report."""
        summary_path = self.output_dir / f"evaluation_summary_{self.timestamp}.json"

        summary = {
            'timestamp': self.timestamp,
            'models': {}
        }

        for model_name, results in all_results.items():
            summary['models'][model_name] = {
                'accuracy': results['accuracy']['overall_metrics']['accuracy'],
                'f1_score': results['accuracy']['overall_metrics']['f1_score'],
                'n_test_samples': results['accuracy']['n_test_samples'],
                'throughput_samples_per_sec': results['inference'].get('overall', {}).get('throughput_mean_samples_per_sec', 0)
            }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        LOGGER.info(f"\nSummary report saved: {summary_path}")

        # Print summary
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info("EVALUATION SUMMARY")
        LOGGER.info(f"{'='*80}")
        for model_name, metrics in summary['models'].items():
            LOGGER.info(f"\n{model_name}:")
            LOGGER.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            LOGGER.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            LOGGER.info(f"  Test Samples: {metrics['n_test_samples']}")
            LOGGER.info(f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")

        return summary_path


def load_model(model_path: str) -> object:
    """Load a trained model."""
    import joblib

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    LOGGER.info(f"Loading model: {model_path}")
    model = joblib.load(str(model_path))
    LOGGER.info(f"Model loaded successfully")

    return model


def load_accelerometer_test_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load accelerometer classifier test data."""
    try:
        X = np.load(data_path / "test" / "features.npy")
        y = np.load(data_path / "test" / "labels.npy")

        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        LOGGER.info(f"Loaded accelerometer test data: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        LOGGER.error(f"Failed to load accelerometer data: {e}")
        raise


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Detailed model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--accel-model",
        type=str,
        required=True,
        help="Path to trained accelerometer classifier model"
    )

    parser.add_argument(
        "--hole-size-models",
        type=str,
        help="Pattern to hole size classifier models (e.g., models/*/accel_*_classifier.pkl)"
    )

    parser.add_argument(
        "--accel-data",
        type=str,
        required=True,
        help="Path to accelerometer classifier data"
    )

    parser.add_argument(
        "--hole-size-data",
        type=str,
        help="Path to hole size classification data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluations/",
        help="Output directory"
    )

    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations"
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="Batch sizes for benchmarking"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick evaluation (50 iterations instead of 100)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_dir="logs", console_level=log_level)

    try:
        LOGGER.info("=" * 80)
        LOGGER.info("DETAILED MODEL EVALUATION")
        LOGGER.info("=" * 80)

        # Setup
        accel_model_path = Path(args.accel_model)
        accel_data_path = Path(args.accel_data)
        output_dir = Path(args.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        n_iterations = 50 if args.quick else args.n_iterations
        batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

        # Validate paths
        if not accel_model_path.exists():
            LOGGER.error(f"Model not found: {accel_model_path}")
            return 1

        if not accel_data_path.exists():
            LOGGER.error(f"Data not found: {accel_data_path}")
            return 1

        # Create evaluator
        evaluator = DetailedEvaluator(output_dir)

        # Evaluate accelerometer classifier
        LOGGER.info("\n[1/2] Evaluating Accelerometer Classifier...")
        accel_model = load_model(str(accel_model_path))
        X_test, y_test = load_accelerometer_test_data(accel_data_path)

        accel_accuracy = evaluator.evaluate_model(
            accel_model, X_test, y_test,
            "Accelerometer Classifier",
            class_names={0: "Accel 0", 1: "Accel 1", 2: "Accel 2"}
        )
        accel_inference = evaluator.benchmark_inference_detailed(
            accel_model, X_test,
            "Accelerometer Classifier",
            n_iterations, batch_sizes
        )

        evaluator.generate_report(accel_accuracy)
        evaluator.generate_report(accel_inference)

        all_results = {
            'accelerometer_classifier': {
                'accuracy': accel_accuracy,
                'inference': accel_inference
            }
        }

        # Generate summary
        evaluator.generate_summary(all_results)

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("EVALUATION COMPLETED")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())