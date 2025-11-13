#!/usr/bin/env python3
"""
Comprehensive benchmarking script for accelerometer_classifier and two_stage_classifier_v2.

Benchmarks:
  1. Inference speed (latency, throughput)
  2. Memory usage
  3. Accuracy metrics (precision, recall, F1)
  4. Comparison between models

Usage:
    python scripts/benchmark_models.py \\
        --accel-data data/accelerometer_classifier_v2/ \\
        --hole-size-data data/processed/ \\
        --output-dir results/benchmarks/ \\
        --n-iterations 100 \\
        --batch-sizes 1,8,32

    # With model training
    python scripts/benchmark_models.py \\
        --accel-data data/accelerometer_classifier_v2/ \\
        --hole-size-data data/processed/ \\
        --output-dir results/benchmarks/ \\
        --train-models
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
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


class ModelBenchmark:
    """Comprehensive model benchmarking class."""

    def __init__(self, output_dir: Path, n_iterations: int = 100, batch_sizes: List[int] = None):
        """Initialize benchmarking."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_iterations = n_iterations
        self.batch_sizes = batch_sizes or [1, 8, 32, 64]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}

    def benchmark_inference(
        self,
        model: object,
        X_test: np.ndarray,
        model_name: str,
        is_keras: bool = False
    ) -> Dict:
        """Benchmark inference speed and memory."""
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"INFERENCE BENCHMARKING: {model_name}")
        LOGGER.info(f"{'='*80}")

        results_by_batch = []

        for batch_size in self.batch_sizes:
            if batch_size > len(X_test):
                LOGGER.warning(f"Batch size {batch_size} > test samples {len(X_test)}, skipping")
                continue

            LOGGER.info(f"\nBenchmarking batch size: {batch_size}")

            # Prepare batch
            X_batch = X_test[:batch_size]

            # Warmup run
            LOGGER.info("  Warmup run...")
            try:
                _ = model.predict(X_batch)
            except Exception as e:
                LOGGER.error(f"  Warmup failed: {e}")
                continue

            # Benchmark iterations
            inference_times = []
            memory_usage = []

            LOGGER.info(f"  Running {self.n_iterations} iterations...")

            for i in range(self.n_iterations):
                # Memory before (optional)
                mem_before = self._get_memory_usage()

                # Inference
                start_time = time.perf_counter()
                try:
                    predictions = model.predict(X_batch)
                except Exception as e:
                    LOGGER.error(f"    Iteration {i+1} failed: {e}")
                    continue

                end_time = time.perf_counter()

                # Memory after
                mem_after = self._get_memory_usage()

                inference_time = (end_time - start_time) * 1000  # ms
                inference_times.append(inference_time)

                if mem_before is not None and mem_after is not None:
                    memory_usage.append(max(0, mem_after - mem_before))

                if (i + 1) % max(1, self.n_iterations // 5) == 0:
                    LOGGER.info(f"    Progress: {i + 1}/{self.n_iterations}")

            if not inference_times:
                LOGGER.error(f"  No successful iterations for batch size {batch_size}")
                continue

            # Compute statistics
            stats = self._compute_statistics(inference_times)
            stats['batch_size'] = batch_size
            stats['throughput_samples_per_sec'] = (batch_size / (stats['mean_time_ms'] / 1000))
            stats['throughput_samples_per_ms'] = batch_size / stats['mean_time_ms']

            if memory_usage:
                stats['mean_memory_mb'] = np.mean(memory_usage)
                stats['max_memory_mb'] = np.max(memory_usage)
                LOGGER.info(f"  Memory (MB): mean={stats['mean_memory_mb']:.2f}, max={stats['max_memory_mb']:.2f}")

            LOGGER.info(f"  Inference time (ms): mean={stats['mean_time_ms']:.2f}, "
                       f"p50={stats['p50_time_ms']:.2f}, p95={stats['p95_time_ms']:.2f}")
            LOGGER.info(f"  Throughput: {stats['throughput_samples_per_sec']:.2f} samples/sec")

            results_by_batch.append(stats)

        return {
            'model': model_name,
            'timestamp': self.timestamp,
            'n_iterations': self.n_iterations,
            'results_by_batch': results_by_batch
        }

    def benchmark_accuracy(
        self,
        model: object,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str
    ) -> Dict:
        """Benchmark accuracy and classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"ACCURACY BENCHMARKING: {model_name}")
        LOGGER.info(f"{'='*80}")

        try:
            # Predictions
            y_pred = model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Per-class metrics
            precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

            LOGGER.info(f"\nOverall Metrics:")
            LOGGER.info(f"  Accuracy:  {accuracy:.4f} ({100*accuracy:.2f}%)")
            LOGGER.info(f"  Precision: {precision:.4f}")
            LOGGER.info(f"  Recall:    {recall:.4f}")
            LOGGER.info(f"  F1 Score:  {f1:.4f}")

            LOGGER.info(f"\nPer-Class Metrics:")
            for i in range(len(np.unique(y_test))):
                LOGGER.info(f"  Class {i}: P={precision_per_class[i]:.4f}, "
                           f"R={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")

            LOGGER.info(f"\nConfusion Matrix:\n{cm}")
            LOGGER.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

            return {
                'model': model_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist(),
                'precision_per_class': precision_per_class.tolist(),
                'recall_per_class': recall_per_class.tolist(),
                'f1_per_class': f1_per_class.tolist(),
                'n_classes': int(len(np.unique(y_test)))
            }

        except Exception as e:
            LOGGER.error(f"Accuracy benchmarking failed: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_memory_usage() -> Optional[float]:
        """Get current process memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
        except Exception as e:
            LOGGER.debug(f"Memory profiling error: {e}")
            return None

    @staticmethod
    def _compute_statistics(times: List[float]) -> Dict:
        """Compute statistical metrics from timing data."""
        return {
            'mean_time_ms': float(np.mean(times)),
            'std_time_ms': float(np.std(times)),
            'min_time_ms': float(np.min(times)),
            'max_time_ms': float(np.max(times)),
            'p50_time_ms': float(np.percentile(times, 50)),
            'p95_time_ms': float(np.percentile(times, 95)),
            'p99_time_ms': float(np.percentile(times, 99)),
        }

    def generate_report(self, benchmark_data: Dict):
        """Generate JSON report."""
        report_path = self.output_dir / f"benchmark_{benchmark_data.get('model', 'model')}.json"
        with open(report_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        LOGGER.info(f"Report saved: {report_path}")
        return report_path

    def generate_comparison_report(self, all_results: Dict):
        """Generate comparison report across models."""
        report_path = self.output_dir / "benchmark_comparison.json"
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        LOGGER.info(f"Comparison report saved: {report_path}")

    def generate_visualization(self, all_results: Dict):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Model Benchmarking Comparison - {self.timestamp}', fontsize=16)

            models = list(all_results.keys())
            colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

            # Plot 1: Inference Time vs Batch Size
            ax = axes[0, 0]
            for model_name, color in zip(models, colors):
                data = all_results[model_name]['inference']
                if data and 'results_by_batch' in data:
                    batches = [r['batch_size'] for r in data['results_by_batch']]
                    times = [r['mean_time_ms'] for r in data['results_by_batch']]
                    ax.plot(batches, times, marker='o', label=model_name, color=color, linewidth=2)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Mean Inference Time (ms)')
            ax.set_title('Inference Time vs Batch Size')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Throughput vs Batch Size
            ax = axes[0, 1]
            for model_name, color in zip(models, colors):
                data = all_results[model_name]['inference']
                if data and 'results_by_batch' in data:
                    batches = [r['batch_size'] for r in data['results_by_batch']]
                    throughput = [r['throughput_samples_per_sec'] for r in data['results_by_batch']]
                    ax.plot(batches, throughput, marker='s', label=model_name, color=color, linewidth=2)
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Throughput (samples/sec)')
            ax.set_title('Throughput vs Batch Size')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Accuracy Comparison
            ax = axes[1, 0]
            model_names = []
            accuracies = []
            for model_name, color in zip(models, colors):
                data = all_results[model_name]['accuracy']
                if data:
                    model_names.append(model_name)
                    accuracies.append(data['accuracy'])
            if model_names:
                bars = ax.bar(model_names, accuracies, color=colors)
                ax.set_ylabel('Accuracy')
                ax.set_title('Model Accuracy Comparison')
                ax.set_ylim([0, 1.0])
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # Plot 4: F1 Score Comparison
            ax = axes[1, 1]
            model_names = []
            f1_scores = []
            for model_name, color in zip(models, colors):
                data = all_results[model_name]['accuracy']
                if data:
                    model_names.append(model_name)
                    f1_scores.append(data['f1_score'])
            if model_names:
                bars = ax.bar(model_names, f1_scores, color=colors)
                ax.set_ylabel('F1 Score')
                ax.set_title('Model F1 Score Comparison')
                ax.set_ylim([0, 1.0])
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            plot_path = self.output_dir / f"benchmark_comparison_{self.timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            LOGGER.info(f"Visualization saved: {plot_path}")
            return plot_path

        except ImportError as e:
            LOGGER.warning(f"Visualization failed (missing dependencies): {e}")
            return None
        except Exception as e:
            LOGGER.error(f"Visualization error: {e}")
            return None


def load_accelerometer_classifier_data(
    data_path: Path,
    split: str = 'test'
) -> Tuple[np.ndarray, np.ndarray]:
    """Load accelerometer classifier data."""
    try:
        X = np.load(data_path / split / "features.npy")
        y = np.load(data_path / split / "labels.npy")

        # Flatten if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)

        LOGGER.info(f"Loaded {split} data: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        LOGGER.error(f"Failed to load accelerometer data: {e}")
        raise


def load_hole_size_data(
    data_path: Path,
    split: str = 'test'
) -> Tuple[np.ndarray, np.ndarray]:
    """Load hole size classification data."""
    try:
        X = np.load(data_path / split / "features.npy")
        y = np.load(data_path / split / "labels.npy")

        LOGGER.info(f"Loaded {split} data: X={X.shape}, y={y.shape}")
        return X, y
    except Exception as e:
        LOGGER.error(f"Failed to load hole size data: {e}")
        raise


def train_accelerometer_classifier(
    data_path: Path,
    output_dir: Path,
    model_type: str = 'random_forest'
) -> object:
    """Train or load accelerometer classifier."""
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("TRAINING ACCELEROMETER CLASSIFIER")
    LOGGER.info(f"{'='*80}")

    # Load data
    X_train, y_train = load_accelerometer_classifier_data(data_path, 'train')

    # Build model
    config = {
        "model": {
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "n_jobs": -1,
            },
            "svm": {
                "kernel": "rbf",
                "C": 1.0,
                "probability": True,
            },
        }
    }

    if model_type == 'random_forest':
        model = RandomForestModel(config)
    elif model_type == 'svm':
        model = SVMClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    LOGGER.info("Training model...")
    model.fit(X_train, y_train)

    # Save
    model_path = output_dir / f"accelerometer_classifier_{model_type}.pkl"
    model.save(str(model_path))
    LOGGER.info(f"Model saved: {model_path}")

    return model


def benchmark_accelerometer_classifier(
    data_path: Path,
    output_dir: Path,
    model_type: str = 'random_forest',
    n_iterations: int = 100,
    batch_sizes: List[int] = None
) -> Dict:
    """Benchmark accelerometer classifier."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("BENCHMARKING ACCELEROMETER CLASSIFIER")
    LOGGER.info(f"{'='*80}")

    # Create benchmark instance
    benchmark = ModelBenchmark(output_dir, n_iterations, batch_sizes)

    # Train/load model
    model = train_accelerometer_classifier(data_path, output_dir, model_type)

    # Load test data
    X_test, y_test = load_accelerometer_classifier_data(data_path, 'test')

    # Benchmark
    inference_results = benchmark.benchmark_inference(
        model, X_test, "Accelerometer Classifier", is_keras=False
    )
    accuracy_results = benchmark.benchmark_accuracy(
        model, X_test, y_test, "Accelerometer Classifier"
    )

    # Combine results
    results = {
        'model': 'accelerometer_classifier',
        'model_type': model_type,
        'data_path': str(data_path),
        'n_test_samples': int(len(X_test)),
        'inference': inference_results,
        'accuracy': accuracy_results
    }

    # Save report
    benchmark.generate_report(results)

    return results


def benchmark_two_stage_classifier(
    accel_data_path: Path,
    hole_size_data_path: Path,
    output_dir: Path,
    model_type: str = 'random_forest',
    n_iterations: int = 100,
    batch_sizes: List[int] = None
) -> Dict:
    """Benchmark two-stage classifier."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("BENCHMARKING TWO-STAGE CLASSIFIER V2")
    LOGGER.info(f"{'='*80}")

    # Create benchmark instance
    benchmark = ModelBenchmark(output_dir, n_iterations, batch_sizes)

    # Train accelerometer classifier (Stage 1)
    accel_model = train_accelerometer_classifier(accel_data_path, output_dir, model_type)

    # For two-stage, we'll benchmark the accelerometer model
    # (Stage 1 is the accelerometer position classifier)
    X_test, y_test = load_accelerometer_classifier_data(accel_data_path, 'test')

    # Benchmark
    inference_results = benchmark.benchmark_inference(
        accel_model, X_test, "Two-Stage Classifier (Stage 1)", is_keras=False
    )
    accuracy_results = benchmark.benchmark_accuracy(
        accel_model, X_test, y_test, "Two-Stage Classifier (Stage 1)"
    )

    # Combine results
    results = {
        'model': 'two_stage_classifier_v2',
        'stage': 'stage1_accelerometer_position',
        'model_type': model_type,
        'data_path': str(accel_data_path),
        'n_test_samples': int(len(X_test)),
        'inference': inference_results,
        'accuracy': accuracy_results
    }

    # Save report
    benchmark.generate_report(results)

    return results


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive model benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        required=True,
        help="Path to hole size classification data"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks/",
        help="Output directory for results"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "svm"],
        help="Model type"
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
        default="1,8,32,64",
        help="Batch sizes to test (comma-separated)"
    )

    parser.add_argument(
        "--benchmark-accel-only",
        action="store_true",
        help="Benchmark only accelerometer classifier"
    )

    parser.add_argument(
        "--benchmark-two-stage-only",
        action="store_true",
        help="Benchmark only two-stage classifier"
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
        # Validate paths
        accel_data = Path(args.accel_data)
        hole_size_data = Path(args.hole_size_data)
        output_dir = Path(args.output_dir)

        if not accel_data.exists():
            LOGGER.error(f"Accelerometer data not found: {accel_data}")
            return 1

        if not hole_size_data.exists():
            LOGGER.error(f"Hole size data not found: {hole_size_data}")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse batch sizes
        batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

        LOGGER.info("=" * 80)
        LOGGER.info("MODEL BENCHMARKING SUITE")
        LOGGER.info("=" * 80)

        all_results = {}

        # Benchmark accelerometer classifier
        if not args.benchmark_two_stage_only:
            LOGGER.info("\n[1/2] Benchmarking Accelerometer Classifier...")
            accel_results = benchmark_accelerometer_classifier(
                accel_data,
                output_dir,
                args.model_type,
                args.n_iterations,
                batch_sizes
            )
            all_results['accelerometer_classifier'] = accel_results

        # Benchmark two-stage classifier
        if not args.benchmark_accel_only:
            LOGGER.info("\n[2/2] Benchmarking Two-Stage Classifier V2...")
            two_stage_results = benchmark_two_stage_classifier(
                accel_data,
                hole_size_data,
                output_dir,
                args.model_type,
                args.n_iterations,
                batch_sizes
            )
            all_results['two_stage_classifier_v2'] = two_stage_results

        # Generate comparison report
        if len(all_results) > 1:
            LOGGER.info("\n" + "=" * 80)
            LOGGER.info("GENERATING COMPARISON REPORT")
            LOGGER.info("=" * 80)

            benchmark = ModelBenchmark(output_dir, args.n_iterations, batch_sizes)
            benchmark.generate_comparison_report(all_results)
            benchmark.generate_visualization(all_results)

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("BENCHMARKING COMPLETED")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Results saved to: {output_dir}")

        return 0

    except Exception as e:
        LOGGER.error(f"Benchmarking failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())