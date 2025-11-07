#!/usr/bin/env python3
"""
Performance benchmarking for inference.

Measures inference speed, memory usage, and accuracy-speed tradeoff.

Usage:
    python scripts/benchmark.py --model-path models/best.h5 --test-data data/processed/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmarking
  python scripts/benchmark.py \\
      --model-path models/best.h5 \\
      --test-data data/processed/test/
  
  # With memory profiling
  python scripts/benchmark.py \\
      --model-path models/best.h5 \\
      --test-data data/processed/test/ \\
      --profile-memory
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Number of inference iterations"
    )
    
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,64",
        help="Batch sizes to test (comma-separated)"
    )
    
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile memory usage"
    )
    
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        help="Profile CPU usage"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def validate_inputs(args) -> bool:
    """Validate command-line arguments."""
    logger = get_logger(__name__)
    
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        return False
    
    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        return False
    
    if args.n_iterations <= 0:
        logger.error(f"Number of iterations must be positive: {args.n_iterations}")
        return False
    
    return True


def run_benchmarks(args):
    """Run performance benchmarks."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Number of iterations: {args.n_iterations}")
        logger.info(f"Batch sizes: {args.batch_sizes}")
        logger.info(f"Profile memory: {args.profile_memory}")
        logger.info(f"Profile CPU: {args.profile_cpu}")

        # Import necessary modules
        import numpy as np
        import tensorflow as tf
        import joblib
        import time
        import json
        from pathlib import Path

        # Load model
        logger.info("Loading model...")
        model_path_obj = Path(args.model_path)

        if model_path_obj.suffix in ['.h5', '.keras']:
            model = tf.keras.models.load_model(str(model_path_obj))
            is_keras_model = True
        elif model_path_obj.suffix in ['.pkl', '.joblib']:
            model = joblib.load(str(model_path_obj))
            is_keras_model = False
        else:
            logger.error(f"Unsupported model format: {model_path_obj.suffix}")
            return 1

        # Load test data
        logger.info("Loading test data...")
        test_path = Path(args.test_data)

        if test_path.is_dir():
            X_test = np.load(test_path / "signals.npy")
            y_test = np.load(test_path / "labels.npy")
        else:
            logger.error(f"Test data directory not found: {args.test_data}")
            return 1

        logger.info(f"Test data shape: {X_test.shape}")

        # Flatten for sklearn models if needed
        if not is_keras_model and len(X_test.shape) > 2:
            X_test = X_test.reshape(X_test.shape[0], -1)

        # Parse batch sizes
        batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
        logger.info(f"Testing batch sizes: {batch_sizes}")

        benchmark_results = []

        # Benchmark for each batch size
        for batch_size in batch_sizes:
            logger.info(f"\nBenchmarking with batch size: {batch_size}")

            # Select subset of data for benchmarking
            n_samples = min(batch_size * 10, len(X_test))
            X_bench = X_test[:n_samples]

            inference_times = []
            memory_usage = []

            # Warmup run
            logger.info("Performing warmup run...")
            if is_keras_model:
                _ = model.predict(X_bench[:batch_size], batch_size=batch_size, verbose=0)
            else:
                _ = model.predict(X_bench[:batch_size])

            # Benchmark iterations
            logger.info(f"Running {args.n_iterations} benchmark iterations...")

            for i in range(args.n_iterations):
                # Measure memory before inference (if profiling enabled)
                if args.profile_memory:
                    try:
                        import psutil
                        import os
                        process = psutil.Process(os.getpid())
                        mem_before = process.memory_info().rss / 1024 / 1024  # MB
                    except ImportError:
                        mem_before = None
                        if i == 0:
                            logger.warning("psutil not installed, memory profiling disabled")

                # Run inference and measure time
                start_time = time.perf_counter()

                if is_keras_model:
                    _ = model.predict(X_bench[:batch_size], batch_size=batch_size, verbose=0)
                else:
                    _ = model.predict(X_bench[:batch_size])

                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000  # Convert to ms

                inference_times.append(inference_time)

                # Measure memory after inference
                if args.profile_memory and mem_before is not None:
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_usage.append(mem_after - mem_before)

                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{args.n_iterations} iterations")

            # Calculate statistics
            mean_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            min_time = np.min(inference_times)
            max_time = np.max(inference_times)
            p50_time = np.percentile(inference_times, 50)
            p95_time = np.percentile(inference_times, 95)
            p99_time = np.percentile(inference_times, 99)

            # Throughput (samples per second)
            throughput = (batch_size / mean_time) * 1000

            logger.info(f"\nResults for batch size {batch_size}:")
            logger.info(f"  Mean inference time: {mean_time:.2f} ms")
            logger.info(f"  Std deviation: {std_time:.2f} ms")
            logger.info(f"  Min time: {min_time:.2f} ms")
            logger.info(f"  Max time: {max_time:.2f} ms")
            logger.info(f"  P50 (median): {p50_time:.2f} ms")
            logger.info(f"  P95: {p95_time:.2f} ms")
            logger.info(f"  P99: {p99_time:.2f} ms")
            logger.info(f"  Throughput: {throughput:.2f} samples/sec")

            result = {
                'batch_size': batch_size,
                'n_iterations': args.n_iterations,
                'mean_time_ms': float(mean_time),
                'std_time_ms': float(std_time),
                'min_time_ms': float(min_time),
                'max_time_ms': float(max_time),
                'p50_time_ms': float(p50_time),
                'p95_time_ms': float(p95_time),
                'p99_time_ms': float(p99_time),
                'throughput_samples_per_sec': float(throughput)
            }

            if args.profile_memory and memory_usage:
                mean_memory = np.mean(memory_usage)
                max_memory = np.max(memory_usage)
                result['mean_memory_mb'] = float(mean_memory)
                result['max_memory_mb'] = float(max_memory)
                logger.info(f"  Mean memory delta: {mean_memory:.2f} MB")
                logger.info(f"  Max memory delta: {max_memory:.2f} MB")

            benchmark_results.append(result)

        # Save benchmark results
        logger.info("\nSaving benchmark results...")
        results_file = output_path / "benchmark_results.json"

        results_summary = {
            'model_path': str(args.model_path),
            'test_data': str(args.test_data),
            'n_iterations': args.n_iterations,
            'profile_memory': args.profile_memory,
            'profile_cpu': args.profile_cpu,
            'results': benchmark_results
        }

        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)

        logger.info(f"Benchmark results saved to {results_file}")

        # Generate summary plot
        logger.info("Generating benchmark visualization...")
        import matplotlib.pyplot as plt

        batch_sizes_list = [r['batch_size'] for r in benchmark_results]
        mean_times = [r['mean_time_ms'] for r in benchmark_results]
        throughputs = [r['throughput_samples_per_sec'] for r in benchmark_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Inference time plot
        ax1.plot(batch_sizes_list, mean_times, marker='o', linewidth=2)
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Mean Inference Time (ms)')
        ax1.set_title('Inference Time vs Batch Size')
        ax1.grid(True, alpha=0.3)

        # Throughput plot
        ax2.plot(batch_sizes_list, throughputs, marker='s', linewidth=2, color='green')
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Throughput vs Batch Size')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / "benchmark_plot.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        logger.info(f"Benchmark plot saved to {plot_path}")
        logger.info("Benchmarking completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("BENCHMARKING - Performance Benchmarking")
    logger.info("=" * 60)
    
    return run_benchmarks(args)


if __name__ == "__main__":
    sys.exit(main())