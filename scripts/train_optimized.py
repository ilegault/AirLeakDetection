#!/usr/bin/env python3
"""
Optimized training script for Lenovo Legion 5 with 16 CPU cores.

This script configures TensorFlow to use all available CPU cores efficiently
and provides options for future GPU training.

Usage:
    python train_optimized.py --models cnn_1d lstm
    python train_optimized.py --all  # Train all models
    python train_optimized.py --parallel  # Train compatible models in parallel
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List
import multiprocessing as mp

# Optimize TensorFlow settings BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
os.environ['OMP_NUM_THREADS'] = '16'  # OpenMP threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'  # Intra-op parallelism
os.environ['TF_NUM_INTEROP_THREADS'] = '2'  # Inter-op parallelism

import tensorflow as tf
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging, get_logger

# Configure TensorFlow for maximum CPU performance
def configure_tensorflow():
    """Configure TensorFlow for optimal CPU performance."""
    logger = get_logger(__name__)

    # Set thread counts
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    # Check available devices
    cpus = tf.config.list_physical_devices('CPU')
    gpus = tf.config.list_physical_devices('GPU')

    logger.info(f"CPU devices: {len(cpus)}")
    logger.info(f"GPU devices: {len(gpus)}")

    if gpus:
        try:
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
    else:
        logger.info("No GPU detected - using CPU with 16 threads")

    # Enable mixed precision for faster training (works on CPU too)
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("Mixed precision enabled (float16)")
    except:
        logger.info("Mixed precision not available, using float32")

    return len(gpus) > 0


def train_single_model(model_type: str, epochs: int = 100, batch_size: int = 64) -> bool:
    """Train a single model with optimized settings."""
    logger = get_logger(__name__)

    logger.info(f"\n{'='*70}")
    logger.info(f"Training {model_type.upper()}")
    logger.info(f"{'='*70}")

    cmd = [
        "python",
        "scripts/train_model.py",
        "--model-type", model_type,
        "--use-fft",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    import subprocess
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env={**os.environ}  # Pass optimized environment variables
        )
        if result.returncode == 0:
            logger.info(f"✓ {model_type.upper()} training completed successfully")
            return True
        else:
            logger.error(f"✗ {model_type.upper()} training failed")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to train {model_type}: {e}")
        return False


def train_sequential(models: List[str], epochs: int, batch_size: int) -> dict:
    """Train models one after another (safest approach)."""
    logger = get_logger(__name__)
    logger.info("Training models SEQUENTIALLY for maximum stability")

    results = {}
    for model in models:
        success = train_single_model(model, epochs, batch_size)
        results[model] = "✓ SUCCESS" if success else "✗ FAILED"

    return results


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Optimized training for Lenovo Legion 5 (16 cores)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models sequentially (recommended for stability)
  python train_optimized.py --all

  # Train specific models
  python train_optimized.py --models cnn_1d lstm

  # Use larger batch size for faster training
  python train_optimized.py --all --batch-size 128

  # Quick training with fewer epochs
  python train_optimized.py --models random_forest svm --epochs 30
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all available models"
    )
    group.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost"],
        help="Specific models to train"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size - larger = faster but more RAM (default: 64)"
    )

    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip data aggregation step"
    )

    parser.add_argument(
        "--force-aggregation",
        action="store_true",
        help="Force re-aggregation of data"
    )

    return parser


def main():
    """Main entry point."""
    setup_logging()
    logger = get_logger(__name__)

    parser = create_parser()
    args = parser.parse_args()

    logger.info("="*70)
    logger.info("OPTIMIZED AIR LEAK DETECTION TRAINING")
    logger.info("Hardware: Lenovo Legion 5 17ITH6 (16 CPU cores)")
    logger.info("="*70)

    # Configure TensorFlow
    has_gpu = configure_tensorflow()

    # Determine which models to train
    if args.all:
        models = ["cnn_1d", "lstm", "random_forest", "svm"]
    else:
        models = args.models

    logger.info(f"\nModels to train: {', '.join(models)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"CPU threads: 16 (intra-op) + 2 (inter-op)")

    # Aggregate data if needed
    if not args.skip_aggregation:
        logger.info("\nStep 1: Aggregating data...")
        # Import aggregate function from train_all_models
        try:
            from train_all_models import aggregate_npz_to_npy
        except ImportError:
            # If direct import fails, use alternative approach
            import importlib.util
            spec = importlib.util.spec_from_file_location("train_all_models", PROJECT_ROOT / "scripts" / "train_all_models.py")
            train_all_models = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_all_models)
            aggregate_npz_to_npy = train_all_models.aggregate_npz_to_npy
        
        if not aggregate_npz_to_npy("data/processed/", args.force_aggregation):
            logger.error("Data aggregation failed!")
            return 1

    # Train models
    logger.info("\nStep 2: Training models...")
    results = train_sequential(models, args.epochs, args.batch_size)

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*70}")
    for model, status in results.items():
        logger.info(f"  {model:20s} : {status}")
    logger.info(f"{'='*70}\n")

    # Return appropriate exit code
    all_success = all("SUCCESS" in v for v in results.values())
    if all_success:
        logger.info("✓ All models trained successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Check results in ./models/ directory")
        logger.info("  2. Run evaluation: python scripts/evaluate_models.py")
        logger.info("  3. To enable GPU, see GPU_SETUP.md")
        return 0
    else:
        logger.warning("⚠ Some models failed to train. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
