#!/usr/bin/env python3
"""
Aggregate processed data and train all models.

This script:
1. Loads individual .npz processed samples
2. Aggregates them into train/val/test .npy arrays
3. Trains all available models in sequence
4. Logs results

Usage:
    python scripts/train_all_models.py
    python scripts/train_all_models.py --models cnn_1d random_forest lstm
    python scripts/train_all_models.py --skip-aggregation  # Skip data prep if already done
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def aggregate_npz_to_npy(
    data_dir: str = "data/processed/",
    force: bool = False
) -> bool:
    """Aggregate individual .npz files into train/val/test .npy arrays."""
    logger = get_logger(__name__)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False
    
    # Check if aggregation already done
    train_features = data_path / "train" / "fft_features.npy"
    if train_features.exists() and not force:
        logger.info("✓ Data already aggregated (use --force to re-aggregate)")
        return True
    
    logger.info("Aggregating processed data from .npz files...")
    
    try:
        aggregated_data = {
            "train": {"features": [], "labels": []},
            "val": {"features": [], "labels": []},
            "test": {"features": [], "labels": []},
        }
        
        # Load all .npz files from each split
        for split in ["train", "val", "test"]:
            split_dir = data_path / split
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}, skipping")
                continue
            
            npz_files = list(split_dir.glob("*.npz"))
            if not npz_files:
                logger.warning(f"No .npz files found in {split_dir}")
                continue
            
            logger.info(f"Loading {len(npz_files)} files from {split}...")
            
            for npz_file in npz_files:
                try:
                    data = np.load(npz_file, allow_pickle=True)
                    
                    # Extract Welch PSD (preferred) or FFT magnitude
                    if "welch_psd" in data:
                        features = data["welch_psd"]
                    elif "fft_magnitude" in data:
                        features = data["fft_magnitude"]
                    else:
                        logger.warning(f"No features in {npz_file.name}, skipping")
                        continue
                    
                    label = int(data["label"])
                    
                    aggregated_data[split]["features"].append(features)
                    aggregated_data[split]["labels"].append(label)
                
                except Exception as e:
                    logger.warning(f"Failed to load {npz_file.name}: {e}")
                    continue
        
        # Convert to numpy arrays and save
        for split in ["train", "val", "test"]:
            if not aggregated_data[split]["features"]:
                logger.warning(f"No data for split: {split}")
                continue
            
            features = np.array(aggregated_data[split]["features"], dtype=np.float32)
            labels = np.array(aggregated_data[split]["labels"], dtype=np.int32)
            
            split_dir = data_path / split
            np.save(split_dir / "fft_features.npy", features)
            np.save(split_dir / "labels.npy", labels)
            
            logger.info(f"  {split}: {features.shape[0]} samples, {features.shape[1:]} feature shape")
        
        logger.info("✓ Data aggregation complete")
        return True
    
    except Exception as e:
        logger.error(f"Data aggregation failed: {e}", exc_info=True)
        return False


def train_model(model_type: str, epochs: int = 50, batch_size: int = 32) -> bool:
    """Train a single model using train_model.py."""
    logger = get_logger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type.upper()}")
    logger.info(f"{'='*60}")
    
    cmd = [
        "python",
        "scripts/train_model.py",
        "--model-type", model_type,
        "--use-fft",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            logger.info(f"✓ {model_type.upper()} training completed successfully")
            return True
        else:
            logger.error(f"✗ {model_type.upper()} training failed with exit code {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to train {model_type}: {e}")
        return False


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Aggregate data and train all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models (default)
  python scripts/train_all_models.py
  
  # Train specific models only
  python scripts/train_all_models.py --models cnn_1d random_forest
  
  # Skip data aggregation (if already done)
  python scripts/train_all_models.py --skip-aggregation
  
  # Re-aggregate and train with custom epochs
  python scripts/train_all_models.py --force-aggregation --epochs 100
        """
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["cnn_1d", "lstm", "random_forest", "svm"],
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost", "ensemble"],
        help="Models to train (default: cnn_1d, lstm, random_forest, svm)"
    )
    
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip data aggregation (if already done)"
    )
    
    parser.add_argument(
        "--force-aggregation",
        action="store_true",
        help="Force re-aggregation even if .npy files exist"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/",
        help="Path to processed data directory"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs for training (default: 50)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def main():
    """Main entry point."""
    setup_logging()
    logger = get_logger(__name__)
    
    parser = create_parser()
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("AIR LEAK DETECTION - TRAINING ALL MODELS")
    logger.info("="*60)
    
    # Step 1: Aggregate data
    if args.skip_aggregation:
        logger.info("Skipping data aggregation (--skip-aggregation)")
    else:
        if not aggregate_npz_to_npy(args.data_dir, force=args.force_aggregation):
            logger.error("Failed to aggregate data. Exiting.")
            return 1
    
    # Step 2: Train models
    logger.info(f"\nTraining models: {', '.join(args.models)}")
    
    results = {}
    for model_type in args.models:
        success = train_model(model_type, args.epochs, args.batch_size)
        results[model_type] = "✓ SUCCESS" if success else "✗ FAILED"
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    for model, status in results.items():
        logger.info(f"  {model:20s} : {status}")
    logger.info(f"{'='*60}\n")
    
    # Check if all succeeded
    all_success = all(v == "✓ SUCCESS" for v in results.values())
    
    if all_success:
        logger.info("✓ All models trained successfully!")
        return 0
    else:
        logger.warning("⚠ Some models failed to train. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())