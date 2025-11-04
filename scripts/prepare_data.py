#!/usr/bin/env python3
"""
Prepare raw data for model training.

Loads raw WebDAQ CSV files, applies preprocessing, and creates
train/validation/test splits. Supports FFT computation and augmentation.

Usage:
    python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare raw data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic data preparation
  python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/
  
  # With custom splits
  python scripts/prepare_data.py \\
      --raw-data data/raw/ \\
      --output-dir data/processed/ \\
      --train-ratio 0.6 \\
      --val-ratio 0.2
        """
    )
    
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw/",
        help="Path to raw data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/",
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    
    parser.add_argument(
        "--compute-fft",
        action="store_true",
        help="Compute FFT features"
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation"
    )
    
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified splitting"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def validate_splits(train_ratio: float, val_ratio: float) -> bool:
    """Validate data split ratios."""
    logger = get_logger(__name__)
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        logger.error("All split ratios must be positive")
        return False
    
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        logger.error("Train and validation ratios must be between 0 and 1")
        return False
    
    logger.info(f"Split ratios - Train: {train_ratio:.2%}, Val: {val_ratio:.2%}, Test: {test_ratio:.2%}")
    return True


def prepare_data(args):
    """Prepare data for training."""
    logger = get_logger(__name__)
    
    try:
        # Validate splits
        if not validate_splits(args.train_ratio, args.val_ratio):
            return 1
        
        # Check input directory
        raw_path = Path(args.raw_data)
        if not raw_path.exists():
            logger.error(f"Raw data directory not found: {args.raw_data}")
            return 1
        
        # Create output directories
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        for split in ["train", "val", "test"]:
            FileUtils.ensure_directory(str(output_path / split))
        
        logger.info(f"Raw data path: {args.raw_data}")
        logger.info(f"Output path: {args.output_dir}")
        
        # TODO: Implement actual data processing logic
        logger.info("Data preparation logic to be implemented with Phase 2 (Data Pipeline)")
        
        # Log configuration
        logger.info(f"Computing FFT: {args.compute_fft}")
        logger.info(f"Data augmentation: {args.augment}")
        logger.info(f"Stratified splitting: {args.stratified}")
        
        logger.info("Data preparation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("DATA PREPARATION - Prepare Data for Training")
    logger.info("=" * 60)
    
    return prepare_data(args)


if __name__ == "__main__":
    sys.exit(main())