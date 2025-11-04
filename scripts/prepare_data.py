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

        # Log configuration
        logger.info(f"Computing FFT: {args.compute_fft}")
        logger.info(f"Data augmentation: {args.augment}")
        logger.info(f"Stratified splitting: {args.stratified}")

        # Import necessary modules
        from src.data.data_loader import WebDAQDataLoader
        from src.data.preprocessor import SignalPreprocessor, PreprocessingConfig
        from src.data.data_splitter import DataSplitter
        from src.data.fft_processor import FlexibleFFTProcessor
        from src.data.augmentor import DataAugmentor
        import numpy as np

        # Load raw data
        logger.info("Loading raw data...")
        data_loader = WebDAQDataLoader(
            data_dir=str(raw_path),
            sample_rate=51200  # Default sample rate for WebDAQ
        )
        signals, labels, file_paths = data_loader.load_data()
        logger.info(f"Loaded {len(signals)} samples from {len(np.unique(labels))} classes")

        # Initialize preprocessor
        preproc_config = PreprocessingConfig(
            detrend_type='linear',
            window_type='hanning',
            fft_size=2048
        )
        preprocessor = SignalPreprocessor(preproc_config)

        # Preprocess signals
        logger.info("Preprocessing signals...")
        processed_signals = []
        for i, signal in enumerate(signals):
            processed = preprocessor.preprocess(signal)
            processed_signals.append(processed)
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(signals)} signals")
        processed_signals = np.array(processed_signals)

        # Compute FFT if requested
        fft_features = None
        if args.compute_fft:
            logger.info("Computing FFT features...")
            fft_processor = FlexibleFFTProcessor(
                fft_size=2048,
                sample_rate=51200,
                method='scipy'
            )
            fft_features = []
            for i, signal in enumerate(processed_signals):
                fft_result = fft_processor.compute_fft(signal)
                fft_features.append(fft_result)
                if (i + 1) % 100 == 0:
                    logger.info(f"Computed FFT for {i + 1}/{len(processed_signals)} signals")
            fft_features = np.array(fft_features)

        # Split data
        logger.info("Splitting data into train/val/test sets...")
        splitter = DataSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            stratified=args.stratified
        )
        train_idx, val_idx, test_idx = splitter.split(processed_signals, labels)

        logger.info(f"Train set: {len(train_idx)} samples")
        logger.info(f"Validation set: {len(val_idx)} samples")
        logger.info(f"Test set: {len(test_idx)} samples")

        # Apply augmentation to training data if requested
        augmented_train_signals = processed_signals[train_idx]
        augmented_train_labels = labels[train_idx]

        if args.augment:
            logger.info("Applying data augmentation to training set...")
            augmentor = DataAugmentor(
                noise_level=0.01,
                shift_range=100,
                scale_range=(0.9, 1.1)
            )

            aug_signals = []
            aug_labels = []
            for i, (signal, label) in enumerate(zip(processed_signals[train_idx], labels[train_idx])):
                # Add original
                aug_signals.append(signal)
                aug_labels.append(label)

                # Add augmented versions
                aug_signal = augmentor.augment(signal)
                aug_signals.append(aug_signal)
                aug_labels.append(label)

            augmented_train_signals = np.array(aug_signals)
            augmented_train_labels = np.array(aug_labels)
            logger.info(f"Augmented training set from {len(train_idx)} to {len(augmented_train_signals)} samples")

        # Save processed data
        logger.info("Saving processed data...")

        # Save train set
        np.save(output_path / "train" / "signals.npy", augmented_train_signals)
        np.save(output_path / "train" / "labels.npy", augmented_train_labels)
        if args.compute_fft:
            train_fft = fft_features[train_idx]
            if args.augment:
                # Duplicate FFT for augmented samples
                train_fft = np.repeat(train_fft, 2, axis=0)
            np.save(output_path / "train" / "fft_features.npy", train_fft)

        # Save validation set
        np.save(output_path / "val" / "signals.npy", processed_signals[val_idx])
        np.save(output_path / "val" / "labels.npy", labels[val_idx])
        if args.compute_fft:
            np.save(output_path / "val" / "fft_features.npy", fft_features[val_idx])

        # Save test set
        np.save(output_path / "test" / "signals.npy", processed_signals[test_idx])
        np.save(output_path / "test" / "labels.npy", labels[test_idx])
        if args.compute_fft:
            np.save(output_path / "test" / "fft_features.npy", fft_features[test_idx])

        # Save metadata
        metadata = {
            'num_classes': len(np.unique(labels)),
            'class_names': [str(c) for c in np.unique(labels)],
            'signal_shape': processed_signals[0].shape,
            'train_samples': len(augmented_train_signals),
            'val_samples': len(val_idx),
            'test_samples': len(test_idx),
            'augmented': args.augment,
            'fft_computed': args.compute_fft,
            'preprocessing': {
                'detrend_type': preproc_config.detrend_type,
                'window_type': preproc_config.window_type,
                'fft_size': preproc_config.fft_size
            }
        }

        import json
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {output_path / 'metadata.json'}")
        
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