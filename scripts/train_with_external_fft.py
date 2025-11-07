#!/usr/bin/env python3
"""
Train model with external (MATLAB) FFT data.

Loads teammate's MATLAB FFT data and trains model, with optional
FFT method comparison.

Usage:
    python scripts/train_with_external_fft.py --fft-source matlab --matlab-path data/matlab_fft/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for external FFT training."""
    parser = argparse.ArgumentParser(
        description="Train with external (MATLAB) FFT data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with MATLAB FFT
  python scripts/train_with_external_fft.py \\
      --fft-source matlab \\
      --matlab-path data/matlab_fft/
  
  # With FFT method comparison
  python scripts/train_with_external_fft.py \\
      --fft-source matlab \\
      --matlab-path data/matlab_fft/ \\
      --compare-methods
        """
    )
    
    parser.add_argument(
        "--fft-source",
        type=str,
        default="matlab",
        choices=["matlab", "scipy", "numpy"],
        help="FFT data source"
    )
    
    parser.add_argument(
        "--matlab-path",
        type=str,
        default="data/matlab_fft/",
        help="Path to MATLAB FFT data"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_1d",
        help="Model type to train"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare different FFT methods"
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
    
    # Check MATLAB path
    matlab_path = Path(args.matlab_path)
    if not matlab_path.exists():
        logger.error(f"MATLAB data path not found: {args.matlab_path}")
        return False
    
    # Validate epochs
    if args.epochs <= 0:
        logger.error(f"Epochs must be positive: {args.epochs}")
        return False
    
    # Validate batch size
    if args.batch_size <= 0:
        logger.error(f"Batch size must be positive: {args.batch_size}")
        return False
    
    return True


def train_with_external_fft(args):
    """Train model with external FFT data."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"FFT source: {args.fft_source}")
        logger.info(f"MATLAB path: {args.matlab_path}")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Compare methods: {args.compare_methods}")

        # Import necessary modules
        import numpy as np
        import json
        from src.data.fft_processor import FlexibleFFTProcessor
        from src.data.data_loader import WebDAQDataLoader
        from src.training.trainer import ModelTrainer
        from src.training.callbacks import get_callbacks
        from src.models.cnn_1d import CNN1DBuilder
        from src.models.cnn_2d import CNN2DBuilder
        from src.models.lstm_model import LSTMBuilder

        matlab_path_obj = Path(args.matlab_path)

        # Load FFT data based on source
        logger.info(f"Loading FFT data from {args.fft_source} source...")

        if args.fft_source == "matlab":
            # Load MATLAB FFT data
            logger.info("Loading MATLAB FFT data...")

            # Try to load MATLAB .mat files from directory
            mat_files = list(matlab_path_obj.glob("*.mat"))

            if not mat_files:
                logger.error(f"No .mat files found in {args.matlab_path}")
                return 1

            logger.info(f"Found {len(mat_files)} MATLAB .mat files")

            # Initialize FFT processor
            fft_processor = FlexibleFFTProcessor(
                fft_size=2048,
                sample_rate=51200,
                method='matlab'
            )

            # Load FFT features and labels from MATLAB files
            fft_features = []
            labels = []

            for mat_file in mat_files:
                try:
                    # Load MATLAB file
                    matlab_fft = fft_processor.load_from_matlab(str(mat_file))
                    fft_features.append(matlab_fft)

                    # Extract label from filename (assuming filename contains class info)
                    # This is a simplified approach - adjust based on your file naming convention
                    label = 0  # Default label
                    labels.append(label)

                except Exception as e:
                    logger.warning(f"Failed to load {mat_file}: {e}")
                    continue

            X_fft = np.array(fft_features)
            y = np.array(labels)

            logger.info(f"Loaded {len(X_fft)} FFT features from MATLAB files")

        elif args.fft_source in ["scipy", "numpy"]:
            # Compute FFT from raw data using specified method
            logger.info(f"Computing FFT using {args.fft_source} method...")

            # Load raw data
            data_loader = WebDAQDataLoader(
                data_dir="data/raw/",  # Adjust path as needed
                sample_rate=51200
            )
            signals, labels, file_paths = data_loader.load_data()

            # Initialize FFT processor
            fft_processor = FlexibleFFTProcessor(
                fft_size=2048,
                sample_rate=51200,
                method=args.fft_source
            )

            # Compute FFT
            fft_features = []
            for i, signal in enumerate(signals):
                fft_result = fft_processor.compute_fft(signal)
                fft_features.append(fft_result)

                if (i + 1) % 100 == 0:
                    logger.info(f"Computed FFT for {i + 1}/{len(signals)} signals")

            X_fft = np.array(fft_features)
            y = labels

            logger.info(f"Computed {len(X_fft)} FFT features")

        else:
            logger.error(f"Unsupported FFT source: {args.fft_source}")
            return 1

        # Compare FFT methods if requested
        if args.compare_methods:
            logger.info("\nComparing FFT computation methods...")

            # Compute using different methods
            fft_scipy = FlexibleFFTProcessor(fft_size=2048, sample_rate=51200, method='scipy')
            fft_numpy = FlexibleFFTProcessor(fft_size=2048, sample_rate=51200, method='numpy')
            fft_matlab = FlexibleFFTProcessor(fft_size=2048, sample_rate=51200, method='matlab')

            # Load sample signal for comparison
            data_loader = WebDAQDataLoader(data_dir="data/raw/", sample_rate=51200)
            signals_sample, _, _ = data_loader.load_data()

            if len(signals_sample) > 0:
                sample_signal = signals_sample[0]

                scipy_fft = fft_scipy.compute_fft(sample_signal)
                numpy_fft = fft_numpy.compute_fft(sample_signal)
                matlab_fft = fft_matlab.compute_fft(sample_signal)

                # Compare
                scipy_numpy = fft_scipy.compare_methods(scipy_fft, numpy_fft)
                scipy_matlab = fft_scipy.compare_methods(scipy_fft, matlab_fft)

                logger.info(f"SciPy vs NumPy - Correlation: {scipy_numpy['correlation']:.6f}, MSE: {scipy_numpy['mse']:.6e}")
                logger.info(f"SciPy vs MATLAB - Correlation: {scipy_matlab['correlation']:.6f}, MSE: {scipy_matlab['mse']:.6e}")

        # Split data into train/val/test
        logger.info("\nSplitting data into train/val/test sets...")
        from sklearn.model_selection import train_test_split

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_fft, y, test_size=0.15, random_state=42, stratify=y
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        num_classes = len(np.unique(y))

        # Build model
        logger.info(f"\nBuilding {args.model_type} model...")

        if args.model_type == "cnn_1d":
            builder = CNN1DBuilder()
            model = builder.build(
                input_shape=X_train.shape[1:],
                num_classes=num_classes,
                conv_filters=[64, 128, 256],
                kernel_sizes=[3, 3, 3],
                dense_units=[128, 64]
            )
            is_deep_learning = True

        elif args.model_type == "cnn_2d":
            builder = CNN2DBuilder()
            # Reshape for 2D CNN
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
            model = builder.build(
                input_shape=X_train.shape[1:],
                num_classes=num_classes,
                conv_filters=[32, 64, 128],
                kernel_sizes=[(3, 3), (3, 3), (3, 3)],
                dense_units=[128, 64]
            )
            is_deep_learning = True

        elif args.model_type == "lstm":
            builder = LSTMBuilder()
            # Reshape for LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            model = builder.build(
                input_shape=X_train.shape[1:],
                num_classes=num_classes,
                lstm_units=[128, 64],
                dense_units=[64]
            )
            is_deep_learning = True

        else:
            logger.error(f"Unsupported model type: {args.model_type}")
            return 1

        # Train model
        logger.info("Training model with external FFT features...")

        trainer = ModelTrainer(model=model, model_name=args.model_type)

        # Compile model
        trainer.compile(
            optimizer='adam',
            learning_rate=0.001,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Get callbacks
        callbacks = get_callbacks(
            model_dir=str(output_path),
            early_stopping_patience=10,
            reduce_lr_patience=5
        )

        # Train
        history = trainer.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks
        )

        # Evaluate on test set
        logger.info("Evaluating model on test set...")
        test_results = trainer.evaluate(X_test, y_test)
        logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")

        # Save model
        model_path = output_path / f"{args.model_type}_external_fft_model.h5"
        trainer.save_checkpoint(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save training info
        training_info = {
            'fft_source': args.fft_source,
            'matlab_path': str(args.matlab_path),
            'model_type': args.model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'test_accuracy': float(test_results['accuracy']),
            'num_classes': int(num_classes)
        }

        info_file = output_path / "training_info.json"
        with open(info_file, 'w') as f:
            json.dump(training_info, f, indent=2)

        logger.info(f"Training info saved to {info_file}")
        logger.info("Training with external FFT completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Training with external FFT failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("EXTERNAL FFT TRAINING - Train with MATLAB FFT")
    logger.info("=" * 60)
    
    return train_with_external_fft(args)


if __name__ == "__main__":
    sys.exit(main())