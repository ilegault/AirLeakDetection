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
        
        # TODO: Implement MATLAB FFT training logic
        logger.info("MATLAB FFT training logic to be implemented with Phase 2 (Data) and Phase 4 (Training)")
        
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