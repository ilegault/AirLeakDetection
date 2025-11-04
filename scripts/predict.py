#!/usr/bin/env python3
"""
Run inference with a trained model.

Performs predictions on single files or batch data with optional
confidence filtering and multiple output formats.

Usage:
    python scripts/predict.py --model-path models/best_model.h5 --input data/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for prediction script."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on single file
  python scripts/predict.py --model-path models/best.h5 --input data/sample.csv
  
  # Batch prediction
  python scripts/predict.py --model-path models/best.h5 --input data/test/
  
  # With confidence threshold
  python scripts/predict.py --model-path models/best.h5 --input data/test/ --confidence-threshold 0.8
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions.json",
        help="Output file for predictions (default: results/predictions.json)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "txt"],
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (default: 0.0)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
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
    
    # Check model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return False
    
    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        return False
    
    # Validate confidence threshold
    if not (0 <= args.confidence_threshold <= 1):
        logger.error(f"Confidence threshold must be between 0 and 1: {args.confidence_threshold}")
        return False
    
    # Validate batch size
    if args.batch_size <= 0:
        logger.error(f"Batch size must be positive: {args.batch_size}")
        return False
    
    return True


def run_predictions(args):
    """Run predictions with trained model."""
    logger = get_logger(__name__)
    
    try:
        # Validate inputs
        if not validate_inputs(args):
            return 1
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Confidence threshold: {args.confidence_threshold}")
        logger.info(f"Output format: {args.output_format}")
        
        # Create output directory
        output_path = Path(args.output)
        FileUtils.ensure_directory(str(output_path.parent))
        
        # TODO: Implement actual prediction logic
        logger.info("Prediction logic to be implemented with Phase 6 (Prediction Pipeline)")
        
        logger.info("Predictions completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("PREDICTION - Run Inference with Trained Model")
    logger.info("=" * 60)
    
    return run_predictions(args)


if __name__ == "__main__":
    sys.exit(main())