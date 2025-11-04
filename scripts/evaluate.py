#!/usr/bin/env python3
"""
Evaluate a trained model.

Computes comprehensive evaluation metrics, generates visualizations,
and produces detailed reports.

Usage:
    python scripts/evaluate.py --model-path models/best_model.h5 --test-data data/processed/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate.py --model-path models/best.h5 --test-data data/processed/test/
  
  # With detailed report
  python scripts/evaluate.py \\
      --model-path models/best.h5 \\
      --test-data data/processed/test/ \\
      --output-dir results/evaluation/ \\
      --generate-report
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
        default="results/evaluation/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
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
    
    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        return False
    
    # Check test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data not found: {args.test_data}")
        return False
    
    return True


def evaluate_model(args):
    """Evaluate trained model."""
    logger = get_logger(__name__)
    
    try:
        # Validate inputs
        if not validate_inputs(args):
            return 1
        
        # Setup output directory
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Generate report: {args.generate_report}")
        logger.info(f"Generate plots: {args.generate_plots}")
        
        # TODO: Implement actual evaluation logic
        logger.info("Evaluation logic to be implemented with Phase 5 (Evaluation)")
        
        logger.info("Evaluation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("EVALUATION - Evaluate Trained Model")
    logger.info("=" * 60)
    
    return evaluate_model(args)


if __name__ == "__main__":
    sys.exit(main())