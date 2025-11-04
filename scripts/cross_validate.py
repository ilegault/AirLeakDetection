#!/usr/bin/env python3
"""
Perform k-fold cross-validation.

Evaluates model stability and generalization across multiple folds.

Usage:
    python scripts/cross_validate.py --model-type cnn_1d --data-path data/processed/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for cross-validation script."""
    parser = argparse.ArgumentParser(
        description="Perform k-fold cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5-fold cross-validation
  python scripts/cross_validate.py --model-type cnn_1d --data-path data/processed/
  
  # With custom k-folds
  python scripts/cross_validate.py \\
      --model-type cnn_1d \\
      --data-path data/processed/ \\
      --k-folds 10
        """
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_1d",
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost"],
        help="Model type to cross-validate"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to processed data"
    )
    
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_validation/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified k-fold"
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
    
    if not Path(args.data_path).exists():
        logger.error(f"Data path not found: {args.data_path}")
        return False
    
    if args.k_folds < 2:
        logger.error(f"k-folds must be at least 2: {args.k_folds}")
        return False
    
    return True


def run_cross_validation(args):
    """Run k-fold cross-validation."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"K-folds: {args.k_folds}")
        logger.info(f"Stratified: {args.stratified}")
        
        # TODO: Implement cross-validation logic
        logger.info("Cross-validation logic to be implemented with Phase 4 (Training)")
        
        logger.info("Cross-validation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION - K-Fold Cross-Validation")
    logger.info("=" * 60)
    
    return run_cross_validation(args)


if __name__ == "__main__":
    sys.exit(main())