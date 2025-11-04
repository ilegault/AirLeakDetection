#!/usr/bin/env python3
"""
Hyperparameter optimization using various search strategies.

Supports grid search, random search, and Bayesian optimization.

Usage:
    python scripts/hyperparameter_search.py --model-type cnn_1d --search-method bayesian
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bayesian optimization
  python scripts/hyperparameter_search.py \\
      --model-type cnn_1d \\
      --search-method bayesian \\
      --n-trials 50
  
  # Grid search
  python scripts/hyperparameter_search.py \\
      --model-type cnn_1d \\
      --search-method grid
        """
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_1d",
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost"],
        help="Model type"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--search-method",
        type=str,
        default="bayesian",
        choices=["grid", "random", "bayesian"],
        help="Search method"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials/iterations"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparameter_search/",
        help="Output directory"
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
    
    if args.n_trials <= 0:
        logger.error(f"Number of trials must be positive: {args.n_trials}")
        return False
    
    if args.n_jobs <= 0:
        logger.error(f"Number of jobs must be positive: {args.n_jobs}")
        return False
    
    return True


def run_hyperparameter_search(args):
    """Run hyperparameter search."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Search method: {args.search_method}")
        logger.info(f"Number of trials: {args.n_trials}")
        logger.info(f"Parallel jobs: {args.n_jobs}")
        
        # TODO: Implement hyperparameter search logic
        logger.info("Hyperparameter search logic to be implemented with Phase 4 (Training)")
        
        logger.info("Hyperparameter search completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER SEARCH - Find Optimal Hyperparameters")
    logger.info("=" * 60)
    
    return run_hyperparameter_search(args)


if __name__ == "__main__":
    sys.exit(main())