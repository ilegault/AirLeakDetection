#!/usr/bin/env python3
"""
Performance benchmarking for inference.

Measures inference speed, memory usage, and accuracy-speed tradeoff.

Usage:
    python scripts/benchmark.py --model-path models/best.h5 --test-data data/processed/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmarking
  python scripts/benchmark.py \\
      --model-path models/best.h5 \\
      --test-data data/processed/test/
  
  # With memory profiling
  python scripts/benchmark.py \\
      --model-path models/best.h5 \\
      --test-data data/processed/test/ \\
      --profile-memory
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
        default="results/benchmarks/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Number of inference iterations"
    )
    
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,64",
        help="Batch sizes to test (comma-separated)"
    )
    
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile memory usage"
    )
    
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        help="Profile CPU usage"
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
    
    if not Path(args.model_path).exists():
        logger.error(f"Model not found: {args.model_path}")
        return False
    
    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        return False
    
    if args.n_iterations <= 0:
        logger.error(f"Number of iterations must be positive: {args.n_iterations}")
        return False
    
    return True


def run_benchmarks(args):
    """Run performance benchmarks."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Number of iterations: {args.n_iterations}")
        logger.info(f"Batch sizes: {args.batch_sizes}")
        logger.info(f"Profile memory: {args.profile_memory}")
        logger.info(f"Profile CPU: {args.profile_cpu}")
        
        # TODO: Implement benchmarking logic
        logger.info("Benchmarking logic to be implemented")
        
        logger.info("Benchmarking completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("BENCHMARKING - Performance Benchmarking")
    logger.info("=" * 60)
    
    return run_benchmarks(args)


if __name__ == "__main__":
    sys.exit(main())