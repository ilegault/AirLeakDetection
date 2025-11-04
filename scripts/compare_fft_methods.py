#!/usr/bin/env python3
"""
Compare different FFT computation methods.

Compares NumPy, SciPy, and MATLAB FFT methods with correlation
and error metrics. Generates visualization and recommendations.

Usage:
    python scripts/compare_fft_methods.py --raw-data data/raw/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for FFT comparison."""
    parser = argparse.ArgumentParser(
        description="Compare different FFT computation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python scripts/compare_fft_methods.py --raw-data data/raw/
  
  # With visualization
  python scripts/compare_fft_methods.py \\
      --raw-data data/raw/ \\
      --output-dir results/fft_comparison/ \\
      --generate-plots
  
  # With MATLAB reference
  python scripts/compare_fft_methods.py \\
      --raw-data data/raw/ \\
      --matlab-path data/matlab_fft/ \\
      --compare-with-matlab
        """
    )
    
    parser.add_argument(
        "--raw-data",
        type=str,
        required=True,
        help="Path to raw data"
    )
    
    parser.add_argument(
        "--matlab-path",
        type=str,
        default=None,
        help="Path to MATLAB FFT data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/fft_comparison/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of samples to compare"
    )
    
    parser.add_argument(
        "--fft-size",
        type=int,
        default=2048,
        help="FFT size"
    )
    
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate comparison plots"
    )
    
    parser.add_argument(
        "--compare-with-matlab",
        action="store_true",
        help="Compare with MATLAB reference"
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
    
    if not Path(args.raw_data).exists():
        logger.error(f"Raw data path not found: {args.raw_data}")
        return False
    
    if args.compare_with_matlab and (args.matlab_path is None or not Path(args.matlab_path).exists()):
        logger.error(f"MATLAB path required and must exist for comparison: {args.matlab_path}")
        return False
    
    if args.n_samples <= 0:
        logger.error(f"Number of samples must be positive: {args.n_samples}")
        return False
    
    if args.fft_size <= 0 or (args.fft_size & (args.fft_size - 1)) != 0:
        logger.error(f"FFT size must be positive and power of 2: {args.fft_size}")
        return False
    
    return True


def compare_fft_methods(args):
    """Compare FFT computation methods."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Raw data path: {args.raw_data}")
        logger.info(f"Number of samples: {args.n_samples}")
        logger.info(f"FFT size: {args.fft_size}")
        logger.info(f"Generate plots: {args.generate_plots}")
        logger.info(f"Compare with MATLAB: {args.compare_with_matlab}")
        
        if args.compare_with_matlab:
            logger.info(f"MATLAB path: {args.matlab_path}")
        
        # TODO: Implement FFT comparison logic
        logger.info("FFT comparison logic to be implemented with Phase 2 (Data Pipeline)")
        
        logger.info("FFT method comparison completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"FFT comparison failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("FFT COMPARISON - Compare FFT Computation Methods")
    logger.info("=" * 60)
    
    return compare_fft_methods(args)


if __name__ == "__main__":
    sys.exit(main())