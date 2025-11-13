#!/usr/bin/env python3
"""
Quick benchmark runner - Cross-platform version of

Usage:
    python run_benchmark.py quick         # Fast test
    python run_benchmark.py standard      # Standard benchmark
    python run_benchmark.py detailed      # Comprehensive benchmark
    python run_benchmark.py --help        # Show all options
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    END = '\033[0m'


def print_header():
    """Print header."""
    print(f"\n{Colors.BLUE}==============================================={Colors.END}")
    print(f"{Colors.BLUE}Model Benchmarking Suite{Colors.END}")
    print(f"{Colors.BLUE}==============================================={Colors.END}\n")


def check_requirements():
    """Check if all requirements are met."""
    project_root = Path(__file__).parent

    # Check Python version
    if sys.version_info < (3, 7):
        print(f"{Colors.RED}Error: Python 3.7+ required{Colors.END}")
        return False

    # Check data directories
    checks = [
        (project_root / "data" / "accelerometer_classifier_v2", "accelerometer_classifier_v2/"),
        (project_root / "data" / "processed", "processed/"),
        (project_root / "scripts" / "benchmark_models.py", "benchmark_models.py"),
    ]

    for path, name in checks:
        if not path.exists():
            print(f"{Colors.RED}Error: {name} not found at {path}{Colors.END}")
            return False

    return True


def run_benchmark(mode: str, verbose: bool = False):
    """Run benchmark with specified mode."""
    print_header()

    print(f"Mode: {Colors.GREEN}{mode}{Colors.END}")
    print(f"Verbose: {Colors.GREEN}{verbose}{Colors.END}\n")

    # Define benchmark configurations
    configs = {
        'quick': {
            'desc': 'QUICK benchmark (50 iterations, batch sizes 1,8,32)',
            'n_iterations': 50,
            'batch_sizes': '1,8,32'
        },
        'standard': {
            'desc': 'STANDARD benchmark (100 iterations, batch sizes 1,8,32,64)',
            'n_iterations': 100,
            'batch_sizes': '1,8,32,64'
        },
        'detailed': {
            'desc': 'DETAILED benchmark (200 iterations, extended batch sizes)',
            'n_iterations': 200,
            'batch_sizes': '1,2,4,8,16,32,64'
        },
    }

    if mode not in configs:
        print(f"{Colors.RED}Unknown mode: {mode}{Colors.END}")
        return False

    config = configs[mode]
    print(f"{Colors.YELLOW}Running {config['desc']}{Colors.END}\n")

    # Build command
    cmd = [
        sys.executable,
        'scripts/benchmark_models.py',
        '--accel-data', 'data/accelerometer_classifier_v2/',
        '--hole-size-data', 'data/processed/',
        '--output-dir', 'results/benchmarks/',
        '--n-iterations', str(config['n_iterations']),
        '--batch-sizes', config['batch_sizes'],
    ]

    if verbose:
        cmd.append('--verbose')

    # Run
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"{Colors.RED}Error running benchmark: {e}{Colors.END}")
        return False


def show_help():
    """Show help message."""
    help_text = f"""
{Colors.GREEN}Usage: python run_benchmark.py [MODE] [OPTIONS]{Colors.END}

Available modes:
  quick           - Fast benchmark (50 iterations) - RECOMMENDED FOR FIRST RUN
  standard        - Standard benchmark (100 iterations)
  detailed        - Detailed benchmark (200 iterations)
  accel-only      - Only accelerometer classifier
  two-stage-only  - Only two-stage classifier
  svm             - Use SVM instead of Random Forest
  help            - Show this message

Options:
  --verbose       - Detailed logging
  --help          - Show this help message

Examples:
  python run_benchmark.py quick           # First-time quick test
  python run_benchmark.py standard        # Standard benchmark
  python run_benchmark.py detailed        # Comprehensive benchmark
  python run_benchmark.py quick --verbose # With detailed logging

Output:
  Results are saved to: results/benchmarks/
  - benchmark_*.json              - Detailed results for each model
  - benchmark_comparison.json     - Comparison metrics
  - benchmark_comparison_*.png    - Visualization plots

{Colors.BLUE}For more information, see docs/BENCHMARKING_GUIDE.md{Colors.END}
    """
    print(help_text)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model Benchmarking Suite",
        add_help=False
    )

    parser.add_argument(
        'mode',
        nargs='?',
        default='quick',
        choices=['quick', 'standard', 'detailed', 'accel-only', 'two-stage-only', 'svm', 'help'],
        help='Benchmark mode'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )

    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help message'
    )

    args = parser.parse_args()

    if args.help or args.mode == 'help':
        show_help()
        return 0

    # Check requirements
    if not check_requirements():
        return 1

    # Run benchmark
    if run_benchmark(args.mode, args.verbose):
        print(f"\n{Colors.GREEN}✓ Benchmarking completed!{Colors.END}\n")

        print("Output files:")
        print("  - results/benchmarks/benchmark_*.json")
        print("  - results/benchmarks/benchmark_comparison.json")
        print("  - results/benchmarks/benchmark_comparison_*.png\n")

        print(f"{Colors.BLUE}Next steps:{Colors.END}")
        print("  1. View results: cat results/benchmarks/benchmark_comparison.json")
        print("  2. Compare models: python run_benchmark.py standard")
        print("  3. For more info: python run_benchmark.py --help\n")

        return 0
    else:
        print(f"\n{Colors.RED}✗ Benchmarking failed!{Colors.END}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())