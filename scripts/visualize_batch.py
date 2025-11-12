#!/usr/bin/env python3
"""Batch visualization of Welch spectra for multiple files."""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def main():
    """Run visualization on multiple CSV files."""
    parser = argparse.ArgumentParser(
        description='Batch visualize Welch spectra for multiple files'
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing CSV files (will search recursively)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots (default: plots)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process'
    )
    parser.add_argument(
        '--class-filter',
        type=str,
        default=None,
        help='Filter by class name (e.g., "3_32", "NOLEAK")'
    )

    args = parser.parse_args()

    # Find all CSV files
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        LOGGER.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    csv_files = sorted(data_dir.rglob("*.csv"))
    
    # Filter by class if specified
    if args.class_filter:
        csv_files = [f for f in csv_files if args.class_filter in str(f)]
        LOGGER.info(f"Filtered to {len(csv_files)} files matching class '{args.class_filter}'")
    
    if not csv_files:
        LOGGER.error(f"No CSV files found in {data_dir}")
        sys.exit(1)

    # Limit number of files if specified
    if args.limit:
        csv_files = csv_files[:args.limit]

    LOGGER.info(f"Found {len(csv_files)} CSV files to process")

    # Get the project root and script path
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "scripts" / "visualize_welch_spectra.py"

    # Process each file
    success_count = 0
    fail_count = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        LOGGER.info(f"Processing {i}/{len(csv_files)}: {csv_file.name}")
        
        try:
            # Run the visualization script with --no-show flag
            cmd = [
                sys.executable,
                str(script_path),
                str(csv_file),
                "--config", args.config,
                "--output-dir", args.output_dir,
                "--no-show"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                success_count += 1
                LOGGER.info(f"✓ Successfully processed {csv_file.name}")
            else:
                fail_count += 1
                LOGGER.error(f"✗ Failed to process {csv_file.name}")
                LOGGER.error(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            fail_count += 1
            LOGGER.error(f"✗ Timeout processing {csv_file.name}")
        except Exception as e:
            fail_count += 1
            LOGGER.error(f"✗ Error processing {csv_file.name}: {e}")

    LOGGER.info(f"\nBatch processing complete!")
    LOGGER.info(f"Success: {success_count}/{len(csv_files)}")
    LOGGER.info(f"Failed: {fail_count}/{len(csv_files)}")
    LOGGER.info(f"Plots saved to: {Path(args.output_dir).resolve()}")


if __name__ == '__main__':
    main()