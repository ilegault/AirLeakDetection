#!/usr/bin/env python3
"""
Prepare data for multi-accelerometer position classification training.

This script creates a dataset for training Stage 1 of the two-stage classifier.
Each NPZ file contains simultaneous recordings from 3 accelerometers at different
positions. This script separates them and labels each by position (0, 1, 2).

Data Flow:
    Input: NPZ files with shape (timesteps, 3) - 3 simultaneous accelerometer recordings
    Process: Extract each accelerometer's data separately
    Output: Individual position samples labeled 0, 1, 2

The trained classifier will later identify which accelerometer position (0, 1, 2)
is closest to a leak source based on signal characteristics.

Usage:
    python scripts/prepare_accelerometer_data.py --input-dir data/processed/ --output-dir data/accelerometer_classifier/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils
from src.data.data_splitter import DataSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare data for accelerometer classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preparation
  python scripts/prepare_accelerometer_data.py --input-dir data/processed/ --output-dir data/accelerometer_classifier/

  # Custom splits
  python scripts/prepare_accelerometer_data.py \\
      --input-dir data/processed/ \\
      --output-dir data/accelerometer_classifier/ \\
      --train-ratio 0.7 \\
      --val-ratio 0.15
        """
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to processed data directory (output from prepare_data.py)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/accelerometer_classifier/",
        help="Output directory for accelerometer classification data"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )

    parser.add_argument(
        "--use-fft",
        action="store_true",
        help="Use FFT features instead of raw signal"
    )

    parser.add_argument(
        "--use-welch",
        action="store_true",
        default=True,
        help="Use Welch PSD features (default: True)"
    )

    parser.add_argument(
        "--use-bandpower",
        action="store_true",
        default=True,
        help="Use band power features (default: True)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )

    return parser


def load_npz_files(input_dir: Path, split: str) -> List[Path]:
    """Load all NPZ files from a split directory.

    Args:
        input_dir: Input directory containing train/val/test splits
        split: Split name ('train', 'val', 'test')

    Returns:
        List of NPZ file paths
    """
    split_dir = input_dir / split
    if not split_dir.exists():
        LOGGER.warning(f"Split directory not found: {split_dir}")
        return []

    npz_files = sorted(split_dir.glob("*.npz"))
    LOGGER.info(f"Found {len(npz_files)} NPZ files in {split}")
    return npz_files


def extract_accelerometer_samples(
    npz_file: Path,
    use_fft: bool = False,
    use_welch: bool = True,
    use_bandpower: bool = True
) -> List[Dict]:
    """Extract individual accelerometer position samples from an NPZ file.

    Each NPZ file contains simultaneous recordings from 3 accelerometers positioned
    at different distances from leak sources. This function separates the data from
    each position and creates individual samples labeled by position (0, 1, 2).

    Args:
        npz_file: Path to NPZ file containing multi-accelerometer data
        use_fft: Use FFT magnitude features
        use_welch: Use Welch PSD features
        use_bandpower: Use band power features

    Returns:
        List of 3 dictionaries (one per accelerometer position) with keys:
        - 'features': Feature array for that position
        - 'accel_id': Position ID (0, 1, 2)
        - 'original_file': Original NPZ file path
        - 'hole_size_label': Original hole size label (for reference)
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        samples = []

        # Get original hole size label for reference
        hole_size_label = int(data["label"]) if "label" in data else -1

        # Extract features for each accelerometer
        for accel_id in range(3):
            features_list = []

            # Option 1: Use raw signal
            if not use_fft and not use_welch and not use_bandpower:
                if "signal" in data:
                    signal = data["signal"]
                    if signal.shape[1] >= 3:
                        features_list.append(signal[:, accel_id])

            # Option 2: Use FFT features
            if use_fft and "fft_magnitude" in data:
                fft_mag = data["fft_magnitude"]
                if fft_mag.shape[1] >= 3:
                    features_list.append(fft_mag[:, accel_id])

            # Option 3: Use Welch PSD features
            if use_welch and "welch_psd" in data:
                welch_psd = data["welch_psd"]
                if welch_psd.shape[1] >= 3:
                    features_list.append(welch_psd[:, accel_id])

            # Option 4: Use band power features
            if use_bandpower and "welch_bandpower" in data:
                bandpower = data["welch_bandpower"]
                if len(bandpower) >= 3:
                    # Band power is a scalar per accelerometer
                    features_list.append(np.array([bandpower[accel_id]]))

            # Concatenate all selected features
            if features_list:
                features = np.concatenate(features_list)

                samples.append({
                    'features': features.astype(np.float32),
                    'accel_id': accel_id,
                    'original_file': str(npz_file),
                    'hole_size_label': hole_size_label
                })

        return samples

    except Exception as e:
        LOGGER.error(f"Failed to extract samples from {npz_file}: {e}")
        return []


def prepare_accelerometer_data(args) -> int:
    """Prepare accelerometer classification data."""
    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if not input_dir.exists():
            LOGGER.error(f"Input directory not found: {input_dir}")
            return 1

        # Create output directories
        for split in ["train", "val", "test"]:
            FileUtils.ensure_directory(str(output_dir / split))
        FileUtils.ensure_directory(str(output_dir / "metadata"))

        LOGGER.info("="*60)
        LOGGER.info("ACCELEROMETER CLASSIFICATION DATA PREPARATION")
        LOGGER.info("="*60)
        LOGGER.info(f"Input directory: {input_dir}")
        LOGGER.info(f"Output directory: {output_dir}")
        LOGGER.info(f"Use FFT: {args.use_fft}")
        LOGGER.info(f"Use Welch PSD: {args.use_welch}")
        LOGGER.info(f"Use Band Power: {args.use_bandpower}")

        # Process each split
        all_samples = {'train': [], 'val': [], 'test': []}

        for split in ["train", "val", "test"]:
            LOGGER.info(f"\n{'='*60}")
            LOGGER.info(f"Processing {split.upper()} split...")
            LOGGER.info(f"{'='*60}")

            npz_files = load_npz_files(input_dir, split)

            for i, npz_file in enumerate(npz_files, 1):
                if i % max(1, len(npz_files) // 10) == 0:
                    LOGGER.info(f"  Progress: {i}/{len(npz_files)}")

                samples = extract_accelerometer_samples(
                    npz_file,
                    use_fft=args.use_fft,
                    use_welch=args.use_welch,
                    use_bandpower=args.use_bandpower
                )

                all_samples[split].extend(samples)

            LOGGER.info(f"Extracted {len(all_samples[split])} accelerometer samples from {split}")

        # Save samples to numpy arrays
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info("Saving processed samples...")
        LOGGER.info(f"{'='*60}")

        for split in ["train", "val", "test"]:
            samples = all_samples[split]

            if not samples:
                LOGGER.warning(f"No samples for {split} split")
                continue

            # Convert to numpy arrays
            X = np.array([s['features'] for s in samples], dtype=np.float32)
            y = np.array([s['accel_id'] for s in samples], dtype=np.int32)
            hole_labels = np.array([s['hole_size_label'] for s in samples], dtype=np.int32)

            # Save features and labels
            np.save(output_dir / split / "features.npy", X)
            np.save(output_dir / split / "labels.npy", y)
            np.save(output_dir / split / "hole_size_labels.npy", hole_labels)

            LOGGER.info(f"{split}: {X.shape[0]} samples, feature shape: {X.shape}")

            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            LOGGER.info(f"  Accelerometer distribution:")
            for accel_id, count in zip(unique, counts):
                LOGGER.info(f"    Accelerometer {accel_id}: {count} samples ({100*count/len(y):.2f}%)")

        # Save metadata
        metadata = {
            "use_fft": args.use_fft,
            "use_welch": args.use_welch,
            "use_bandpower": args.use_bandpower,
            "n_accelerometers": 3,
            "accelerometer_labels": {
                "0": "Accelerometer 0 (Closest)",
                "1": "Accelerometer 1 (Middle)",
                "2": "Accelerometer 2 (Farthest)"
            },
            "splits": {
                "train": len(all_samples['train']),
                "val": len(all_samples['val']),
                "test": len(all_samples['test'])
            }
        }

        metadata_file = output_dir / "metadata" / "info.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        LOGGER.info(f"\nMetadata saved to {metadata_file}")

        LOGGER.info("="*60)
        LOGGER.info("ACCELEROMETER DATA PREPARATION COMPLETED")
        LOGGER.info("="*60)

        return 0

    except Exception as e:
        LOGGER.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)

    return prepare_accelerometer_data(args)


if __name__ == "__main__":
    sys.exit(main())
