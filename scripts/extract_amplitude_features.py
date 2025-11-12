#!/usr/bin/env python3
"""
Extract amplitude-based features for accelerometer classification.

The Welch PSD features are nearly identical across all accelerometers because
they all detect the same frequencies. The key difference is AMPLITUDE.

This script extracts amplitude-based features that capture signal strength:
- RMS (Root Mean Square)
- Standard Deviation
- Peak amplitude
- Signal energy
- FFT magnitude statistics
- Band power (absolute, not normalized)

Usage:
    python scripts/extract_amplitude_features.py \\
        --input-dir data/processed/ \\
        --output-dir data/accelerometer_classifier_v2/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, FileUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def extract_amplitude_features(signal: np.ndarray, sample_rate: int = 10000) -> np.ndarray:
    """Extract amplitude-based features from a single accelerometer signal.

    Args:
        signal: 1D array of signal values
        sample_rate: Sampling rate in Hz

    Returns:
        Feature vector combining multiple amplitude metrics
    """
    features = []

    # 1. RMS (Root Mean Square) - Overall signal strength
    rms = np.sqrt(np.mean(signal ** 2))
    features.append(rms)

    # 2. Standard Deviation - Signal variability
    std = np.std(signal)
    features.append(std)

    # 3. Peak amplitude - Maximum excursion
    peak = np.max(np.abs(signal))
    features.append(peak)

    # 4. Signal energy - Total power
    energy = np.sum(signal ** 2)
    features.append(energy)

    # 5. Peak-to-peak amplitude
    peak_to_peak = np.max(signal) - np.min(signal)
    features.append(peak_to_peak)

    # 6. Mean absolute value
    mean_abs = np.mean(np.abs(signal))
    features.append(mean_abs)

    # 7. Crest factor (peak / rms) - Signal dynamics
    crest_factor = peak / (rms + 1e-10)
    features.append(crest_factor)

    # 8. Kurtosis - Tail behavior (how spiky)
    from scipy.stats import kurtosis
    kurt = kurtosis(signal)
    features.append(kurt)

    # 9. Skewness - Asymmetry
    from scipy.stats import skew
    skewness = skew(signal)
    features.append(skewness)

    # 10-12. FFT Magnitude statistics
    fft_result = np.fft.rfft(signal)
    fft_mag = np.abs(fft_result)
    fft_mean = np.mean(fft_mag)
    fft_max = np.max(fft_mag)
    fft_std = np.std(fft_mag)
    features.extend([fft_mean, fft_max, fft_std])

    # 13-18. Band power in different frequency ranges
    # Low: 50-500 Hz, Mid: 500-1500 Hz, High: 1500-4000 Hz
    freqs = np.fft.rfftfreq(len(signal), d=1.0/sample_rate)

    for f_min, f_max in [(50, 500), (500, 1500), (1500, 4000)]:
        mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(mask):
            band_power = np.sum(fft_mag[mask] ** 2)
        else:
            band_power = 0.0
        features.append(band_power)

    # 19-21. Welch PSD statistics (using absolute values)
    try:
        freqs_welch, psd_welch = scipy_signal.welch(
            signal,
            fs=sample_rate,
            nperseg=min(len(signal), 2048),
            noverlap=None,
            scaling='spectrum'  # Use 'spectrum' for absolute power
        )

        # Filter to useful range
        mask = (freqs_welch >= 50) & (freqs_welch <= 4000)
        psd_band = psd_welch[mask]

        welch_mean = np.mean(psd_band)
        welch_max = np.max(psd_band)
        welch_total_power = np.sum(psd_band)

        features.extend([welch_mean, welch_max, welch_total_power])
    except:
        # If Welch fails, use zeros
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def extract_features_from_npz(npz_file: Path) -> Dict:
    """Extract amplitude features for all 3 accelerometers from an NPZ file.

    Args:
        npz_file: Path to NPZ file

    Returns:
        Dictionary with features for each accelerometer
    """
    try:
        data = np.load(npz_file, allow_pickle=True)

        # Get raw signal
        if "signal" not in data:
            LOGGER.error(f"No signal data in {npz_file.name}")
            return None

        signal = data["signal"]

        if signal.shape[1] != 3:
            LOGGER.error(f"Expected 3 accelerometers, got {signal.shape[1]}")
            return None

        # Extract features for each accelerometer
        samples = []
        for accel_id in range(3):
            accel_signal = signal[:, accel_id]
            features = extract_amplitude_features(accel_signal)

            samples.append({
                'features': features,
                'accel_id': accel_id,
                'original_file': str(npz_file),
                'hole_size_label': int(data.get("label", -1))
            })

        return samples

    except Exception as e:
        LOGGER.error(f"Failed to process {npz_file}: {e}")
        return None


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract amplitude-based features for accelerometer classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to processed data directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/accelerometer_classifier_v2/",
        help="Output directory for features"
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        LOGGER.error(f"Input directory not found: {input_dir}")
        return 1

    # Create output directories
    for split in ["train", "val", "test"]:
        FileUtils.ensure_directory(str(output_dir / split))

    LOGGER.info("="*80)
    LOGGER.info("AMPLITUDE-BASED FEATURE EXTRACTION FOR ACCELEROMETER CLASSIFICATION")
    LOGGER.info("="*80)
    LOGGER.info(f"Input directory: {input_dir}")
    LOGGER.info(f"Output directory: {output_dir}")

    # Process each split
    for split in ["train", "val", "test"]:
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Processing {split.upper()} split...")
        LOGGER.info(f"{'='*80}")

        split_dir = input_dir / split
        if not split_dir.exists():
            LOGGER.warning(f"Split directory not found: {split_dir}")
            continue

        npz_files = sorted(split_dir.glob("*.npz"))
        LOGGER.info(f"Found {len(npz_files)} NPZ files")

        all_samples = []
        for i, npz_file in enumerate(npz_files, 1):
            if i % max(1, len(npz_files) // 10) == 0:
                LOGGER.info(f"  Progress: {i}/{len(npz_files)}")

            samples = extract_features_from_npz(npz_file)
            if samples:
                all_samples.extend(samples)

        if not all_samples:
            LOGGER.warning(f"No samples extracted for {split}")
            continue

        # Convert to numpy arrays
        X = np.array([s['features'] for s in all_samples], dtype=np.float32)
        y = np.array([s['accel_id'] for s in all_samples], dtype=np.int32)
        hole_labels = np.array([s['hole_size_label'] for s in all_samples], dtype=np.int32)

        # Save
        np.save(output_dir / split / "features.npy", X)
        np.save(output_dir / split / "labels.npy", y)
        np.save(output_dir / split / "hole_size_labels.npy", hole_labels)

        LOGGER.info(f"{split}: {X.shape[0]} samples, {X.shape[1]} features")

        # Check statistics
        LOGGER.info(f"\nFeature statistics for {split}:")
        for accel_id in range(3):
            mask = y == accel_id
            if np.any(mask):
                accel_features = X[mask]
                LOGGER.info(f"  Accelerometer {accel_id}:")
                LOGGER.info(f"    Count: {np.sum(mask)}")
                LOGGER.info(f"    Mean: {np.mean(accel_features):.6e}")
                LOGGER.info(f"    Std: {np.std(accel_features):.6e}")
                LOGGER.info(f"    Min: {np.min(accel_features):.6e}")
                LOGGER.info(f"    Max: {np.max(accel_features):.6e}")

        # Check if features are different
        LOGGER.info(f"\nChecking feature differences:")
        for i in range(3):
            for j in range(i+1, 3):
                mask_i = y == i
                mask_j = y == j
                if np.any(mask_i) and np.any(mask_j):
                    mean_i = np.mean(X[mask_i])
                    mean_j = np.mean(X[mask_j])
                    diff = abs(mean_i - mean_j)
                    LOGGER.info(f"  Accel {i} vs Accel {j}: Mean diff = {diff:.6e}")
                    if diff > 1e-6:
                        LOGGER.info(f"    ✓ Features are DIFFERENT (good!)")
                    else:
                        LOGGER.warning(f"    ✗ Features are still IDENTICAL (bad!)")

    # Save metadata
    metadata = {
        "feature_extraction": "amplitude_based",
        "n_features": 21,
        "feature_names": [
            "rms", "std", "peak", "energy", "peak_to_peak", "mean_abs",
            "crest_factor", "kurtosis", "skewness",
            "fft_mean", "fft_max", "fft_std",
            "bandpower_50_500", "bandpower_500_1500", "bandpower_1500_4000",
            "welch_mean", "welch_max", "welch_total_power",
            "welch_psd_mean", "welch_psd_max", "welch_psd_total"
        ],
        "description": "Amplitude-based features capturing signal strength differences"
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    LOGGER.info(f"\nMetadata saved to {metadata_file}")
    LOGGER.info("="*80)
    LOGGER.info("FEATURE EXTRACTION COMPLETED")
    LOGGER.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
