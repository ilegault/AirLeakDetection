#!/usr/bin/env python3
"""
Verify that NPZ files contain valid data for all 3 accelerometers.

This script loads processed NPZ files and checks:
1. All 3 accelerometer channels have non-zero data
2. FFT magnitude contains data for all 3 channels
3. Welch PSD contains data for all 3 channels
4. Band power is computed for all 3 channels
5. Print statistics for each accelerometer

Usage:
    python scripts/verify_accelerometer_data.py data/processed/train/sample_001_processed.npz
    python scripts/verify_accelerometer_data.py --all data/processed/train/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def verify_npz_file(npz_path: Path) -> Dict:
    """Verify a single NPZ file contains valid data for all 3 accelerometers.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dictionary with verification results
    """
    results = {
        "file": npz_path.name,
        "valid": True,
        "issues": [],
        "stats": {}
    }

    try:
        # Load NPZ file
        data = np.load(npz_path, allow_pickle=True)

        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Verifying: {npz_path.name}")
        LOGGER.info(f"{'='*60}")

        # List all keys
        LOGGER.info(f"Keys in file: {list(data.keys())}")

        # 1. Check signal data
        if "signal" in data:
            signal = data["signal"]
            LOGGER.info(f"\nSignal shape: {signal.shape}")

            if signal.ndim != 2 or signal.shape[1] != 3:
                results["valid"] = False
                results["issues"].append(f"Signal shape is {signal.shape}, expected (timesteps, 3)")

            # Check each accelerometer channel
            for i in range(min(3, signal.shape[1])):
                mean = np.mean(signal[:, i])
                std = np.std(signal[:, i])
                min_val = np.min(signal[:, i])
                max_val = np.max(signal[:, i])
                non_zero = np.count_nonzero(signal[:, i])

                LOGGER.info(f"  Accelerometer {i}:")
                LOGGER.info(f"    Mean: {mean:.6f}, Std: {std:.6f}")
                LOGGER.info(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
                LOGGER.info(f"    Non-zero samples: {non_zero}/{signal.shape[0]} ({100*non_zero/signal.shape[0]:.2f}%)")

                results["stats"][f"accel_{i}_mean"] = mean
                results["stats"][f"accel_{i}_std"] = std

                if non_zero == 0:
                    results["valid"] = False
                    results["issues"].append(f"Accelerometer {i} has all zeros in signal data")

        # 2. Check FFT magnitude
        if "fft_magnitude" in data:
            fft_mag = data["fft_magnitude"]
            LOGGER.info(f"\nFFT magnitude shape: {fft_mag.shape}")

            if fft_mag.ndim != 2 or fft_mag.shape[1] != 3:
                results["valid"] = False
                results["issues"].append(f"FFT magnitude shape is {fft_mag.shape}, expected (n_freqs, 3)")

            # Check each accelerometer FFT
            for i in range(min(3, fft_mag.shape[1])):
                mean = np.mean(fft_mag[:, i])
                max_val = np.max(fft_mag[:, i])
                non_zero = np.count_nonzero(fft_mag[:, i])

                LOGGER.info(f"  FFT Accelerometer {i}:")
                LOGGER.info(f"    Mean: {mean:.6e}, Max: {max_val:.6e}")
                LOGGER.info(f"    Non-zero bins: {non_zero}/{fft_mag.shape[0]} ({100*non_zero/fft_mag.shape[0]:.2f}%)")

                results["stats"][f"fft_accel_{i}_mean"] = mean
                results["stats"][f"fft_accel_{i}_max"] = max_val

                if non_zero == 0:
                    results["valid"] = False
                    results["issues"].append(f"Accelerometer {i} has all zeros in FFT magnitude")

        # 3. Check Welch PSD
        if "welch_psd" in data:
            welch_psd = data["welch_psd"]
            LOGGER.info(f"\nWelch PSD shape: {welch_psd.shape}")

            if welch_psd.ndim != 2 or welch_psd.shape[1] != 3:
                results["valid"] = False
                results["issues"].append(f"Welch PSD shape is {welch_psd.shape}, expected (n_freqs, 3)")

            # Check each accelerometer PSD
            for i in range(min(3, welch_psd.shape[1])):
                mean = np.mean(welch_psd[:, i])
                max_val = np.max(welch_psd[:, i])
                non_zero = np.count_nonzero(welch_psd[:, i])

                LOGGER.info(f"  Welch PSD Accelerometer {i}:")
                LOGGER.info(f"    Mean: {mean:.6e}, Max: {max_val:.6e}")
                LOGGER.info(f"    Non-zero bins: {non_zero}/{welch_psd.shape[0]} ({100*non_zero/welch_psd.shape[0]:.2f}%)")

                results["stats"][f"welch_accel_{i}_mean"] = mean
                results["stats"][f"welch_accel_{i}_max"] = max_val

                if non_zero == 0:
                    results["valid"] = False
                    results["issues"].append(f"Accelerometer {i} has all zeros in Welch PSD")

        # 4. Check band power
        if "welch_bandpower" in data:
            bandpower = data["welch_bandpower"]
            LOGGER.info(f"\nBand power shape: {bandpower.shape}")
            LOGGER.info(f"Band power values:")

            if bandpower.shape[0] != 3:
                results["valid"] = False
                results["issues"].append(f"Band power shape is {bandpower.shape}, expected (3,)")

            for i in range(min(3, len(bandpower))):
                LOGGER.info(f"  Accelerometer {i}: {bandpower[i]:.6e}")
                results["stats"][f"bandpower_accel_{i}"] = bandpower[i]

                if bandpower[i] == 0:
                    results["valid"] = False
                    results["issues"].append(f"Accelerometer {i} has zero band power")

        # 5. Check other metadata
        if "label" in data:
            LOGGER.info(f"\nLabel: {data['label']}")
        if "class_name" in data:
            LOGGER.info(f"Class name: {data['class_name']}")

        # Summary
        LOGGER.info(f"\n{'='*60}")
        if results["valid"]:
            LOGGER.info("✓ VERIFICATION PASSED - All 3 accelerometers have valid data")
        else:
            LOGGER.warning("✗ VERIFICATION FAILED")
            for issue in results["issues"]:
                LOGGER.warning(f"  - {issue}")
        LOGGER.info(f"{'='*60}\n")

    except Exception as e:
        LOGGER.error(f"Error loading {npz_path}: {e}", exc_info=True)
        results["valid"] = False
        results["issues"].append(f"Failed to load file: {e}")

    return results


def plot_accelerometer_comparison(npz_path: Path, output_dir: Path = None) -> None:
    """Create visualization comparing all 3 accelerometers.

    Args:
        npz_path: Path to NPZ file
        output_dir: Directory to save plots (optional)
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Three Accelerometer Data - {npz_path.name}', fontsize=16, fontweight='bold')

        accel_names = ['Accelerometer 0 (Closest)', 'Accelerometer 1 (Middle)', 'Accelerometer 2 (Farthest)']
        colors = ['blue', 'green', 'red']

        # Plot signal data
        if "signal" in data:
            signal = data["signal"]
            for i in range(3):
                axes[i, 0].plot(signal[:, i], color=colors[i], linewidth=0.5, alpha=0.7)
                axes[i, 0].set_title(f'{accel_names[i]} - Time Domain', fontsize=11, fontweight='bold')
                axes[i, 0].set_xlabel('Sample')
                axes[i, 0].set_ylabel('Acceleration (g)')
                axes[i, 0].grid(True, alpha=0.3)

        # Plot FFT or Welch PSD with SEPARATE Y-AXIS SCALES
        if "welch_psd" in data and "welch_frequencies" in data:
            psd = data["welch_psd"]
            frequencies = data["welch_frequencies"]

            for i in range(3):
                # Use semi-log scale to show signals with different magnitudes
                axes[i, 1].semilogy(frequencies, psd[:, i], color=colors[i], linewidth=1.0)
                axes[i, 1].set_title(f'{accel_names[i]} - Welch PSD (Log Scale)', fontsize=11, fontweight='bold')
                axes[i, 1].set_xlabel('Frequency (Hz)')
                axes[i, 1].set_ylabel('PSD (g²/Hz) [log scale]')
                axes[i, 1].grid(True, alpha=0.3, which='both')

                # Show min/max for this accelerometer
                min_val = np.min(psd[:, i][psd[:, i] > 0])
                max_val = np.max(psd[:, i])
                axes[i, 1].text(0.02, 0.98, f'Range: {min_val:.2e} to {max_val:.2e}',
                              transform=axes[i, 1].transAxes,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        elif "fft_magnitude" in data and "frequencies_fft" in data:
            fft_mag = data["fft_magnitude"]
            frequencies = data["frequencies_fft"]

            for i in range(3):
                # Use semi-log scale to show signals with different magnitudes
                axes[i, 1].semilogy(frequencies, fft_mag[:, i], color=colors[i], linewidth=1.0)
                axes[i, 1].set_title(f'{accel_names[i]} - FFT (Log Scale)', fontsize=11, fontweight='bold')
                axes[i, 1].set_xlabel('Frequency (Hz)')
                axes[i, 1].set_ylabel('Magnitude [log scale]')
                axes[i, 1].grid(True, alpha=0.3, which='both')

                # Show min/max for this accelerometer
                min_val = np.min(fft_mag[:, i][fft_mag[:, i] > 0])
                max_val = np.max(fft_mag[:, i])
                axes[i, 1].text(0.02, 0.98, f'Range: {min_val:.2e} to {max_val:.2e}',
                              transform=axes[i, 1].transAxes,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{npz_path.stem}_verification.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            LOGGER.info(f"Plot saved to {output_path}")

        plt.show()

    except Exception as e:
        LOGGER.error(f"Error plotting {npz_path}: {e}", exc_info=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Verify NPZ files contain valid data for all 3 accelerometers'
    )
    parser.add_argument(
        'npz_path',
        type=str,
        help='Path to NPZ file or directory'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Verify all NPZ files in directory'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create verification plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots/verification',
        help='Output directory for plots (default: plots/verification)'
    )

    args = parser.parse_args()

    npz_path = Path(args.npz_path)

    if not npz_path.exists():
        LOGGER.error(f"Path does not exist: {npz_path}")
        return 1

    # Verify files
    if args.all and npz_path.is_dir():
        # Verify all NPZ files in directory
        npz_files = list(npz_path.glob("*.npz"))
        LOGGER.info(f"Found {len(npz_files)} NPZ files in {npz_path}")

        results = []
        for npz_file in npz_files:
            result = verify_npz_file(npz_file)
            results.append(result)

        # Summary
        valid_count = sum(1 for r in results if r["valid"])
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"SUMMARY: {valid_count}/{len(results)} files passed verification")
        LOGGER.info(f"{'='*60}")

        if valid_count < len(results):
            LOGGER.warning("\nFiles with issues:")
            for result in results:
                if not result["valid"]:
                    LOGGER.warning(f"  {result['file']}")
                    for issue in result["issues"]:
                        LOGGER.warning(f"    - {issue}")

    elif npz_path.is_file():
        # Verify single file
        result = verify_npz_file(npz_path)

        if args.plot:
            output_dir = Path(args.output_dir) if args.output_dir else None
            plot_accelerometer_comparison(npz_path, output_dir)

        return 0 if result["valid"] else 1

    else:
        LOGGER.error(f"Invalid path: {npz_path}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
