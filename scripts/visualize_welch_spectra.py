#!/usr/bin/env python3
"""Visualize spectral analysis using Welch's method.

This script demonstrates the professor's specific Welch's method parameters:
- Hamming window for better spectral leakage reduction
- 16 segments with 50% overlap
- Zero-padding for frequency resolution
- Band power analysis for classification
"""

import argparse
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import WebDAQDataLoader
from src.data.preprocessor import SignalPreprocessor
from src.data.fft_processor import FlexibleFFTProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_welch_spectra(
    frequencies: np.ndarray,
    psd: np.ndarray,
    title: str,
    output_path: Path = None,
    axis_limits: list = None
) -> None:
    """Create linear and semi-log plots of Welch PSD for all accelerometers.

    Args:
        frequencies: Frequency array
        psd: PSD array, shape (n_freqs, 3) for 3 accelerometers
        title: Plot title
        output_path: Path to save plot (optional)
        axis_limits: [xmin, xmax, ymin, ymax] for linear plot
    """
    # Create 3x1 subplots to compare all accelerometers
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{title} - Welch Power Spectral Density', fontsize=14, fontweight='bold')

    accel_names = ['Accelerometer 0', 'Accelerometer 1', 'Accelerometer 2']
    colors = ['blue', 'green', 'red']

    for i, (ax, name, color) in enumerate(zip(axes, accel_names, colors)):
        # Linear plot
        ax.plot(frequencies, psd[:, i], color=color, linewidth=1.0, label=name)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (g²/Hz)')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set axis limits if provided (professor's recommendation: [30, 4500, 0, 7e-5])
        if axis_limits:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Linear plot saved to {output_path}")

    plt.show()


def plot_welch_spectra_semilog(
    frequencies: np.ndarray,
    psd: np.ndarray,
    title: str,
    output_path: Path = None
) -> None:
    """Create semi-log plots of Welch PSD for better visibility of small features.

    Args:
        frequencies: Frequency array
        psd: PSD array, shape (n_freqs, 3) for 3 accelerometers
        title: Plot title
        output_path: Path to save plot (optional)
    """
    # Create 3x1 subplots to compare all accelerometers
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{title} - Welch PSD (Semi-log)', fontsize=14, fontweight='bold')

    accel_names = ['Accelerometer 0', 'Accelerometer 1', 'Accelerometer 2']
    colors = ['blue', 'green', 'red']

    for i, (ax, name, color) in enumerate(zip(axes, accel_names, colors)):
        # Semi-log plot
        ax.semilogy(frequencies, psd[:, i], color=color, linewidth=1.0, label=name)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (g²/Hz) [log scale]')
        ax.set_title(name)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Semi-log plot saved to {output_path}")

    plt.show()


def plot_bandpower_comparison(
    bandpower: np.ndarray,
    freq_range: tuple,
    title: str,
    output_path: Path = None
) -> None:
    """Create bar chart showing relative band power differences.

    Args:
        bandpower: Band power array, shape (3,) for 3 accelerometers
        freq_range: (min_freq, max_freq) tuple
        title: Plot title
        output_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    accel_names = ['Accelerometer 0', 'Accelerometer 1', 'Accelerometer 2']
    colors = ['blue', 'green', 'red']

    x_pos = np.arange(len(accel_names))
    bars = ax.bar(x_pos, bandpower, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, power in zip(bars, bandpower):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{power:.3e}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    ax.set_xlabel('Accelerometer', fontsize=12)
    ax.set_ylabel('Band Power (g²)', fontsize=12)
    ax.set_title(
        f'{title}\nBand Power ({freq_range[0]}-{freq_range[1]} Hz)',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(accel_names)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Band power plot saved to {output_path}")

    plt.show()


def plot_all_accelerometers_overlay(
    frequencies: np.ndarray,
    psd: np.ndarray,
    title: str,
    output_path: Path = None
) -> None:
    """Create overlay plot comparing all three accelerometers.

    Args:
        frequencies: Frequency array
        psd: PSD array, shape (n_freqs, 3) for 3 accelerometers
        title: Plot title
        output_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{title} - All Accelerometers Comparison', fontsize=14, fontweight='bold')

    accel_names = ['Accelerometer 0', 'Accelerometer 1', 'Accelerometer 2']
    colors = ['blue', 'green', 'red']

    # Linear plot
    for i, (name, color) in enumerate(zip(accel_names, colors)):
        ax1.plot(frequencies, psd[:, i], color=color, linewidth=1.5, label=name, alpha=0.7)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('PSD (g²/Hz)')
    ax1.set_title('Linear Scale')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Semi-log plot
    for i, (name, color) in enumerate(zip(accel_names, colors)):
        ax2.semilogy(frequencies, psd[:, i], color=color, linewidth=1.5, label=name, alpha=0.7)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('PSD (g²/Hz) [log scale]')
    ax2.set_title('Semi-log Scale')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        LOGGER.info(f"Overlay plot saved to {output_path}")

    plt.show()


def main():
    """Main function to visualize Welch spectral analysis."""
    parser = argparse.ArgumentParser(
        description='Visualize spectral analysis using Welch\'s method'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV file with accelerometer data'
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
        '--num-segments',
        type=int,
        default=16,
        help='Number of segments for Welch\'s method (default: 16)'
    )
    parser.add_argument(
        '--freq-range',
        type=float,
        nargs=2,
        default=[50.0, 4000.0],
        help='Frequency range for band power analysis (default: 50 4000)'
    )
    parser.add_argument(
        '--axis-limits',
        type=float,
        nargs=4,
        default=None,
        help='Axis limits for linear plot: xmin xmax ymin ymax (e.g., 30 4500 0 7e-5)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save)'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to project root
        config_path = project_root / args.config
    config = load_config(config_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        LOGGER.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    LOGGER.info(f"Loading data from {csv_path}")
    loader = WebDAQDataLoader(config)
    time_data, signal_data, label = loader.load(csv_path)

    LOGGER.info(
        f"Loaded data: time shape {time_data.shape}, "
        f"signal shape {signal_data.shape}, label: {label}"
    )

    # Preprocess data
    LOGGER.info("Preprocessing signal data")
    preprocessor = SignalPreprocessor(config)
    processed_signal = preprocessor.preprocess(signal_data)

    # Compute Welch PSD
    LOGGER.info("Computing Welch Power Spectral Density")
    fft_processor = FlexibleFFTProcessor(config)
    frequencies, psd = fft_processor.compute_welch_psd(
        processed_signal,
        num_segments=args.num_segments
    )

    LOGGER.info(f"PSD shape: {psd.shape}, frequency range: [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] Hz")

    # Compute band power
    LOGGER.info(f"Computing band power in range {args.freq_range[0]}-{args.freq_range[1]} Hz")
    bandpower = fft_processor.compute_bandpower_welch(
        processed_signal,
        freq_range=tuple(args.freq_range),
        num_segments=args.num_segments
    )

    # Extract filename for title
    filename = csv_path.stem
    title = f"File: {filename} (Label: {label})"

    # Create plots
    if not args.no_show:
        # Linear plot with 3x1 subplots
        plot_welch_spectra(
            frequencies,
            psd,
            title,
            output_path=output_dir / f"{filename}_welch_linear.png",
            axis_limits=args.axis_limits
        )

        # Semi-log plot with 3x1 subplots
        plot_welch_spectra_semilog(
            frequencies,
            psd,
            title,
            output_path=output_dir / f"{filename}_welch_semilog.png"
        )

        # Band power comparison
        plot_bandpower_comparison(
            bandpower,
            tuple(args.freq_range),
            title,
            output_path=output_dir / f"{filename}_bandpower.png"
        )

        # All accelerometers overlay
        plot_all_accelerometers_overlay(
            frequencies,
            psd,
            title,
            output_path=output_dir / f"{filename}_overlay.png"
        )
    else:
        # Save only mode (for batch processing)
        LOGGER.info("Saving plots without display...")
        plt.ioff()

        plot_welch_spectra(
            frequencies,
            psd,
            title,
            output_path=output_dir / f"{filename}_welch_linear.png",
            axis_limits=args.axis_limits
        )
        plt.close()

        plot_welch_spectra_semilog(
            frequencies,
            psd,
            title,
            output_path=output_dir / f"{filename}_welch_semilog.png"
        )
        plt.close()

        plot_bandpower_comparison(
            bandpower,
            tuple(args.freq_range),
            title,
            output_path=output_dir / f"{filename}_bandpower.png"
        )
        plt.close()

        plot_all_accelerometers_overlay(
            frequencies,
            psd,
            title,
            output_path=output_dir / f"{filename}_overlay.png"
        )
        plt.close()

        LOGGER.info("All plots saved successfully")

    LOGGER.info("Spectral analysis complete!")


if __name__ == '__main__':
    main()
