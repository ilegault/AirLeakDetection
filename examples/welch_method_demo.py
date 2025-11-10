#!/usr/bin/env python3
"""Demonstration of Welch's method for spectral analysis.

This example script shows how to:
1. Load accelerometer data from CSV
2. Apply preprocessing (detrending, windowing)
3. Compute Power Spectral Density using Welch's method
4. Calculate band power in specific frequency ranges
5. Compare Welch's method with standard FFT

Usage:
    python examples/welch_method_demo.py data/raw/NOLEAK/sample.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import WebDAQDataLoader
from src.data.preprocessor import SignalPreprocessor
from src.data.fft_processor import FlexibleFFTProcessor


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print('=' * 80)


def main():
    """Demonstrate Welch's method for spectral analysis."""
    if len(sys.argv) < 2:
        print("Usage: python examples/welch_method_demo.py <csv_file>")
        print("\nExample:")
        print("  python examples/welch_method_demo.py data/raw/NOLEAK/noleak_trial1.csv")
        sys.exit(1)

    csv_file = Path(sys.argv[1])
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    # Load configuration
    config_path = project_root / "config.yaml"
    config = load_config(config_path)

    print_section("WELCH'S METHOD DEMONSTRATION")
    print(f"Input file: {csv_file}")
    print(f"Configuration: {config_path}")

    # ==================================================================================
    # Step 1: Load Data
    # ==================================================================================
    print_section("Step 1: Load Data")

    loader = WebDAQDataLoader(config)
    time_data, signal_data, label = loader.load(csv_file)

    print(f"  Time data shape:   {time_data.shape}")
    print(f"  Signal data shape: {signal_data.shape}")
    print(f"  Label:             {label}")
    print(f"  Sample rate:       {config['data']['sample_rate']} Hz")
    print(f"  Duration:          {len(time_data) / config['data']['sample_rate']:.2f} seconds")

    # ==================================================================================
    # Step 2: Preprocess Data
    # ==================================================================================
    print_section("Step 2: Preprocess Data")

    preprocessor = SignalPreprocessor(config)
    processed_signal = preprocessor.preprocess(signal_data)

    print(f"  Preprocessed signal shape: {processed_signal.shape}")
    print(f"  Preprocessing steps:")
    print(f"    - Detrending (remove DC offset)")
    print(f"    - Normalization")

    # ==================================================================================
    # Step 3: Compute Welch PSD (Professor's Method)
    # ==================================================================================
    print_section("Step 3: Compute Welch Power Spectral Density")

    fft_processor = FlexibleFFTProcessor(config)

    # Get Welch parameters from config
    welch_params = config.get('preprocessing', {}).get('welch', {})
    num_segments = welch_params.get('num_segments', 16)

    print(f"  Welch's Method Parameters:")
    print(f"    - Number of segments: {num_segments}")
    print(f"    - Window type:        {welch_params.get('window_type', 'hamming')}")
    print(f"    - Overlap ratio:      {welch_params.get('overlap_ratio', 0.5)}")

    # Calculate expected parameters
    Nx = processed_signal.shape[0]
    segment_length = int(np.floor(Nx / (num_segments / 2 + 0.5)))
    num_overlap = segment_length // 2
    nextpow2 = int(np.ceil(np.log2(segment_length)))
    nfft = max(256, 2**nextpow2)

    print(f"\n  Calculated Parameters:")
    print(f"    - Signal length (Nx):     {Nx} samples")
    print(f"    - Segment length:         {segment_length} samples")
    print(f"    - Overlap samples:        {num_overlap} samples")
    print(f"    - FFT size (with padding): {nfft} points")

    # Compute Welch PSD
    frequencies_welch, psd_welch = fft_processor.compute_welch_psd(
        processed_signal,
        num_segments=num_segments
    )

    print(f"\n  Welch PSD Output:")
    print(f"    - Frequency array shape: {frequencies_welch.shape}")
    print(f"    - PSD array shape:       {psd_welch.shape}")
    print(f"    - Frequency range:       [{frequencies_welch[0]:.2f}, {frequencies_welch[-1]:.2f}] Hz")
    print(f"    - Frequency resolution:  {frequencies_welch[1] - frequencies_welch[0]:.4f} Hz")

    # ==================================================================================
    # Step 4: Compute Band Power
    # ==================================================================================
    print_section("Step 4: Compute Band Power")

    freq_min = welch_params.get('bandpower_freq_min', 50)
    freq_max = welch_params.get('bandpower_freq_max', 4000)

    bandpower = fft_processor.compute_bandpower_welch(
        processed_signal,
        freq_range=(freq_min, freq_max),
        num_segments=num_segments
    )

    print(f"  Band Power Analysis ({freq_min}-{freq_max} Hz):")
    print(f"    - Accelerometer 0: {bandpower[0]:.6e} g²")
    print(f"    - Accelerometer 1: {bandpower[1]:.6e} g²")
    print(f"    - Accelerometer 2: {bandpower[2]:.6e} g²")
    print(f"\n  Relative Power:")
    max_power = np.max(bandpower)
    for i in range(3):
        relative = (bandpower[i] / max_power) * 100
        print(f"    - Accelerometer {i}: {relative:.2f}% of maximum")

    # ==================================================================================
    # Step 5: Compare with Standard FFT
    # ==================================================================================
    print_section("Step 5: Compare Welch's Method vs Standard FFT")

    # Compute standard FFT for comparison
    frequencies_fft, magnitude_fft = fft_processor.compute_scipy_fft(processed_signal)

    print(f"  Standard FFT:")
    print(f"    - Frequency array shape: {frequencies_fft.shape}")
    print(f"    - Magnitude array shape: {magnitude_fft.shape}")
    print(f"\n  Welch's Method:")
    print(f"    - Frequency array shape: {frequencies_welch.shape}")
    print(f"    - PSD array shape:       {psd_welch.shape}")

    print(f"\n  Key Differences:")
    print(f"    - Welch's method averages {num_segments} overlapping segments")
    print(f"    - This reduces spectral variance and noise")
    print(f"    - Makes leak signatures more distinct")
    print(f"    - Better for classification tasks")

    # ==================================================================================
    # Step 6: Visualize Results
    # ==================================================================================
    print_section("Step 6: Visualize Results")

    # Create comparison plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(
        f'Welch\'s Method vs Standard FFT\nFile: {csv_file.name} (Label: {label})',
        fontsize=14,
        fontweight='bold'
    )

    accel_names = ['Accelerometer 0', 'Accelerometer 1', 'Accelerometer 2']
    colors = ['blue', 'green', 'red']

    for i, (name, color) in enumerate(zip(accel_names, colors)):
        # Left column: Standard FFT magnitude
        ax_fft = axes[i, 0]
        ax_fft.plot(frequencies_fft, magnitude_fft[:, i], color=color, linewidth=0.8, alpha=0.7)
        ax_fft.set_xlabel('Frequency (Hz)')
        ax_fft.set_ylabel('FFT Magnitude')
        ax_fft.set_title(f'{name} - Standard FFT')
        ax_fft.grid(True, alpha=0.3)

        # Right column: Welch PSD
        ax_welch = axes[i, 1]
        ax_welch.plot(frequencies_welch, psd_welch[:, i], color=color, linewidth=1.0)
        ax_welch.set_xlabel('Frequency (Hz)')
        ax_welch.set_ylabel('PSD (g²/Hz)')
        ax_welch.set_title(f'{name} - Welch PSD')
        ax_welch.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create band power bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(3)
    bars = ax.bar(x_pos, bandpower, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels
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
        f'Band Power Comparison ({freq_min}-{freq_max} Hz)\n'
        f'File: {csv_file.name} (Label: {label})',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(accel_names)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\n  Plots displayed successfully!")

    # ==================================================================================
    # Summary
    # ==================================================================================
    print_section("SUMMARY")

    print("  Why Welch's Method Works Better:")
    print("    1. Reduces spectral variance by averaging multiple segments")
    print("    2. Hamming window reduces spectral leakage better than rectangular")
    print("    3. Proper overlap ensures all data is used while getting good averaging")
    print("    4. Makes leak signatures more distinct and easier to classify")
    print("    5. Band power provides single metric for ML classification")

    print("\n  Next Steps:")
    print("    - Use visualize_welch_spectra.py for detailed analysis")
    print("    - Integrate Welch features into model training pipeline")
    print("    - Compare classification accuracy with standard FFT")
    print("    - Analyze band power patterns across different leak types")

    print("\n" + "=" * 80)
    print("  Demo complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
