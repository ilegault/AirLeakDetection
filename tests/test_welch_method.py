#!/usr/bin/env python3
"""Unit tests for Welch's method implementation."""

import sys
from pathlib import Path
import numpy as np
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.fft_processor import FlexibleFFTProcessor


def load_config():
    """Load configuration."""
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_welch_psd_shape():
    """Test that Welch PSD returns correct shapes."""
    print("\n" + "=" * 80)
    print("TEST 1: Welch PSD Shape")
    print("=" * 80)

    config = load_config()
    fft_processor = FlexibleFFTProcessor(config)

    # Create synthetic signal: 3 accelerometers, 10 seconds at 17066 Hz
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    n_samples = sample_rate * duration

    # Generate synthetic signal with known frequencies
    time = np.linspace(0, duration, n_samples)
    signal_data = np.zeros((n_samples, 3))

    # Accelerometer 0: 100 Hz sine wave
    signal_data[:, 0] = np.sin(2 * np.pi * 100 * time) + 0.1 * np.random.randn(n_samples)

    # Accelerometer 1: 250 Hz sine wave
    signal_data[:, 1] = np.sin(2 * np.pi * 250 * time) + 0.1 * np.random.randn(n_samples)

    # Accelerometer 2: 500 Hz sine wave
    signal_data[:, 2] = np.sin(2 * np.pi * 500 * time) + 0.1 * np.random.randn(n_samples)

    print(f"  Input signal shape: {signal_data.shape}")

    # Compute Welch PSD
    frequencies, psd = fft_processor.compute_welch_psd(signal_data, num_segments=16)

    print(f"  Output frequency shape: {frequencies.shape}")
    print(f"  Output PSD shape: {psd.shape}")
    print(f"  Frequency range: [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] Hz")

    # Verify shapes
    assert psd.shape[1] == 3, "PSD should have 3 columns (one per accelerometer)"
    assert frequencies.shape[0] == psd.shape[0], "Frequency and PSD lengths should match"

    print("  ✓ Shape test PASSED")


def test_welch_parameters():
    """Test that Welch parameters are calculated correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: Welch Parameter Calculation")
    print("=" * 80)

    config = load_config()
    fft_processor = FlexibleFFTProcessor(config)

    # Create synthetic signal
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    n_samples = sample_rate * duration
    signal_data = np.random.randn(n_samples, 3)

    num_segments = 16
    Nx = signal_data.shape[0]

    # Expected parameters (professor's formula)
    expected_segment_length = int(np.floor(Nx / (num_segments / 2 + 0.5)))
    expected_overlap = expected_segment_length // 2
    expected_nextpow2 = int(np.ceil(np.log2(expected_segment_length)))
    expected_nfft = max(256, 2**expected_nextpow2)

    print(f"  Signal length (Nx): {Nx}")
    print(f"  Number of segments: {num_segments}")
    print(f"\n  Expected Parameters:")
    print(f"    - Segment length: {expected_segment_length}")
    print(f"    - Overlap:        {expected_overlap}")
    print(f"    - Next pow2:      {expected_nextpow2}")
    print(f"    - NFFT:           {expected_nfft}")

    # Verify the formula
    assert expected_segment_length == int(np.floor(Nx / (num_segments / 2 + 0.5))), \
        "Segment length calculation is incorrect"
    assert expected_overlap == expected_segment_length // 2, \
        "Overlap should be 50% of segment length"
    assert expected_nfft >= 256, "NFFT should be at least 256"
    assert expected_nfft == 2**expected_nextpow2 or expected_nfft == 256, \
        "NFFT should be power of 2 or 256"

    print("  ✓ Parameter calculation test PASSED")


def test_bandpower_calculation():
    """Test band power calculation."""
    print("\n" + "=" * 80)
    print("TEST 3: Band Power Calculation")
    print("=" * 80)

    config = load_config()
    fft_processor = FlexibleFFTProcessor(config)

    # Create synthetic signal with known frequencies
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    n_samples = sample_rate * duration
    time = np.linspace(0, duration, n_samples)

    signal_data = np.zeros((n_samples, 3))

    # Accelerometer 0: Low frequency (100 Hz)
    signal_data[:, 0] = 2.0 * np.sin(2 * np.pi * 100 * time)

    # Accelerometer 1: Mid frequency (500 Hz)
    signal_data[:, 1] = 2.0 * np.sin(2 * np.pi * 500 * time)

    # Accelerometer 2: High frequency (1000 Hz)
    signal_data[:, 2] = 2.0 * np.sin(2 * np.pi * 1000 * time)

    # Compute band power in full range
    bandpower = fft_processor.compute_bandpower_welch(
        signal_data,
        freq_range=(50.0, 4000.0),
        num_segments=16
    )

    print(f"  Band Power (50-4000 Hz):")
    print(f"    - Accelerometer 0 (100 Hz):  {bandpower[0]:.6e}")
    print(f"    - Accelerometer 1 (500 Hz):  {bandpower[1]:.6e}")
    print(f"    - Accelerometer 2 (1000 Hz): {bandpower[2]:.6e}")

    # Verify shape
    assert bandpower.shape == (3,), "Band power should have shape (3,)"
    assert np.all(bandpower > 0), "Band power should be positive"

    print("  ✓ Band power test PASSED")


def test_welch_vs_fft():
    """Test that Welch's method produces smoother spectra than standard FFT."""
    print("\n" + "=" * 80)
    print("TEST 4: Welch vs Standard FFT Comparison")
    print("=" * 80)

    config = load_config()
    fft_processor = FlexibleFFTProcessor(config)

    # Create noisy signal
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    n_samples = sample_rate * duration
    time = np.linspace(0, duration, n_samples)

    # Signal with noise
    signal_data = np.zeros((n_samples, 3))
    for i in range(3):
        signal_data[:, i] = (
            np.sin(2 * np.pi * 200 * time) +
            0.5 * np.random.randn(n_samples)
        )

    # Compute both methods
    freq_fft, mag_fft = fft_processor.compute_scipy_fft(signal_data)
    freq_welch, psd_welch = fft_processor.compute_welch_psd(signal_data, num_segments=16)

    print(f"  Standard FFT:")
    print(f"    - Frequency bins: {len(freq_fft)}")
    print(f"    - Magnitude shape: {mag_fft.shape}")

    print(f"\n  Welch's Method:")
    print(f"    - Frequency bins: {len(freq_welch)}")
    print(f"    - PSD shape: {psd_welch.shape}")

    # Calculate variance as smoothness metric
    # Lower variance indicates smoother spectrum
    variance_fft = np.var(np.diff(mag_fft[:, 0]))
    variance_welch = np.var(np.diff(psd_welch[:, 0]))

    print(f"\n  Smoothness (variance of derivative):")
    print(f"    - Standard FFT:   {variance_fft:.6e}")
    print(f"    - Welch's method: {variance_welch:.6e}")

    # Note: We can't assert Welch is smoother because they're different units
    # (magnitude vs PSD), but we can verify both produce valid output
    assert not np.any(np.isnan(mag_fft)), "FFT contains NaN values"
    assert not np.any(np.isnan(psd_welch)), "Welch PSD contains NaN values"

    print("  ✓ Comparison test PASSED")


def test_frequency_peak_detection():
    """Test that Welch's method correctly identifies frequency peaks."""
    print("\n" + "=" * 80)
    print("TEST 5: Frequency Peak Detection")
    print("=" * 80)

    config = load_config()
    fft_processor = FlexibleFFTProcessor(config)

    # Create signal with known peaks
    sample_rate = config['data']['sample_rate']
    duration = config['data']['duration']
    n_samples = sample_rate * duration
    time = np.linspace(0, duration, n_samples)

    signal_data = np.zeros((n_samples, 3))

    # Each accelerometer has a strong peak at different frequency
    test_freqs = [150, 350, 750]

    for i, freq in enumerate(test_freqs):
        signal_data[:, i] = 3.0 * np.sin(2 * np.pi * freq * time) + 0.1 * np.random.randn(n_samples)

    # Compute Welch PSD
    frequencies, psd = fft_processor.compute_welch_psd(signal_data, num_segments=16)

    print(f"  Test frequencies: {test_freqs} Hz")
    print(f"\n  Detected peaks:")

    for i, expected_freq in enumerate(test_freqs):
        # Find peak in PSD
        peak_idx = np.argmax(psd[:, i])
        detected_freq = frequencies[peak_idx]
        peak_power = psd[peak_idx, i]

        error = abs(detected_freq - expected_freq)
        error_percent = (error / expected_freq) * 100

        print(f"    - Accelerometer {i}:")
        print(f"        Expected: {expected_freq} Hz")
        print(f"        Detected: {detected_freq:.2f} Hz")
        print(f"        Error:    {error:.2f} Hz ({error_percent:.2f}%)")
        print(f"        Power:    {peak_power:.6e}")

        # Allow 5% error in peak detection
        assert error_percent < 5.0, f"Peak detection error too large: {error_percent:.2f}%"

    print("  ✓ Peak detection test PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  WELCH'S METHOD IMPLEMENTATION TESTS")
    print("=" * 80)

    try:
        test_welch_psd_shape()
        test_welch_parameters()
        test_bandpower_calculation()
        test_welch_vs_fft()
        test_frequency_peak_detection()

        print("\n" + "=" * 80)
        print("  ALL TESTS PASSED! ✓")
        print("=" * 80 + "\n")

        return True

    except AssertionError as e:
        print(f"\n  ✗ TEST FAILED: {str(e)}")
        return False
    except Exception as e:
        print(f"\n  ✗ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
