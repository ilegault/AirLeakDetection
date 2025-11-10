# Welch's Method for Spectral Analysis

## Overview

This document describes the implementation of Welch's method for Power Spectral Density (PSD) analysis in the Air Leak Detection system. The implementation follows specific parameters recommended by the professor for optimal frequency separation and cleaner spectra.

## Why Welch's Method?

Welch's method provides significant advantages over standard FFT for air leak detection:

1. **Reduced Spectral Variance**: Averages multiple overlapping segments to reduce noise
2. **Cleaner Frequency Separation**: Makes leak signatures more distinct
3. **Better Spectral Leakage Reduction**: Hamming window reduces artifacts
4. **Improved Classification**: Smoother spectra lead to better feature extraction

## Implementation Details

### Key Parameters

The implementation uses the following parameters as specified by the professor:

| Parameter | Formula/Value | Description |
|-----------|---------------|-------------|
| **Segment Length** | `floor(Nx / (numSegments/2 + 0.5))` | Number of samples per segment |
| **Number of Segments** | 16 (default) | Number of segments to average |
| **Window Type** | Hamming | Better spectral leakage reduction than Hanning |
| **Overlap** | 50% | `numOverlap = segmentLength / 2` |
| **FFT Points** | `max(256, 2^nextpow2(segmentLength))` | Zero-padding for frequency resolution |

### Calculation Example

For a 10-second recording at 17066 Hz (WebDAQ sample rate):
- **Nx** = 170,660 samples
- **Segment Length** = floor(170660 / (16/2 + 0.5)) = 20,077 samples
- **Overlap** = 10,038 samples (50%)
- **Next Power of 2** = 15
- **FFT Points** = 32,768

This results in approximately 3,783 frequency bins between 30-2000 Hz with excellent frequency resolution.

## Usage

### 1. Basic Usage in Code

```python
from src.data.fft_processor import FlexibleFFTProcessor
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize processor
fft_processor = FlexibleFFTProcessor(config)

# Compute Welch PSD for 3-accelerometer signal
# signal_data shape: (timesteps, 3)
frequencies, psd = fft_processor.compute_welch_psd(
    signal_data,
    num_segments=16,
    window_type='hamming'
)

# Output shapes:
# - frequencies: (n_freqs,)
# - psd: (n_freqs, 3) - one PSD per accelerometer
```

### 2. Band Power Calculation

Calculate total energy in a specific frequency range (useful for ML classification):

```python
# Calculate band power between 50-4000 Hz
bandpower = fft_processor.compute_bandpower_welch(
    signal_data,
    freq_range=(50.0, 4000.0),
    num_segments=16
)

# Output: bandpower shape (3,) - one value per accelerometer
# Each value represents the integrated power in the frequency band
```

### 3. Visualization Script

Generate comprehensive spectral analysis plots:

```bash
# Basic usage
python scripts/visualize_welch_spectra.py data/raw/NOLEAK/sample.csv

# With custom parameters
python scripts/visualize_welch_spectra.py data/raw/1_16/sample.csv \
    --num-segments 16 \
    --freq-range 50 4000 \
    --axis-limits 30 4500 0 7e-5 \
    --output-dir plots/leak_analysis

# Batch processing (no display, save only)
python scripts/visualize_welch_spectra.py data/raw/sample.csv --no-show
```

The script generates four types of plots:
1. **Linear Plot** (3x1 subplots): One subplot per accelerometer with linear scale
2. **Semi-log Plot** (3x1 subplots): Log scale for better visibility of small features
3. **Band Power Bar Chart**: Comparison of total power across accelerometers
4. **Overlay Plot**: All accelerometers compared on same axes (linear and log)

### 4. Demo Script

Run the interactive demonstration:

```bash
python examples/welch_method_demo.py data/raw/NOLEAK/sample.csv
```

This script provides:
- Step-by-step walkthrough of the processing pipeline
- Parameter calculations and explanations
- Comparison with standard FFT
- Interactive visualizations

## Configuration

The Welch's method parameters are configured in `config.yaml`:

```yaml
preprocessing:
  # ... existing preprocessing config ...

  # Welch's method parameters
  welch:
    num_segments: 16              # Number of segments for averaging
    window_type: "hamming"        # Window function type
    overlap_ratio: 0.5            # 50% overlap
    bandpower_freq_min: 50        # Band power minimum (Hz)
    bandpower_freq_max: 4000      # Band power maximum (Hz)
```

## Testing

Comprehensive unit tests are provided in `tests/test_welch_method.py`:

```bash
# Run all tests
python tests/test_welch_method.py
```

Tests include:
1. **Shape Verification**: Ensures correct output dimensions
2. **Parameter Calculation**: Validates professor's formula
3. **Band Power**: Tests power calculation accuracy
4. **Welch vs FFT**: Compares smoothness of spectra
5. **Peak Detection**: Verifies frequency peak identification

All tests use synthetic signals with known properties for validation.

## Integration with ML Pipeline

### Feature Extraction

Band power from Welch's PSD can be used as features for classification:

```python
# Extract band power features for each recording
features = []
for signal in dataset:
    bandpower = fft_processor.compute_bandpower_welch(
        signal,
        freq_range=(50.0, 4000.0)
    )
    features.append(bandpower)  # Shape: (3,) per sample

# features shape: (n_samples, 3)
# Ready for classification model
```

### Data Preparation Pipeline

To integrate Welch's method into the data preparation pipeline, modify `scripts/prepare_data.py`:

```python
# Add Welch PSD computation alongside standard FFT
frequencies, psd = fft_processor.compute_welch_psd(preprocessed_signal)
bandpower = fft_processor.compute_bandpower_welch(preprocessed_signal)

# Save both standard FFT and Welch features
np.savez(
    output_file,
    signal=preprocessed_signal,
    fft_magnitude=fft_magnitude,
    welch_psd=psd,
    welch_bandpower=bandpower,
    frequencies=frequencies,
    label=label
)
```

## API Reference

### FlexibleFFTProcessor.compute_welch_psd()

```python
def compute_welch_psd(
    self,
    signal_data: np.ndarray,
    num_segments: int = 16,
    window_type: str = 'hamming'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        signal_data: Shape (timesteps, 3) for 3 accelerometers
        num_segments: Number of segments for averaging (default: 16)
        window_type: Window function ('hamming', 'hanning', etc.)

    Returns:
        frequencies: Shape (n_freqs,) - frequency bins in Hz
        psd: Shape (n_freqs, 3) - PSD for each accelerometer in g²/Hz
    """
```

### FlexibleFFTProcessor.compute_bandpower_welch()

```python
def compute_bandpower_welch(
    self,
    signal_data: np.ndarray,
    freq_range: Tuple[float, float] = (50.0, 4000.0),
    num_segments: int = 16
) -> np.ndarray:
    """
    Calculate band power in specific frequency range using Welch's PSD.

    Args:
        signal_data: Shape (timesteps, 3) for 3 accelerometers
        freq_range: (min_freq, max_freq) in Hz
        num_segments: Number of segments for Welch's method

    Returns:
        bandpower: Shape (3,) - integrated power per accelerometer in g²
    """
```

## Comparison: Welch vs Standard FFT

| Aspect | Standard FFT | Welch's Method |
|--------|-------------|----------------|
| **Noise Reduction** | No averaging | Averages 16 segments |
| **Spectral Variance** | High | Low (reduced by ~4x) |
| **Frequency Resolution** | Good | Excellent (with zero-padding) |
| **Transient Noise** | Sensitive | Robust |
| **Computation Time** | Fast | Moderate |
| **Best For** | Clean signals | Noisy signals, classification |

## Theoretical Background

### Periodogram Averaging

Welch's method computes the periodogram (squared magnitude of FFT) for multiple overlapping segments and averages them:

```
PSD(f) = (1/K) * Σ |FFT_k(f)|²
```

where K is the number of segments.

### Variance Reduction

For K independent segments, variance is reduced by factor K:

```
Var(PSD_welch) ≈ Var(PSD_periodogram) / K
```

With 16 segments, this provides ~4x reduction in standard deviation.

### Window Function

The Hamming window is defined as:

```
w(n) = 0.54 - 0.46 * cos(2πn / (N-1))
```

It provides better sidelobe suppression than Hanning window, reducing spectral leakage.

## Visualization Recommendations

For consistent results across different leak types:

1. **Linear Scale**: Use axis limits [30, 4500, 0, 7e-5] for comparison
2. **Semi-log Scale**: Better for identifying small peaks
3. **3x1 Subplots**: Compare accelerometers side-by-side
4. **Band Power Bar Chart**: Quick comparison of relative power

## Troubleshooting

### Issue: Frequency resolution too coarse

**Solution**: Increase `num_segments` or check `nfft` calculation

### Issue: Spectra too smooth, losing peak information

**Solution**: Decrease `num_segments` (e.g., 8 instead of 16)

### Issue: Band power values seem incorrect

**Solution**: Verify frequency range is within computed PSD range (30-2000 Hz by default)

### Issue: Memory error with large signals

**Solution**: Process in batches or reduce `nfft` parameter

## References

1. Welch, P. D. (1967). "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms". IEEE Transactions on Audio and Electroacoustics.

2. Harris, F. J. (1978). "On the use of windows for harmonic analysis with the discrete Fourier transform". Proceedings of the IEEE.

3. SciPy Documentation: `scipy.signal.welch` - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

## Authors and Acknowledgments

Implementation by Claude based on professor's specifications for the Air Leak Detection project.

Special thanks to the professor for providing the optimal parameters and methodology.
