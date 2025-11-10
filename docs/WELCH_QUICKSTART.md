# Welch's Method Quick Start Guide

## TL;DR

```bash
# Run demo on your data
python examples/welch_method_demo.py data/raw/NOLEAK/sample.csv

# Generate visualizations
python scripts/visualize_welch_spectra.py data/raw/NOLEAK/sample.csv

# Run tests
python tests/test_welch_method.py
```

## Quick Code Example

```python
from src.data.fft_processor import FlexibleFFTProcessor
import yaml

# Load config and initialize
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
fft_processor = FlexibleFFTProcessor(config)

# Compute Welch PSD (signal_data shape: timesteps x 3)
frequencies, psd = fft_processor.compute_welch_psd(signal_data)

# Compute band power (50-4000 Hz)
bandpower = fft_processor.compute_bandpower_welch(signal_data)

# Result shapes:
# - frequencies: (n_freqs,)
# - psd: (n_freqs, 3) - one per accelerometer
# - bandpower: (3,) - one per accelerometer
```

## Key Features

✅ **Professor's Parameters**: Implements exact specifications
✅ **Hamming Window**: Better spectral leakage reduction
✅ **16 Segments**: Optimal noise reduction with 50% overlap
✅ **Band Power**: Single metric per accelerometer for ML
✅ **Comprehensive Viz**: Linear, semi-log, and comparison plots
✅ **Well Tested**: 5 unit tests covering all functionality

## Parameters

| Parameter | Value | Why? |
|-----------|-------|------|
| Segments | 16 | Reduces variance by ~4x |
| Window | Hamming | Better sidelobe suppression |
| Overlap | 50% | Optimal data usage |
| FFT Size | `max(256, 2^nextpow2(L))` | Better frequency resolution |
| Band Range | 50-4000 Hz | Captures leak signatures |

## What's Included

### Files Added/Modified

```
src/data/fft_processor.py              # ✨ NEW: Welch methods
config.yaml                            # ✨ UPDATED: Welch config
scripts/visualize_welch_spectra.py     # ✨ NEW: Visualization script
examples/welch_method_demo.py          # ✨ NEW: Demo script
tests/test_welch_method.py             # ✨ NEW: Unit tests
docs/WELCH_METHOD.md                   # ✨ NEW: Full documentation
docs/WELCH_QUICKSTART.md               # ✨ NEW: This file
```

### New Methods

1. **`compute_welch_psd()`** - Compute PSD using Welch's method
2. **`compute_bandpower_welch()`** - Calculate band power for classification

## Why Use This?

| Before (Standard FFT) | After (Welch's Method) |
|----------------------|----------------------|
| ❌ Noisy spectra | ✅ Smooth, clean spectra |
| ❌ Hard to identify peaks | ✅ Clear frequency separation |
| ❌ Sensitive to transients | ✅ Robust to noise |
| ❌ Variable results | ✅ Consistent, reproducible |

## Test Results

All tests passing! ✅

```
TEST 1: Welch PSD Shape                 ✓
TEST 2: Welch Parameter Calculation      ✓
TEST 3: Band Power Calculation           ✓
TEST 4: Welch vs Standard FFT            ✓
TEST 5: Frequency Peak Detection         ✓
```

## Integration Ideas

### 1. Add to Data Pipeline

```python
# In prepare_data.py, add:
frequencies, psd = fft_processor.compute_welch_psd(signal)
bandpower = fft_processor.compute_bandpower_welch(signal)
np.savez(output, ..., welch_psd=psd, welch_bandpower=bandpower)
```

### 2. Use as ML Features

```python
# Extract features for classification
features = fft_processor.compute_bandpower_welch(signal)  # Shape: (3,)
# Now you have 3 features per sample for your ML model
```

### 3. Batch Visualization

```bash
# Process all files in a directory
for file in data/raw/NOLEAK/*.csv; do
    python scripts/visualize_welch_spectra.py "$file" --no-show
done
```

## Command Line Examples

### Basic Visualization

```bash
python scripts/visualize_welch_spectra.py data/raw/NOLEAK/trial1.csv
```

### Custom Parameters

```bash
python scripts/visualize_welch_spectra.py data/raw/1_16/trial1.csv \
    --num-segments 16 \
    --freq-range 50 4000 \
    --axis-limits 30 4500 0 7e-5 \
    --output-dir plots/leak_analysis
```

### Batch Processing (No GUI)

```bash
python scripts/visualize_welch_spectra.py data/raw/sample.csv \
    --no-show \
    --output-dir plots
```

## Troubleshooting

**Q: ImportError when running scripts?**
A: Install requirements: `pip install -r requirements.txt`

**Q: No data files to test with?**
A: Use the demo with synthetic data: `python tests/test_welch_method.py`

**Q: Plots don't display?**
A: Use `--no-show` flag or check matplotlib backend

**Q: Spectra look wrong?**
A: Verify signal shape is (timesteps, 3) and sample rate is correct

## Next Steps

1. ✅ Implementation complete
2. ✅ Tests passing
3. ✅ Documentation written
4. ⏭️ Run on real data
5. ⏭️ Compare classification accuracy
6. ⏭️ Integrate into training pipeline

## Learn More

- Full documentation: `docs/WELCH_METHOD.md`
- Source code: `src/data/fft_processor.py`
- Tests: `tests/test_welch_method.py`
- Examples: `examples/welch_method_demo.py`

---

**Questions?** Check the full documentation or the source code comments!
