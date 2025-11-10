# Update: Handling 3 Separate Accelerometers (NOT 3 Axes)

## Problem Identified
- Your data has **3 separate accelerometers** (columns: Acceleration 0, 1, 2)
- NOT 3 axes (X, Y, Z) from a single accelerometer
- Previous code averaged all 3 channels before computing FFT → Loss of information
- Each accelerometer should have its own FFT computed independently

## Files Updated

### 1. `.zencoder/rules/repo.md` (Documentation Updated)
- Clarified that data comes from 3 **separate accelerometers** collected via WebDAQ
- Updated `data_loader.py` spec to parse WebDAQ headers and extract actual sample rate (~17066 Hz)
- Updated `fft_processor.py` spec: **CRITICAL - compute FFT for EACH accelerometer separately**
- Updated `cnn_1d.py` spec: input shape now (403 frequencies, 3 accelerometers)

### 2. `src/data/fft_processor.py` (Code Updated)
**Key changes:**
- `compute_matlab_style_fft()`: Now computes 3 separate FFTs (one per accelerometer)
  - Returns shape: `(n_freqs, 3)` instead of `(n_freqs,)`
- `compute_scipy_fft()`: Same improvement
  - Returns shape: `(n_freqs, 3)`
- `compute_numpy_fft()`: Same improvement  
  - Returns shape: `(n_freqs, 3)`
- `compare_methods()`: Updated to compare per-accelerometer FFTs

**Result format:**
```
FFT output shape: (403, 3)
├─ Column 0: FFT magnitude for Accelerometer 0
├─ Column 1: FFT magnitude for Accelerometer 1
└─ Column 2: FFT magnitude for Accelerometer 2
```

### 3. `config.yaml` (Configuration Updated)
- `sample_rate`: Changed from 10000 to 17066 (actual WebDAQ sample rate)
- `n_channels: 3` clarified as "3 separate accelerometers"
- Added `freq_min: 30` for frequency range limiting

## Current Data Status
- **Processed files**: 1,009 NPZ files already created (699 in train, rest in val/test)
- **Format**: Each file contains:
  ```python
  {
    'signal': shape (170660, 3),            # Raw acceleration from 3 accelerometers at 17066 Hz
    'fft_magnitude': shape (237, 3),        # OLD: Single averaged FFT (will be outdated after reprocessing)
    'frequencies': shape (237,),            # Frequency bins from 30-2000 Hz
    'label': int,
    'class_name': str
  }
  ```
- **Actual Frequency Bins**: 237 (not 403) 
  - Sample rate: 17066 Hz
  - Frequency range: 30-2000 Hz  
  - FFT size: 2048
  - Formula: Each bin = 17066 / 2048 ≈ 8.3 Hz

## Next Steps

### Option 1: Reprocess All Data (Recommended)
Clean old data and regenerate with correct 3-accelerometer FFTs:
```bash
# Clear old processed data
rm -rf data/processed/*

# Run prepare_data with updated code
python scripts/prepare_data.py \
  --raw-data data/raw \
  --output-dir data/processed \
  --compute-fft \
  --verbose
```

**Duration**: ~20-30 minutes for 1,900 files (CPU-intensive FFT computation)

### Option 2: Update Model Input Shape Only
If you want to use existing data, just update your CNN model to handle 3D input:
```python
# CNN Input layer
input_shape = (403, 3)  # frequencies × accelerometers
```

## Model Input Expected Shape
**CNN 1D architecture:**
- Input: `(batch_size, 237, 3)`
  - 237 frequency bins (30-2000 Hz at 17066 Hz sample rate)
  - 3 channels (one per accelerometer)
- Output: 4 classes (NOLEAK, 1/16", 3/32", 1/8")

## Why This Matters
1. **More information**: Each accelerometer captures different vibration patterns
2. **Better features**: Model can learn cross-accelerometer patterns
3. **Realistic**: Matches actual hardware (3 independent sensors)
4. **ML-friendly**: More input dimensions = more feature extraction capability

## Files to Keep Updated
- `src/data/fft_processor.py` ✅ Updated
- `src/models/cnn_1d.py` → Needs input shape adjustment (403, 3)
- `src/data/dataset_generator.py` → Verify batch generation works with (*, 403, 3)
- `src/training/trainer.py` → Verify model compilation with correct shapes