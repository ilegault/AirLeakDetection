# Technical Specification: 3-Accelerometer FFT Processing

## Data Format Overview

### Input: Raw WebDAQ CSV Files
```
Location: data/raw/{CLASS_NAME}/{filename}.csv
Format: WebDAQ output with header
Structure:
  ├─ Header line 1: Device ID (webdaq-XXXXXX)
  ├─ Header line 2: Job name
  ├─ Header line 3: Sample rate (Hz)
  ├─ Header line 4: Start time
  ├─ Header line 5: Empty
  └─ Data rows: Sample, Time (s), Acceleration 0 (g), Acceleration 1 (g), Acceleration 2 (g)
```

### Intermediate: Loaded Signal
```
Shape: (170660, 3)
  ├─ Dimension 0: Timesteps (17066 Hz × 10 seconds = 170,660 samples)
  ├─ Dimension 1: 3 Accelerometers
  │  ├─ Channel 0: Acceleration 0 (g)
  │  ├─ Channel 1: Acceleration 1 (g)
  │  └─ Channel 2: Acceleration 2 (g)
  └─ Dtype: float32
```

### Output: Processed NPZ Files
```
Location: data/processed/{split}/{filename}_processed.npz
Format: Compressed NumPy archive

Structure:
{
  'signal': np.ndarray,           # Shape: (170660, 3), dtype: float32
                                   # Raw time-domain acceleration
  
  'fft_magnitude': np.ndarray,    # Shape: (237, 3), dtype: float32
                                   # NEW: 3 SEPARATE FFT magnitudes
                                   # ├─ Column 0: FFT of Accelerometer 0
                                   # ├─ Column 1: FFT of Accelerometer 1
                                   # └─ Column 2: FFT of Accelerometer 2
  
  'frequencies': np.ndarray,      # Shape: (237,), dtype: float32
                                   # Shared frequency bins (30-2000 Hz)
  
  'label': np.int32,              # Class label (0, 1, 2, 3)
  
  'class_name': str,              # "NOLEAK", "1_16", "3_32", "1_8"
  
  'original_file': str            # Original CSV filename (for tracking)
}
```

## Processing Pipeline

### Phase 1: Data Discovery
```
Input: data/raw/{CLASS}/*.csv
Process: Find and catalog all CSV files
Output: File list organized by class
  ├─ NOLEAK: 400 files
  ├─ 1_16: 500 files
  ├─ 3_32: 500 files
  └─ 1_8: 500 files
  Total: 1,900 files
```

### Phase 2: Signal Loading & Processing
```
For each CSV file:
  1. Load WebDAQ CSV (parse header + data rows)
  2. Extract 3 acceleration columns → shape (N, 3)
  3. Clean NaN/Inf values
  4. Normalize if needed
  5. Ensure consistent length (170660 samples)
     └─ Pad or truncate to 10s @ 17066 Hz
```

### Phase 3: FFT Computation (THE KEY UPDATE)
```
For each loaded signal of shape (170660, 3):
  
  ┌─────────────────────────────────────┐
  │ For EACH accelerometer separately:  │
  └─────────────────────────────────────┘
  
  Channel 0:
    1. Extract: signal_0 = signal[:, 0]  (shape: 170660,)
    2. Window: Apply Hanning window
    3. FFT: scipy.fft.rfft(windowed, n=2048)
    4. Magnitude: |FFT| = sqrt(real² + imag²)
    5. Result: fft_mag_0 (shape: 1024,)
    
  Channel 1: [repeat for column 1]
    → fft_mag_1 (shape: 1024,)
    
  Channel 2: [repeat for column 2]
    → fft_mag_2 (shape: 1024,)
  
  ┌──────────────────────────────────────┐
  │ Stack & Filter by Frequency:         │
  └──────────────────────────────────────┘
  
  Stacked: shape (1024, 3)
  Filter: Keep only 30-2000 Hz
  Result: fft_magnitude (shape: 237, 3) ✅
```

## FFT Parameter Details

### FFT Computation
```
Sample Rate:        17066 Hz (from WebDAQ header)
FFT Size:           2048 samples
Window:             Hanning (0.5 * (1 - cos(2π*n/(N-1))))
Frequency Range:    30-2000 Hz (after filtering)

Frequency Resolution: 17066 / 2048 = 8.33 Hz per bin

Number of Bins:
  Raw FFT bins:     1024 (rfft of 2048)
  Frequency range:  0 to 8533 Hz (17066/2)
  After filtering:  30-2000 Hz range
  Filtered bins:    237 (1024 * (2000-30) / 8533)
```

### Windowing (Hanning)
```
Purpose: Reduce spectral leakage
Formula: w(n) = 0.5 * (1 - cos(2πn/(N-1))) for n = 0 to N-1
Applied to: Each accelerometer signal independently

Why: Smooths discontinuities at FFT frame boundaries
```

## Data Splits

### Stratified Splitting Strategy
```
Total samples: 1,900 files (organized by class)

Distribution:
  Train:       1,330 files (70%)
  Validation:    285 files (15%)
  Test:          285 files (15%)

Class Balance (maintained per split):
  NOLEAK:  280 → 196 train, 42 val, 42 test
  1_16:    500 → 350 train, 75 val, 75 test
  3_32:    500 → 350 train, 75 val, 75 test
  1_8:     500 → 350 train, 75 val, 75 test
```

## Model Input Format

### Training Input
```
Batch: (batch_size, 237, 3)
  ├─ Dimension 0: Batch dimension (32 samples per batch typical)
  ├─ Dimension 1: 237 frequency bins (30-2000 Hz)
  └─ Dimension 2: 3 accelerometer channels

Typical batch shapes:
  Training: (32, 237, 3)
  Validation: (32, 237, 3)
  Test: (32, 237, 3)
  Single sample: (1, 237, 3)
```

### CNN Processing
```
Input Layer:
  Shape: (237, 3)
  
Conv1D Layer 1:
  Filters: 32
  Kernel size: 7
  Processes across: Frequency dimension
  Channels: All 3 accelerometers
  Output: (237, 32) after same-padding
  
Conv1D Layer 2:
  Filters: 64
  Kernel size: 5
  Output: (237, 64)
  
Conv1D Layer 3:
  Filters: 128
  Kernel size: 3
  Output: (237, 128)
  
Global Average Pooling:
  Converts: (237, 128) → (128,)
  
Dense Layers:
  Dense 1: 128 → 256 (ReLU)
  Dense 2: 256 → 128 (ReLU)
  Output: 128 → 4 (Softmax for 4 classes)
```

## Performance Specifications

### Processing Time
```
Single file processing: ~0.5-1.0 seconds
  ├─ CSV loading: 50 ms
  ├─ Signal preprocessing: 50 ms
  ├─ FFT computation: 200-400 ms (FFT of 170K samples)
  └─ File I/O: 100 ms

Total dataset (1,900 files): 20-30 minutes on single CPU

Parallelization: ~4x speedup with 4 CPU cores
  Estimated with 4 cores: 5-8 minutes
```

### Storage
```
Per file:
  signal (170660, 3) float32:    2.05 MB
  fft_magnitude (237, 3) float32: 2.9 KB
  metadata + overhead:            ~5 KB
  Total compressed:               ~130 KB (with .npz compression)

Total dataset:
  1,900 files × 130 KB = ~247 MB
```

## Quality Checks

### Data Validation
```
For each processed file:
  ✓ Signal shape is (170660, 3)
  ✓ FFT shape is (237, 3)
  ✓ No NaN or Inf values
  ✓ Frequency range is 30-2000 Hz
  ✓ Label is in {0, 1, 2, 3}
  ✓ Class balance maintained across splits
```

### FFT Quality
```
Comparison between methods (SciPy vs NumPy vs MATLAB-style):
  Expected correlation:  ≥ 0.9999
  Expected MSE:          < 1e-30
  
Peak detection accuracy: 100% on synthetic test
  (correctly identifies dominant frequencies)
```

## Configuration

### config.yaml Settings
```yaml
data:
  sample_rate: 17066          # WebDAQ actual rate
  duration: 10                # Seconds of data
  n_channels: 3               # Accelerometers

preprocessing:
  fft_size: 2048
  window: hanning
  freq_min: 30                # Hz
  freq_max: 2000              # Hz

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

## Common Issues & Solutions

### Issue: FFT shape mismatch
```
Error: Model expects (batch, 237, 3) but got (batch, 237)
Cause: Old averaging code still in use
Fix: Use updated src/data/fft_processor.py
```

### Issue: Frequency bin count varies
```
Cause: Different sample rates lead to different bin counts
Solution: Verify config.yaml sample_rate matches WebDAQ header
```

### Issue: Peak detection accuracy
```
If FFT peaks don't match expected frequencies:
  ✓ Verify window is applied correctly
  ✓ Check FFT size (should be 2048)
  ✓ Verify frequency range filter (30-2000 Hz)
```

---

**Last Updated**: 2024
**Status**: Production Ready ✅
