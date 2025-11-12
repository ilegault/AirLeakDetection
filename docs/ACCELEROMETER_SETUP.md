# Accelerometer Configuration and Data Structure

## Overview

This document clarifies the accelerometer setup used in the Air Leak Detection system to avoid confusion about the data structure.

## Sensor Configuration

### Hardware Setup

The system uses **3 single-axis accelerometers** (NOT 3-axis accelerometers) mounted at different positions along the pipe:

```
                    PIPE
    [Accel 0]----[Accel 1]--------[Accel 2]
    (Closest)    (Middle)         (Farthest)
        ^            ^                ^
        |            |                |
     Leak       (if present)
```

- **Accelerometer 0**: Positioned closest to potential leak sources
- **Accelerometer 1**: Positioned at a middle distance
- **Accelerometer 2**: Positioned farthest from leak sources

### Why This Configuration?

The different positions allow the system to:
1. **Detect leak location** - Signal strength varies by distance from leak
2. **Improve classification accuracy** - Different sensors see different signal characteristics
3. **Provide redundancy** - Multiple sensors increase reliability

### Expected Signal Behavior

Since accelerometers are at different distances from a leak:

- **Accelerometer 0** (closest):
  - Highest signal magnitude
  - Strongest frequency components
  - Best signal-to-noise ratio

- **Accelerometer 1** (middle):
  - Moderate signal magnitude
  - Good frequency resolution

- **Accelerometer 2** (farthest):
  - **Lower signal magnitude** (this is expected!)
  - May have smaller frequency peaks
  - Signal attenuation due to distance

**⚠️ Important**: If Accelerometer 2 shows much smaller magnitudes than Accelerometer 0, this is **normal physical behavior** due to signal attenuation over distance. This is NOT a data processing error.

## Data Format

### WebDAQ CSV Format

Raw CSV files contain columns:
```
Time, Acceleration 0, Acceleration 1, Acceleration 2
0.0,  0.123,         0.098,          0.045
0.001, 0.125,        0.101,          0.047
...
```

### Processed NPZ Format

After processing, NPZ files contain:

```python
{
    'signal': np.ndarray,          # Shape: (timesteps, 3) - raw time-domain signals
    'fft_magnitude': np.ndarray,   # Shape: (n_freqs, 3) - FFT for each accelerometer
    'frequencies_fft': np.ndarray, # Shape: (n_freqs,) - frequency bins
    'welch_psd': np.ndarray,      # Shape: (n_freqs, 3) - Welch PSD for each accelerometer
    'welch_frequencies': np.ndarray, # Shape: (n_freqs,) - Welch frequency bins
    'welch_bandpower': np.ndarray, # Shape: (3,) - band power for each accelerometer
    'label': int,                  # Class label
    'class_name': str             # Class name string
}
```

### Model Input Shape

Deep learning models receive:
- **Input shape**: `(n_samples, n_frequencies, 3)` or `(n_samples, timesteps, 3)`
- The last dimension (3) corresponds to the 3 accelerometer channels
- All 3 channels are fed to the model simultaneously

## Data Processing Pipeline

### 1. Data Loading (src/data/data_loader.py)

```python
# Loads last 3 columns from CSV
signal = numeric_df.iloc[:, -3:].to_numpy(dtype=np.float32)
# Result shape: (timesteps, 3)
```

### 2. FFT Processing (src/data/fft_processor.py)

FFT is computed **separately for each accelerometer**:

```python
# Compute FFT for EACH accelerometer separately
magnitude_list = []
for i in range(3):  # Loop over 3 accelerometers
    single_channel = signal_data[:, i]
    fft_result = rfft(windowed, n=fft_size)
    magnitude = np.abs(fft_result)
    magnitude_list.append(magnitude)

# Stack magnitudes: shape (n_freqs, 3)
magnitude_stacked = np.column_stack(magnitude_list)
```

### 3. Welch PSD Processing

Welch's method is also computed **separately for each accelerometer**:

```python
# Compute Welch PSD for EACH accelerometer separately
psd_list = []
for i in range(3):
    single_channel = signal_data[:, i]
    frequencies, psd = scipy.signal.welch(
        single_channel,
        fs=sample_rate,
        window='hamming',
        nperseg=segment_length,
        noverlap=num_overlap,
        nfft=nfft
    )
    psd_list.append(psd)

# Stack PSDs: shape (n_freqs, 3)
psd_stacked = np.column_stack(psd_list)
```

### 4. Model Training

Models use all 3 channels:

```python
# CNN1D example
input_shape = (n_frequencies, 3)  # 3 channels
inputs = Input(shape=input_shape, name="signals")
x = Conv1D(filters=32, kernel_size=7, padding="same")(inputs)
# Conv1D processes all 3 channels automatically
```

## Visualization Best Practices

### ✅ Correct: Separate Y-Axis Scales

Since accelerometers have different signal magnitudes, use **separate subplots** with independent y-axes:

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for i in range(3):
    axes[i].plot(frequencies, psd[:, i])
    axes[i].set_ylabel(f'Accelerometer {i}')
    # Each subplot has its own scale
```

### ✅ Correct: Log Scale for Comparison

Use semi-log plots to visualize all accelerometers together:

```python
for i in range(3):
    plt.semilogy(frequencies, psd[:, i], label=f'Accel {i}')
plt.legend()
```

### ❌ Incorrect: Single Linear Scale for All

Don't plot all accelerometers on the same linear scale - Accelerometer 2 may be invisible:

```python
# BAD - Accelerometer 2 signal will be hard to see
for i in range(3):
    plt.plot(frequencies, psd[:, i])
```

## Verification Scripts

### Check NPZ Files

Verify that processed NPZ files contain valid data for all 3 accelerometers:

```bash
python scripts/verify_accelerometer_data.py data/processed/train/sample_001.npz --plot
```

This will:
1. Load the NPZ file
2. Check that all 3 accelerometer channels have non-zero data
3. Print statistics (mean, std, min, max) for each channel
4. Create visualization plots with proper scaling

### Batch Verification

Check all files in a directory:

```bash
python scripts/verify_accelerometer_data.py data/processed/train/ --all
```

## Common Issues and Solutions

### Issue 1: "Third accelerometer appears to have no data"

**Cause**: Using the same y-axis scale for all accelerometers in visualization

**Solution**: Use separate subplots or log scale (see Visualization Best Practices above)

### Issue 2: "Accelerometer 2 has much smaller magnitude"

**Cause**: This is **expected behavior** - Accelerometer 2 is farthest from the leak source

**Solution**: This is not an error. The model learns to use the relative differences between accelerometers.

### Issue 3: "Model only seems to use Accelerometer 0"

**Cause**: Model may be overfitting to the strongest signal

**Solutions**:
1. Use normalization per channel
2. Try feature engineering that captures cross-channel relationships
3. Ensure the model architecture can process all channels (e.g., Conv1D with proper input shape)

## Configuration

In `config.yaml`:

```yaml
data:
  n_channels: 3  # 3 separate single-axis accelerometers (NOT 3-axis)
  sample_rate: 17066
  duration: 10
```

## References

- Data Loader: `src/data/data_loader.py:104` - Column selection logic
- FFT Processor: `src/data/fft_processor.py:62-96` - Per-accelerometer FFT computation
- Preprocessor: `src/data/preprocessor.py:75-111` - Channel handling
- Visualization: `scripts/visualize_welch_spectra.py:48-76` - Proper plotting with separate scales
- Verification: `scripts/verify_accelerometer_data.py` - Data validation tool
