# Accelerometer Classifier Fix

## Problem Identified

The accelerometer classifier achieved only 33% accuracy (random guessing) because **all three accelerometers have nearly identical frequency-domain features**.

### Root Cause Analysis

From the diagnostic output:

```
Raw Signal (Time Domain):
  Accel 0: std=0.000912  ← DIFFERENT
  Accel 1: std=0.000920  ← DIFFERENT
  Accel 2: std=0.000833  ← DIFFERENT

Welch PSD (Frequency Domain):
  Accel 0: mean=0.000000, std=0.000000  ← IDENTICAL
  Accel 1: mean=0.000000, std=0.000000  ← IDENTICAL
  Accel 2: mean=0.000000, std=0.000000  ← IDENTICAL
```

### Why This Happens

1. **All accelerometers detect the same frequencies** - They're all on the same pipe picking up the same vibrations
2. **Distance affects AMPLITUDE, not FREQUENCY** - The farther accelerometer has weaker signal, but same frequency content
3. **Welch PSD focuses on spectral shape** - It shows which frequencies are present, normalizing out amplitude differences
4. **The differentiating information is LOST** - The key signal (amplitude attenuation) is removed by frequency-domain transformation

## The Solution: Amplitude-Based Features

Instead of using Welch PSD which captures "what frequencies", we need features that capture "how strong".

### New Feature Set (21 features)

#### Time-Domain Statistics (9 features)
1. **RMS** (Root Mean Square) - Overall signal strength
2. **Standard Deviation** - Signal variability
3. **Peak Amplitude** - Maximum excursion
4. **Signal Energy** - Total power
5. **Peak-to-Peak** - Dynamic range
6. **Mean Absolute Value** - Average magnitude
7. **Crest Factor** - Peak/RMS ratio (signal dynamics)
8. **Kurtosis** - Tail behavior (spikiness)
9. **Skewness** - Asymmetry

#### Frequency-Domain Statistics (12 features)
10-12. **FFT Statistics** - Mean, max, std of FFT magnitude (ABSOLUTE values)
13-15. **Band Power** - Low (50-500Hz), Mid (500-1500Hz), High (1500-4000Hz)
16-18. **Welch Statistics** - Mean, max, total power using 'spectrum' scaling (ABSOLUTE values)

**Key difference**: We use ABSOLUTE power values, not normalized spectral density!

## Implementation

### New Script: `extract_amplitude_features.py`

```bash
python scripts/extract_amplitude_features.py \
    --input-dir data/processed/ \
    --output-dir data/accelerometer_classifier_v2/
```

This script:
- Reads NPZ files from `data/processed/`
- Extracts 21 amplitude-based features per accelerometer
- Saves to `data/accelerometer_classifier_v2/`
- Creates train/val/test splits with proper labels

### Training with New Features

```bash
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier_v2/ \
    --model-type random_forest
```

## Expected Improvement

### Before (Welch PSD features):
```
Mean differences: ~0.000000 (identical)
Validation Accuracy: 33% (random guessing)
```

### After (Amplitude features):
```
Mean differences: >0.01 (distinct)
Validation Accuracy: >70% (expected)
```

The farther accelerometer should have:
- **Lower RMS** (weaker vibration)
- **Lower peak amplitude**
- **Lower energy**
- **Lower band power** in all frequency ranges

## Why This Will Work

### Physical Basis
Vibrations attenuate with distance along the pipe:
- **Accelerometer 0** (closest): Strongest signal → High RMS, high peak
- **Accelerometer 1** (middle): Medium signal → Medium RMS, medium peak
- **Accelerometer 2** (farthest): Weakest signal → Low RMS, low peak

### Feature Discriminability
Amplitude-based features capture this directly:
```
Expected pattern (for leak at position 0):
  Accel 0: RMS=0.0015, Peak=0.008  ← Strongest
  Accel 1: RMS=0.0012, Peak=0.006  ← Medium
  Accel 2: RMS=0.0009, Peak=0.004  ← Weakest
```

Random Forest can easily learn: `if RMS > threshold1: predict 0, elif RMS > threshold2: predict 1, else: predict 2`

## Validation Steps

After extracting new features, verify they're different:

```python
import numpy as np

# Load features
X_train = np.load('../data/accelerometer_classifier_v2/train/features.npy')
y_train = np.load('../data/accelerometer_classifier_v2/train/labels.npy')

# Check first feature (RMS) for each accelerometer
for accel_id in range(3):
    mask = y_train == accel_id
    rms_values = X_train[mask, 0]  # First feature is RMS
    print(f"Accel {accel_id}: RMS mean={rms_values.mean():.6f}, std={rms_values.std():.6f}")

# Should see DIFFERENT means!
```

## Files Created

- ✅ `scripts/extract_amplitude_features.py` - New feature extraction script
- ✅ `ACCELEROMETER_FIX.md` - This documentation
- ✅ Data will be saved to `data/accelerometer_classifier_v2/`

## Next Steps

1. **Extract new features**:
   ```bash
   python scripts/extract_amplitude_features.py \
       --input-dir data/processed/ \
       --output-dir data/accelerometer_classifier_v2/
   ```

2. **Verify features are different**:
   - Check the output log for "Mean diff" values
   - Should see differences >1e-6 (ideally >1e-3)

3. **Train classifier**:
   ```bash
   python scripts/train_accelerometer_classifier.py \
       --data-path data/accelerometer_classifier_v2/ \
       --model-type random_forest
   ```

4. **Expected accuracy**: >70% (compared to 33% before)

5. **If still low**: Check if leak position is randomized in your data, or if all accelerometers really are equidistant

## Alternative Approaches (If This Doesn't Work)

If amplitude features still don't discriminate:

### Option 1: Use Signal Ratios
Instead of absolute values, use ratios:
```python
features = [
    signal0_rms / signal1_rms,
    signal1_rms / signal2_rms,
    signal0_peak / signal1_peak,
    ...
]
```

### Option 2: Phase-Based Features
Use cross-correlation or phase delay:
```python
from scipy.signal import correlate
lag_0_1 = np.argmax(correlate(signal0, signal1))
features.append(lag_0_1)
```

### Option 3: Use Raw Signal Segments
Feed small windows of raw signal directly into a CNN:
```python
# Shape: (n_samples, 3, window_size)
X = signal.reshape(-1, 3, 1000)
```

## Technical Details

### Why Welch PSD Failed

Welch's Power Spectral Density uses:
```python
scipy.signal.welch(..., scaling='density')
```

This produces **power per Hz**, normalized by frequency resolution. The normalization removes absolute amplitude information!

### Why Amplitude Features Work

We use:
```python
scipy.signal.welch(..., scaling='spectrum')
```

This produces **absolute power**, preserving amplitude differences!

Plus time-domain features (RMS, peak, etc.) directly measure signal strength.

## References

- See `ACCELEROMETER_DEBUG_GUIDE.md` for diagnostic procedures
- See `debug_output/step5_pca_visualization.png` for visual confirmation of identical features
- Original issue: All three accelerometers cluster together in PCA space
