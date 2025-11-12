# Accelerometer Classifier Debugging Guide

This guide documents the debugging enhancements added to identify why the accelerometer classifier achieves only 33% accuracy (random guessing).

## Problem Statement

The accelerometer classifier (Random Forest) cannot distinguish between the 3 accelerometers (0, 1, 2), suggesting:
1. Features are identical across all accelerometers, OR
2. The data pipeline is not correctly separating accelerometer data, OR
3. Features don't capture the distance-based differences

## Debugging Enhancements Added

### Step 1: Data Loading Diagnostics (`train_accelerometer_classifier.py`)

**Location**: `scripts/train_accelerometer_classifier.py:132-174`

**What it checks**:
- ✓ Shape of `features.npy` and `labels.npy`
- ✓ Unique values in labels (should be [0, 1, 2])
- ✓ Sample features to ensure they're not all zeros
- ✓ Feature statistics (mean, std, min, max) per accelerometer
- ✓ Feature differences between accelerometer pairs

**How to use**:
```bash
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier/ \
    --model-type random_forest
```

**What to look for in output**:
```
STEP 1: DATA LOADING DIAGNOSTICS
Unique labels in training set: [0 1 2]  ← Should see all 3
Training features all zeros: False       ← Must be False
Mean difference: X.XXXXXX               ← Should be > 0.000001
```

### Step 2: Feature Extraction Diagnostics (`train_two_stage_classifier.py`)

**Location**: `scripts/train_two_stage_classifier.py:146-164, 213-221`

**What it checks**:
- ✓ Keys available in NPZ files
- ✓ Shape and dtype of all arrays
- ✓ Values for all 3 accelerometers
- ✓ Extraction statistics per accelerometer

**How to use**:
```bash
python scripts/train_two_stage_classifier.py \
    --hole-size-data data/processed/ \
    --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl
```

**What to look for**:
```
STEP 2: FEATURE EXTRACTION DIAGNOSTICS
Keys available in NPZ file: ['signal', 'welch_psd', 'welch_bandpower', 'label']
welch_bandpower: Values for all 3 accelerometers: [0.123, 0.456, 0.789]  ← Should be DIFFERENT
```

### Step 3-7: Comprehensive Data Verification (`debug_accelerometer_data.py`)

**Location**: `scripts/debug_accelerometer_data.py`

This script performs a complete diagnostic analysis:

#### Step 3: Check Data Flow
- Verifies accelerometer labels correspond to actual positions
- Confirms we have 3 samples per recording (one per accelerometer)
- Checks class balance

#### Step 4: Restructure Check
- Verifies each sample represents ONE accelerometer
- Confirms labels indicate which accelerometer (0, 1, 2)
- Estimates recordings from sample count

#### Step 5: Verify Feature Differences ⚠️ **CRITICAL**
- Groups features by accelerometer label
- Calculates mean and std for each group
- **These MUST be different if distance-based assumption is correct**
- Generates visualizations:
  - `step5_feature_distributions.png` - Histogram per accelerometer
  - `step5_pca_visualization.png` - 2D PCA projection

#### Step 6: Check Pipeline
- Inspects source NPZ files from `data/processed/`
- Verifies 3 accelerometer columns exist
- **Checks if `welch_bandpower` values are identical (THE KEY CHECK)**
- Verifies sample count (should be 3x number of NPZ files)

#### Step 7: Random Forest Specific Issues
- Checks feature scaling and value ranges
- Identifies constant features
- Quick RF training to verify learnability
- Shows feature importances

**How to use**:
```bash
python scripts/debug_accelerometer_data.py \
    --accelerometer-data data/accelerometer_classifier/ \
    --processed-data data/processed/ \
    --output-dir debug_output/
```

**Expected output locations**:
- Console: Detailed diagnostic logs
- `debug_output/step5_feature_distributions.png`
- `debug_output/step5_pca_visualization.png`

## Common Issues and Solutions

### Issue 1: All Features Identical Across Accelerometers

**Symptoms**:
```
✗ Means are IDENTICAL (bad!)
✗ Stds are IDENTICAL (bad!)
PROBLEM FOUND: Features are IDENTICAL across accelerometers!
```

**Root Cause**: The feature extraction in `prepare_accelerometer_data.py` is not correctly extracting per-accelerometer features.

**Solution**:
1. Check `scripts/prepare_accelerometer_data.py:133-207`
2. Verify `welch_bandpower[accel_id]` extracts different values
3. Inspect source NPZ files: `data/processed/train/*.npz`
4. Ensure `prepare_data.py` generates different values per accelerometer

### Issue 2: Using Wrong Feature Index

**Symptoms**:
```
welch_bandpower: [0.123, 0.123, 0.123]  ← All same!
```

**Root Cause**: Bug in `load_data_for_accelerometer()` - using wrong index or aggregating instead of selecting.

**Current code** (line 153-156 in `train_two_stage_classifier.py`):
```python
bandpower = data["welch_bandpower"]
if len(bandpower) >= 3:
    features = np.array([bandpower[accel_id]], dtype=np.float32)
```

**Check**: Is `bandpower` already per-accelerometer or aggregated?

### Issue 3: Features Have No Variation

**Symptoms**:
```
Value range: 0.000000
✗ ✗ ✗ Feature values have almost no variation!
```

**Root Cause**: Features are constant (all same value).

**Solution**:
1. Check Welch PSD calculation in feature extraction
2. Verify signal processing is working correctly
3. May need to use different features (FFT, PSD instead of just band power)

### Issue 4: Data Structure Mismatch

**Symptoms**:
```
Expected samples (3 per file): 300
Actual samples: 100
✗ Sample count mismatch!
```

**Root Cause**: Not creating 3 samples per recording (one per accelerometer).

**Solution**:
1. Review `prepare_accelerometer_data.py:133-207`
2. Ensure loop creates 3 samples: `for accel_id in range(3):`
3. Each should have label = accel_id

## Quick Diagnostic Workflow

1. **First, prepare the data**:
   ```bash
   # Generate processed NPZ files
   python scripts/prepare_data.py --input-dir data/raw/ --output-dir data/processed/

   # Generate accelerometer classification data
   python scripts/prepare_accelerometer_data.py \
       --input-dir data/processed/ \
       --output-dir data/accelerometer_classifier/
   ```

2. **Run comprehensive diagnostics**:
   ```bash
   python scripts/debug_accelerometer_data.py \
       --accelerometer-data data/accelerometer_classifier/ \
       --processed-data data/processed/ \
       --output-dir debug_output/
   ```

3. **Check the output**:
   - If "IDENTICAL" appears → Problem in NPZ generation
   - If "no variation" appears → Problem in feature extraction
   - If "cannot learn" appears → Features don't capture differences

4. **Inspect source NPZ files manually**:
   ```python
   import numpy as np
   data = np.load('data/processed/train/sample_001.npz')
   print(data['welch_bandpower'])  # Should show 3 DIFFERENT values
   ```

5. **Train with diagnostics**:
   ```bash
   python scripts/train_accelerometer_classifier.py \
       --data-path data/accelerometer_classifier/ \
       --model-type random_forest
   ```

6. **Review diagnostic output in logs** - Look for:
   - Mean differences between accelerometers
   - Feature value ranges
   - Training accuracy on quick RF

## Expected Results (Good Data)

```
STEP 5: VERIFYING FEATURE DIFFERENCES
Accelerometer 0:
  Mean: 0.123456
  Std: 0.045678

Accelerometer 1:
  Mean: 0.234567  ← DIFFERENT from 0
  Std: 0.056789

Accelerometer 2:
  Mean: 0.345678  ← DIFFERENT from 0 and 1
  Std: 0.067890

Accel 0 vs Accel 1:
  Mean difference: 0.111111  ← > 0.000001
  ✓ Means are DIFFERENT (good!)

STEP 7: RANDOM FOREST SPECIFIC CHECKS
Quick RF training accuracy: 0.8500 (85.00%)
✓ RF can learn from the data
```

## Next Steps After Debugging

Once you identify the root cause:

1. **If NPZ files have identical values**: Fix `prepare_data.py` feature extraction
2. **If prepare_accelerometer_data.py extracts wrong**: Fix indexing/selection
3. **If features work but RF fails**: Try different features or more complex model

## ✅ ROOT CAUSE IDENTIFIED AND FIXED

**Problem**: All three accelerometers have nearly IDENTICAL frequency-domain features (Welch PSD). The raw signals have different amplitudes, but Welch PSD captures only spectral shape, not amplitude.

**Solution**: Use amplitude-based features instead! See `ACCELEROMETER_FIX.md` for complete details.

**Quick Fix**:
```bash
# 1. Extract new amplitude-based features
python scripts/extract_amplitude_features.py \
    --input-dir data/processed/ \
    --output-dir data/accelerometer_classifier_v2/

# 2. Train with new features
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier_v2/ \
    --model-type random_forest
```

**Expected result**: Accuracy should improve from 33% to >70%

## Files Modified

- ✅ `scripts/train_accelerometer_classifier.py` - Added Step 1 diagnostics
- ✅ `scripts/train_two_stage_classifier.py` - Added Step 2 diagnostics
- ✅ `scripts/debug_accelerometer_data.py` - **NEW** comprehensive diagnostic script

## Visualization Outputs

The debug script generates:
1. **Feature distributions** - Shows if each accelerometer has different feature values
2. **PCA visualization** - Shows if accelerometers cluster separately in feature space

If accelerometers overlap completely in PCA → features don't capture differences!

## Contact/Support

If issues persist after running diagnostics, share:
1. Output from `debug_accelerometer_data.py`
2. Sample NPZ file inspection results
3. Training log with Step 1 diagnostics
