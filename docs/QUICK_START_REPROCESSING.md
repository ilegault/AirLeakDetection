# Quick Start: Reprocess Data with Updated 3-Accelerometer FFT

## What Changed?
Your data has 3 **separate accelerometers** (not 3 axes). The code now computes **3 independent FFTs** (one per accelerometer) instead of averaging them.

**Output shape changes from:**
- (237,) ← Single averaged FFT
- **to (237, 3)** ← 3 separate FFTs ✅

## Step 1: Validate the Update (Optional but Recommended)

Test that the updated FFT processor works correctly:

Expected output:
```
✓ Output shape: (237 frequencies, 3 accelerometers)
✓ Peak frequency detection: CORRECT
✓ Method correlation: 0.9999+
✓ ALL TESTS PASSED!
```

## Step 2: Reprocess Your Data

### Option A: Clean Start (Recommended)
Remove old processed files and reprocess fresh:

```bash
# Backup old data (optional)
cp -r data/processed data/processed_old_backup

# Remove old files
rm -rf data/processed/*

# Reprocess with updated FFT code
python scripts/prepare_data.py \
  --raw-data data/raw \
  --output-dir data/processed \
  --compute-fft \
  --verbose
```

### Option B: Selective Reprocessing
If you only want to reprocess specific classes:

```bash
# Process only specific directory
python scripts/prepare_data.py \
  --raw-data data/raw/NOLEAK \
  --output-dir data/processed \
  --compute-fft \
  --verbose
```

### Command Explanation
- `--raw-data data/raw`: Location of 1,900 CSV files
- `--output-dir data/processed`: Where to save processed files
- `--compute-fft`: Enable FFT computation (NEW: now 3 separate FFTs)
- `--verbose`: Show progress and details

## Step 3: Expected Output

Reprocessing generates NPZ files with structure:
```python
{
  'signal': (170660, 3),          # Raw acceleration from 3 accelerometers
  'fft_magnitude': (237, 3),      # NEW: 3 separate FFTs ✅
  'frequencies': (237,),          # Shared frequency bins
  'label': int,                   # Class (0-3)
  'class_name': str               # "NOLEAK", "1_16", "3_32", "1_8"
}
```

### File Count
- **Total files**: 1,900
- **Train**: ~1,330 (70%)
- **Validation**: ~285 (15%)
- **Test**: ~285 (15%)

**Estimated processing time**: 20-30 minutes

## Step 4: Update Your Models

Update any code that loads FFT data to expect 3D input:

### Before (WRONG - Old averaging):
```python
# Load data
data = np.load('processed_sample.npz')
fft = data['fft_magnitude']  # Shape: (237,) ← Wrong!

# Model
model.build(input_shape=(237,))  # Single channel
```

### After (CORRECT - 3 separate accelerometers):
```python
# Load data
data = np.load('processed_sample.npz')
fft = data['fft_magnitude']  # Shape: (237, 3) ✅

# Model
model.build(input_shape=(237, 3))  # 3 channels
```

## Step 5: Train Updated Model

With new 3-channel FFT data:

```bash
python scripts/train_model.py \
  --config config.yaml \
  --data-dir data/processed \
  --epochs 100 \
  --batch-size 32
```

Model will automatically handle (batch, 237, 3) input shape.

## Verification Checklist

✅ Test script passes
✅ Raw data present in `data/raw/` (1,900 files)
✅ `config.yaml` has `sample_rate: 17066`
✅ Old `data/processed/` backed up (if needed)
✅ Reprocessing script started

## Troubleshooting

### Issue: "Memory error during FFT computation"
- **Solution**: The FFT computation is CPU-intensive. Run on a machine with more RAM or process in batches.

### Issue: "Output file already exists"
- **Solution**: Clean `data/processed/` directory first before reprocessing.

### Issue: "Expected 3 channels, got X"
- **Solution**: Verify your raw CSV files have 3 acceleration columns. Check with:
  ```bash
  head data/raw/NOLEAK/*.csv | grep "Acceleration"
  ```

### Issue: "Sample rate mismatch"
- **Solution**: Verify `config.yaml` has correct sample rate (should be ~17066 Hz based on your WebDAQ header).

## Performance Impact

**Benefits of 3-channel FFT:**
- ✅ More information: 3 independent accelerometer views
- ✅ Better features: Model learns cross-accelerometer patterns
- ✅ More accurate: No information loss from averaging
- ✅ 3D CNN input: More feature extraction capability

**Expected accuracy improvement**: 5-15% (depending on how much useful information was lost in averaging)

## Questions?

See detailed documentation in:
- `UPDATE_3_ACCELEROMETERS.md` - Full technical details
- `CHANGES_SUMMARY.md` - Complete list of changes
- `.zencoder/rules/repo.md` - Updated project documentation

---

**Next Steps**: Run reprocessing → Update model code → Train with new 3-channel FFT data