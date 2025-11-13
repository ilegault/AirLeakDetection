# Air Leak Detection Model Export Guide

## Overview

This guide explains how to export your two-stage Random Forest classifier system for air leak detection and hole size classification.

## System Architecture

Your system has **4 Random Forest classifiers** organized in two stages:

```
Stage 1: Leak Detection (1 model)
├─ Input: Accelerometer FFT features
└─ Output: "leak" or "no_leak"
    │
    └─► If "leak" detected
        │
        Stage 2: Hole Size Classification (3 models)
        ├─ Small holes classifier
        ├─ Medium holes classifier  
        └─ Large holes classifier
            │
            └─► Output: "small", "medium", or "large"
```

## Why Use Pickle/Joblib?

For Random Forest models (sklearn), **pickle with joblib** is the best choice because:

✅ **Fast**: Optimized for NumPy arrays and sklearn models  
✅ **Reliable**: Preserves exact model state  
✅ **Complete**: Saves hyperparameters, tree structures, everything  
✅ **Flexible**: Easy to load and use in production  
✅ **Standard**: Industry standard for sklearn models  

**Don't use** TFLite, ONNX, or Keras formats - those are for neural networks, not Random Forests!

## Quick Start

### Option 1: Export Complete System (Recommended)

Export all 4 models as a single bundle:

```bash
python export_air_leak_models.py --bundle \
    --stage1-model models/leak_detector.pkl \
    --stage2-models models/small_holes.pkl models/medium_holes.pkl models/large_holes.pkl \
    --output-dir deployment/models/
```

This creates:
- `air_leak_system_bundle.pkl` - Complete system in one file
- `individual_models/` - Each model separately (backup)
- `system_metadata.json` - Model information
- `DEPLOYMENT_GUIDE.md` - Usage instructions

### Option 2: Export Individual Model

Export a single model:

```bash
python export_air_leak_models.py \
    --model-path models/leak_detector.pkl \
    --output-dir deployment/models/
```

## Detailed Instructions

### Step 1: Organize Your Models

Make sure your trained models are saved as `.pkl` files:

```
models/
├── leak_detector.pkl           # Stage 1: Leak detection
├── small_holes.pkl             # Stage 2: Small hole classifier
├── medium_holes.pkl            # Stage 2: Medium hole classifier
└── large_holes.pkl             # Stage 2: Large hole classifier
```

If you haven't saved them yet:

```python
import joblib

# Save your trained models
joblib.dump(stage1_model, 'models/leak_detector.pkl')
joblib.dump(small_classifier, 'models/small_holes.pkl')
joblib.dump(medium_classifier, 'models/medium_holes.pkl')
joblib.dump(large_classifier, 'models/large_holes.pkl')
```

### Step 2: Export the System

Run the export script with the bundle option:

```bash
python export_air_leak_models.py --bundle \
    --stage1-model models/leak_detector.pkl \
    --stage2-models models/small_holes.pkl models/medium_holes.pkl models/large_holes.pkl \
    --output-dir deployment/models/ \
    --compression 3 \
    --verbose
```

**Parameters explained**:
- `--bundle`: Export all models together
- `--stage1-model`: Path to your leak detection model
- `--stage2-models`: Paths to your three hole size classifiers (order: small, medium, large)
- `--output-dir`: Where to save exported models
- `--compression`: Compression level 0-9 (3 is good balance)
- `--verbose`: Show detailed output

### Step 3: Verify Export

Check that everything exported correctly:

```bash
ls -lh deployment/models/
```

You should see:
- `air_leak_system_bundle.pkl` (~10-50 MB depending on model size)
- `system_metadata.json` (text file with model info)
- `DEPLOYMENT_GUIDE.md` (usage instructions)
- `individual_models/` (directory with separate models)

## Using Exported Models

### In Python (Production Code)

```python
import joblib
import numpy as np

# Load the complete system
system = joblib.load('deployment/models/air_leak_system_bundle.pkl')

# Get the models
stage1_model = system['stage1_classifier']
stage2_models = system['stage2_classifiers']

# Make predictions
def detect_leak(accelerometer_features):
    # Stage 1: Check for leak
    has_leak = stage1_model.predict([accelerometer_features])[0]
    
    if has_leak == 'no_leak':
        return 'No leak detected'
    
    # Stage 2: Determine hole size
    small_pred = stage2_models['small'].predict([accelerometer_features])
    medium_pred = stage2_models['medium'].predict([accelerometer_features])
    large_pred = stage2_models['large'].predict([accelerometer_features])
    
    # Majority vote
    predictions = [small_pred[0], medium_pred[0], large_pred[0]]
    hole_size = max(set(predictions), key=predictions.count)
    
    return f'Leak detected - Hole size: {hole_size}'

# Example usage
features = extract_fft_features(raw_accelerometer_data)
result = detect_leak(features)
print(result)
```

### Using the Example Script

Test your exported models:

```bash
python example_usage.py
```

This demonstrates:
- Loading the system bundle
- Single predictions
- Batch predictions
- Real-time monitoring simulation

## Advanced Options

### Custom Compression

Higher compression = smaller files but slower to load:

```bash
# No compression (fastest, largest)
python export_air_leak_models.py --bundle ... --compression 0

# Maximum compression (slowest, smallest)
python export_air_leak_models.py --bundle ... --compression 9

# Balanced (recommended)
python export_air_leak_models.py --bundle ... --compression 3
```

### Export to Different Location

```bash
# Export to specific directory
python export_air_leak_models.py --bundle ... --output-dir /path/to/production/models/

# Export to cloud storage mount
python export_air_leak_models.py --bundle ... --output-dir /mnt/s3/models/
```

## Troubleshooting

### "Model file not found"
- Check that your model paths are correct
- Use absolute paths or ensure you're in the right directory

### "Failed to load model"
- Verify the file is a valid pickle file
- Make sure you saved it with `joblib.dump()` or `pickle.dump()`
- Check that sklearn version matches (export and import should use same version)

### Import Errors
- Install required packages: `pip install joblib scikit-learn numpy`
- Ensure sklearn version compatibility

### Large File Sizes
- Use compression: `--compression 9`
- Consider feature selection to reduce model complexity
- Use smaller tree depths when training

## Best Practices

1. **Version Control**: Save metadata.json with your models
2. **Testing**: Always test exported models before deployment
3. **Backup**: Keep individual models as backup
4. **Documentation**: Update deployment guide with your specific feature engineering
5. **Monitoring**: Track model performance in production

## Project Context

Based on your Tennessee Tech air leak detection project:
- Uses accelerometers mounted on pipes (80 PSI)
- FFT analysis of vibration data
- Targets 90%+ accuracy
- Two-stage classification approach

The export format preserves:
- ✅ All tree structures and decision rules
- ✅ Feature importance rankings
- ✅ Hyperparameters (n_estimators, max_depth, etc.)
- ✅ Training metadata
- ✅ Class labels and mappings

## Next Steps

1. ✅ Export your models using this guide
2. ✅ Test the exported system with example_usage.py
3. ✅ Integrate into your data acquisition pipeline
4. ✅ Deploy to your testing environment
5. ✅ Monitor performance and iterate

## Questions?

Since you're learning about functions in Python, remember:
- **joblib.dump()** = saves your model to a file (like "save as")
- **joblib.load()** = loads your model from a file (like "open")
- The bundle is just a Python dictionary containing your 4 models

Think of it like exporting a game save file - everything gets packed up so you can load it exactly as it was later!
