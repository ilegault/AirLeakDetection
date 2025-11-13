#!/usr/bin/env python3
"""
Export Air Leak Detection Models - Two-Stage Classifier System

This script exports a complete two-stage classification system:
- Stage 1: Main classifier (accelerometer detection)
- Stage 2: Three specialized classifiers (one per accelerometer)

The script can export individual models or the entire system as a bundle.

Usage:
    # Export entire system as a bundle with detected models
    python export_air_leak_models.py --bundle --auto-detect
    
    # Export entire system as a bundle (explicit paths)
    python export_air_leak_models.py --bundle \
        --stage1-model models/accelerometer_classifier/model_20251112_164556/random_forest_accelerometer.pkl \
        --stage2-models models/two_stage_classifier/model_20251112_154823/accel_0_classifier.pkl \
                        models/two_stage_classifier/model_20251112_154823/accel_1_classifier.pkl \
                        models/two_stage_classifier/model_20251112_154823/accel_2_classifier.pkl \
        --output-dir deployment/models/
    
    # Export individual model
    python export_air_leak_models.py --model-path models/accelerometer_classifier/model_20251112_164556/random_forest_accelerometer.pkl \
        --output-dir deployment/models/
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from glob import glob

import joblib
import numpy as np


def auto_detect_models() -> tuple:
    """Auto-detect the latest trained models."""
    print("\n🔍 Auto-detecting trained models...")
    
    # Find latest accelerometer classifier (Stage 1: position detector)
    accel_pattern = "/home/viscosity_c/PycharmProjects/AirLeakDetection/data/accelerometer_classifier/*/random_forest_accelerometer.pkl"
    accel_files = sorted(glob(accel_pattern), reverse=True)
    
    if not accel_files:
        print("  ✗ No Stage 1 accelerometer position detector found in data/accelerometer_classifier/")
        return None, None
    
    stage1 = accel_files[0]
    print(f"  ✓ Stage 1 (Position Detector): {Path(stage1).parent.name}")
    
    # Find latest position-specific classifiers (Stage 2)
    # Looking for any .pkl file matching accel_* in v2 directory
    v2_pattern = "/home/viscosity_c/PycharmProjects/AirLeakDetection/data/accelerometer_classifier_v2/*/*.pkl"
    v2_files = glob(v2_pattern)
    
    if not v2_files:
        print("  ✗ No Stage 2 position-specific classifiers found in data/accelerometer_classifier_v2/")
        return stage1, None
    
    # Extract unique parent directories and get latest
    base_dirs = sorted(set(str(Path(f).parent) for f in v2_files), reverse=True)
    
    if not base_dirs:
        print("  ✗ Could not determine Stage 2 model directory")
        return stage1, None
    
    base_dir = base_dirs[0]
    stage2_files = [
        str(Path(base_dir) / "accel_0_classifier.pkl"),
        str(Path(base_dir) / "accel_1_classifier.pkl"),
        str(Path(base_dir) / "accel_2_classifier.pkl")
    ]
    
    # Verify all files exist
    missing = [f for f in stage2_files if not Path(f).exists()]
    if missing:
        print(f"  ✗ Missing files: {missing}")
        print(f"     Expected in: {base_dir}")
        print(f"     Available files: {sorted(glob(str(Path(base_dir) / '*.pkl')))}")
        return stage1, None
    
    print(f"  ✓ Stage 2 (Position-Specific Classifiers 0,1,2): {Path(base_dir).name}")
    return stage1, stage2_files


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for model export."""
    parser = argparse.ArgumentParser(
        description="Export air leak detection models (two-stage classifier system)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with auto-detected models (recommended)
  python export_air_leak_models.py --bundle --auto-detect
  
  # Export complete two-stage system as bundle (explicit)
  python export_air_leak_models.py --bundle \\
      --stage1-model models/accelerometer_classifier/model_20251112_164556/random_forest_accelerometer.pkl \\
      --stage2-models models/two_stage_classifier/model_20251112_154823/accel_0_classifier.pkl \\
                      models/two_stage_classifier/model_20251112_154823/accel_1_classifier.pkl \\
                      models/two_stage_classifier/model_20251112_154823/accel_2_classifier.pkl \\
      --output-dir deployment/
  
  # Export single model
  python export_air_leak_models.py \\
      --model-path models/accelerometer_classifier/model_20251112_164556/random_forest_accelerometer.pkl \\
      --output-dir deployment/
        """
    )
    
    # Single model export options
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to single trained model (for individual export)"
    )
    
    # Bundle export options
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Export all models as a complete system bundle"
    )
    
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect the latest trained models"
    )
    
    parser.add_argument(
        "--stage1-model",
        type=str,
        help="Path to Stage 1 classifier (accelerometer detection)"
    )
    
    parser.add_argument(
        "--stage2-models",
        type=str,
        nargs=3,
        metavar=('ACCEL0', 'ACCEL1', 'ACCEL2'),
        help="Paths to Stage 2 classifiers (accel_0, accel_1, accel_2)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment/models/",
        help="Output directory for exported models"
    )
    
    parser.add_argument(
        "--compression",
        type=int,
        default=3,
        choices=range(0, 10),
        help="Compression level for pickle (0=none, 9=max, 3=default)"
    )
    
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=True,
        help="Include metadata file with model export"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def validate_model_file(model_path: str) -> bool:
    """Validate that model file exists and is readable."""
    path = Path(model_path)
    if not path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return False
    if not path.is_file():
        print(f"ERROR: Path is not a file: {model_path}")
        return False
    if path.suffix not in ['.pkl', '.joblib']:
        print(f"WARNING: Unexpected file extension: {path.suffix}")
    return True


def load_model(model_path: str):
    """Load a pickled model using joblib."""
    print(f"Loading model from: {model_path}")
    try:
        model = joblib.load(model_path)
        print(f"  ✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        raise


def export_single_model(model_path: str, output_dir: str, compression: int = 3):
    """Export a single model."""
    print(f"\n{'='*60}")
    print("EXPORTING SINGLE MODEL")
    print(f"{'='*60}\n")
    
    # Validate input
    if not validate_model_file(model_path):
        return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(model_path)
    
    # Generate output filename
    input_path = Path(model_path)
    output_file = output_path / f"{input_path.stem}_exported.pkl"
    
    # Export model
    print(f"\nExporting model to: {output_file}")
    joblib.dump(model, output_file, compress=compression)
    print(f"  ✓ Model exported successfully")
    
    # Get model info
    model_info = get_model_info(model)
    
    # Save metadata
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'original_model_path': str(model_path),
        'exported_model_path': str(output_file),
        'compression_level': compression,
        'model_info': model_info
    }
    
    metadata_file = output_path / f"{input_path.stem}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to: {metadata_file}")
    
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    
    return True


def export_system_bundle(stage1_path: str, stage2_paths: list, 
                        output_dir: str, compression: int = 3):
    """Export complete two-stage classifier system as a bundle."""
    print(f"\n{'='*60}")
    print("EXPORTING TWO-STAGE CLASSIFIER SYSTEM BUNDLE")
    print(f"{'='*60}\n")
    
    # Validate all model files
    all_paths = [stage1_path] + stage2_paths
    for path in all_paths:
        if not validate_model_file(path):
            return False
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all models
    print("\nLoading Stage 1 classifier (accelerometer position detector)...")
    stage1_model = load_model(stage1_path)
    
    print("\nLoading Stage 2 classifiers (position-specific hole size detectors)...")
    stage2_models = {
        '0': load_model(stage2_paths[0]),
        '1': load_model(stage2_paths[1]),
        '2': load_model(stage2_paths[2])
    }
    
    # Create system bundle
    system_bundle = {
        'stage1_classifier': stage1_model,
        'stage2_classifiers': stage2_models,
        'version': '2.0',
        'export_date': datetime.now().isoformat(),
        'description': 'Two-stage air leak detection: position detection → hole size classification',
        'architecture': 'Position-based routing system'
    }
    
    # Export bundle
    bundle_file = output_path / "air_leak_system_bundle.pkl"
    print(f"\nExporting system bundle to: {bundle_file}")
    joblib.dump(system_bundle, bundle_file, compress=compression)
    print(f"  ✓ System bundle exported successfully")
    
    # Also export individual models for flexibility
    print("\nExporting individual models...")
    individual_dir = output_path / "individual_models"
    individual_dir.mkdir(exist_ok=True)
    
    joblib.dump(stage1_model, individual_dir / "stage1_position_detector.pkl", compress=compression)
    print(f"  ✓ Stage 1 (position detector) model saved")
    
    for pos, model in stage2_models.items():
        joblib.dump(model, individual_dir / f"stage2_accel_{pos}_hole_classifier.pkl", compress=compression)
        print(f"  ✓ Stage 2 (accelerometer {pos}) model saved")
    
    # Create comprehensive metadata
    metadata = {
        'system_info': {
            'version': '2.0',
            'export_timestamp': datetime.now().isoformat(),
            'architecture': 'two-stage position-based routing',
            'description': 'Air leak detection: Stage 1 detects which accelerometer is closest to leak, Stage 2 classifies hole size for that position'
        },
        'stage1': {
            'purpose': 'Detect which accelerometer (0, 1, or 2) is closest to the leak source',
            'input': 'Accelerometer features (all 3 channels)',
            'output_classes': ['0', '1', '2'],
            'output_description': 'Position of closest accelerometer',
            'model_path': str(stage1_path),
            'model_info': get_model_info(stage1_model)
        },
        'stage2': {
            'purpose': 'Classify hole size based on accelerometer position',
            'input': 'Accelerometer features (position-specific)',
            'output_classes': ['No Leak', '1/16"', '3/32"', '1/8"'],
            'routing': 'Route input to Stage 2 classifier matching Stage 1 output position',
            'models': {
                '0': {
                    'description': 'Hole size classifier for accelerometer position 0',
                    'model_path': str(stage2_paths[0]),
                    'model_info': get_model_info(stage2_models['0'])
                },
                '1': {
                    'description': 'Hole size classifier for accelerometer position 1',
                    'model_path': str(stage2_paths[1]),
                    'model_info': get_model_info(stage2_models['1'])
                },
                '2': {
                    'description': 'Hole size classifier for accelerometer position 2',
                    'model_path': str(stage2_paths[2]),
                    'model_info': get_model_info(stage2_models['2'])
                }
            }
        },
        'deployment_info': {
            'bundle_file': str(bundle_file),
            'individual_models_dir': str(individual_dir),
            'compression_level': compression
        }
    }
    
    # Save metadata
    metadata_file = output_path / "system_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  ✓ System metadata saved to: {metadata_file}")
    
    # Create deployment guide
    create_deployment_guide(output_path)
    
    print(f"\n{'='*60}")
    print("SYSTEM BUNDLE EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"\nBundle location: {bundle_file}")
    print(f"Individual models: {individual_dir}")
    print(f"Metadata: {metadata_file}")
    
    return True


def get_model_info(model) -> dict:
    """Extract information about the model."""
    info = {
        'model_type': type(model).__name__,
        'model_class': str(type(model))
    }
    
    # Try to get Random Forest specific info
    try:
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = int(model.n_estimators)
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = int(model.n_features_in_)
        if hasattr(model, 'n_classes_'):
            info['n_classes'] = int(model.n_classes_)
        if hasattr(model, 'classes_'):
            info['classes'] = [str(c) for c in model.classes_]
    except Exception as e:
        info['extraction_error'] = str(e)
    
    return info


def create_deployment_guide(output_dir: Path):
    """Create a deployment guide for using the exported models."""
    guide = """# Air Leak Detection System - Deployment Guide

## System Architecture

This is a **two-stage position-based routing system** for air leak detection:

**Stage 1: Accelerometer Position Detector**
- **Input**: Accelerometer features from all 3 channels
- **Output**: Which accelerometer (0, 1, or 2) is closest to the leak
- **Purpose**: Determine which sensor detected the strongest signal
- **Routing**: Directs to the corresponding Stage 2 classifier

**Stage 2: Position-Specific Hole Size Classifiers** (3 models)
- **Input**: Accelerometer features (routed by Stage 1 output)
- **Output**: Hole size classification (No Leak, 1/16", 3/32", 1/8")
- **Model 0**: Classifier trained on accelerometer position 0
- **Model 1**: Classifier trained on accelerometer position 1
- **Model 2**: Classifier trained on accelerometer position 2

## Workflow

```
Raw Accelerometer Data (3 channels)
           ↓
   Feature Extraction
           ↓
  Stage 1: Position Detector
  "Which accelerometer is closest?"
           ↓
      Position: 0, 1, or 2
           ↓
   Route to Stage 2 Classifier
   (matching detected position)
           ↓
  Stage 2: Hole Size Classifier
  "What's the hole size?"
           ↓
  Final Output: No Leak / 1/16" / 3/32" / 1/8"
```

## File Structure

```
deployment/models/
├── air_leak_system_bundle.pkl          # Complete system (recommended)
├── system_metadata.json                 # Detailed system information
├── DEPLOYMENT_GUIDE.md                  # This file
└── individual_models/                   # Individual models (alternative)
    ├── stage1_position_detector.pkl
    ├── stage2_accel_0_hole_classifier.pkl
    ├── stage2_accel_1_hole_classifier.pkl
    └── stage2_accel_2_hole_classifier.pkl
```

## Usage Examples

### Option 1: Load Complete Bundle (Recommended)

```python
import joblib
import numpy as np

# Load the complete system
system = joblib.load('air_leak_system_bundle.pkl')

stage1_model = system['stage1_classifier']
stage2_models = system['stage2_classifiers']  # Dict with keys '0', '1', '2'

def predict_air_leak(accelerometer_features):
    '''
    Args:
        accelerometer_features: numpy array of shape (n_features,)
        containing preprocessed features from all 3 accelerometers
    
    Returns:
        dict with keys:
            - 'detected_position': Which accelerometer is closest (0, 1, or 2)
            - 'hole_size': Detected hole size (No Leak / 1/16" / 3/32" / 1/8")
            - 'confidence': Prediction confidence (0-1)
    '''
    
    # Stage 1: Detect which accelerometer is closest
    position = stage1_model.predict([accelerometer_features])[0]
    position_proba = stage1_model.predict_proba([accelerometer_features])[0]
    position_confidence = np.max(position_proba)
    
    # Stage 2: Classify hole size using position-specific classifier
    classifier = stage2_models[str(position)]
    hole_size = classifier.predict([accelerometer_features])[0]
    hole_proba = classifier.predict_proba([accelerometer_features])[0]
    hole_confidence = np.max(hole_proba)
    
    return {
        'detected_position': int(position),
        'hole_size': hole_size,
        'position_confidence': float(position_confidence),
        'hole_confidence': float(hole_confidence),
        'overall_confidence': float((position_confidence + hole_confidence) / 2)
    }

# Example usage
features = np.random.rand(18)  # Adjust to match your feature count
result = predict_air_leak(features)
print(f"Closest accelerometer: {result['detected_position']}")
print(f"Hole size: {result['hole_size']}")
print(f"Confidence: {result['overall_confidence']:.2%}")
```

### Option 2: Load Individual Models

```python
import joblib

# Load models separately
stage1 = joblib.load('individual_models/stage1_position_detector.pkl')
stage2_0 = joblib.load('individual_models/stage2_accel_0_hole_classifier.pkl')
stage2_1 = joblib.load('individual_models/stage2_accel_1_hole_classifier.pkl')
stage2_2 = joblib.load('individual_models/stage2_accel_2_hole_classifier.pkl')

stage2_models = {0: stage2_0, 1: stage2_1, 2: stage2_2}

# Make prediction
features = [...preprocessed accelerometer data...]
position = stage1.predict([features])[0]
hole_size = stage2_models[position].predict([features])[0]
```

### Option 3: Get Predictions with Probabilities

```python
import joblib
import numpy as np

system = joblib.load('air_leak_system_bundle.pkl')
stage1 = system['stage1_classifier']
stage2_models = system['stage2_classifiers']

features = [...preprocessed data...]

# Stage 1: Get position probabilities
position_probs = stage1.predict_proba([features])[0]
positions = stage1.classes_  # Usually [0, 1, 2]

# Stage 2: Get hole size probabilities for detected position
best_position = np.argmax(position_probs)
hole_probs = stage2_models[str(best_position)].predict_proba([features])[0]
hole_classes = stage2_models[str(best_position)].classes_

# Print results
for pos, prob in zip(positions, position_probs):
    print(f"Position {pos}: {prob:.2%}")
    
for hole_class, prob in zip(hole_classes, hole_probs):
    print(f"{hole_class}: {prob:.2%}")
```

## Input Data Format

Your accelerometer data must be preprocessed into features before prediction:

**Expected:**
- Feature vector of shape: (n_features,)
- Typically 18 features per the training configuration
- Should include data from all 3 accelerometers
- Normalized/scaled according to training preprocessing

**Example preprocessing:**

```python
import numpy as np
from scipy.fft import fft
from scipy.signal import hann

def extract_features(accelerometer_data_3ch, sampling_rate=17066):
    '''
    Extract features from 3-channel accelerometer data
    
    Args:
        accelerometer_data_3ch: shape (n_samples, 3)
        sampling_rate: sampling frequency in Hz
    
    Returns:
        features: 1D array of features for prediction
    '''
    features = []
    
    for ch in range(3):
        signal = accelerometer_data_3ch[:, ch]
        
        # Time domain features
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(np.max(signal))
        features.append(np.min(signal))
        features.append(np.ptp(signal))  # peak-to-peak
        
        # Frequency domain features
        window = hann(len(signal))
        fft_vals = np.abs(fft(signal * window))
        fft_freqs = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        features.append(np.max(fft_vals))
        features.append(fft_freqs[np.argmax(fft_vals)])
    
    return np.array(features)
```

## Performance Optimization

For real-time or high-throughput predictions:

1. **Load models once**: Load at startup, reuse for multiple predictions
2. **Batch processing**: Accumulate multiple prediction requests
3. **Confidence thresholds**: Use predict_proba() to set decision thresholds
4. **Caching**: Cache repeated feature extractions if input data is similar
5. **Warm-up**: Make test predictions to initialize any runtime caches

```python
import joblib

# Load once at initialization
system = joblib.load('air_leak_system_bundle.pkl')
stage1 = system['stage1_classifier']
stage2_models = system['stage2_classifiers']

# Make many predictions without reloading
def batch_predict(features_list):
    results = []
    for features in features_list:
        position = stage1.predict([features])[0]
        hole_size = stage2_models[str(position)].predict([features])[0]
        results.append((position, hole_size))
    return results
```

## Model Versioning & Metadata

- **System Version**: 2.0 (Position-based routing)
- **Export Date**: See system_metadata.json
- **Architecture**: Two-stage position-based classification
- **Stage 1 Output**: Accelerometer position (3 classes: 0, 1, 2)
- **Stage 2 Output**: Hole size (4 classes: No Leak, 1/16", 3/32", 1/8")

For detailed model information, see `system_metadata.json`.

## Troubleshooting

**Issue: Models not loading**
- Verify file paths are correct
- Check that .pkl files are not corrupted
- Ensure joblib version matches export version

**Issue: Unexpected predictions**
- Verify input features have same format/scale as training data
- Check that feature extraction preprocessing matches training
- Review system_metadata.json for model input requirements

**Issue: Low confidence scores**
- May indicate out-of-distribution data
- Consider retraining or data augmentation
- Check if input data quality matches training data

## Support

For issues or questions, refer to system_metadata.json for detailed configuration details or contact the development team.
"""
    
    guide_file = output_dir / "DEPLOYMENT_GUIDE.md"
    with open(guide_file, 'w') as f:
        f.write(guide)
    print(f"  ✓ Deployment guide created: {guide_file}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Determine export mode
    if args.bundle:
        # Bundle export mode
        stage1 = args.stage1_model
        stage2 = args.stage2_models
        
        # Auto-detect if requested
        if args.auto_detect:
            stage1, stage2 = auto_detect_models()
            if not stage1 or not stage2:
                print("ERROR: Could not auto-detect models")
                return 1
        
        # Validate that we have the required models
        if not stage1 or not stage2:
            print("ERROR: --bundle requires --stage1-model and --stage2-models (or --auto-detect)")
            return 1
        
        success = export_system_bundle(
            stage1,
            stage2,
            args.output_dir,
            args.compression
        )
    elif args.model_path:
        # Single model export mode
        success = export_single_model(
            args.model_path,
            args.output_dir,
            args.compression
        )
    else:
        print("ERROR: Must specify either --model-path or --bundle mode")
        print("Use --help for usage information")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
