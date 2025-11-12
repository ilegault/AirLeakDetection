# Two-Stage Classifier Guide

## Overview

The **Two-Stage Classifier** system combines accelerometer identification with hole size detection to create a comprehensive air leak detection solution.

### Architecture

```
Input Signal (3 accelerometers)
        ↓
┌──────────────────────────┐
│ Stage 1: Accelerometer   │
│    Identification        │
│ (Which sensor? 0, 1, 2)  │
└──────────────────────────┘
        ↓
┌──────────────────────────┐
│ Stage 2: Hole Size       │
│    Detection             │
│ (NOLEAK, 1/16", 3/32",   │
│  1/8")                   │
└──────────────────────────┘
        ↓
Combined Prediction
(Accelerometer ID + Hole Size)
```

### Why Two-Stage?

1. **Sensor Position Matters**: Each accelerometer is at a different distance from the leak source, resulting in different signal characteristics
2. **Specialized Models**: Train separate hole-size classifiers optimized for each accelerometer's unique signal profile
3. **Higher Accuracy**: Avoids the confusion of training a single model on data from all 3 accelerometers mixed together
4. **Interpretability**: Know which sensor detected the leak and what size it is

## Components

### 1. Data Preparation Scripts

#### `prepare_accelerometer_data.py`
Prepares data for training the accelerometer classifier (Stage 1).

**Input**: Processed NPZ files from `prepare_data.py` (shape: timesteps × 3 channels)

**Output**: Individual accelerometer samples labeled with accelerometer ID (0, 1, 2)

**Usage**:
```bash
python scripts/prepare_accelerometer_data.py \
    --input-dir data/processed/ \
    --output-dir data/accelerometer_classifier/ \
    --use-welch \
    --use-bandpower
```

**Options**:
- `--use-fft`: Use FFT features
- `--use-welch`: Use Welch PSD features (default: True)
- `--use-bandpower`: Use band power features (default: True)
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)

### 2. Training Scripts

#### `train_accelerometer_classifier.py`
Trains a classifier to identify which accelerometer (0, 1, 2) a sample came from.

**Usage**:
```bash
# Train Random Forest
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier/ \
    --model-type random_forest \
    --output-dir models/accelerometer_classifier/

# Train SVM
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier/ \
    --model-type svm \
    --output-dir models/accelerometer_classifier/
```

**Output**:
- Trained accelerometer classifier (`.pkl` file)
- Training metadata (accuracy, confusion matrix)

#### `train_two_stage_classifier.py`
Trains the complete two-stage system:
1. Uses pre-trained accelerometer classifier (Stage 1)
2. Trains separate hole-size classifiers for each accelerometer (Stage 2)
3. Combines them into a unified system

**Usage**:
```bash
python scripts/train_two_stage_classifier.py \
    --hole-size-data data/processed/ \
    --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \
    --model-type random_forest \
    --output-dir models/two_stage_classifier/
```

**Output**:
- 3 hole-size classifiers (one per accelerometer): `accel_0_classifier.pkl`, `accel_1_classifier.pkl`, `accel_2_classifier.pkl`
- Two-stage configuration: `two_stage_config.json`
- Metadata: `metadata.json`

### 3. Evaluation Script

#### `evaluate_two_stage_classifier.py`
Evaluates the two-stage classifier and generates detailed performance metrics.

**Usage**:
```bash
python scripts/evaluate_two_stage_classifier.py \
    --model-dir models/two_stage_classifier/model_*/ \
    --test-data data/processed/test/ \
    --accelerometer-test-data data/accelerometer_classifier/test/ \
    --plot \
    --output-dir results/two_stage_evaluation/
```

**Output**:
- Evaluation results JSON
- Stage 1 confusion matrix (accelerometer identification)
- Stage 2 confusion matrix (hole size detection)
- Confidence distribution plots
- Per-accelerometer accuracy plots

## Complete Workflow

### Step 1: Prepare Original Data
```bash
# If you haven't already, prepare the raw data
python scripts/prepare_data.py \
    --raw-data data/raw/ \
    --output-dir data/processed/ \
    --compute-fft \
    --augment
```

### Step 2: Prepare Accelerometer Classification Data
```bash
python scripts/prepare_accelerometer_data.py \
    --input-dir data/processed/ \
    --output-dir data/accelerometer_classifier/ \
    --use-welch \
    --use-bandpower
```

### Step 3: Train Accelerometer Classifier (Stage 1)
```bash
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier/ \
    --model-type random_forest \
    --output-dir models/accelerometer_classifier/
```

Expected output: **~95-100% accuracy** distinguishing between the 3 accelerometers

### Step 4: Train Two-Stage Classifier
```bash
python scripts/train_two_stage_classifier.py \
    --hole-size-data data/processed/ \
    --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \
    --model-type random_forest \
    --output-dir models/two_stage_classifier/
```

This trains 3 separate hole-size classifiers (one per accelerometer).

### Step 5: Evaluate Two-Stage System
```bash
python scripts/evaluate_two_stage_classifier.py \
    --model-dir models/two_stage_classifier/model_*/ \
    --test-data data/processed/test/ \
    --accelerometer-test-data data/accelerometer_classifier/test/ \
    --plot \
    --output-dir results/two_stage_evaluation/
```

## Python API Usage

### Using TwoStageClassifier in Code

```python
from src.models.two_stage_classifier import TwoStageClassifier
import numpy as np

# Load the classifier
classifier = TwoStageClassifier(
    accelerometer_classifier_path="models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl",
    hole_size_classifier_paths={
        0: "models/two_stage_classifier/model_*/accel_0_classifier.pkl",
        1: "models/two_stage_classifier/model_*/accel_1_classifier.pkl",
        2: "models/two_stage_classifier/model_*/accel_2_classifier.pkl",
    },
    class_names={
        0: "NOLEAK",
        1: "1_16",
        2: "3_32",
        3: "1_8"
    }
)

# Predict on a single sample
features = np.load("sample.npy")
result = classifier.predict_single(features)

print(f"Accelerometer: {result['accelerometer_name']}")
print(f"Hole Size: {result['hole_size_name']}")
print(f"Combined Confidence: {result['combined_confidence']:.3f}")

# Predict on batch
features_batch = np.load("batch.npy")  # shape: (n_samples, n_features)
results = classifier.predict_batch(features_batch)

for i, result in enumerate(results):
    print(f"Sample {i}: {result['accelerometer_name']} -> {result['hole_size_name']}")
```

### Evaluation Example

```python
from src.models.two_stage_classifier import TwoStageClassifier
import numpy as np

# Load test data
X_test = np.load("data/accelerometer_classifier/test/features.npy")
y_accel_test = np.load("data/accelerometer_classifier/test/labels.npy")
y_hole_test = np.load("data/accelerometer_classifier/test/hole_size_labels.npy")

# Flatten if needed
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Evaluate
results = classifier.evaluate(X_test_flat, y_accel_test, y_hole_test)

print(f"Stage 1 Accuracy: {results['stage1_accuracy']:.3f}")
print(f"Stage 2 Accuracy: {results['stage2_accuracy']:.3f}")
print(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
```

## Expected Performance

Based on your previous work achieving **100% classification for hole sizes**, you can expect:

- **Stage 1 (Accelerometer ID)**: 95-100% accuracy
  - Each accelerometer has distinct signal characteristics based on distance from leak

- **Stage 2 (Hole Size)**: 90-100% accuracy per accelerometer
  - Performance depends on how well each accelerometer's signal distinguishes hole sizes
  - Accelerometer 0 (closest) likely performs best

- **Overall (End-to-End)**: 90-100% accuracy
  - Product of both stage accuracies
  - If Stage 1 = 100% and Stage 2 = 95%, Overall ≈ 95%

## Tips for Best Performance

1. **Feature Selection**:
   - Use Welch PSD features for frequency analysis
   - Band power (50-4000 Hz) is excellent for distinguishing accelerometers
   - Experiment with different frequency ranges

2. **Model Selection**:
   - **Random Forest**: Fast, interpretable, works well with small datasets
   - **SVM**: Better for complex decision boundaries
   - Try both and compare!

3. **Data Quality**:
   - Ensure all 3 accelerometers have good signal quality
   - Verify using `scripts/verify_accelerometer_data.py`
   - Check for class imbalance

4. **Hyperparameter Tuning**:
   - Adjust `n_estimators` for Random Forest (default: 300)
   - Try different kernels for SVM (`rbf`, `linear`, `poly`)
   - Use `class_weight='balanced'` for imbalanced data

## Troubleshooting

### Low Stage 1 Accuracy
- Check if accelerometer signals are truly distinct
- Verify data preparation (are channels mixed up?)
- Try different features (raw signal vs FFT vs Welch)

### Low Stage 2 Accuracy
- Some accelerometers may not capture leak information well
- Try training on Accelerometer 0 (closest) only
- Consider combining multiple accelerometers

### Low Overall Accuracy
- Errors compound across stages
- Focus on improving the weaker stage first
- Consider ensemble methods or data augmentation

## Next Steps

1. **Real-Time Deployment**: Use `TwoStageClassifier` in production pipeline
2. **Feature Engineering**: Experiment with additional features (time-domain stats, wavelet transforms)
3. **Model Fusion**: Combine predictions from multiple accelerometers for robustness
4. **Continuous Learning**: Retrain periodically with new data

## References

- Original data preparation: `scripts/prepare_data.py`
- Model architectures: `src/models/`
- Configuration: `config.yaml`
