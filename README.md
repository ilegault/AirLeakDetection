# Air Leak Detection ML System

A machine learning system for detecting and classifying air leaks using multi-accelerometer arrays and spectral analysis. The system uses a **two-stage classification approach** to first identify the optimal sensor position and then classify leak severity.

## Overview

This project implements a complete ML pipeline for industrial air leak detection, capable of classifying leaks into 4 categories:

| Class | Description |
|-------|-------------|
| **NOLEAK** | Normal operation, no leak detected |
| **1/16"** | Small hole leak (1/16 inch) |
| **3/32"** | Medium hole leak (3/32 inch) |
| **1/8"** | Large hole leak (1/8 inch) |

### Key Results

The system achieves **100% accuracy** on the test set using Random Forest and SVM classifiers with amplitude-based features extracted from accelerometer signals.

## Sensor Configuration

The system uses **3 single-axis accelerometers** mounted at different distances along the pipe:

| Accelerometer | Position | WebDAQ Column |
|---------------|----------|---------------|
| **Accelerometer 0** | Closest to leak source | `Acceleration 0` |
| **Accelerometer 1** | Middle position | `Acceleration 1` |
| **Accelerometer 2** | Farthest from leak source | `Acceleration 2` |

> **Note**: These are 3 separate single-axis accelerometers at different physical positions, NOT a single 3-axis accelerometer. Each sensor measures acceleration in a single direction.

## Two-Stage Classification Architecture

```
Multi-Accelerometer Array (3 sensors recording simultaneously)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Stage 1: Position Classifier │
    │  Identify which accelerometer │
    │  is closest to leak source    │
    └───────────────────────────────┘
                    │
                    ▼
         Position ID (0, 1, or 2)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Stage 2: Hole Size Classifier│
    │  Position-specific model for  │
    │  leak severity classification │
    └───────────────────────────────┘
                    │
                    ▼
    Leak Size Prediction (NOLEAK, 1/16", 3/32", 1/8")
```

### Why Two Stages?

1. **Signal Strength Varies with Distance**: The accelerometer closest to a leak will have the strongest signal
2. **Position-Specific Models**: Each position has unique signal characteristics that benefit from specialized classifiers
3. **Improved Accuracy**: By first identifying the optimal sensor, the system can apply the most appropriate classification model

## Project Structure

```
AirLeakDetection/
├── src/                          # Main source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── data_loader.py        # WebDAQ CSV loading
│   │   ├── fft_processor.py      # FFT/Welch PSD processing
│   │   ├── preprocessor.py       # Signal preprocessing
│   │   ├── feature_extractor.py  # Feature extraction
│   │   └── ...
│   ├── models/                   # Model implementations
│   │   ├── two_stage_classifier.py  # Main two-stage classifier
│   │   ├── random_forest.py      # Random Forest (best performer)
│   │   ├── svm_classifier.py     # SVM classifier
│   │   ├── cnn_1d.py             # 1D CNN for FFT data
│   │   ├── lstm_model.py         # LSTM model
│   │   └── ensemble_model.py     # Ensemble methods
│   ├── training/                 # Training pipeline
│   ├── evaluation/               # Evaluation and metrics
│   ├── prediction/               # Inference pipeline
│   └── utils/                    # Utilities and configuration
├── scripts/                      # Executable scripts
│   ├── train_two_stage_classifier_v2.py  # Train two-stage system
│   ├── train_accelerometer_classifier.py # Train Stage 1
│   ├── extract_amplitude_features.py     # Feature extraction
│   └── ...
├── tests/                        # Test suite
├── docs/                         # Documentation
├── config.yaml                   # Configuration file
└── requirements.txt              # Dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- See `requirements.txt` for dependencies

### Installation

```bash
# Clone repository
git clone https://github.com/ilegault/AirLeakDetection.git
cd AirLeakDetection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

#### 1. Prepare Your Data

Place raw WebDAQ CSV files in `data/raw/` organized by class:
```
data/raw/
├── NOLEAK/
│   ├── sample1.csv
│   └── ...
├── 1_16/
├── 3_32/
└── 1_8/
```

#### 2. Process Data and Extract Features

```bash
# Prepare processed data
python scripts/prepare_data.py --input-dir data/raw/ --output-dir data/processed/

# Extract amplitude-based features for accelerometer classification
python scripts/extract_amplitude_features.py \
    --input-dir data/processed/ \
    --output-dir data/accelerometer_classifier_v2/
```

#### 3. Train the Two-Stage Classifier

```bash
# Train Stage 1: Accelerometer position classifier
python scripts/train_accelerometer_classifier.py \
    --data-path data/accelerometer_classifier_v2/ \
    --model-type random_forest

# Train Stage 2: Two-stage classifier with hole size models
python scripts/train_two_stage_classifier_v2.py \
    --accelerometer-data data/accelerometer_classifier_v2/ \
    --accelerometer-classifier models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \
    --hole-size-data data/processed/ \
    --output-dir models/two_stage_classifier_v2/
```

#### 4. Evaluate

```bash
# Evaluate the two-stage classifier
python src/evaluation/evaluate_two_stage_classifier_v2.py \
    --config models/two_stage_classifier_v2/model_*/two_stage_config.json \
    --hole-size-data data/processed/ \
    --output-dir results/two_stage_classifier_v2/
```

## Feature Extraction

The system extracts amplitude-based features that capture signal strength differences between accelerometers:

| Feature | Description |
|---------|-------------|
| RMS | Root Mean Square - overall signal strength |
| Standard Deviation | Signal variability |
| Peak Amplitude | Maximum excursion |
| Signal Energy | Total power |
| Peak-to-Peak | Maximum range |
| Crest Factor | Peak / RMS ratio |
| Kurtosis | Tail behavior (spikiness) |
| Skewness | Signal asymmetry |
| FFT Statistics | Mean, max, std of FFT magnitude |
| Band Power | Power in frequency bands (50-500Hz, 500-1500Hz, 1500-4000Hz) |
| Welch PSD | Power spectral density statistics |

## Models

### Available Models

| Model | Type | Best Use Case |
|-------|------|---------------|
| **Random Forest** | Traditional ML | Best overall performance (100% accuracy) |
| **SVM** | Traditional ML | Excellent performance (100% accuracy) |
| CNN-1D | Deep Learning | FFT magnitude classification |
| LSTM | Deep Learning | Sequential signal data |
| Ensemble | Combined | Model combination strategies |

### Model Performance Summary

| Model | Test Accuracy | Status |
|-------|---------------|--------|
| Random Forest | **100%** | Production Ready |
| SVM | **100%** | Production Ready |
| CNN-1D | 26% | Needs Improvement |
| LSTM | 26% | Needs Improvement |

> Traditional ML models significantly outperform deep learning models on this task. The FFT and amplitude features are highly discriminative, making Random Forest and SVM ideal choices.

## Configuration

Configuration is managed via `config.yaml`:

```yaml
data:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  sample_rate: 17066  # WebDAQ sample rate (Hz)
  duration: 10        # Recording duration (seconds)
  n_channels: 3       # Number of accelerometers

preprocessing:
  fft_size: 2048
  window: "hanning"
  freq_min: 30        # Minimum frequency (Hz)
  freq_max: 2000      # Maximum frequency (Hz)

  # Welch's method parameters
  welch:
    num_segments: 16
    window_type: "hamming"
    overlap_ratio: 0.5
    bandpower_freq_min: 50
    bandpower_freq_max: 4000

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.15
  test_split: 0.15

classes:
  0: "NOLEAK"
  1: "1_16"
  2: "3_32"
  3: "1_8"
```

## FFT Processing Methods

The system supports multiple FFT computation methods:

| Method | Description |
|--------|-------------|
| **Welch PSD** | Power Spectral Density with averaging (recommended) |
| SciPy FFT | Standard FFT with windowing |
| NumPy FFT | Basic FFT implementation |
| MATLAB Import | Load pre-computed FFT from .mat files |

## Data Directory

The following directories are **excluded from Git** (see `.gitignore`):

- `data/raw/` - Raw accelerometer CSV files
- `data/processed/` - Processed NPZ files
- `models/` - Trained model weights
- `results/` - Evaluation results
- `logs/` - Log files

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_evaluation.py -v
```

## Documentation

Additional documentation is available in the `docs/` directory:

- `PHASES.md` - Development phase breakdown
- `TRAINING_GUIDE.md` - Detailed training instructions
- `BENCHMARKING_GUIDE.md` - Performance benchmarking guide
- `WELCH_METHOD.md` - Welch's method implementation details
- `ACCELEROMETER_SETUP.md` - Sensor configuration details

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Isaac Legault

## References

- FFT Processing: `src/data/fft_processor.py`
- WebDAQ Data Format: `src/data/data_loader.py`
- Two-Stage Classifier: `src/models/two_stage_classifier.py`
- Feature Extraction: `scripts/extract_amplitude_features.py`
