# Air Leak Detection ML System

A machine learning system to detect and classify air leaks using accelerometer data converted to FFT spectral analysis.

## Overview

This project builds a complete ML pipeline for detecting and classifying air leaks in 4 categories:
- **No Leak** - Normal operation
- **1/16" leak** - Small hole
- **3/32" leak** - Medium hole  
- **1/8" leak** - Large hole

The system supports accelerometer data from WebDAQ sensors (9 channels: 3 accelerometers × 3 axes) and integrates with both external MATLAB FFT processing and internal NumPy/SciPy FFT implementations.

## 🚀 Phased Implementation

This project follows a **multi-phase development approach** to incrementally build the complete system. Each phase has specific goals and deliverables.

> **Status**: Currently in early phases with foundational code scaffolding in place. See [`PHASES.md`](docs/PHASES.md) for detailed phase breakdown and current progress.

### Key Features of Phased Approach:
- ✅ Clear phase boundaries with specific deliverables
- ✅ Modular implementation allowing parallel development  
- ✅ Well-documented TODO markers linking to phase numbers
- ✅ Allows pushing to GitHub before all features are implemented
- ✅ Easy to track progress and manage dependencies

## Project Structure

```
air-leak-detection-ml/
├── src/                      # Main source code
│   ├── data/                # Data loading, preprocessing, FFT
│   ├── models/              # Model architectures (CNN, LSTM, RF, SVM, etc.)
│   ├── training/            # Training pipeline
│   ├── evaluation/          # Metrics and visualization
│   ├── prediction/          # Inference and deployment
│   └── utils/               # Configuration, logging, utilities
├── scripts/                 # Executable scripts for training, evaluation, prediction
├── tests/                   # Test suite
├── notebooks/               # Jupyter notebooks for exploration
├── config/                  # Configuration files
├── data/                    # Data directory (ignored in git)
│   ├── raw/                # Raw WebDAQ CSV files (ignored)
│   ├── processed/          # Processed data
│   └── metadata/           # Data metadata
├── models/                  # Trained models (ignored in git)
├── results/                 # Results, figures, reports
└── PHASES.md               # Detailed phase breakdown
```

## Getting Started

### Prerequisites
- Python 3.8+
- See `requirements.txt` for dependencies

### Installation
```bash
# Clone repository
git clone <repo-url>
cd air-leak-detection-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Current Phase Status

**Phase 3** (Models): Core model architectures defined  
**Phase 4** (Training): Training pipeline scaffolding ready  
**Phase 5** (Evaluation): Evaluation framework scaffolding ready  
**Phase 6** (Prediction): Inference pipeline scaffolding ready  

Check [`PHASES.md`](docs/PHASES.md) for:
- Detailed breakdown of each phase
- List of TODOs with their phase assignments
- Current implementation status
- Recommended development order

## Key Modules

### Data Pipeline (`src/data/`)
- **data_loader.py** - Load raw WebDAQ CSV files
- **fft_processor.py** - Flexible FFT processing (MATLAB, NumPy, SciPy)
- **preprocessor.py** - Signal preprocessing (bandpass filter, normalization)
- **feature_extractor.py** - Extract time/frequency domain features
- **dataset_generator.py** - Create TensorFlow/PyTorch datasets

### Models (`src/models/`)
- **cnn_1d.py** - 1D CNN for FFT classification
- **cnn_2d.py** - 2D CNN for spectrograms
- **lstm_model.py** - LSTM for sequential data
- **random_forest_model.py** - Random Forest baseline
- **svm_model.py** - SVM classifier
- **xgboost_model.py** - Gradient boosting
- **ensemble_model.py** - Ensemble methods

### Scripts (`scripts/`)
Execute high-level operations:
```bash
python scripts/prepare_data.py          # Prepare dataset
python scripts/train_model.py           # Train model
python scripts/evaluate.py              # Evaluate performance
python scripts/predict.py               # Run inference
python scripts/compare_fft_methods.py   # Compare FFT approaches
```

See `PHASES.md` for which scripts are currently functional vs. in development.

## Development Notes

### About TODOs
This codebase intentionally contains TODO comments marking unimplemented functionality:
- These are **not bugs** - they are **intentional development markers**
- Each TODO is tied to a specific phase (see `PHASES.md`)
- Scripts are scaffolded to run but will indicate "logic to be implemented"
- This allows pushing to GitHub and collaborating before all features are done

### Before Using
1. Prepare your raw accelerometer data in `data/raw/`
2. Reference `PHASES.md` to understand what's ready vs. in-progress
3. Check phase-specific docs in `/docs/` for implementation details

## Data Directory

The following directories are **excluded from Git** (see `.gitignore`):
- `data/raw/` - Raw accelerometer CSV files
- `models/` - Trained model weights
- `experiments/` - Experiment tracking artifacts
- `*.log` - Log files

This keeps the repository lightweight and prevents accidental commits of large data files.

## Contributing

1. Check `PHASES.md` for the current phase status
2. Create branches for specific phases: `phase-4-training`, `phase-5-evaluation`, etc.
3. Include phase number in commit messages
4. Update `PHASES.md` when completing TODO items

## License

[Add your license here]

## Authors

Isaac Legault

## References

- MATLAB FFT Integration: See `src/data/fft_processor.py`
- WebDAQ Format: See `src/data/data_loader.py`
- Model Architectures: See `src/models/`