# Phase 8: Executable Scripts & CLI Framework

## Overview

Phase 8 implements all executable scripts as specified in repo.md. These scripts serve as the CLI interface to the Air Leak Detection ML system, enabling:

- Training models with various configurations
- Evaluating model performance
- Making predictions on new data
- Data preparation and validation
- Hyperparameter optimization
- Cross-validation studies
- Model export for deployment
- Performance benchmarking

---

## 📋 Scripts Overview

### 1. **train_model.py** - Main Training Script
```bash
python scripts/train_model.py \
    --model-type cnn_1d \
    --data-path data/processed/ \
    --config config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --output-dir models/
```

**Features:**
- Support all model types (CNN, LSTM, RF, SVM, XGBoost, Ensemble)
- Load raw or FFT data
- Automatic experiment tracking (MLflow/WandB)
- Checkpoint saving
- Logging to file and console

---

### 2. **train_with_external_fft.py** - MATLAB FFT Training
```bash
python scripts/train_with_external_fft.py \
    --fft-source matlab \
    --matlab-path data/matlab_fft/ \
    --compare-methods
```

**Features:**
- Load teammate's MATLAB FFT
- Compare different FFT methods
- Train on external FFT data
- Performance comparison

---

### 3. **compare_fft_methods.py** - FFT Method Comparison
```bash
python scripts/compare_fft_methods.py \
    --raw-data data/raw/ \
    --output-dir results/fft_comparison/
```

**Features:**
- Compare numpy, scipy, and MATLAB FFT
- Generate correlation metrics
- Create visualizations
- Recommend best method

---

### 4. **predict.py** - Inference Script
```bash
python scripts/predict.py \
    --model-path models/best_model.h5 \
    --input data/test/ \
    --output results/predictions.json \
    --confidence-threshold 0.8
```

**Features:**
- Single file or batch prediction
- Multiple output formats (JSON, CSV)
- Confidence scoring
- Threshold filtering

---

### 5. **evaluate.py** - Model Evaluation
```bash
python scripts/evaluate.py \
    --model-path models/best_model.h5 \
    --test-data data/processed/test/ \
    --output-dir results/evaluation/
```

**Features:**
- Compute all metrics
- Generate confusion matrix
- ROC curves
- Detailed error analysis
- HTML report generation

---

### 6. **prepare_data.py** - Data Preparation
```bash
python scripts/prepare_data.py \
    --raw-data data/raw/ \
    --output-dir data/processed/ \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

**Features:**
- Load and validate raw data
- Apply preprocessing
- Create train/val/test splits
- Stratified sampling
- Data augmentation (optional)

---

### 7. **hyperparameter_search.py** - Hyperparameter Optimization
```bash
python scripts/hyperparameter_search.py \
    --model-type cnn_1d \
    --search-method bayesian \
    --n-trials 50 \
    --output-dir results/hpo/
```

**Features:**
- Grid search
- Random search
- Bayesian optimization (Optuna)
- Parallel trials
- Best model selection

---

### 8. **cross_validate.py** - K-Fold Cross-Validation
```bash
python scripts/cross_validate.py \
    --model-type cnn_1d \
    --data-path data/processed/ \
    --k-folds 5 \
    --output-dir results/cv/
```

**Features:**
- Stratified k-fold splitting
- Model stability assessment
- Cross-validation metrics
- Fold-wise analysis

---

### 9. **export_model.py** - Model Export
```bash
python scripts/export_model.py \
    --model-path models/best_model.h5 \
    --format tflite \
    --output-dir deployment/models/
```

**Features:**
- Export to TensorFlow Lite
- ONNX format
- TorchScript (PyTorch)
- Quantization options
- Model optimization

---

### 10. **benchmark.py** - Performance Benchmarking
```bash
python scripts/benchmark.py \
    --model-path models/best_model.h5 \
    --test-data data/processed/test/ \
    --output-dir results/benchmarks/
```

**Features:**
- Inference speed benchmarking
- Memory usage profiling
- Accuracy-speed tradeoff analysis
- Hardware compatibility testing

---

## 🚀 Common Workflows

### Complete Training Pipeline
```bash
# 1. Prepare data
python scripts/prepare_data.py \
    --raw-data data/raw/ \
    --output-dir data/processed/

# 2. Train model
python scripts/train_model.py \
    --model-type cnn_1d \
    --data-path data/processed/ \
    --epochs 100 \
    --output-dir models/

# 3. Evaluate
python scripts/evaluate.py \
    --model-path models/best_model.h5 \
    --test-data data/processed/test/ \
    --output-dir results/evaluation/

# 4. Predict on new data
python scripts/predict.py \
    --model-path models/best_model.h5 \
    --input data/new_samples/ \
    --output results/predictions.json
```

### Hyperparameter Optimization
```bash
# Find best hyperparameters
python scripts/hyperparameter_search.py \
    --model-type cnn_1d \
    --search-method bayesian \
    --n-trials 50 \
    --output-dir results/hpo/

# Cross-validate best model
python scripts/cross_validate.py \
    --model-type cnn_1d \
    --data-path data/processed/ \
    --k-folds 5 \
    --output-dir results/cv/
```

### MATLAB Integration
```bash
# Compare FFT methods
python scripts/compare_fft_methods.py \
    --raw-data data/raw/ \
    --output-dir results/fft_comparison/

# Train with MATLAB FFT
python scripts/train_with_external_fft.py \
    --fft-source matlab \
    --matlab-path data/matlab_fft/
```

### Model Export & Deployment
```bash
# Benchmark before export
python scripts/benchmark.py \
    --model-path models/best_model.h5 \
    --test-data data/processed/test/

# Export for deployment
python scripts/export_model.py \
    --model-path models/best_model.h5 \
    --format tflite \
    --output-dir deployment/models/
```

---

## 🧪 Testing

All scripts are tested with comprehensive unit tests:

```bash
# Run all script tests
pytest tests/test_scripts_phase8.py -v

# Run specific script tests
pytest tests/test_scripts_phase8.py::TestTrainModel -v

# Run with coverage
pytest tests/test_scripts_phase8.py --cov=scripts --cov-report=html
```

**Test Coverage:**
- Input validation (CLI arguments)
- File I/O operations
- Error handling
- Integration with core modules
- Output format verification
- Mock model testing

---

## 📦 Requirements

The scripts require all Phase 7 utilities and the complete project structure:

```
src/
├── data/          # Phase 2 data pipeline
├── models/        # Phase 3 models
├── training/      # Phase 4 training
├── evaluation/    # Phase 5 evaluation
├── prediction/    # Phase 6 prediction
└── utils/         # Phase 7 utilities

scripts/           # Phase 8 (this phase)
tests/             # Test suite
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔧 Script Architecture

Each script follows this pattern:

```python
#!/usr/bin/env python3
"""
Script description.
Usage: python script.py --arg value
"""

import argparse
import logging
from src.utils import setup_logging, get_logger, ConfigManager

def create_parser():
    """Create and return argument parser."""
    parser = argparse.ArgumentParser(
        description="Script description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples: ..."
    )
    # Add arguments
    return parser

def main():
    """Main entry point."""
    args = create_parser().parse_args()
    logger = setup_logging(log_dir="logs", console_level="INFO")
    
    logger.info(f"Starting {args.command}")
    try:
        # Main logic
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
```

---

## 🎯 Best Practices

1. **Always validate inputs** before processing
2. **Use ConfigManager** for configuration
3. **Setup logging** at the start of each script
4. **Handle errors gracefully** with meaningful messages
5. **Create output directories** automatically
6. **Log progress** for long-running operations
7. **Support both single and batch modes** when applicable
8. **Document arguments** with help messages

---

## 🚨 Error Handling

Each script includes:
- Input validation with meaningful error messages
- Try-catch blocks with proper logging
- Exit codes (0=success, 1=error)
- Graceful degradation
- Automatic retry logic for I/O operations

Example:
```python
try:
    model = load_model(args.model_path)
except FileNotFoundError:
    logger.error(f"Model not found: {args.model_path}")
    return 1
```

---

## 🔄 Integration Points

- **Phase 2 (Data)**: Data loading and preprocessing
- **Phase 3 (Models)**: Model loading and training
- **Phase 4 (Training)**: Training execution
- **Phase 5 (Evaluation)**: Metrics and visualization
- **Phase 6 (Prediction)**: Inference pipeline
- **Phase 7 (Utils)**: Config, logging, file operations
- **Phase 8+ (API)**: Scripts are called by API endpoints

---

## 📊 Progress Tracking

Scripts are implemented incrementally:

1. ✅ Phase 8a: Core training script (train_model.py)
2. ✅ Phase 8b: Data preparation (prepare_data.py)
3. ✅ Phase 8c: Evaluation (evaluate.py, predict.py)
4. ✅ Phase 8d: Advanced (cross_validate.py, hyperparameter_search.py)
5. ✅ Phase 8e: Deployment (export_model.py, benchmark.py)
6. ✅ Phase 8f: MATLAB integration (train_with_external_fft.py, compare_fft_methods.py)
7. ✅ Phase 8g: Testing (test_scripts_phase8.py)

---

## 🆘 Troubleshooting

### Script not found
```bash
python -m scripts.train_model --help  # Alternative invocation
```

### Import errors
```bash
export PYTHONPATH="${PYTHONPATH}:/home/user/AirLeakDetection"
```

### Permission denied
```bash
chmod +x scripts/*.py
```

### Missing arguments
Check help message:
```bash
python scripts/train_model.py --help
```

---

## 📝 Logging Output

Each script creates logs in `logs/` directory:

```
logs/
├── train_model_20240101_120000.log
├── predict_20240101_120500.log
├── evaluate_20240101_121000.log
└── ...
```

View logs:
```bash
tail -f logs/train_model_*.log
```

---

**Phase 8 Status**: READY FOR IMPLEMENTATION ✅