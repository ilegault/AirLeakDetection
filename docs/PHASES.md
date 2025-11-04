# Development Phases & TODO Mapping

This document maps all TODO items in the codebase to their respective development phases. Use this to understand what's complete, in-progress, and planned.

## Phase Overview

| Phase | Name | Status | Key Deliverables |
|-------|------|--------|------------------|
| 1 | Project Setup | ✅ Complete | Git repo, directory structure, requirements |
| 2 | Data Pipeline | 🔄 In Progress | Data loading, preprocessing, FFT processing |
| 3 | Model Architectures | ✅ Scaffolded | CNN 1D/2D, LSTM, RF, SVM, XGBoost, Ensemble |
| 4 | Training Pipeline | 🔄 In Progress | Trainer, callbacks, losses, optimizers, CV |
| 5 | Evaluation Suite | 🔄 In Progress | Metrics, visualizer, reports, comparator |
| 6 | Prediction Pipeline | 🔄 In Progress | Predictor, real-time, batch processor, calibrator |
| 7 | Utilities | ✅ Scaffolded | Config, logging, file utils, MATLAB bridge |
| 8 | Scripts | 🔄 In Progress | Training, evaluation, prediction executables |
| 9 | Deployment | 🔄 In Progress | Model export, benchmarking, optimization |

---

## Phase 2: Data Pipeline

**Status**: 🔄 In Progress  
**Location**: `src/data/`  
**Purpose**: Load, preprocess, and convert accelerometer data to FFT features

### Implemented ✅
- `data_loader.py` - WebDAQ CSV loading logic
- `fft_processor.py` - FFT processing framework
- `hybrid_loader.py` - Multi-source data loading
- `preprocessor.py` - Signal preprocessing
- `feature_extractor.py` - Feature extraction
- `data_splitter.py` - Train/val/test splitting
- `augmentor.py` - Data augmentation
- `dataset_generator.py` - TensorFlow/PyTorch datasets
- `validator.py` - Data quality validation
- `cache_manager.py` - Caching processed data

### TODOs (Phase 2)
None currently - Phase 2 is scaffolded but internal implementation depends on data availability.

### What's Needed
- Actual WebDAQ CSV test files to validate loading
- MATLAB `.mat` FFT files for FFT comparison
- Testing with real accelerometer data

---

## Phase 3: Model Architectures

**Status**: ✅ Scaffolded  
**Location**: `src/models/`  
**Purpose**: Define deep learning and ML model architectures

### Implemented ✅
- `cnn_1d.py` - 1D CNN for FFT (3 conv blocks, 9 channels)
- `cnn_2d.py` - 2D CNN for spectrograms
- `lstm_model.py` - LSTM with optional attention
- `random_forest_model.py` - Random Forest baseline
- `svm_model.py` - SVM with RBF kernel
- `xgboost_model.py` - Gradient boosting
- `ensemble_model.py` - Voting/stacking ensembles

### TODOs (Phase 3)
None - all model classes are defined and ready for Phase 4 training integration

### What's Needed
- Integration with Phase 4 Training Pipeline
- Hyperparameter tuning
- Model-specific optimization

---

## Phase 4: Training Pipeline

**Status**: 🔄 In Progress  
**Location**: `src/training/`  
**Scripts**: `scripts/train_model.py`, `scripts/train_with_external_fft.py`  
**Purpose**: Implement the training orchestration and utilities

### Implemented ✅
- `trainer.py` - Base trainer class (scaffolded)
- `callbacks.py` - Training callbacks
- `losses.py` - Custom loss functions
- `optimizers.py` - Optimizer configurations
- `cross_validator.py` - K-fold cross-validation
- `hyperparameter_tuner.py` - Hyperparameter optimization framework

### TODOs (Phase 4)

#### 1. `scripts/train_model.py` - Line 233
```python
# TODO: Implement actual training logic
```
**Task**: Integrate model training with data loading and evaluation  
**Blocking**: Phase 5, Phase 6  
**Script Status**: Argument parsing ✅, Validation ✅, Framework ✅ | Training logic ❌

#### 2. `scripts/train_with_external_fft.py` - Line 140
```python
# TODO: Implement MATLAB FFT training logic
```
**Task**: Load teammate's MATLAB FFT (.mat files) and train models  
**Requires**: Data pipeline Phase 2 complete  
**Script Status**: Argument parsing ✅, Validation ✅ | FFT loading & training ❌

#### 3. `scripts/hyperparameter_search.py` - Line 128
```python
# TODO: Implement hyperparameter search logic
```
**Task**: Integrate hyperparameter tuner with training pipeline  
**Requires**: Training logic from train_model.py  
**Script Status**: Framework ✅ | Search integration ❌

#### 4. `scripts/cross_validate.py` - Line 114
```python
# TODO: Implement cross-validation logic
```
**Task**: Run stratified K-fold CV evaluation  
**Requires**: Trainer and cross-validator integration  
**Script Status**: Framework ✅ | CV logic ❌

### What's Needed
- Load training data (Phase 2 output)
- Instantiate model from Phase 3
- Execute training loop
- Log metrics with MLflow/WandB
- Save checkpoints and best model

---

## Phase 5: Evaluation Suite

**Status**: 🔄 In Progress  
**Location**: `src/evaluation/`  
**Scripts**: `scripts/evaluate.py`, `scripts/compare_fft_methods.py`, `scripts/benchmark.py`  
**Purpose**: Evaluate models, generate reports, and compare approaches

### Implemented ✅
- `metrics.py` - Metrics computation framework
- `visualizer.py` - Result visualization framework
- `report_generator.py` - Report generation framework
- `model_comparator.py` - Model comparison framework
- `error_analyzer.py` - Error analysis framework

### TODOs (Phase 5)

#### 1. `scripts/evaluate.py` - Line 127
```python
# TODO: Implement actual evaluation logic
```
**Task**: Load test data, run inference, compute metrics  
**Requires**: Trained model from Phase 4, Phase 2 test data  
**Script Status**: Framework ✅, Metrics loading ✅ | Evaluation ❌

#### 2. `scripts/compare_fft_methods.py` - Line 144
```python
# TODO: Implement FFT comparison logic
```
**Task**: Compare MATLAB vs NumPy vs SciPy FFT on same data  
**Requires**: Phase 2 FFT processor complete  
**Script Status**: Framework ✅ | Comparison logic ❌

#### 3. `scripts/benchmark.py` - Line 133
```python
# TODO: Implement benchmarking logic
```
**Task**: Measure inference speed and memory usage  
**Requires**: Trained models from Phase 4  
**Script Status**: Framework ✅ | Benchmarking ❌

### What's Needed
- Load test set from Phase 2
- Run model inference
- Compute accuracy, precision, recall, F1, confusion matrix
- Generate plots and reports
- Compare FFT methods
- Benchmark inference performance

---

## Phase 6: Prediction Pipeline

**Status**: 🔄 In Progress  
**Location**: `src/prediction/`  
**Scripts**: `scripts/predict.py`  
**Purpose**: Production inference and deployment preparation

### Implemented ✅
- `predictor.py` - Main inference class (scaffolded)
- `real_time_predictor.py` - Streaming inference framework
- `batch_processor.py` - Batch processing framework
- `confidence_calibrator.py` - Confidence calibration framework

### TODOs (Phase 6)

#### 1. `scripts/predict.py` - Line 139
```python
# TODO: Implement actual prediction logic
```
**Task**: Load model and run inference on new data  
**Requires**: Phase 4 trained model, Phase 2 preprocessor  
**Script Status**: Argument parsing ✅, Validation ✅ | Inference ❌

### What's Needed
- Load trained model and preprocessor
- Read input CSV file or directory
- Preprocess data (same as training pipeline)
- Run inference
- Output predictions with confidence scores
- Support batch and single-file modes

---

## Phase 7: Utilities & Configuration

**Status**: ✅ Mostly Complete  
**Location**: `src/utils/`  
**Purpose**: Supporting utilities for all phases

### Implemented ✅
- `config_manager.py` - YAML config loading
- `logger.py` - Logging setup
- `file_utils.py` - File operations
- `matlab_bridge.py` - MATLAB .mat file handling
- `visualization_utils.py` - Plotting utilities
- `math_utils.py` - Signal processing helpers
- `reproducibility.py` - Seed management

### TODOs (Phase 7)
None - Phase 7 is complete

### Status
✅ Ready for use by all phases

---

## Phase 8: Executable Scripts

**Status**: 🔄 In Progress  
**Location**: `scripts/`  
**Purpose**: User-facing scripts for common operations

### Overview
All scripts are scaffolded with:
- ✅ Argument parsing
- ✅ Input validation
- ✅ Logging framework
- ✅ Error handling
- ❌ Core logic (marked TODO)

### Script Status Summary

| Script | Status | Phase | TODO Line | Core Task |
|--------|--------|-------|-----------|-----------|
| `train_model.py` | 🔄 | Phase 4 | 233 | Train any model type |
| `train_with_external_fft.py` | 🔄 | Phase 4 | 140 | Train with MATLAB FFT |
| `predict.py` | 🔄 | Phase 6 | 139 | Run inference |
| `evaluate.py` | 🔄 | Phase 5 | 127 | Evaluate model |
| `compare_fft_methods.py` | 🔄 | Phase 5 | 144 | Compare FFT approaches |
| `benchmark.py` | 🔄 | Phase 5 | 133 | Benchmark performance |
| `prepare_data.py` | 🔄 | Phase 2 | 140 | Prepare dataset |
| `hyperparameter_search.py` | 🔄 | Phase 4 | 128 | Hyperparameter tuning |
| `cross_validate.py` | 🔄 | Phase 4 | 114 | Cross-validation |
| `export_model.py` | 🔄 | Phase 9 | 114 | Export for deployment |

### What's Needed
Implement core logic for each script (see TODO mappings above)

---

## Phase 9: Deployment & Export

**Status**: 🔄 In Progress  
**Location**: `scripts/export_model.py`  
**Purpose**: Prepare models for production deployment

### TODOs (Phase 9)

#### 1. `scripts/export_model.py` - Line 114
```python
# TODO: Implement model export logic
```
**Task**: Export trained models to TFLite, ONNX, or TorchScript  
**Requires**: Trained model from Phase 4  
**Format Support**:
- TensorFlow Lite (.tflite)
- ONNX (.onnx)
- TorchScript (.pt)

### What's Needed
- Load trained model
- Convert to target format
- Save exported model
- Generate deployment documentation
- Test model compatibility

---

## TODO Summary by Phase

### Quick Reference: All TODOs

| Phase | File | Line | Task |
|-------|------|------|------|
| Phase 2 | `prepare_data.py` | 140 | Implement data processing |
| Phase 4 | `train_model.py` | 233 | Implement training logic |
| Phase 4 | `train_with_external_fft.py` | 140 | Implement MATLAB FFT training |
| Phase 4 | `hyperparameter_search.py` | 128 | Implement hyperparameter search |
| Phase 4 | `cross_validate.py` | 114 | Implement cross-validation |
| Phase 5 | `evaluate.py` | 127 | Implement evaluation logic |
| Phase 5 | `compare_fft_methods.py` | 144 | Implement FFT comparison |
| Phase 5 | `benchmark.py` | 133 | Implement benchmarking |
| Phase 6 | `predict.py` | 139 | Implement prediction logic |
| Phase 9 | `export_model.py` | 114 | Implement model export |

### Total: 10 TODO items across 5 active phases

---

## Recommended Implementation Order

### Priority 1: Foundation (Required for everything)
1. **Phase 2 - Data**: Load and preprocess data
2. **Phase 3 - Models**: Models already defined ✅

### Priority 2: Training & Evaluation (Core ML)
3. **Phase 4 - Training**: Implement model training
4. **Phase 5 - Evaluation**: Implement evaluation metrics

### Priority 3: Prediction & Export (Production)
5. **Phase 6 - Prediction**: Implement inference
6. **Phase 9 - Export**: Implement model export

### Optional: Utilities & Advanced
7. **Phase 7 - Utilities**: Mostly complete ✅
8. **Phase 8 - Scripts**: Depends on Phases 2-6

---

## Development Workflow

### Before Starting Phase N:
1. Check this file for phase requirements
2. Review `/docs/PHASE{N}_IMPLEMENTATION.md` if available
3. Understand blocking dependencies

### While Working on Phase N:
1. Create branch: `phase-{N}-{feature-name}`
2. Include phase number in commit messages: `[Phase N] Description`
3. When implementing TODO, replace it with actual code
4. Update this file and mark phase complete

### After Completing Phase N:
1. Run tests: `pytest tests/`
2. Update status in this file (✅ Complete)
3. Create pull request with phase number in title
4. Update README.md if needed
5. Mark next phase as ready in this file

---

## Notes for Collaborators

### About TODOs
- **They are NOT bugs** - they are intentional development markers
- Scripts run and provide framework, but indicate "logic to be implemented"
- This allows pushing to GitHub before features are complete
- Each TODO clearly indicates which phase implements it

### Dependencies
- Phases must be completed in order (Phase 2 → Phase 4 → Phase 5/6)
- Phase 3 (models) is largely independent
- Phase 7 (utils) is mostly complete and can be used anytime
- Phase 8 (scripts) depends on all other phases

### Testing Strategy
- Each phase should have corresponding tests in `tests/`
- Run `pytest` before committing
- Use test fixtures from `tests/conftest.py`

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete, ready to use |
| 🔄 | In progress, scaffolded with TODOs |
| ❌ | Not started |
| 🔗 | Blocked by another phase |

---

## Quick Links

- **Repository Structure**: See `README.md`
- **Project Overview**: See `.zencoder/rules/repo.md`
- **Phase Docs**: Check `/docs/PHASE{N}_*.md` files
- **Run Tests**: `pytest tests/`
- **View Logs**: `logs/` directory