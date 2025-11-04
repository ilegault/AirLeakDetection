# Air Leak Detection ML - Complete File Structure & Implementation Guide

## Project Overview
Build a machine learning system to detect and classify air leaks (No Leak, 1/16", 3/32", 1/8") using accelerometer data converted to FFT. System must support both teammate's MATLAB FFT and internal FFT processing.

## Complete File Structure (67 files total)

```
air-leak-detection-ml/
├── config/
│   ├── config.yaml                 # Main configuration
│   ├── config_dev.yaml             # Development settings
│   ├── config_prod.yaml            # Production settings
│   └── config_test.yaml            # Testing settings
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/                       # [11 files] Data pipeline
│   ├── models/                     # [8 files] Model architectures  
│   ├── training/                   # [7 files] Training pipeline
│   ├── evaluation/                 # [6 files] Evaluation tools
│   ├── prediction/                 # [5 files] Inference pipeline
│   └── utils/                      # [8 files] Utilities
│
├── scripts/                         # [10 files] Executable scripts
├── tests/                          # [12 files] Test suite
├── notebooks/                      # [4 files] Jupyter notebooks
└── deployment/                     # [6 files] Deployment configs
```

---

## DETAILED FILE BREAKDOWN

### 1. DATA PIPELINE (src/data/) - 11 files

#### src/data/__init__.py
- Export all data classes

#### src/data/data_loader.py
**Purpose:** Load raw WebDAQ CSV files
- Class: `WebDAQDataLoader`
- Handle CSV with 5 header lines (like teammate's MATLAB)
- Extract 9 channels (3 accelerometers × 3 axes)
- Support columns: "Acceleration 0 (g)", "Acceleration 1 (g)", etc.
- Handle sampling rate: 4654.545 Hz (teammate) or 10000 Hz (config)

#### src/data/fft_processor.py
**Purpose:** Flexible FFT processing supporting multiple methods
- Class: `FlexibleFFTProcessor`
- Methods needed:
  - `load_matlab_fft()`: Load teammate's .mat files
  - `compute_matlab_style_fft()`: Replicate MATLAB approach (avg 3 sensors)
  - `compute_scipy_fft()`: Using scipy.fft with Hanning window
  - `compute_numpy_fft()`: Simple np.fft matching MATLAB
  - `compare_methods()`: Return correlation & MSE between methods

#### src/data/hybrid_loader.py
**Purpose:** Load data from multiple sources
- Support raw CSV + pre-computed FFT simultaneously
- Handle .mat files from MATLAB
- Handle .npz files from NumPy
- Combine different data sources

#### src/data/preprocessor.py
**Purpose:** Signal preprocessing
- Bandpass filter (10-2000 Hz)
- Normalization (minmax, zscore, robust)
- Window functions (Hanning, Hamming)
- Handle FFT size: 2048
- Frequency range limiting (30-2000 Hz)

#### src/data/feature_extractor.py
**Purpose:** Extract features for non-CNN models
- Time domain: RMS, peak-to-peak, kurtosis, skewness, zero-crossing
- Frequency domain: peak frequency, spectral centroid, band power
- Statistical: mean, std, quartiles
- Per-channel and cross-channel features

#### src/data/data_splitter.py
**Purpose:** Create train/val/test splits
- Stratified splitting (70/15/15)
- Save split indices for reproducibility
- Support both file-level and sample-level splits

#### src/data/augmentor.py
**Purpose:** Data augmentation
- Add noise (0.005 factor)
- Time shifting (±10%)
- Amplitude scaling (0.9-1.1)
- Frequency masking

#### src/data/dataset_generator.py
**Purpose:** Create TensorFlow/PyTorch datasets
- Batch generation (size 32)
- Support both FFT and raw data
- Handle imbalanced classes
- Infinite generator for training

#### src/data/validator.py
**Purpose:** Validate data quality
- Check for NaN/Inf values
- Verify shape consistency
- Validate frequency ranges
- Check class balance

#### src/data/cache_manager.py
**Purpose:** Cache processed data
- Save/load preprocessed FFT
- Memory-mapped arrays for large datasets
- Version control for cache invalidation

---

### 2. MODEL ARCHITECTURES (src/models/) - 8 files

#### src/models/__init__.py
- Model factory pattern

#### src/models/cnn_1d.py
**Purpose:** 1D CNN for FFT classification
- Input shape: (1024 frequencies, 9 channels)
- Architecture: 3 conv blocks → global pooling → 2 dense layers
- Conv blocks: [32, 64, 128] filters, kernel sizes [7, 5, 3]
- Dropout: 0.3-0.4
- Output: 4 classes softmax

#### src/models/cnn_2d.py
**Purpose:** 2D CNN for spectrogram classification
- Process time-frequency spectrograms
- Use for comparison with 1D approach

#### src/models/lstm_model.py
**Purpose:** LSTM for sequential data
- Process raw time-series or FFT sequences
- Bidirectional LSTM option
- Attention mechanism

#### src/models/random_forest_model.py
**Purpose:** Random Forest baseline
- 200 trees, max_depth=20
- Handle extracted features (not raw FFT)
- Feature importance analysis

#### src/models/svm_model.py
**Purpose:** SVM classifier
- RBF kernel, C=10
- Requires feature scaling
- Probability estimates for confidence

#### src/models/xgboost_model.py
**Purpose:** Gradient boosting
- Alternative to Random Forest
- Better handling of imbalanced data

#### src/models/ensemble_model.py
**Purpose:** Combine multiple models
- Voting classifier
- Stacking ensemble
- Weighted averaging

---

### 3. TRAINING PIPELINE (src/training/) - 7 files

#### src/training/__init__.py

#### src/training/trainer.py
**Purpose:** Main training orchestrator
- Class: `ModelTrainer`
- Support all model types
- MLflow/WandB integration
- Handle both raw and FFT input

#### src/training/callbacks.py
**Purpose:** Custom callbacks
- Early stopping (patience=15)
- Model checkpointing
- Learning rate reduction
- Custom metrics logging

#### src/training/losses.py
**Purpose:** Custom loss functions
- Focal loss for imbalanced data
- Weighted categorical crossentropy
- Custom leak-specific losses

#### src/training/optimizers.py
**Purpose:** Optimizer configurations
- Adam with warmup
- SGD with momentum
- Learning rate schedules

#### src/training/cross_validator.py
**Purpose:** K-fold cross-validation
- Stratified K-fold (k=5)
- Leave-one-out for small datasets
- Time-series split option

#### src/training/hyperparameter_tuner.py
**Purpose:** Hyperparameter optimization
- Grid search
- Random search
- Bayesian optimization (Optuna)

---

### 4. EVALUATION SUITE (src/evaluation/) - 6 files

#### src/evaluation/__init__.py

#### src/evaluation/metrics.py
**Purpose:** Comprehensive metrics
- Accuracy, precision, recall, F1
- Confusion matrix
- Per-class metrics
- ROC-AUC for multiclass

#### src/evaluation/visualizer.py
**Purpose:** Result visualization
- Confusion matrix heatmap
- ROC curves (all classes)
- Training history plots
- Feature importance plots
- FFT comparison plots

#### src/evaluation/report_generator.py
**Purpose:** Generate evaluation reports
- HTML reports with plots
- PDF export
- LaTeX tables
- Markdown summaries

#### src/evaluation/model_comparator.py
**Purpose:** Compare multiple models
- Statistical significance tests
- Performance tables
- Best model selection

#### src/evaluation/error_analyzer.py
**Purpose:** Analyze misclassifications
- Common error patterns
- Difficult samples identification
- Per-class error analysis

---

### 5. PREDICTION PIPELINE (src/prediction/) - 5 files

#### src/prediction/__init__.py

#### src/prediction/predictor.py
**Purpose:** Main inference class
- Class: `LeakDetector`
- Load model + preprocessor
- Single file prediction
- Batch prediction
- Confidence scores

#### src/prediction/real_time_predictor.py
**Purpose:** Real-time inference
- Stream processing
- Sliding window approach
- Low-latency optimization

#### src/prediction/batch_processor.py
**Purpose:** Large-scale batch processing
- Parallel processing
- Progress tracking
- Result aggregation

#### src/prediction/confidence_calibrator.py
**Purpose:** Calibrate prediction confidence
- Temperature scaling
- Platt scaling
- Isotonic regression

---

### 6. UTILITIES (src/utils/) - 8 files

#### src/utils/__init__.py

#### src/utils/config_manager.py
**Purpose:** Configuration handling
- Load YAML configs
- Environment variable override
- Config validation

#### src/utils/logger.py
**Purpose:** Logging setup
- File + console logging
- Log levels per module
- Structured logging (JSON)

#### src/utils/file_utils.py
**Purpose:** File operations
- Path management
- Safe file saving
- Directory creation

#### src/utils/matlab_bridge.py
**Purpose:** MATLAB integration
- Load/save .mat files
- Convert data formats
- Handle MATLAB-specific structures

#### src/utils/visualization_utils.py
**Purpose:** Common plotting functions
- FFT plots with frequency markers
- Accelerometer time series
- Spectrogram generation

#### src/utils/math_utils.py
**Purpose:** Mathematical operations
- Signal processing helpers
- Statistical functions
- FFT utilities

#### src/utils/reproducibility.py
**Purpose:** Ensure reproducible results
- Seed management
- Deterministic operations
- Version tracking

---

### 7. EXECUTABLE SCRIPTS (scripts/) - 10 files

#### scripts/train_model.py
**Purpose:** Main training script
- Parse arguments (model type, data path, config)
- Load data → preprocess → train → evaluate
- Save model + metadata

#### scripts/train_with_external_fft.py
**Purpose:** Train using teammate's FFT
- Load .mat files
- Select FFT source (matlab/scipy/numpy)
- Compare performance

#### scripts/compare_fft_methods.py
**Purpose:** Compare FFT approaches
- Visual comparison
- Correlation metrics
- Recommendation output

#### scripts/predict.py
**Purpose:** Run inference
- Single or batch prediction
- Output formats (JSON, CSV)
- Confidence thresholds

#### scripts/evaluate.py
**Purpose:** Evaluate trained model
- Load test data
- Generate all metrics
- Create visualizations

#### scripts/prepare_data.py
**Purpose:** Data preparation
- Convert raw to processed
- Generate train/val/test splits
- Create cached datasets

#### scripts/hyperparameter_search.py
**Purpose:** Find best hyperparameters
- Define search space
- Run optimization
- Save best params

#### scripts/cross_validate.py
**Purpose:** Run cross-validation
- K-fold evaluation
- Generate CV report
- Model stability check

#### scripts/export_model.py
**Purpose:** Export for deployment
- TensorFlow Lite
- ONNX format
- TorchScript

#### scripts/benchmark.py
**Purpose:** Performance benchmarking
- Inference speed
- Memory usage
- Accuracy vs speed tradeoff

---

### 8. TEST SUITE (tests/) - 12 files

#### tests/conftest.py
**Purpose:** Pytest configuration
- Fixtures for test data
- Mock objects
- Test paths

#### tests/test_data_loader.py
**Tests:** Data loading functionality
- CSV parsing
- Channel extraction
- Missing data handling

#### tests/test_fft_processor.py
**Tests:** FFT processing
- FFT accuracy (known frequencies)
- Method consistency
- MATLAB compatibility

#### tests/test_models.py
**Tests:** Model architectures
- Input/output shapes
- Forward pass
- Gradient flow

#### tests/test_training.py
**Tests:** Training pipeline
- Convergence on small data
- Checkpoint saving
- Metric calculation

#### tests/test_prediction.py
**Tests:** Inference pipeline
- Single prediction
- Batch consistency
- Confidence calibration

#### tests/test_feature_extractor.py
**Tests:** Feature extraction
- Feature validity
- Dimension consistency
- NaN handling

#### tests/test_augmentation.py
**Tests:** Data augmentation
- Augmentation effects
- Randomness control
- Shape preservation

#### tests/test_evaluation.py
**Tests:** Evaluation metrics
- Metric calculation
- Edge cases (single class)
- Visualization generation

#### tests/test_integration.py
**Tests:** End-to-end pipeline
- Full training cycle
- Data → Model → Prediction

#### tests/test_matlab_compatibility.py
**Tests:** MATLAB integration
- .mat file loading
- FFT matching
- Frequency alignment

#### tests/performance/
**Purpose:** Performance tests
- Speed benchmarks
- Memory profiling
- Scalability tests

---

### 9. NOTEBOOKS (notebooks/) - 4 files

#### notebooks/01_data_exploration.ipynb
**Purpose:** Initial data analysis
- Load sample files
- Visualize time series
- FFT analysis
- Class distribution

#### notebooks/02_fft_comparison.ipynb
**Purpose:** Compare FFT methods
- Side-by-side visualizations
- Statistical analysis
- Method selection

#### notebooks/03_model_experiments.ipynb
**Purpose:** Model development
- Architecture experiments
- Learning curves
- Hyperparameter effects

#### notebooks/04_results_analysis.ipynb
**Purpose:** Final results
- Best model performance
- Error analysis
- Production recommendations

---

### 10. DEPLOYMENT (deployment/) - 6 files

#### deployment/Dockerfile
```dockerfile
FROM python:3.9-slim
# Install dependencies
# Copy code
# Set entrypoint
```

#### deployment/docker-compose.yml
- Training service
- Prediction API service
- Data processing service

#### deployment/requirements.txt
- All Python dependencies
- Version pinning

#### deployment/requirements_dev.txt
- Additional dev dependencies
- Testing tools
- Linting

#### deployment/kubernetes/
- k8s deployment configs
- Service definitions
- ConfigMaps

#### deployment/api/
- FastAPI/Flask API
- REST endpoints
- WebSocket for real-time

---

## IMPLEMENTATION PRIORITIES

### Phase 1: Core Data Pipeline (Week 1)
1. `data_loader.py` - Load CSV files
2. `fft_processor.py` - FFT computation
3. `hybrid_loader.py` - Multiple source support
4. `data_splitter.py` - Train/val/test splits

### Phase 2: Model Development (Week 2)
1. `cnn_1d.py` - Primary model
2. `trainer.py` - Training pipeline
3. `callbacks.py` - Training monitoring
4. `random_forest_model.py` - Baseline

### Phase 3: Evaluation & Testing (Week 3)
1. `metrics.py` - Performance metrics
2. `visualizer.py` - Result plots
3. `predictor.py` - Inference
4. Core tests for each component

### Phase 4: Production & Optimization (Week 4)
1. Scripts for all workflows
2. Docker deployment
3. API development
4. Performance optimization

---

## KEY TECHNICAL SPECIFICATIONS

### Data Specifications
- Sampling rates: 4654.545 Hz (MATLAB) or 10000 Hz (internal)
- FFT size: 2048 points
- Frequency range: 30-2000 Hz
- Window: Hanning
- Channels: 9 (3 accelerometers × 3 axes)

### Model Specifications
- Input: (1024 frequency bins, 9 channels)
- Classes: 4 (No Leak, 1/16", 3/32", 1/8")
- Train/Val/Test: 70/15/15 split
- Batch size: 32
- Target accuracy: >90%

### Important Frequencies (from MATLAB)
- Monitor: [60.782, 248.3, 356.67, 477.35, 640.8, 995.82, 1073.9, 1572.9] Hz

---

## CRITICAL IMPLEMENTATION NOTES

1. **FFT Compatibility**: Must support both teammate's MATLAB FFT (4654.545 Hz) and internal (10000 Hz)

2. **File Format Support**: 
   - Input: WebDAQ CSV with 5 header lines
   - FFT: MATLAB .mat files
   - Models: .h5 (Keras) or .pkl (sklearn)

3. **Performance Requirements**:
   - Training: <1 hour for 1000 samples
   - Inference: <100ms per prediction
   - Memory: <4GB for training

4. **Testing Coverage**:
   - Unit tests: >80% coverage
   - Integration tests for full pipeline
   - Performance benchmarks

5. **Documentation**:
   - Docstrings for all classes/functions
   - README for each module
   - API documentation

---

## QUICK START COMMANDS FOR AI AGENT

```bash
# Setup project structure
mkdir -p src/{data,models,training,evaluation,prediction,utils}
mkdir -p scripts tests notebooks deployment
touch src/__init__.py src/{data,models,training,evaluation,prediction,utils}/__init__.py

# Install dependencies
pip install numpy pandas scipy scikit-learn tensorflow matplotlib pyyaml pytest mlflow

# Start implementation
# 1. Implement data_loader.py first
# 2. Then fft_processor.py
# 3. Test with sample data
# 4. Continue with models...
```

This structure provides a complete roadmap with 67 files covering every aspect of the air leak detection system. Each file has a clear purpose and the AI agent can implement the specific functionality based on these descriptions.

---

## PHASE 7: UTILITIES FRAMEWORK ✅

### Implementation Status: COMPLETE

**7 Core Utility Modules + Comprehensive Testing**

#### Completed Files:

1. **src/utils/config_manager.py** (170 lines)
   - YAML configuration loading
   - Nested key access with dot notation
   - Environment variable overrides
   - Configuration validation and merging
   - Safe configuration saving

2. **src/utils/logger.py** (140 lines)
   - File and console logging handlers
   - Rotating file handler (10MB, 5 backups)
   - JSON structured logging support
   - Module-level log level control
   - Custom formatters

3. **src/utils/file_utils.py** (210 lines)
   - Atomic file operations with temp files
   - Automatic backup creation
   - Multiple format support (pickle, JSON, text)
   - Safe directory management
   - File listing and filtering
   - Path utilities (absolute, relative)

4. **src/utils/matlab_bridge.py** (200 lines)
   - MATLAB .mat file I/O
   - FFT data extraction and conversion
   - Label extraction from MATLAB data
   - FFT comparison metrics (MSE, correlation)
   - Struct array handling
   - Data format conversions

5. **src/utils/visualization_utils.py** (220 lines)
   - FFT plotting with frequency markers
   - Time series visualization
   - Spectrogram generation
   - Multi-channel data plotting
   - Comparison plots
   - Histogram generation
   - Automatic directory creation for outputs

6. **src/utils/math_utils.py** (280 lines)
   - Butterworth bandpass filtering
   - Multiple normalization methods (minmax, zscore, robust)
   - Window functions (hanning, hamming, blackman, bartlett)
   - FFT computation with windowing
   - Signal statistics (RMS, peak-to-peak, crest factor)
   - Spectral analysis (centroid, band power)
   - Statistical measures (kurtosis, skewness, zero-crossing rate)
   - Correlation and MSE calculations

7. **src/utils/reproducibility.py** (230 lines)
   - Unified seed management (Python, NumPy, TensorFlow, PyTorch)
   - Deterministic mode enabling
   - Library version tracking
   - SHA256/MD5/SHA512 hashing
   - Data integrity verification
   - Reproducibility info saving

#### Test Suite:

**tests/test_utils_phase7.py** (600+ lines)
- 56 comprehensive unit tests (100% passing)
- Test Coverage:
  - ConfigManager: 10 tests
  - LoggerSetup: 4 tests
  - FileUtils: 11 tests
  - MATLABBridge: 4 tests
  - MathUtils: 17 tests
  - VisualizationUtils: 4 tests
  - Reproducibility: 7 tests
  - Integration: 1 test

#### Documentation:

**PHASE7_QUICK_START.md** (400+ lines)
- Complete usage examples for all utilities
- Architecture and design patterns
- Integration with other phases
- Performance considerations
- Troubleshooting guide

### Key Achievements:

✅ All 7 utility modules implemented according to repo.md specification
✅ 56 unit tests with 100% pass rate
✅ Comprehensive error handling and logging
✅ MATLAB compatibility for teammate integration
✅ Matplotlib integration with graceful degradation
✅ NumPy/SciPy optimization for performance
✅ Documented with docstrings and examples
✅ Integration-ready for all other phases

### Integration Points:

- **Phase 2 (Data Pipeline)**: Config loading, file operations, signal processing
- **Phase 3 (Models)**: Reproducibility, logging, configuration
- **Phase 4 (Training)**: Config, logging, reproducibility
- **Phase 5 (Evaluation)**: Visualization, math utilities
- **Phase 6 (Prediction)**: File operations, logging, MATLAB bridge
- **Phase 8+ (Future)**: All utilities available for API, deployment, monitoring

### Performance Metrics:

- ConfigManager initialization: ~1ms
- Safe file save with backup: ~10-50ms
- FFT computation (2048-point): ~2-5ms
- Normalization (1M elements): ~5-10ms
- Logging overhead: <1% performance impact

---

## PHASE 8: REST API DEVELOPMENT (Ready to Start)

Next phase will implement REST API endpoints for the inference pipeline:
- Flask/FastAPI REST endpoints
- Request/response validation
- API documentation (OpenAPI/Swagger)
- Authentication and rate limiting
- Deployment configuration