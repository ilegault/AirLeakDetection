Repository Overview: AirLeakDetection

General Information
Root Path: /home/viscosity_c/PycharmProjects/AirLeakDetection
Primary Language: Python
Objective: Detect and classify air leaks using vibration data from WebDAQ accelerometer systems.
Pipeline: Data acquisition → Preprocessing (FFT/filtering) → Feature extraction → Model training → Evaluation → Inference.

Repository Structure

air-leak-detection-ml/
│
├── src/ Core source modules
│ ├── data/ Data loading, preprocessing, feature extraction
│ ├── models/ CNN, SVM, Random Forest architectures
│ ├── training/ Training logic and callbacks
│ ├── evaluation/ Metrics and visualization
│ └── utils/ Logging, configuration, and helpers
│
├── data/ Local datasets (gitignored)
│ ├── raw/ Raw WebDAQ CSVs
│ ├── processed/ FFT and time-domain feature arrays
│ └── metadata/ Data splits and normalization
│
├── scripts/ Executable pipeline scripts
│ ├── prepare_data.py
│ ├── train_model.py
│ ├── evaluate.py
│ └── predict.py
│
├── notebooks/ Analysis and visualization (Jupyter)
├── tests/ Automated validation (pytest)
├── models/ Checkpoints and final trained networks
├── results/ Logs, figures, and reports
└── config.yaml Centralized parameters

Data Schema

Source: WebDAQ accelerometer (multi-channel vibration signals)
Sampling Rate: Typically 10,000 Hz
Signal Duration: 10 seconds per sample
Channels: 9 total (3 accelerometers × 3 axes)

Leak Classes:
NOLEAK → baseline condition
SMALL_1_16 → 1/16 inch orifice
MEDIUM_3_32 → 3/32 inch orifice
LARGE_1_8 → 1/8 inch orifice

Configuration Notes

data:
raw_data_path: "data/raw"
processed_data_path: "data/processed"
sample_rate: 10000
duration: 10
n_channels: 9

preprocessing:
fft_size: 2048
window: "hanning"
overlap: 0.5
freq_max: 2000
normalize: true

training:
batch_size: 32
epochs: 100
learning_rate: 0.001
validation_split: 0.15
test_split: 0.15
early_stopping_patience: 10

classes:
0: "NOLEAK"
1: "SMALL_1_16"
2: "MEDIUM_3_32"
3: "LARGE_1_8"

Key Components

Data Loader: src/data/data_loader.py
Function: Parse and structure WebDAQ CSV files by class

Preprocessor: src/data/preprocessor.py
Function: Apply FFT, windowing, normalization

Feature Extractor: src/data/feature_extractor.py
Function: Compute time and frequency domain metrics

Model Factory: src/models/model_factory.py
Function: Build CNN, SVM, or Random Forest based on config

Trainer: src/training/trainer.py
Function: Manage data batching, callbacks, checkpoints

Evaluator: src/evaluation/metrics.py
Function: Compute accuracy, precision, recall, confusion matrix

Inference Engine: scripts/predict.py
Function: Classify unseen signals into leak categories

Workflow Summary

Step 1: Prepare Data
Command: python scripts/prepare_data.py --config config.yaml

Step 2: Train Model
Command: python scripts/train_model.py --config config.yaml

Step 3: Evaluate Model
Command: python scripts/evaluate.py --model models/production/final_model.h5

Step 4: Predict Leak Type
Command: python scripts/predict.py --signal path/to/new/sample.csv

Testing and Validation

Test Framework: pytest
Key Tests: FFT computation, data loader integrity, model reproducibility
Command: pytest tests/

Implementation Status

Data ingestion and preprocessing complete
Feature extraction verified (time and frequency domain)
CNN model trained and validated
Deployment and optimization pending

Zencoder Execution Map

Phase 1: Environment and Setup
Goal: Initialize repository and dependencies.
Files:

requirements.txt

config.yaml
Output: Base folder tree and configuration.

Phase 2: Data Infrastructure
Goal: Implement input data handling.
Files to Generate:

src/data/data_loader.py

src/data/preprocessor.py

src/data/feature_extractor.py
Input: CSV files in data/raw/
Output: .npy feature arrays in data/processed/

Phase 3: Model Development
Goal: Implement ML model architectures.
Files to Generate:

src/models/cnn_1d.py

src/models/random_forest.py

src/models/svm_classifier.py
Output: Model definitions and saved weights in models/checkpoints/

Phase 4: Training Pipeline
Goal: Integrate training logic and callbacks.
Files to Generate:

src/training/trainer.py

scripts/train_model.py
Output: Trained model at models/production/final_model.h5

Phase 5: Evaluation and Inference
Goal: Create evaluation metrics and prediction interface.
Files to Generate:

scripts/evaluate.py

scripts/predict.py
Output:

results/reports/metrics.json

results/figures/confusion_matrix.png

Phase 6: Testing
Goal: Implement basic unit tests.
Files to Generate:

tests/test_data_loader.py

tests/test_preprocessor.py

tests/test_models.py
Validation: Check data shapes, FFT correctness, and model reproducibility.

Zencoder Prompt Reference Block

Phase 1
Zencoder, generate the setup phase (Phase 1) as defined in repo.md. Create requirements.txt and config.yaml using the specifications provided.

Phase 2
Zencoder, implement Phase 2 from repo.md. Generate code for data_loader.py, preprocessor.py, and feature_extractor.py according to the described functionality and configuration.

Phase 3
Zencoder, complete Phase 3 from repo.md. Implement CNN, Random Forest, and SVM models as outlined under src/models/.

Phase 4
Zencoder, complete Phase 4 from repo.md by writing trainer.py and train_model.py that follow the described workflow.

Phase 5
Zencoder, implement Phase 5 from repo.md. Create evaluate.py and predict.py with output paths and metrics logging.

Phase 6
Zencoder, generate Phase 6 from repo.md. Write tests for data, preprocessing, and models to ensure reproducibility and data integrity.