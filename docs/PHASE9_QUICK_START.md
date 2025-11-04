# Phase 9: Jupyter Notebooks & Analysis

## Overview

Phase 9 provides comprehensive Jupyter notebooks for data exploration, FFT method comparison, model experimentation, and results analysis. These notebooks serve as:

- **Interactive documentation** of the system
- **Development sandbox** for prototyping
- **Visualization tools** for insights
- **Analysis pipeline** for results

---

## 📓 Notebooks Overview

### 1. **01_data_exploration.ipynb** - Initial Data Analysis

**Purpose:** Load and explore raw sensor data

**Sections:**
1. **Setup & Imports**
   - Load configuration
   - Setup logging
   - Define paths

2. **Data Loading**
   - Load sample WebDAQ CSV files
   - Display file structure
   - Show data shape and types

3. **Time Series Visualization**
   - Plot raw accelerometer signals
   - Show all 9 channels
   - Highlight patterns per leak class

4. **Statistical Analysis**
   - Descriptive statistics per class
   - Class distribution histograms
   - Signal amplitude analysis

5. **FFT Quick Look**
   - Compute basic FFT
   - Identify dominant frequencies
   - Compare across leak classes

6. **Data Quality Checks**
   - Missing values analysis
   - Outlier detection
   - Sampling rate validation

**Key Outputs:**
- Summary statistics table
- Time series plots (9 subplots)
- Class distribution bar chart
- FFT magnitude plots
- Data quality report

---

### 2. **02_fft_comparison.ipynb** - FFT Method Comparison

**Purpose:** Compare different FFT computation methods

**Sections:**
1. **Setup & Load Data**
   - Load sample raw data
   - Load MATLAB FFT (if available)
   - Define comparison parameters

2. **NumPy FFT**
   - Basic np.fft.fft computation
   - Averaging 3 accelerometers
   - Frequency binning (30-2000 Hz)

3. **SciPy FFT**
   - Using scipy.fft with Hanning window
   - Welch's method (averaged periodogram)
   - Compare with NumPy

4. **MATLAB FFT**
   - Load .mat files from teammate
   - Extract frequency and magnitude
   - Resample to common frequency grid

5. **Method Comparison**
   - Visual side-by-side plots
   - Correlation analysis
   - Mean squared error (MSE)
   - Frequency alignment check

6. **Recommendations**
   - Performance comparison table
   - Method pros/cons
   - Best method selection
   - Consistency analysis

**Key Outputs:**
- Comparison plots (NumPy vs SciPy vs MATLAB)
- Correlation heatmap
- MSE analysis table
- Frequency alignment plots
- Recommendation report

---

### 3. **03_model_experiments.ipynb** - Model Development & Hyperparameter Tuning

**Purpose:** Experiment with different architectures and hyperparameters

**Sections:**
1. **Setup & Data Preparation**
   - Load processed training data
   - Define train/val/test split
   - Data normalization

2. **Architecture Experiments**
   - Build different CNN architectures
   - LSTM variations
   - Compare learning curves

3. **Training Experiments**
   - Different optimizers (Adam, SGD, RMSprop)
   - Learning rate schedules
   - Batch size effects

4. **Regularization Study**
   - Dropout rates
   - L1/L2 penalties
   - Early stopping comparison

5. **Class Imbalance Handling**
   - Focal loss effects
   - Weighted loss comparison
   - Class weights tuning

6. **Visualization**
   - Learning curves
   - Validation metrics over time
   - Hyperparameter sensitivity analysis

**Key Outputs:**
- Training history plots
- Learning curve comparison
- Hyperparameter sensitivity charts
- Best configuration summary
- Model comparison table

---

### 4. **04_results_analysis.ipynb** - Final Results & Production Recommendations

**Purpose:** Analyze best model performance and generate production insights

**Sections:**
1. **Load Best Model**
   - Load trained model from checkpoints
   - Display model architecture
   - Show training metadata

2. **Test Set Evaluation**
   - Compute all metrics (Acc, Precision, Recall, F1)
   - Confusion matrix
   - Per-class performance

3. **ROC & AUC Analysis**
   - ROC curves for all classes
   - AUC scores
   - Operating point selection

4. **Error Analysis**
   - Analyze misclassifications
   - Common confusion patterns
   - Difficult samples identification

5. **Production Readiness**
   - Model size and latency
   - Memory requirements
   - Deployment checklist

6. **Recommendations**
   - Strengths and weaknesses
   - Further improvements
   - Deployment strategy
   - Monitoring suggestions

**Key Outputs:**
- Comprehensive evaluation report
- Confusion matrix heatmap
- ROC curves (all classes)
- Error distribution analysis
- Production checklist
- Deployment recommendations

---

## 🚀 Usage

### Run Individual Notebooks

```bash
# Navigate to notebooks directory
cd notebooks/

# Start Jupyter
jupyter notebook

# Or with lab
jupyter lab
```

### Run All Notebooks (Batch Mode)

```bash
# Convert and run
jupyter nbconvert --to notebook --execute 01_data_exploration.ipynb
jupyter nbconvert --to notebook --execute 02_fft_comparison.ipynb
jupyter nbconvert --to notebook --execute 03_model_experiments.ipynb
jupyter nbconvert --to notebook --execute 04_results_analysis.ipynb

# Export to HTML reports
jupyter nbconvert --to html 01_data_exploration.ipynb
jupyter nbconvert --to html 02_fft_comparison.ipynb
jupyter nbconvert --to html 03_model_experiments.ipynb
jupyter nbconvert --to html 04_results_analysis.ipynb
```

### Recommended Workflow

1. **Start with 01_data_exploration.ipynb**
   - Understand your data
   - Check for quality issues
   - Identify patterns

2. **Compare FFT methods with 02_fft_comparison.ipynb**
   - Validate FFT approach
   - Check MATLAB compatibility
   - Select best method

3. **Experiment with 03_model_experiments.ipynb**
   - Try different architectures
   - Tune hyperparameters
   - Compare performance

4. **Analyze results with 04_results_analysis.ipynb**
   - Evaluate best model
   - Generate reports
   - Plan deployment

---

## 📊 Expected Outputs

### Data Exploration
- Time series plots showing 9 accelerometer channels
- FFT magnitude plots per leak class
- Statistical summary table
- Data quality assessment

### FFT Comparison
- Side-by-side FFT plots (NumPy, SciPy, MATLAB)
- Correlation matrix showing method agreement
- MSE comparison table
- Frequency alignment analysis

### Model Experiments
- Learning curves for different models
- Hyperparameter sensitivity plots
- Performance comparison table
- Best configuration recommendation

### Results Analysis
- Confusion matrix heatmap
- ROC curves for all 4 classes
- Per-class metrics table
- Error distribution analysis
- Production deployment checklist

---

## 🔧 Requirements

### Python Packages
```
jupyter>=1.0
jupyterlab>=3.0
ipywidgets>=7.0
matplotlib>=3.0
seaborn>=0.11
numpy>=1.19
scipy>=1.5
pandas>=1.1
scikit-learn>=0.24
tensorflow>=2.5 (optional, for CNN evaluation)
```

### Data Requirements
- Processed data in `data/processed/`
- Test data with labels
- (Optional) MATLAB FFT files in `data/matlab_fft/`

### Disk Space
- Approximate: 500MB-1GB for notebook outputs
- Additional space for generated plots and reports

---

## 📝 Notebook Features

Each notebook includes:

1. **Clear sections** with descriptive markdown
2. **Code organization** with comments
3. **Error handling** for missing data
4. **Reproducibility** with seeds
5. **Progress indicators** for long operations
6. **Configurable parameters** at the top
7. **Output saving** (plots, tables, reports)
8. **Interactive visualizations** where applicable

---

## 🎯 Best Practices

1. **Run in order** - 01 → 02 → 03 → 04
2. **Save outputs** - Each notebook exports results
3. **Document findings** - Add markdown cells for notes
4. **Use version control** - Commit significant notebooks
5. **Keep data fresh** - Rerun after data updates
6. **Export for sharing** - Use nbconvert to generate HTML/PDF

---

## 🔄 Integration with Pipeline

Notebooks integrate with the main pipeline:

```
scripts/prepare_data.py
    ↓
notebooks/01_data_exploration.ipynb  ← Validate data
    ↓
notebooks/02_fft_comparison.ipynb    ← Select FFT method
    ↓
scripts/train_model.py
    ↓
notebooks/03_model_experiments.ipynb ← Optimize hyperparams
    ↓
scripts/evaluate.py
    ↓
notebooks/04_results_analysis.ipynb  ← Analyze results
```

---

## 📦 File Structure

```
notebooks/
├── 01_data_exploration.ipynb
├── 02_fft_comparison.ipynb
├── 03_model_experiments.ipynb
├── 04_results_analysis.ipynb
└── outputs/
    ├── exploration/
    ├── fft_comparison/
    ├── experiments/
    └── results/
```

---

## 🚨 Troubleshooting

### Notebook won't start
```bash
jupyter notebook --ip=0.0.0.0
```

### Data not found
- Ensure `data/processed/` exists
- Run `scripts/prepare_data.py` first

### Missing MATLAB files
- Optional - notebooks continue without MATLAB data
- Set `skip_matlab=True` in notebook configuration

### Memory issues
- Reduce batch sizes in notebooks
- Process smaller subsets of data
- Use data sampling/downsampling

### Plot rendering issues
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for headless
```

---

## 📞 Support

For issues or questions:
1. Check notebook error messages
2. Review section comments
3. Check data file paths
4. Verify dependencies installed
5. Review logs in `logs/` directory

---

## ✅ Validation Checklist

- [ ] All notebooks run without errors
- [ ] Data loaded successfully
- [ ] FFT comparison shows expected patterns
- [ ] Model experiments show convergence
- [ ] Results analysis produces valid metrics
- [ ] All outputs saved to disk
- [ ] HTML reports generated successfully

---

**Status:** Phase 9 - Jupyter Notebooks (Complete)