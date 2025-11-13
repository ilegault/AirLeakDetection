# Model Benchmarking Guide

This guide explains how to benchmark your accelerometer classifier and two-stage classifier models.

## Quick Start

### 1. Comprehensive Benchmarking (Recommended)

This script automatically trains models (if needed) and benchmarks both inference speed and accuracy:

```bash
python scripts/benchmark_models.py \
    --accel-data data/accelerometer_classifier_v2/ \
    --hole-size-data data/processed/ \
    --output-dir results/benchmarks/ \
    --n-iterations 100 \
    --batch-sizes 1,8,32,64
```

**What it does:**
- ✅ Trains accelerometer classifier
- ✅ Trains two-stage classifier components
- ✅ Measures inference latency & throughput for different batch sizes
- ✅ Computes accuracy, precision, recall, F1 scores
- ✅ Generates confusion matrices
- ✅ Creates comparison visualizations (PNG plots)
- ✅ Outputs detailed JSON reports

**Output files:**
- `benchmark_accelerometer_classifier.json` - Detailed results
- `benchmark_two_stage_classifier_v2.json` - Detailed results
- `benchmark_comparison.json` - Side-by-side comparison
- `benchmark_comparison_*.png` - 4-plot visualization

---

### 2. Quick Evaluation (Faster)

For a faster evaluation with 50 iterations instead of 100:

```bash
python scripts/benchmark_models.py \
    --accel-data data/accelerometer_classifier_v2/ \
    --hole-size-data data/processed/ \
    --output-dir results/benchmarks/ \
    --quick
```

---

### 3. Detailed Model Evaluation

For comprehensive per-class metrics and error analysis:

```bash
# First train/obtain your models
python scripts/evaluate_models_detailed.py \
    --accel-model models/accelerometer_classifier/model_*/random_forest_accelerometer.pkl \
    --accel-data data/accelerometer_classifier_v2/ \
    --hole-size-data data/processed/ \
    --output-dir results/evaluations/ \
    --n-iterations 100
```

**What it provides:**
- Per-class accuracy, precision, recall, F1
- Confusion matrix analysis
- Misclassification statistics
- Latency percentiles (p50, p95, p99)
- Throughput measurements
- Error pattern analysis

**Output files:**
- `eval_accelerometer_classifier_*.json` - Detailed accuracy metrics
- `evaluation_summary_*.json` - Executive summary
- Misclassification breakdown

---

## Benchmark Metrics Explained

### Inference Performance

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Mean Latency** | Average inference time in ms | Overall performance |
| **P95 Latency** | 95th percentile response time | Performance under load |
| **P99 Latency** | 99th percentile response time | Worst-case scenarios |
| **Throughput** | Samples per second | Batch processing capacity |

### Accuracy Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Precision** | TP / (TP + FP) | False positive rate |
| **Recall** | TP / (TP + FN) | False negative rate (coverage) |
| **F1 Score** | 2 × (P × R) / (P + R) | Balanced measure |

---

## Command-Line Options

### `benchmark_models.py`

```
--accel-data PATH              Data directory for accelerometer classifier
--hole-size-data PATH          Data directory for hole size classification
--output-dir PATH              Output directory (default: results/benchmarks/)
--model-type {rf,svm}          Model type (default: random_forest)
--n-iterations INT             Benchmark iterations (default: 100)
--batch-sizes STR              Comma-separated batch sizes (default: 1,8,32,64)
--benchmark-accel-only         Only benchmark accelerometer classifier
--benchmark-two-stage-only     Only benchmark two-stage classifier
--verbose                      Detailed logging
```

### `evaluate_models_detailed.py`

```
--accel-model PATH             Path to trained model
--hole-size-models PATTERN     Optional: hole size model pattern
--accel-data PATH              Accelerometer data directory
--hole-size-data PATH          Optional: hole size data directory
--output-dir PATH              Output directory
--n-iterations INT             Iterations for inference benchmark
--batch-sizes STR              Batch sizes to test
--quick                        Use 50 iterations instead of 100
--verbose                      Detailed logging
```

---

## Example Workflow

### 1. Train and Benchmark

```bash
# Full benchmark cycle
python scripts/benchmark_models.py \
    --accel-data data/accelerometer_classifier_v2/ \
    --hole-size-data data/processed/ \
    --output-dir results/benchmarks_v1/ \
    --n-iterations 200 \
    --batch-sizes 1,2,4,8,16,32,64
```

### 2. Compare With Different Model Type

```bash
# Compare SVM vs Random Forest
python scripts/benchmark_models.py \
    --accel-data data/accelerometer_classifier_v2/ \
    --hole-size-data data/processed/ \
    --output-dir results/benchmarks_svm/ \
    --model-type svm \
    --n-iterations 100
```

### 3. Detailed Analysis

```bash
# Get detailed error analysis
python scripts/evaluate_models_detailed.py \
    --accel-model results/benchmarks_v1/model_*/random_forest_accelerometer.pkl \
    --accel-data data/accelerometer_classifier_v2/ \
    --output-dir results/analysis/ \
    --n-iterations 100 \
    --verbose
```

---

## Understanding Results

### Visualization (PNG Output)

The generated comparison plot shows 4 subplots:

1. **Top-Left: Inference Time vs Batch Size**
   - Lower is better
   - Should be roughly linear with batch size

2. **Top-Right: Throughput vs Batch Size**
   - Higher is better
   - Shows batching efficiency

3. **Bottom-Left: Accuracy Comparison**
   - Taller bars are better
   - Should be close to 1.0 for good models

4. **Bottom-Right: F1 Score Comparison**
   - Taller bars are better
   - Balances precision and recall

### JSON Report Structure

```json
{
  "model": "accelerometer_classifier",
  "n_test_samples": 150,
  "inference": {
    "results_by_batch": [
      {
        "batch_size": 1,
        "mean_time_ms": 5.2,
        "throughput_samples_per_sec": 192.3,
        ...
      }
    ]
  },
  "accuracy": {
    "accuracy": 0.95,
    "f1_score": 0.94,
    "confusion_matrix": [...]
  }
}
```

---

## Tips for Benchmarking

### 1. Warm-up Runs
- Both scripts include warmup iterations before measuring
- This gives the model time to load and optimize

### 2. Batch Size Selection
- Start with `--batch-sizes 1,4,8,16,32`
- Adjust based on your hardware
- Larger batches usually show throughput benefits

### 3. Iteration Count
- Use `--quick` flag for initial exploration (50 iterations)
- Use `--n-iterations 100` for standard benchmarking
- Use `--n-iterations 200+` for production releases

### 4. Multiple Runs
- Run benchmarks multiple times for statistical significance
- Output directories are timestamped, so results won't overwrite
- Compare results across runs with `benchmark_comparison.json`

---

## Performance Interpretation

### Good vs. Poor Performance

**Accelerometer Classifier (3-class problem):**
- ✅ Good: Accuracy > 90%, Latency < 5ms
- ⚠️ Okay: Accuracy 80-90%, Latency 5-10ms
- ❌ Poor: Accuracy < 80%, Latency > 10ms

**Two-Stage Classifier:**
- ✅ Good: Overall accuracy > 85%, Stage 1 > 90%
- ⚠️ Okay: 75-85% overall
- ❌ Poor: < 75% overall

---

## Troubleshooting

### Issue: Model not found
```
Error: Model not found: models/...
```
**Solution:** Run `benchmark_models.py` first without pre-trained models to train them.

### Issue: Data shape mismatch
```
Error: Feature shape mismatch...
```
**Solution:** Verify data preprocessing pipeline:
```bash
python scripts/prepare_accelerometer_data.py
```

### Issue: Out of memory
```
MemoryError during benchmarking
```
**Solution:** 
- Use smaller batch sizes
- Reduce n_iterations
- Run with `--quick` flag

### Issue: Very slow inference
```
Mean latency > 100ms for batch_size=1
```
**Solutions:**
- Check CPU/GPU availability
- Close other applications
- Verify data loading is not bottleneck
- Profile with CPU profiler: `--verbose`

---

## Performance Profiling

To get more detailed profiling:

```bash
# With verbose output
python scripts/benchmark_models.py \
    ... \
    --verbose

# Check memory usage
python -m memory_profiler scripts/benchmark_models.py ...
```

---

## Next Steps

1. **Baseline:** Run benchmarks on current models
   ```bash
   python scripts/benchmark_models.py \
       --accel-data data/accelerometer_classifier_v2/ \
       --hole-size-data data/processed/ \
       --output-dir results/baseline/
   ```

2. **Compare:** Try different hyperparameters
   ```bash
   # Edit config.yaml to change model parameters
   python scripts/benchmark_models.py --output-dir results/v2/
   python scripts/benchmark_models.py --output-dir results/v3/
   ```

3. **Analyze:** Review JSON reports
   ```bash
   cat results/baseline/benchmark_comparison.json | python -m json.tool
   ```

4. **Optimize:** Based on bottlenecks
   - If accuracy is low: improve data or features
   - If latency is high: simplify model or batch
   - If throughput is low: increase batch size

---

## See Also

- [Training Guide](TRAINING_GUIDE.md)
- [Evaluation Summary](EVALUATION_SUMMARY.md)
- [Two-Stage Classifier Guide](TWO_STAGE_CLASSIFIER_GUIDE.md)