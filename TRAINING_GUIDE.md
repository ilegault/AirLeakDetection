# Complete Training Guide - Optimized for Lenovo Legion 5

This guide walks you through the complete workflow from raw data to trained models, optimized for your hardware (16 CPU cores).

## Quick Start

```bash
# 1. Prepare your data (place CSV files in data/raw/)
python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft

# 2. Train all models (optimized for 16 cores)
python train_optimized.py --all

# 3. Evaluate results
python scripts/evaluate.py
```

## Detailed Workflow

### Step 1: Prepare Your Raw Data

**Required:** Place your WebDAQ CSV files in the `data/raw/` directory

```bash
# Create data directory if it doesn't exist
mkdir -p data/raw/

# Your directory structure should look like:
data/raw/
├── NOLEAK/          # No leak samples
├── 1_16/            # Leak type 1/16
├── 3_32/            # Leak type 3/32
└── 1_8/             # Leak type 1/8
```

Each subdirectory should contain CSV files from your WebDAQ measurements.

### Step 2: Process Data with Welch's Method

Run the data preparation script:

```bash
# Full processing with FFT and Welch's method
python scripts/prepare_data.py \
    --raw-data data/raw/ \
    --output-dir data/processed/ \
    --compute-fft

# With data augmentation (recommended for better model performance)
python scripts/prepare_data.py \
    --raw-data data/raw/ \
    --output-dir data/processed/ \
    --compute-fft \
    --augment
```

**What this does:**
- Loads raw CSV files (17066 Hz sampling rate, 3 accelerometers)
- Applies Welch's method with professor's parameters:
  - 16 segments
  - Hamming window
  - 50% overlap
  - Band power: 50-4000 Hz
- Creates train/val/test splits (70%/15%/15%)
- Saves processed features as `.npz` files

**Expected time:** 5-15 minutes depending on data size

### Step 3: Train Models (Optimized)

Now train your models using the optimized script:

```bash
# Train all models (recommended)
python train_optimized.py --all

# Or train specific models
python train_optimized.py --models cnn_1d lstm random_forest svm

# Quick test with fewer epochs
python train_optimized.py --all --epochs 30

# If data already processed, skip aggregation
python train_optimized.py --all --skip-aggregation
```

**Training optimizations applied:**
- ✅ 16 CPU threads for TensorFlow operations
- ✅ Optimized batch size (64) for CPU
- ✅ Multi-threaded data loading (8 workers)
- ✅ oneDNN optimizations enabled
- ✅ Efficient memory management

**Expected training times (CPU only):**

| Model | Time (100 epochs) | Time (30 epochs) |
|-------|-------------------|------------------|
| Random Forest | 3-5 minutes | 1-2 minutes |
| SVM | 5-10 minutes | 2-3 minutes |
| CNN 1D | 20-30 minutes | 8-12 minutes |
| LSTM | 30-45 minutes | 12-18 minutes |

**Total for all 4 models:** ~60-90 minutes (100 epochs), ~25-35 minutes (30 epochs)

### Step 4: Monitor Training Progress

During training, you'll see:

```
==================================================
Training CNN_1D
==================================================

Epoch 1/100
  Training samples: 1234
  Validation samples: 264

  128/128 [==============================] - 12s 94ms/step
  loss: 1.2345 - accuracy: 0.4567 - val_loss: 1.1234 - val_accuracy: 0.5678

Epoch 2/100
  ...
```

**Monitor CPU usage in another terminal:**
```bash
htop  # Watch all 16 cores being utilized
```

### Step 5: View Results

After training completes:

```bash
# Check trained models
ls -lh models/

# View training logs
ls -lh experiments/

# Evaluate all models
python scripts/evaluate.py
```

## Optimization Tips

### For Faster Training

1. **Reduce epochs for testing:**
   ```bash
   python train_optimized.py --all --epochs 30
   ```

2. **Train only fast models first:**
   ```bash
   python train_optimized.py --models random_forest svm
   ```

3. **Increase batch size (if you have enough RAM):**
   ```bash
   python train_optimized.py --all --batch-size 128
   ```

### For Better Performance

1. **Use data augmentation:**
   ```bash
   python scripts/prepare_data.py \
       --raw-data data/raw/ \
       --output-dir data/processed/ \
       --compute-fft \
       --augment
   ```

2. **Train with more epochs:**
   ```bash
   python train_optimized.py --all --epochs 200
   ```

3. **Enable GPU (see GPU_SETUP.md)** - 6-10x faster for deep learning models

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Already fixed!** Dependencies are installed.

### Issue: "Data directory not found"
**Solution:** Run data preparation first:
```bash
python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft
```

### Issue: Training is slow
**Current status:** Optimized for 16 CPU cores
- Random Forest/SVM: Normal (CPU-bound, can't be sped up much)
- CNN/LSTM: Will be 6-10x faster with GPU (see GPU_SETUP.md)

**Temporary solution:** Train faster models first
```bash
python train_optimized.py --models random_forest svm
```

### Issue: Out of memory
**Solution:** Reduce batch size
```bash
python train_optimized.py --all --batch-size 32
```

### Issue: PyCharm integration
To run in PyCharm:
1. Open `train_optimized.py`
2. Right-click > "Run 'train_optimized'"
3. Or set up a Run Configuration:
   - Script path: `/home/user/AirLeakDetection/train_optimized.py`
   - Parameters: `--all --epochs 100`
   - Working directory: `/home/user/AirLeakDetection`

## Performance Comparison

### CPU Only (Current Setup)
- Using all 16 cores efficiently
- Deep learning: 20-45 min per model
- Traditional ML: 3-10 min per model
- **Total: ~60-90 minutes for all models**

### With GPU (After Setup - See GPU_SETUP.md)
- Using RTX 30-series + 16 cores
- Deep learning: 3-7 min per model (6-10x faster!)
- Traditional ML: 3-10 min per model (same)
- **Total: ~15-30 minutes for all models**

## Next Steps

1. ✅ **Run data preparation** (if you haven't already)
2. ✅ **Train models with `train_optimized.py`**
3. ⏭️ **Evaluate results** with `python scripts/evaluate.py`
4. ⏭️ **Set up GPU** (optional, see GPU_SETUP.md) for 6-10x speedup
5. ⏭️ **Hyperparameter tuning** with `python scripts/hyperparameter_search.py`

## Files Created

This optimization created:

- `train_optimized.py` - Optimized training script for your hardware
- `GPU_SETUP.md` - Guide to enable GPU (6-10x faster)
- `TRAINING_GUIDE.md` - This file
- `config.yaml` - Updated with optimized batch sizes

## Configuration Details

Your optimized configuration (`config.yaml`):

```yaml
training:
  batch_size: 64        # Optimized for 16 CPU cores
  epochs: 100           # Configurable via --epochs
  workers: 8            # 8 parallel data loaders
  max_queue_size: 16    # Efficient data pipeline
```

Environment variables set by `train_optimized.py`:
```bash
TF_NUM_INTRAOP_THREADS=16  # Use all 16 cores
TF_NUM_INTEROP_THREADS=2    # Thread scheduling
TF_ENABLE_ONEDNN_OPTS=1     # Intel optimizations
OMP_NUM_THREADS=16          # OpenMP parallelism
```

---

**Questions? Issues?**
- Check GPU_SETUP.md for GPU setup
- Check logs in `experiments/` directory
- Review TensorFlow output for errors
