# Quick Start - Air Leak Detection Training

## TL;DR - Fast Track

```bash
# 1. Put your CSV files in data/raw/
mkdir -p data/raw/{NOLEAK,1_16,3_32,1_8}

# 2. Process data
python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft

# 3. Train all models (optimized for 16 cores)
python train_optimized.py --all

# Done! Check results in models/ and experiments/
```

## System Info

- **CPU:** 16 cores (optimized)
- **GPU:** Not detected (see GPU_SETUP.md to enable 6-10x speedup)
- **RAM:** Use `--batch-size 32` if you run out of memory
- **OS:** Zorin Linux
- **TensorFlow:** 2.20.0 ✅ installed

## Training Commands

### Recommended (100 epochs, all models)
```bash
python train_optimized.py --all
```
**Time:** ~60-90 minutes on CPU

### Quick Test (30 epochs)
```bash
python train_optimized.py --all --epochs 30
```
**Time:** ~25-35 minutes on CPU

### Fast Models Only
```bash
python train_optimized.py --models random_forest svm
```
**Time:** ~8-15 minutes

### Specific Models
```bash
python train_optimized.py --models cnn_1d lstm --epochs 50
```

## Expected Training Times (CPU - 16 cores)

| Model | 30 epochs | 100 epochs |
|-------|-----------|------------|
| Random Forest | 1-2 min | 3-5 min |
| SVM | 2-3 min | 5-10 min |
| CNN 1D | 8-12 min | 20-30 min |
| LSTM | 12-18 min | 30-45 min |
| **All 4 models** | **25-35 min** | **60-90 min** |

## Speed Up Training

### Option 1: Train Fast Models First
```bash
# Takes 8-15 minutes
python train_optimized.py --models random_forest svm

# Then train deep learning models separately
python train_optimized.py --models cnn_1d lstm
```

### Option 2: Use Fewer Epochs for Testing
```bash
python train_optimized.py --all --epochs 30  # 25-35 minutes
```

### Option 3: Enable GPU (6-10x Faster!)
```bash
# See GPU_SETUP.md for instructions
# With GPU: 15-30 minutes for all models (vs 60-90 min CPU)
```

## Common Issues

### "Data directory not found"
```bash
# Make sure you've run data preparation:
python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft
```

### "Out of memory"
```bash
# Reduce batch size:
python train_optimized.py --all --batch-size 32
```

### Want GPU Speedup?
```bash
# See GPU_SETUP.md - one-time setup for 6-10x faster training
cat GPU_SETUP.md
```

## Files

- `train_optimized.py` - **Main training script** (use this!)
- `TRAINING_GUIDE.md` - Detailed guide
- `GPU_SETUP.md` - GPU setup for 6-10x speedup
- `config.yaml` - Configuration (already optimized)

## Help

```bash
# See all options
python train_optimized.py --help

# Data preparation options
python scripts/prepare_data.py --help
```

---

**Ready to train?** Run: `python train_optimized.py --all`
