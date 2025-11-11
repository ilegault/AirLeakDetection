# GPU Setup Guide for Lenovo Legion 5 17ITH6

Your Lenovo Legion 5 17ITH6 likely has an NVIDIA RTX 30-series GPU that isn't currently being detected. Here's how to fix it:

## Current Status

```
❌ CUDA drivers: Not found
❌ GPU detection: Failed
✅ CPU: 16 cores available
✅ TensorFlow: 2.20.0 installed
```

## Quick Fix Steps

### 1. Check Your GPU Model

```bash
lspci | grep -i nvidia
```

### 2. Install NVIDIA Drivers (Zorin Linux)

#### Option A: Using Zorin's Driver Manager (Recommended)
1. Open "Additional Drivers" from the menu
2. Select the recommended NVIDIA driver (usually nvidia-driver-550 or newer)
3. Click "Apply Changes"
4. Reboot your system

#### Option B: Command Line Installation
```bash
# Add NVIDIA PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install recommended driver
sudo ubuntu-drivers devices  # See available drivers
sudo apt install nvidia-driver-550  # Or latest version shown

# Reboot
sudo reboot
```

### 3. Install CUDA Toolkit

```bash
# Install CUDA 12.x (compatible with TensorFlow 2.20)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 4. Install cuDNN

```bash
# Download cuDNN from NVIDIA (requires free account)
# https://developer.nvidia.com/cudnn

# Or install via apt (if available)
sudo apt install libcudnn9-cuda-12
```

### 5. Verify Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Should show your GPU, driver version, and CUDA version

# Test TensorFlow GPU
python3 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Expected Performance Gains

With GPU enabled on RTX 30-series:

| Model Type | CPU (16 cores) | GPU (RTX 30xx) | Speedup |
|------------|----------------|----------------|---------|
| CNN 1D     | ~30 min        | ~3-5 min       | 6-10x   |
| LSTM       | ~45 min        | ~5-7 min       | 6-9x    |
| Random Forest | ~5 min      | ~5 min         | 1x*     |
| SVM        | ~10 min        | ~10 min        | 1x*     |

\* Traditional ML models (RF, SVM) run on CPU regardless

## Optimized Training Commands

### With GPU (After Setup)
```bash
# GPU will be automatically detected and used
python train_optimized.py --all --batch-size 128

# Larger batches work better with GPU
python train_optimized.py --models cnn_1d lstm --batch-size 256
```

### Current CPU-Only Setup
```bash
# Already optimized for 16 cores
python train_optimized.py --all --batch-size 64
```

## Troubleshooting

### Issue: "nvidia-smi" not found after installation
**Solution:** Reboot your system

### Issue: CUDA version mismatch
**Solution:** Check TensorFlow compatibility
```bash
python3 -c "import tensorflow as tf; print(tf.version.CUDA_VERSION)"
```

### Issue: GPU detected but not used
**Solution:** Check TensorFlow is using GPU
```bash
python3 -c "
import tensorflow as tf
print('Built with CUDA:', tf.test.is_built_with_cuda())
print('GPU available:', tf.config.list_physical_devices('GPU'))
"
```

### Issue: Out of memory errors
**Solution:** Reduce batch size
```bash
python train_optimized.py --all --batch-size 64  # Instead of 128
```

## Performance Monitoring

### Monitor GPU usage during training:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Monitor CPU usage:
```bash
htop  # Press F2 > Display > Show custom thread names
```

## Current Working Configuration (CPU Only)

Until GPU is set up, you can train efficiently with:

```bash
# Best performance on CPU
python train_optimized.py --all --batch-size 64 --epochs 100

# Quick test run
python train_optimized.py --models random_forest svm --epochs 30

# Train deep learning models
python train_optimized.py --models cnn_1d lstm --batch-size 64
```

## Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [cuDNN Installation](https://developer.nvidia.com/cudnn)
- [Zorin Forum - NVIDIA Drivers](https://forum.zorinos.com/)

---

**Note:** The optimized training script already configures TensorFlow to automatically use GPU if available. Once you complete the GPU setup, simply run the same commands and training will be 6-10x faster!
