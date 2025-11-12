# Welch Spectra Visualization Guide

This guide explains how to visualize the Welch Power Spectral Density (PSD) analysis for your air leak detection data.

## Quick Start

### Visualize a Single File

```bash
python scripts/visualize_welch_spectra.py "data/raw/3_32/Air_Leak_Test_Three_Thirtyseconds_2025-11-06T15-17-06-657.csv"
```

This will:
- Load the CSV file
- Compute Welch PSD for all 3 accelerometers
- Generate 4 plots and save them to the `plots/` directory
- Display the plots (if GUI is available)

### Visualize Multiple Files (Batch Mode)

```bash
# Process all files in a directory
python scripts/visualize_batch.py data/raw/3_32/

# Process only first 5 files
python scripts/visualize_batch.py data/raw/1_8/ --limit 5

# Process files from a specific class
python scripts/visualize_batch.py data/raw/ --class-filter "NOLEAK"
```

## Generated Plots

For each CSV file, 4 plots are generated:

1. **`*_welch_linear.png`** - Linear scale PSD for each accelerometer (3 subplots)
   - Shows the power spectral density in linear scale
   - Good for seeing overall magnitude differences

2. **`*_welch_semilog.png`** - Semi-log scale PSD for each accelerometer (3 subplots)
   - Y-axis is logarithmic
   - Better for seeing small features and wide dynamic range

3. **`*_bandpower.png`** - Band power comparison (bar chart)
   - Shows integrated power in the specified frequency range
   - Useful for comparing accelerometer responses

4. **`*_overlay.png`** - All accelerometers overlaid (2 subplots: linear and semi-log)
   - Direct comparison of all 3 accelerometers
   - Both linear and log scales

## Command Line Options

### Single File Visualization

```bash
python scripts/visualize_welch_spectra.py <csv_file> [OPTIONS]
```

**Options:**
- `--config <path>` - Configuration file (default: `config.yaml`)
- `--output-dir <path>` - Output directory for plots (default: `plots`)
- `--num-segments <int>` - Number of segments for Welch's method (default: 16)
- `--freq-range <min> <max>` - Frequency range for band power (default: 50 4000)
- `--axis-limits <xmin> <xmax> <ymin> <ymax>` - Axis limits for linear plot
- `--no-show` - Don't display plots, only save them

**Examples:**

```bash
# Custom frequency range for band power
python scripts/visualize_welch_spectra.py data.csv --freq-range 100 3000

# Custom axis limits (professor's recommendation)
python scripts/visualize_welch_spectra.py data.csv --axis-limits 30 4500 0 7e-5

# More segments for better frequency resolution
python scripts/visualize_welch_spectra.py data.csv --num-segments 32

# Save only, don't display
python scripts/visualize_welch_spectra.py data.csv --no-show
```

### Batch Visualization

```bash
python scripts/visualize_batch.py <data_dir> [OPTIONS]
```

**Options:**
- `--config <path>` - Configuration file (default: `config.yaml`)
- `--output-dir <path>` - Output directory for plots (default: `plots`)
- `--limit <int>` - Limit number of files to process
- `--class-filter <name>` - Filter by class name (e.g., "3_32", "NOLEAK")

**Examples:**

```bash
# Process all files in raw data directory
python scripts/visualize_batch.py data/raw/

# Process only NOLEAK class
python scripts/visualize_batch.py data/raw/ --class-filter NOLEAK

# Process first 10 files from 3_32 class
python scripts/visualize_batch.py data/raw/3_32/ --limit 10

# Save to custom directory
python scripts/visualize_batch.py data/raw/ --output-dir analysis_plots/
```

## Understanding the Welch Method

The visualization uses **Welch's method** with these parameters (from `config.yaml`):

- **Number of segments**: 16 (configurable)
- **Window type**: Hamming (better spectral leakage reduction)
- **Overlap**: 50% between segments
- **Zero-padding**: Automatic for better frequency resolution
- **Frequency range**: 30-2000 Hz (from config)

### What is Welch's Method?

Welch's method is a spectral estimation technique that:
1. Divides the signal into overlapping segments
2. Applies a window function to each segment
3. Computes FFT for each segment
4. Averages the power spectra

**Benefits:**
- Reduces noise and variance in the spectrum
- Provides smoother, more reliable spectral estimates
- Better for classification and feature extraction

## Interpreting the Plots

### Linear Scale PSD
- **Peaks** indicate dominant frequencies
- **Height** shows power at that frequency
- Compare peaks across accelerometers to identify leak signatures

### Semi-log Scale PSD
- Better for seeing small features
- Useful when power varies over several orders of magnitude
- Helps identify subtle differences between classes

### Band Power
- Single metric per accelerometer
- Useful for ML classification
- Higher values indicate more vibration energy in that frequency range

### Overlay Plots
- Direct comparison of all 3 accelerometers
- Look for differences in:
  - Peak locations
  - Overall power levels
  - Frequency distribution

## Troubleshooting

### "No display" warning
If you see: `Matplotlib is currently using agg, which is a non-GUI backend`

This is normal when running without a GUI. The plots are still saved to files.

### NaN values in band power
If you see `Accel2=nan`, check:
- The CSV file has valid data for all 3 accelerometers
- No missing or corrupted values in the data

### Memory issues with batch processing
If processing many files:
- Use `--limit` to process in smaller batches
- The script automatically closes plots to free memory

## Output Location

By default, plots are saved to:
```
plots/
├── <filename>_welch_linear.png
├── <filename>_welch_semilog.png
├── <filename>_bandpower.png
└── <filename>_overlay.png
```

You can change this with `--output-dir` option.

## Next Steps

After visualizing the data:
1. Compare spectra across different leak classes
2. Identify characteristic frequency ranges for each class
3. Use band power features for ML classification
4. Adjust frequency ranges based on observed patterns

## Configuration

Edit `config.yaml` to change default parameters:

```yaml
preprocessing:
  freq_min: 30       # Minimum frequency (Hz)
  freq_max: 2000     # Maximum frequency (Hz)
  
  welch:
    num_segments: 16        # Number of segments
    window_type: "hamming"  # Window type
    overlap_ratio: 0.5      # 50% overlap
    bandpower_freq_min: 50  # Band power min freq
    bandpower_freq_max: 4000 # Band power max freq
```