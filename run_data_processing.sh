#!/bin/bash
# Process raw data with Welch method implementation
cd /home/viscosity_c/PycharmProjects/AirLeakDetection

echo "Starting data processing with Welch method..."
echo "Processing 1,900 CSV files (1300 train, 280 val, 320 test)"
echo "Each file: signal preprocessing + FFT + Welch PSD + band power extraction"
echo ""

python scripts/prepare_data.py \
    --raw-data data/raw/ \
    --output-dir data/processed/ \
    --compute-fft \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --verbose > data/processing_log.txt 2>&1

echo "Data processing completed!"
echo "Check data/processing_log.txt for details"
echo "Results saved to data/processed/"