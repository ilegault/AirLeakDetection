#!/usr/bin/env python3
"""
Prepare raw data for model training with Welch's method spectral analysis.

Loads raw WebDAQ CSV files, applies preprocessing, and computes:
1. Standard FFT (for comparison)
2. Welch Power Spectral Density (professor's parameters: 16 segments, Hamming window)
3. Band power features 50-4000 Hz (for ML classification)

Creates train/validation/test splits. Supports data augmentation.

Usage:
    python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/
    python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft --augment
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils
from src.data.data_loader import WebDAQDataLoader
from src.data.preprocessor import SignalPreprocessor
from src.data.fft_processor import FlexibleFFTProcessor
from src.data.data_splitter import DataSplitter
from src.data.augmentor import DataAugmentor


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare raw data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic data preparation (no FFT)
  python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/
  
  # With FFT computation
  python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft
  
  # With FFT and augmentation
  python scripts/prepare_data.py --raw-data data/raw/ --output-dir data/processed/ --compute-fft --augment
  
  # Custom splits
  python scripts/prepare_data.py \\
      --raw-data data/raw/ \\
      --output-dir data/processed/ \\
      --compute-fft \\
      --train-ratio 0.6 \\
      --val-ratio 0.2 \\
      --augment
        """
    )
    
    parser.add_argument(
        "--raw-data",
        type=str,
        default="data/raw/",
        help="Path to raw data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/",
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    
    parser.add_argument(
        "--compute-fft",
        action="store_true",
        help="Compute FFT features"
    )
    
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation to training set"
    )
    
    parser.add_argument(
        "--augmentation-multiplier",
        type=int,
        default=2,
        help="How many times to augment each training sample (default: 2)"
    )
    
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified splitting"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def validate_splits(train_ratio: float, val_ratio: float) -> bool:
    """Validate data split ratios."""
    logger = get_logger(__name__)
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        logger.error("All split ratios must be positive")
        return False
    
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        logger.error("Train and validation ratios must be between 0 and 1")
        return False
    
    logger.info(f"Split ratios - Train: {train_ratio:.2%}, Val: {val_ratio:.2%}, Test: {test_ratio:.2%}")
    return True


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    logger = get_logger(__name__)
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def discover_data_files(raw_data_path: Path, config: Dict) -> Dict[str, List[Path]]:
    """Discover all CSV files organized by class."""
    logger = get_logger(__name__)
    
    classes = config.get("classes", {})
    file_mapping: Dict[str, List[Path]] = {}
    
    for class_id, class_name in classes.items():
        class_dir = raw_data_path / class_name
        
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue
        
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {class_dir}")
            continue
        
        file_mapping[class_name] = csv_files
        logger.info(f"Found {len(csv_files)} files for class '{class_name}'")
    
    if not file_mapping:
        raise ValueError(f"No CSV files found in {raw_data_path}")
    
    return file_mapping


def load_csv_signal(file_path: Path, config: Dict) -> np.ndarray:
    """Load a single CSV file and return signal data.
    
    Handles two formats:
    1. Files with 5-line header (WebDAQ format with metadata)
    2. Raw 3-column data files with no header
    
    Args:
        file_path: Path to CSV file
        config: Configuration dictionary
        
    Returns:
        Signal array of shape (timesteps, 3) for X, Y, Z axes
    """
    logger = get_logger(__name__)
    n_channels = config.get("data", {}).get("n_channels", 3)
    
    try:
        import pandas as pd
        
        # Try format 1: WebDAQ format with headers (skiprows=5)
        try:
            df = pd.read_csv(file_path, skiprows=5, skipinitialspace=True)
            accel_cols = [col for col in df.columns if 'Acceleration' in col]
            
            if len(accel_cols) >= n_channels:
                # Successfully found header format
                data = df[accel_cols[:n_channels]].to_numpy(dtype=np.float32)
                logger.debug(f"Loaded {file_path.name} with header format ({data.shape[0]} samples)")
            else:
                # Header format didn't work, try raw format
                raise ValueError("No acceleration columns found in header format")
        
        except (pd.errors.ParserError, ValueError, KeyError):
            # Try format 2: Raw 3-column data (no header)
            data = np.genfromtxt(file_path, delimiter=',', dtype=np.float32)
            
            # Handle different shapes
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Remove trailing comma columns (NaN columns)
            data = data[:, :n_channels]
            
            logger.debug(f"Loaded {file_path.name} with raw format ({data.shape[0]} samples)")
        
        # Clean up: remove rows with NaN or Inf
        mask = ~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))
        data = data[mask]
        
        if len(data) == 0:
            raise ValueError(f"No valid data rows in {file_path.name}")
        
        # Ensure correct number of channels
        if data.shape[1] < n_channels:
            logger.warning(
                f"File {file_path.name}: Got {data.shape[1]} channels, "
                f"expected {n_channels}. Padding with zeros."
            )
            pad_width = ((0, 0), (0, n_channels - data.shape[1]))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        elif data.shape[1] > n_channels:
            data = data[:, :n_channels]
        
        return data.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        raise


def process_signal(
    signal: np.ndarray,
    config: Dict,
    compute_fft: bool = False,
) -> Dict:
    """Process a signal with preprocessing and FFT/Welch computation.
    
    Implements proper data treatment using Welch's method (professor's parameters):
    - Computes Welch Power Spectral Density with 16 segments
    - Computes band power as ML features
    - Also computes standard FFT for comparison
    
    Args:
        signal: Input signal of shape (timesteps, 3)
        config: Configuration dictionary
        compute_fft: Whether to compute FFT/Welch features
        
    Returns:
        Dictionary with 'signal' and optionally:
        - 'fft_magnitude': Standard FFT magnitude
        - 'frequencies_fft': FFT frequencies
        - 'welch_psd': Welch Power Spectral Density
        - 'welch_frequencies': Welch frequency bins
        - 'welch_bandpower': Band power features (3,)
    """
    logger = get_logger(__name__)
    
    # Create preprocessor
    preprocessor = SignalPreprocessor(config)
    
    # Detrend signal
    detrended = preprocessor.detrend(signal)
    
    # Ensure length
    duration = config.get("data", {}).get("duration", 10)
    sample_rate = config.get("data", {}).get("sample_rate", 10000)
    expected_samples = int(sample_rate * duration)
    
    if detrended.shape[0] < expected_samples:
        pad_width = ((0, expected_samples - detrended.shape[0]), (0, 0))
        detrended = np.pad(detrended, pad_width, mode='constant', constant_values=0)
    elif detrended.shape[0] > expected_samples:
        detrended = detrended[:expected_samples, :]
    
    result = {"signal": detrended}
    
    # Compute FFT and Welch if requested
    if compute_fft:
        fft_processor = FlexibleFFTProcessor(config)
        
        # 1. Compute standard FFT (for comparison)
        frequencies_fft, fft_magnitude = fft_processor.compute_scipy_fft(detrended)
        result["fft_magnitude"] = fft_magnitude.astype(np.float32)
        result["frequencies_fft"] = frequencies_fft.astype(np.float32)
        logger.debug(f"  Computed FFT: shape={fft_magnitude.shape}")
        
        # 2. Compute Welch's PSD (professor's parameters: 16 segments, Hamming window)
        welch_config = config.get("preprocessing", {}).get("welch", {})
        num_segments = welch_config.get("num_segments", 16)
        window_type = welch_config.get("window_type", "hamming")
        
        frequencies_welch, welch_psd = fft_processor.compute_welch_psd(
            detrended,
            num_segments=num_segments,
            window_type=window_type
        )
        result["welch_psd"] = welch_psd.astype(np.float32)
        result["welch_frequencies"] = frequencies_welch.astype(np.float32)
        logger.debug(f"  Computed Welch PSD: shape={welch_psd.shape} (frequency range: {frequencies_welch[0]:.2f}-{frequencies_welch[-1]:.2f} Hz)")
        
        # 3. Compute band power (50-4000 Hz) for ML features
        freq_min = welch_config.get("bandpower_freq_min", 50.0)
        freq_max = welch_config.get("bandpower_freq_max", 4000.0)
        
        bandpower = fft_processor.compute_bandpower_welch(
            detrended,
            freq_range=(freq_min, freq_max),
            num_segments=num_segments
        )
        result["welch_bandpower"] = bandpower.astype(np.float32)
        logger.debug(f"  Computed band power ({freq_min}-{freq_max} Hz): {bandpower}")
    
    return result


def save_processed_sample(
    output_file: Path,
    signal: np.ndarray,
    label: int,
    class_name: str,
    original_file: Path,
    fft_magnitude: np.ndarray = None,
    frequencies_fft: np.ndarray = None,
    welch_psd: np.ndarray = None,
    welch_frequencies: np.ndarray = None,
    welch_bandpower: np.ndarray = None,
) -> None:
    """Save processed sample to disk with FFT and Welch features.
    
    Args:
        output_file: Output file path (.npz)
        signal: Processed signal
        label: Class label
        class_name: Class name string
        original_file: Original file path (for reference)
        fft_magnitude: Optional FFT magnitude (standard)
        frequencies_fft: Optional FFT frequencies
        welch_psd: Optional Welch Power Spectral Density
        welch_frequencies: Optional Welch frequency bins
        welch_bandpower: Optional band power features
    """
    logger = get_logger(__name__)
    
    # Save as NPZ file
    save_data = {
        "signal": signal,
        "label": np.array(label, dtype=np.int32),
        "class_name": class_name,
        "original_file": str(original_file),
    }
    
    # Standard FFT data
    if fft_magnitude is not None:
        save_data["fft_magnitude"] = fft_magnitude
    if frequencies_fft is not None:
        save_data["frequencies_fft"] = frequencies_fft
    
    # Welch's method data (NEW)
    if welch_psd is not None:
        save_data["welch_psd"] = welch_psd
    if welch_frequencies is not None:
        save_data["welch_frequencies"] = welch_frequencies
    if welch_bandpower is not None:
        save_data["welch_bandpower"] = welch_bandpower
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_file), **save_data)
    
    logger.debug(f"Saved processed sample to {output_file.name}")


def prepare_data(args) -> int:
    """Prepare data for training."""
    logger = get_logger(__name__)
    
    try:
        # Validate splits
        if not validate_splits(args.train_ratio, args.val_ratio):
            return 1
        
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Data configuration: {config.get('data')}")
        
        # Check input directory
        raw_path = Path(args.raw_data)
        if not raw_path.exists():
            logger.error(f"Raw data directory not found: {args.raw_data}")
            return 1
        
        # Create output directories
        output_path = Path(args.output_dir)
        for split in ["train", "val", "test"]:
            FileUtils.ensure_directory(str(output_path / split))
        
        # Add metadata directory for split information
        FileUtils.ensure_directory(str(output_path / "metadata"))
        
        logger.info(f"Raw data path: {args.raw_data}")
        logger.info(f"Output path: {args.output_dir}")
        logger.info(f"Computing FFT: {args.compute_fft}")
        logger.info(f"Data augmentation: {args.augment}")
        
        # ============================================================
        # PHASE 1: Discover and load all data
        # ============================================================
        logger.info("="*60)
        logger.info("PHASE 1: Discovering data files...")
        logger.info("="*60)
        
        file_mapping = discover_data_files(raw_path, config)
        
        total_files = sum(len(files) for files in file_mapping.values())
        logger.info(f"Total files discovered: {total_files}")
        
        # ============================================================
        # PHASE 2: Process all signals
        # ============================================================
        logger.info("="*60)
        logger.info("PHASE 2: Processing signals...")
        logger.info("="*60)
        
        processed_samples = []
        labels = []
        file_paths = []
        
        sample_count = 0
        for class_id_str, class_name in config.get("classes", {}).items():
            class_id = int(class_id_str)
            
            if class_name not in file_mapping:
                logger.warning(f"No files for class '{class_name}'")
                continue
            
            csv_files = file_mapping[class_name]
            logger.info(f"\nProcessing class '{class_name}' ({len(csv_files)} files)...")
            
            for i, file_path in enumerate(csv_files, 1):
                if i % max(1, len(csv_files) // 5) == 0:
                    logger.info(f"  Progress: {i}/{len(csv_files)}")
                
                try:
                    # Load signal
                    signal = load_csv_signal(file_path, config)
                    
                    # Process signal
                    processed = process_signal(signal, config, compute_fft=args.compute_fft)
                    
                    # Store for later splitting
                    processed_samples.append(processed)
                    labels.append(class_id)
                    file_paths.append(file_path)
                    sample_count += 1
                
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue
        
        if sample_count == 0:
            logger.error("No samples were processed successfully")
            return 1
        
        logger.info(f"\nSuccessfully processed {sample_count} samples")
        
        # ============================================================
        # PHASE 3: Create train/val/test splits
        # ============================================================
        logger.info("="*60)
        logger.info("PHASE 3: Creating train/val/test splits...")
        logger.info("="*60)
        
        labels_array = np.array(labels, dtype=np.int32)
        splitter = DataSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            random_seed=42
        )
        
        train_idx, val_idx, test_idx = splitter.split_stratified(labels_array, file_level=True)
        
        split_stats = splitter.get_split_statistics(
            labels_array, train_idx, val_idx, test_idx
        )
        
        logger.info(f"Train set: {len(train_idx)} samples")
        logger.info(f"  Class distribution: {split_stats['train']['class_distribution']}")
        logger.info(f"Val set: {len(val_idx)} samples")
        logger.info(f"  Class distribution: {split_stats['val']['class_distribution']}")
        logger.info(f"Test set: {len(test_idx)} samples")
        logger.info(f"  Class distribution: {split_stats['test']['class_distribution']}")
        
        # Save split indices
        split_indices = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        splitter.save_split_indices(
            split_indices,
            output_path / "metadata" / "split_indices.json"
        )
        
        # ============================================================
        # PHASE 4: Save processed samples
        # ============================================================
        logger.info("="*60)
        logger.info("PHASE 4: Saving processed samples...")
        logger.info("="*60)
        
        def save_split_samples(indices, split_name):
            """Save samples for a specific split."""
            logger.info(f"\nSaving {split_name} set ({len(indices)} samples)...")
            
            for i, idx in enumerate(indices):
                if (i + 1) % max(1, len(indices) // 5) == 0:
                    logger.info(f"  Progress: {i+1}/{len(indices)}")
                
                processed = processed_samples[idx]
                label = labels[idx]
                class_id = int(class_id_str)
                class_name = [name for cid, name in config.get("classes", {}).items() 
                             if int(cid) == label][0]
                original_file = file_paths[idx]
                
                # Generate output filename
                output_filename = f"{original_file.stem}_processed.npz"
                output_file = output_path / split_name / output_filename
                
                # Save sample (with standard FFT and Welch features)
                save_processed_sample(
                    output_file,
                    processed["signal"],
                    label,
                    class_name,
                    original_file,
                    fft_magnitude=processed.get("fft_magnitude"),
                    frequencies_fft=processed.get("frequencies_fft"),
                    welch_psd=processed.get("welch_psd"),
                    welch_frequencies=processed.get("welch_frequencies"),
                    welch_bandpower=processed.get("welch_bandpower"),
                )
        
        # Save each split
        save_split_samples(train_idx, "train")
        save_split_samples(val_idx, "val")
        save_split_samples(test_idx, "test")
        
        # ============================================================
        # PHASE 5: Optional data augmentation
        # ============================================================
        if args.augment:
            logger.info("="*60)
            logger.info("PHASE 5: Augmenting training data...")
            logger.info("="*60)
            
            augmentor = DataAugmentor(config)
            
            # Get list of training files
            train_files = list((output_path / "train").glob("*_processed.npz"))
            logger.info(f"Augmenting {len(train_files)} training samples ({args.augmentation_multiplier}x)...")
            
            for i, npz_file in enumerate(train_files, 1):
                if (i) % max(1, len(train_files) // 5) == 0:
                    logger.info(f"  Progress: {i}/{len(train_files)}")
                
                try:
                    # Load original sample
                    data = np.load(npz_file)
                    original_signal = data["signal"]
                    label = int(data["label"])
                    class_name = str(data["class_name"])
                    
                    # Create augmented versions
                    for aug_num in range(1, args.augmentation_multiplier):
                        augmented_signal = augmentor.augment_batch(
                            original_signal[np.newaxis, :, :],
                            augmentation_type="all"
                        )[0]
                        
                        # Generate augmented filename
                        aug_filename = f"{npz_file.stem}_aug{aug_num}.npz"
                        aug_file = npz_file.parent / aug_filename
                        
                        # Save augmented sample (preserve Welch features from original)
                        save_processed_sample(
                            aug_file,
                            augmented_signal,
                            label,
                            class_name,
                            npz_file,
                            fft_magnitude=data.get("fft_magnitude"),
                            frequencies_fft=data.get("frequencies_fft"),
                            welch_psd=data.get("welch_psd"),
                            welch_frequencies=data.get("welch_frequencies"),
                            welch_bandpower=data.get("welch_bandpower"),
                        )
                
                except Exception as e:
                    logger.error(f"Failed to augment {npz_file}: {e}")
                    continue
            
            logger.info(f"Data augmentation completed")
        
        # ============================================================
        # SAVE SUMMARY STATISTICS
        # ============================================================
        logger.info("="*60)
        logger.info("Saving summary statistics...")
        logger.info("="*60)
        
        summary = {
            "total_samples_processed": sample_count,
            "compute_fft": args.compute_fft,
            "data_augmentation": args.augment,
            "augmentation_multiplier": args.augmentation_multiplier if args.augment else 1,
            "splits": {
                "train": {
                    "count": len(train_idx),
                    "class_distribution": split_stats["train"]["class_distribution"],
                },
                "val": {
                    "count": len(val_idx),
                    "class_distribution": split_stats["val"]["class_distribution"],
                },
                "test": {
                    "count": len(test_idx),
                    "class_distribution": split_stats["test"]["class_distribution"],
                },
            },
            "configuration": {
                "data": config.get("data"),
                "preprocessing": config.get("preprocessing"),
            },
        }
        
        summary_file = output_path / "metadata" / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary saved to {summary_file}")
        
        logger.info("="*60)
        logger.info("DATA PREPARATION COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Processed samples saved to: {args.output_dir}")
        logger.info(f"  - Train: {len(train_idx)} samples")
        logger.info(f"  - Val: {len(val_idx)} samples")
        logger.info(f"  - Test: {len(test_idx)} samples")
        
        return 0
    
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("DATA PREPARATION - Prepare Data for Training")
    logger.info("=" * 60)
    
    return prepare_data(args)


if __name__ == "__main__":
    sys.exit(main())