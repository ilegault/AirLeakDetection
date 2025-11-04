"""Load data from multiple sources (CSV, .mat, .npz files)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class HybridDataLoader:
    """Load data from multiple sources and combine them."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize hybrid loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        data_cfg = config.get("data", {})
        self.raw_data_dir = Path(data_cfg.get("raw_data_path", "data/raw"))
        self.processed_data_dir = Path(data_cfg.get("processed_data_path", "data/processed"))

    def load_csv(self, csv_path: Path) -> np.ndarray:
        """Load raw CSV data.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Data array
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            numeric_df = df.select_dtypes(include=["number"])
            return numeric_df.to_numpy(dtype=np.float32)
        except Exception as e:
            LOGGER.error(f"Failed to load CSV {csv_path}: {e}")
            raise

    def load_mat(self, mat_path: Path) -> np.ndarray:
        """Load MATLAB .mat file.
        
        Args:
            mat_path: Path to .mat file
            
        Returns:
            Data array
        """
        try:
            import scipy.io as sio
            mat_data = sio.loadmat(mat_path)
            
            # Look for common variable names
            for key in ['data', 'fft', 'FFT', 'spectrum', 'magnitude']:
                if key in mat_data:
                    return np.asarray(mat_data[key], dtype=np.float32)
            
            # Return first non-meta variable
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    return np.asarray(value, dtype=np.float32)
                    
            raise ValueError("No suitable data found in .mat file")
        except Exception as e:
            LOGGER.error(f"Failed to load .mat file {mat_path}: {e}")
            raise

    def load_npz(self, npz_path: Path) -> Dict[str, np.ndarray]:
        """Load NumPy .npz file.
        
        Args:
            npz_path: Path to .npz file
            
        Returns:
            Dictionary of arrays
        """
        try:
            data = np.load(npz_path, allow_pickle=False)
            return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
        except Exception as e:
            LOGGER.error(f"Failed to load .npz file {npz_path}: {e}")
            raise

    def load_raw_csv_dataset(self, raw_csv_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load all raw CSV files from a directory.
        
        Args:
            raw_csv_dir: Directory containing class subdirectories with CSV files
            
        Returns:
            signals (n_samples, timesteps, channels), labels (n_samples,)
        """
        if raw_csv_dir is None:
            raw_csv_dir = self.raw_data_dir

        signals: List[np.ndarray] = []
        labels: List[int] = []

        class_dirs = sorted([d for d in Path(raw_csv_dir).iterdir() if d.is_dir()])
        
        for class_id, class_dir in enumerate(class_dirs):
            csv_files = sorted(class_dir.glob("*.csv"))
            for csv_file in csv_files:
                try:
                    signal = self.load_csv(csv_file)
                    signals.append(signal)
                    labels.append(class_id)
                except Exception as e:
                    LOGGER.warning(f"Skipping {csv_file}: {e}")

        if not signals:
            LOGGER.warning("No signals loaded")
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        signals_array = np.stack(signals, axis=0)
        labels_array = np.array(labels, dtype=np.int32)
        
        return signals_array, labels_array

    def load_precomputed_fft(self, fft_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-computed FFT data from .npz files.
        
        Args:
            fft_dir: Directory containing .npz files with FFT data
            
        Returns:
            fft_data (n_samples, frequencies, channels), labels
        """
        npz_files = sorted(fft_dir.glob("*.npz"))
        fft_data: List[np.ndarray] = []
        labels: List[int] = []

        for idx, npz_file in enumerate(npz_files):
            try:
                data = self.load_npz(npz_file)
                if 'fft' in data:
                    fft_data.append(data['fft'])
                    if 'label' in data:
                        labels.append(int(data['label'].flat[0]))
                    else:
                        labels.append(idx)
            except Exception as e:
                LOGGER.warning(f"Skipping {npz_file}: {e}")

        if not fft_data:
            LOGGER.warning("No FFT data loaded")
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        fft_array = np.stack(fft_data, axis=0)
        labels_array = np.array(labels, dtype=np.int32)
        
        return fft_array, labels_array

    def combine_sources(
        self,
        raw_csv_dir: Optional[Path] = None,
        fft_dir: Optional[Path] = None,
        fft_processor: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load and combine data from multiple sources.
        
        Args:
            raw_csv_dir: Directory with raw CSV files
            fft_dir: Directory with pre-computed FFT files
            fft_processor: FFT processor to compute FFT from raw data if needed
            
        Returns:
            raw_signals, labels, fft_data (if available)
        """
        raw_signals = None
        labels = None
        fft_data = None

        # Load raw CSV data
        if raw_csv_dir is not None or self.raw_data_dir.exists():
            csv_dir = raw_csv_dir or self.raw_data_dir
            LOGGER.info(f"Loading raw CSV data from {csv_dir}")
            raw_signals, labels = self.load_raw_csv_dataset(csv_dir)

        # Load pre-computed FFT
        if fft_dir is not None and Path(fft_dir).exists():
            LOGGER.info(f"Loading pre-computed FFT from {fft_dir}")
            fft_data, fft_labels = self.load_precomputed_fft(Path(fft_dir))
            if labels is None:
                labels = fft_labels

        # Compute FFT if needed
        if raw_signals is not None and fft_data is None and fft_processor is not None:
            LOGGER.info("Computing FFT from raw signals")
            fft_list = []
            for signal in raw_signals:
                _, magnitude = fft_processor.compute_scipy_fft(signal)
                fft_list.append(magnitude)
            fft_data = np.stack(fft_list, axis=0)

        return raw_signals, labels, fft_data