"""Utilities for discovering and loading WebDAQ vibration datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


class WebDAQDataLoader:
    """Load and structure vibration signals captured with a WebDAQ system."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as cfg_file:
            self.config: Dict[str, Any] = yaml.safe_load(cfg_file)

        self._config_dir = self.config_path.parent
        self.raw_data_dir = (self._config_dir / self.config["data"]["raw_data_path"]).resolve()
        self.processed_data_dir = (self._config_dir / self.config["data"]["processed_data_path"]).resolve()

        if not self.raw_data_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory does not exist: {self.raw_data_dir}. "
                "Ensure the data files are available before attempting to load them."
            )

        classes_cfg = self.config.get("classes", {})
        if not classes_cfg:
            raise ValueError("No class mapping found in configuration under `classes`." )

        self.class_id_to_name: Dict[int, str] = {int(idx): name for idx, name in classes_cfg.items()}
        self.class_name_to_id: Dict[str, int] = {name: idx for idx, name in self.class_id_to_name.items()}

        data_cfg = self.config.get("data", {})
        self.sample_rate: int = int(data_cfg.get("sample_rate", 0))
        self.n_channels: int = int(data_cfg.get("n_channels", 1))
        duration = float(data_cfg.get("duration", 0))
        self.duration: Optional[float] = duration if duration > 0 else None
        expected_samples = int(self.sample_rate * duration) if self.duration is not None else 0
        self._expected_samples: Optional[int] = expected_samples if expected_samples > 0 else None

        LOGGER.debug(
            "Initialized WebDAQDataLoader with raw_dir=%s, processed_dir=%s, classes=%s",
            self.raw_data_dir,
            self.processed_data_dir,
            list(self.class_name_to_id.keys()),
        )

    def available_classes(self) -> List[str]:
        """Return the ordered list of class names present in the configuration."""
        return [self.class_id_to_name[idx] for idx in sorted(self.class_id_to_name)]

    def discover_files(self) -> Dict[str, List[Path]]:
        """Locate CSV files for each leak class."""
        file_mapping: Dict[str, List[Path]] = {}
        for class_name in self.available_classes():
            class_dir = self.raw_data_dir / class_name
            if not class_dir.exists():
                LOGGER.warning("Class directory missing: %s", class_dir)
                continue

            csv_files = sorted(class_dir.glob("*.csv"))
            if not csv_files:
                LOGGER.warning("No CSV files found for class '%s' in %s", class_name, class_dir)
                continue

            file_mapping[class_name] = csv_files

        if not file_mapping:
            LOGGER.warning("No data files discovered under %s", self.raw_data_dir)

        return file_mapping

    def iter_samples(self) -> Iterator[Tuple[int, str, Path]]:
        """Yield (class_id, class_name, file_path) tuples for each discovered sample."""
        for class_name, files in self.discover_files().items():
            class_id = self.class_name_to_id[class_name]
            for file_path in files:
                yield class_id, class_name, file_path

    def load_sample(self, file_path: Path) -> np.ndarray:
        """Load a single CSV file into a 2D NumPy array shaped (samples, channels)."""
        if not file_path.exists():
            raise FileNotFoundError(f"Data file does not exist: {file_path}")

        df = pd.read_csv(file_path)
        numeric_df = df.select_dtypes(include=["number"])
        if numeric_df.shape[1] < self.n_channels:
            raise ValueError(
                "Insufficient numeric columns in file "
                f"{file_path}. Expected at least {self.n_channels} numeric channels."
            )

        signal = numeric_df.iloc[:, -self.n_channels :].to_numpy(dtype=np.float32, copy=False)
        signal = np.ascontiguousarray(signal)

        if self._expected_samples is not None:
            signal = self._ensure_length(signal)

        return signal

    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
        """Load the full dataset into memory.

        Returns:
            signals: Array shaped (n_samples, n_timesteps, n_channels)
            labels: Array of integer class labels with shape (n_samples,)
            file_paths: Original file locations for reference
        """
        signals: List[np.ndarray] = []
        labels: List[int] = []
        file_paths: List[Path] = []

        for class_id, class_name, file_path in self.iter_samples():
            try:
                signal = self.load_sample(file_path)
            except Exception as exc:  # pragma: no cover - surface helpful error details
                LOGGER.error("Failed to load %s (%s): %s", file_path, class_name, exc)
                raise

            signals.append(signal)
            labels.append(class_id)
            file_paths.append(file_path)

        if not signals:
            LOGGER.warning("No signals loaded; returning empty dataset.")
            empty_signals = np.empty((0, self._expected_samples or 0, self.n_channels), dtype=np.float32)
            return empty_signals, np.empty((0,), dtype=np.int32), file_paths

        stacked_signals = np.stack(signals, axis=0)
        label_array = np.asarray(labels, dtype=np.int32)
        return stacked_signals, label_array, file_paths

    def _ensure_length(self, signal: np.ndarray) -> np.ndarray:
        """Pad or trim signals so every sample has the expected number of timesteps."""
        if self._expected_samples is None:
            return signal

        current_length = signal.shape[0]
        if current_length == self._expected_samples:
            return signal

        if current_length < self._expected_samples:
            pad_width = self._expected_samples - current_length
            LOGGER.debug("Padding signal from %s to %s samples", current_length, self._expected_samples)
            return np.pad(signal, ((0, pad_width), (0, 0)), mode="constant", constant_values=0.0)

        LOGGER.debug("Truncating signal from %s to %s samples", current_length, self._expected_samples)
        if self.duration and self.sample_rate > 0:
            try:
                return self._resample_signal(signal, self._expected_samples)
            except Exception as exc:  # pragma: no cover - surface resampling issues
                LOGGER.warning("Resampling failed (%s); falling back to simple truncation.", exc)

        return signal[: self._expected_samples, :]

    def _resample_signal(self, signal: np.ndarray, target_samples: int) -> np.ndarray:
        """Resample the signal using linear interpolation to preserve early sample values."""
        original_samples = signal.shape[0]
        if original_samples == target_samples:
            return signal

        # Generate integer-based indices to preserve the first samples exactly.
        target_indices = np.linspace(0, original_samples - 1, target_samples)

        resampled = np.empty((target_samples, signal.shape[1]), dtype=np.float32)
        for channel_idx in range(signal.shape[1]):
            # Interpolate along each channel independently with float precision.
            resampled[:, channel_idx] = np.interp(
                target_indices,
                np.arange(original_samples),
                signal[:, channel_idx].astype(np.float64, copy=False),
            ).astype(np.float32, copy=False)

        return resampled

    def dataset_summary(self) -> Dict[str, Any]:
        """Provide a quick overview of discovered data."""
        summary: Dict[str, Any] = {
            "raw_data_dir": str(self.raw_data_dir),
            "processed_data_dir": str(self.processed_data_dir),
            "n_channels": self.n_channels,
            "sample_rate": self.sample_rate,
            "expected_samples": self._expected_samples,
            "class_distribution": {},
        }

        for class_name, files in self.discover_files().items():
            summary["class_distribution"][class_name] = len(files)

        return summary