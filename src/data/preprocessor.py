"""Signal preprocessing utilities for WebDAQ vibration data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq


@dataclass
class PreprocessingConfig:
    """Lightweight container for preprocessing parameters."""

    sample_rate: int
    fft_size: int
    window: str
    overlap: float
    freq_max: Optional[float]
    normalize: bool


class SignalPreprocessor:
    """Prepare time-series signals for downstream feature extraction."""

    def __init__(self, config: Dict[str, Any]):
        data_cfg = config.get("data", {})
        prep_cfg = config.get("preprocessing", {})

        self.sample_rate = int(data_cfg.get("sample_rate", 0))
        self.n_channels = int(data_cfg.get("n_channels", 1))

        self.params = PreprocessingConfig(
            sample_rate=self.sample_rate,
            fft_size=int(prep_cfg.get("fft_size", 2048)),
            window=str(prep_cfg.get("window", "hanning")),
            overlap=float(prep_cfg.get("overlap", 0.0)),
            freq_max=float(prep_cfg.get("freq_max")) if "freq_max" in prep_cfg else None,
            normalize=bool(prep_cfg.get("normalize", False)),
        )

        self._window_cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Basic signal operations
    # ------------------------------------------------------------------
    def detrend(self, signal_data: np.ndarray, type_: str = "constant") -> np.ndarray:
        """Remove the trend component from each channel."""
        return scipy_signal.detrend(signal_data, axis=0, type=type_)

    def apply_window(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply the configured window function to each channel."""
        n_samples = signal_data.shape[0]
        window = self._get_window(n_samples)
        return signal_data * window[:, None]

    def _get_window(self, length: int) -> np.ndarray:
        if length not in self._window_cache:
            window_type = self.params.window.lower()
            if window_type in {"hann", "hanning"}:
                self._window_cache[length] = np.hanning(length)
            elif window_type == "hamming":
                self._window_cache[length] = np.hamming(length)
            elif window_type == "blackman":
                self._window_cache[length] = np.blackman(length)
            else:
                self._window_cache[length] = np.ones(length)
        return self._window_cache[length]

    # ------------------------------------------------------------------
    # FFT computation
    # ------------------------------------------------------------------
    def compute_fft(self, signal_data: np.ndarray, apply_window: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the one-sided FFT magnitude for each channel."""
        if signal_data.ndim != 2:
            raise ValueError("`signal_data` must be a 2D array of shape (timesteps, channels)")

        n_samples, n_channels = signal_data.shape
        if n_channels != self.n_channels:
            raise ValueError(
                f"Unexpected number of channels: got {n_channels}, expected {self.n_channels}"
            )

        values = signal_data.astype(np.float32, copy=False)
        if apply_window:
            values = self.apply_window(values)

        fft_size = self.params.fft_size
        fft_magnitude = np.empty((n_channels, fft_size // 2 + 1), dtype=np.float32)

        for idx in range(n_channels):
            spectrum = rfft(values[:, idx], n=fft_size)
            magnitude = np.abs(spectrum)
            fft_magnitude[idx] = magnitude

        frequencies = rfftfreq(fft_size, d=1.0 / self.params.sample_rate)

        if self.params.freq_max is not None:
            mask = frequencies <= self.params.freq_max
            frequencies = frequencies[mask]
            fft_magnitude = fft_magnitude[:, mask]

        if self.params.normalize:
            fft_magnitude = self.normalize_features(fft_magnitude)

        if self.params.freq_max is not None:
            frequencies = frequencies.astype(np.float32, copy=False)

        return frequencies, fft_magnitude

    # ------------------------------------------------------------------
    # Feature scaling
    # ------------------------------------------------------------------
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features per-channel using min-max scaling after log compression."""
        log_scaled = np.log1p(features)
        mins = log_scaled.min(axis=1, keepdims=True)
        maxs = log_scaled.max(axis=1, keepdims=True)
        denominator = np.clip(maxs - mins, a_min=1e-8, a_max=None)
        normalized = (log_scaled - mins) / denominator
        return normalized

    # ------------------------------------------------------------------
    # Sliding window segmentation
    # ------------------------------------------------------------------
    def segment_signal(self, signal_data: np.ndarray, window_size: int, step_size: Optional[int] = None) -> np.ndarray:
        """Segment continuous recordings into overlapping windows."""
        if step_size is None:
            step_size = max(1, int(window_size * (1.0 - self.params.overlap)))

        n_samples, n_channels = signal_data.shape
        if window_size > n_samples:
            raise ValueError("Window size cannot exceed number of samples in signal")

        segments: List[np.ndarray] = []  # type: ignore[name-defined]
        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            segments.append(signal_data[start:end])

        if not segments:
            raise ValueError("Segmentation produced 0 windows; adjust parameters.")

        return np.stack(segments, axis=0)