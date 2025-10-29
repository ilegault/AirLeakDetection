"""Feature extraction utilities for vibration signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np
from scipy import stats


@dataclass
class FeatureExtractorConfig:
    """Configuration container for feature extraction."""

    band_edges: Iterable[Tuple[float, float]]


class FeatureExtractor:
    """Compute descriptive statistics in time and frequency domains."""

    def __init__(self, config: Dict[str, Any]) -> None:
        classes_cfg = config.get("classes", {})
        self.class_mapping = {int(idx): name for idx, name in classes_cfg.items()}
        prep_cfg = config.get("preprocessing", {})

        default_bands = [(0.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0)]
        bands = prep_cfg.get("bands", default_bands)
        self.params = FeatureExtractorConfig(band_edges=[(float(lo), float(hi)) for lo, hi in bands])

    # ------------------------------------------------------------------
    # Time-domain features
    # ------------------------------------------------------------------
    def extract_time_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute basic descriptive statistics from a 1D signal."""
        if signal.ndim != 1:
            raise ValueError("`signal` must be a 1D array for time-domain features")

        features: Dict[str, float] = {}
        features["mean"] = float(np.mean(signal))
        features["std"] = float(np.std(signal))
        features["rms"] = float(np.sqrt(np.mean(signal**2)))
        features["peak_to_peak"] = float(np.ptp(signal))
        features["kurtosis"] = float(stats.kurtosis(signal, fisher=True, bias=False))
        features["skewness"] = float(stats.skew(signal, bias=False))

        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features["zero_crossing_rate"] = float(len(zero_crossings) / max(1, signal.size))

        crest = np.max(np.abs(signal)) / max(np.finfo(np.float32).eps, features["rms"])
        features["crest_factor"] = float(crest)

        return features

    def extract_time_features_multichannel(self, signals: np.ndarray) -> Dict[str, float]:
        """Aggregate time-domain features across all channels."""
        if signals.ndim != 2:
            raise ValueError("`signals` must be 2D with shape (timesteps, channels)")

        aggregated: Dict[str, float] = {}
        for ch_idx in range(signals.shape[1]):
            channel_feats = self.extract_time_features(signals[:, ch_idx])
            for key, value in channel_feats.items():
                aggregated[f"ch{ch_idx}_{key}"] = value
        return aggregated

    # ------------------------------------------------------------------
    # Frequency-domain features
    # ------------------------------------------------------------------
    def extract_frequency_features(
        self, frequencies: np.ndarray, magnitude: np.ndarray
    ) -> Dict[str, float]:
        """Compute frequency-domain descriptors for a single spectrum."""
        if magnitude.ndim != 1:
            raise ValueError("`magnitude` must be a 1D array for frequency features")

        if frequencies.shape[0] != magnitude.shape[0]:
            raise ValueError("`frequencies` and `magnitude` must have the same length")

        features: Dict[str, float] = {}

        peak_idx = int(np.argmax(magnitude))
        features["peak_frequency"] = float(frequencies[peak_idx])
        features["peak_magnitude"] = float(magnitude[peak_idx])

        power = magnitude**2
        total_power = float(np.sum(power))
        if total_power <= 0:
            total_power = float(np.finfo(np.float32).eps)

        features["spectral_centroid"] = float(np.sum(frequencies * power) / total_power)
        features["spectral_spread"] = float(np.sqrt(np.sum(((frequencies - features["spectral_centroid"]) ** 2) * power) / total_power))

        cumulative_power = np.cumsum(power) / total_power
        features["spectral_rolloff_85"] = float(frequencies[np.searchsorted(cumulative_power, 0.85)])

        for idx, (low, high) in enumerate(self.params.band_edges):
            mask = (frequencies >= low) & (frequencies < high)
            features[f"band_power_{idx}"] = float(np.sum(power[mask]))

        return features

    def extract_frequency_features_multichannel(
        self, frequencies: np.ndarray, magnitudes: np.ndarray
    ) -> Dict[str, float]:
        """Aggregate spectral features across channels."""
        if magnitudes.ndim != 2:
            raise ValueError("`magnitudes` must be 2D with shape (channels, freqs)")

        aggregated: Dict[str, float] = {}
        for ch_idx in range(magnitudes.shape[0]):
            channel_features = self.extract_frequency_features(frequencies, magnitudes[ch_idx])
            for key, value in channel_features.items():
                aggregated[f"ch{ch_idx}_{key}"] = value
        return aggregated

    # ------------------------------------------------------------------
    # Combined feature vectors
    # ------------------------------------------------------------------
    def combine_features(
        self,
        time_features: Dict[str, float],
        frequency_features: Dict[str, float],
        label: Optional[int] = None,
    ) -> Dict[str, float]:
        """Merge time and frequency features into one vector."""
        features = {**time_features, **frequency_features}
        if label is not None:
            features["label"] = float(label)
            if label in self.class_mapping:
                features["label_name"] = self.class_mapping[label]
        return features