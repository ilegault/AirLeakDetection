"""Validate data quality and consistency."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality, shape consistency, and class balance."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        data_cfg = config.get("data", {})
        self.expected_channels = int(data_cfg.get("n_channels", 9))
        self.expected_sample_rate = int(data_cfg.get("sample_rate", 10000))

    def check_nan_inf(self, signals: np.ndarray) -> Dict[str, Any]:
        """Check for NaN/Inf values in signals.
        
        Args:
            signals: Input signals array
            
        Returns:
            Dictionary with validation results
        """
        has_nan = np.any(np.isnan(signals))
        has_inf = np.any(np.isinf(signals))
        
        nan_count = np.sum(np.isnan(signals))
        inf_count = np.sum(np.isinf(signals))

        result = {
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "total_elements": int(signals.size),
        }

        if has_nan or has_inf:
            LOGGER.warning(f"Data contains NaN/Inf values: {result}")

        return result

    def check_shape_consistency(self, signals: np.ndarray) -> Dict[str, Any]:
        """Verify shape consistency across samples.
        
        Args:
            signals: Input signals array (n_samples, timesteps, channels)
            
        Returns:
            Dictionary with shape validation results
        """
        if signals.ndim < 2:
            return {"valid": False, "error": "Signals must be at least 2D"}

        n_samples = signals.shape[0]
        expected_shape = signals[0].shape

        inconsistent_samples = []
        for i in range(1, n_samples):
            if signals[i].shape != expected_shape:
                inconsistent_samples.append(i)

        result = {
            "valid": len(inconsistent_samples) == 0,
            "n_samples": n_samples,
            "expected_shape": expected_shape,
            "inconsistent_samples": inconsistent_samples,
        }

        if inconsistent_samples:
            LOGGER.warning(f"Found {len(inconsistent_samples)} samples with inconsistent shapes")

        return result

    def check_frequency_range(self, frequencies: np.ndarray) -> Dict[str, Any]:
        """Validate frequency range.
        
        Args:
            frequencies: Frequency array
            
        Returns:
            Dictionary with frequency validation results
        """
        prep_cfg = self.config.get("preprocessing", {})
        freq_min = float(prep_cfg.get("freq_min", 30.0))
        freq_max = float(prep_cfg.get("freq_max", 2000.0))

        actual_min = float(np.min(frequencies))
        actual_max = float(np.max(frequencies))

        result = {
            "expected_min": freq_min,
            "expected_max": freq_max,
            "actual_min": actual_min,
            "actual_max": actual_max,
            "within_range": (actual_min >= freq_min) and (actual_max <= freq_max),
        }

        return result

    def check_class_balance(
        self,
        labels: np.ndarray,
        min_samples_per_class: int = 10,
    ) -> Dict[str, Any]:
        """Check class balance and minimum samples per class.
        
        Args:
            labels: Array of class labels
            min_samples_per_class: Minimum required samples per class
            
        Returns:
            Dictionary with class balance information
        """
        unique_classes, counts = np.unique(labels, return_counts=True)

        imbalanced_classes = [c for c, count in zip(unique_classes, counts) if count < min_samples_per_class]

        result = {
            "n_classes": len(unique_classes),
            "class_distribution": {int(c): int(count) for c, count in zip(unique_classes, counts)},
            "min_class_size": int(np.min(counts)),
            "max_class_size": int(np.max(counts)),
            "imbalance_ratio": float(np.max(counts) / np.min(counts)),
            "insufficient_classes": imbalanced_classes,
            "balanced": len(imbalanced_classes) == 0,
        }

        if not result["balanced"]:
            LOGGER.warning(f"Imbalanced classes detected: {result}")

        return result

    def check_value_range(
        self,
        signals: np.ndarray,
        expected_range: tuple[float, float] = (-10.0, 10.0),
    ) -> Dict[str, Any]:
        """Check if values are within expected range.
        
        Args:
            signals: Input signals
            expected_range: Tuple of (min, max) expected values
            
        Returns:
            Dictionary with range validation results
        """
        actual_min = float(np.min(signals))
        actual_max = float(np.max(signals))
        
        out_of_range = (actual_min < expected_range[0]) or (actual_max > expected_range[1])

        result = {
            "expected_range": expected_range,
            "actual_min": actual_min,
            "actual_max": actual_max,
            "out_of_range": out_of_range,
        }

        if out_of_range:
            LOGGER.warning(f"Data contains values outside expected range: {result}")

        return result

    def validate_dataset(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Comprehensive dataset validation.
        
        Args:
            signals: Input signals
            labels: Class labels
            frequencies: Optional frequency array for FFT data
            
        Returns:
            Comprehensive validation report
        """
        report = {
            "nan_inf": self.check_nan_inf(signals),
            "shape_consistency": self.check_shape_consistency(signals),
            "class_balance": self.check_class_balance(labels),
            "value_range": self.check_value_range(signals),
        }

        if frequencies is not None:
            report["frequency_range"] = self.check_frequency_range(frequencies)

        # Overall validity
        report["valid"] = all([
            not report["nan_inf"]["has_nan"],
            not report["nan_inf"]["has_inf"],
            report["shape_consistency"]["valid"],
            report["class_balance"]["balanced"],
            not report["value_range"]["out_of_range"],
        ])

        return report