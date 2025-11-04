"""Calibrate and adjust prediction confidence scores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.special import expit, softmax
from scipy.optimize import minimize

from src.prediction.predictor import LeakDetector


class ConfidenceCalibrator:
    """Calibrate model predictions to improve confidence estimates."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor: Optional[Any] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize confidence calibrator.

        Args:
            model_path: Path to trained model
            preprocessor: Preprocessor instance
            class_names: Dictionary mapping class indices to names
        """
        self.leak_detector = LeakDetector(model_path, preprocessor, class_names)
        self.temperature = 1.0
        self.calibration_method = None
        self.calibration_params = {}

    def _temperature_scale(
        self, logits: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling.

        Args:
            logits: Raw model logits
            temperature: Temperature parameter

        Returns:
            Scaled probabilities
        """
        return softmax(logits / temperature, axis=1)

    def _platt_scale(
        self, confidences: np.ndarray, weights: Tuple[float, float]
    ) -> np.ndarray:
        """Apply Platt scaling.

        Args:
            confidences: Raw confidence scores
            weights: Scaling parameters (a, b)

        Returns:
            Scaled confidences
        """
        a, b = weights
        return expit(a * confidences + b)

    def _isotonic_regression_scale(
        self, confidences: np.ndarray, thresholds: list[float]
    ) -> np.ndarray:
        """Apply isotonic regression scaling.

        Args:
            confidences: Raw confidence scores
            thresholds: Threshold values for isotonic regression

        Returns:
            Scaled confidences
        """
        scaled = np.zeros_like(confidences)
        sorted_idx = np.argsort(confidences)

        for i, threshold in enumerate(thresholds):
            mask = confidences >= threshold
            scaled[mask] = i / len(thresholds)

        return scaled

    def calibrate_temperature(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        verbose: bool = False,
    ) -> float:
        """Find optimal temperature scaling.

        Args:
            val_data: Validation data
            val_labels: Validation labels
            verbose: Whether to print optimization details

        Returns:
            Optimal temperature
        """
        # Get raw predictions
        result = self.leak_detector.predict_batch(val_data)
        proba = np.array(result["probabilities"])

        # Get log likelihood for different temperatures
        def neg_log_likelihood(temp):
            if temp <= 0:
                return 1e10
            scaled_proba = softmax(np.log(proba) / temp, axis=1)
            # Clip to avoid log(0)
            scaled_proba = np.clip(scaled_proba, 1e-7, 1 - 1e-7)
            log_likelihood = -np.mean(np.log(scaled_proba[range(len(val_labels)), val_labels]))
            return log_likelihood

        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=np.array([1.0]),
            bounds=[(0.1, 10.0)],
            method="L-BFGS-B",
        )

        self.temperature = float(result.x[0])
        self.calibration_method = "temperature"

        if verbose:
            print(f"Optimal temperature: {self.temperature:.4f}")

        return self.temperature

    def calibrate_platt(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """Find optimal Platt scaling parameters.

        Args:
            val_data: Validation data
            val_labels: Validation labels
            verbose: Whether to print optimization details

        Returns:
            Tuple of (a, b) scaling parameters
        """
        # Get raw predictions
        result = self.leak_detector.predict_batch(val_data)
        confidences = np.array(result["confidences"])

        # Get log likelihood for different parameters
        def neg_log_likelihood(params):
            a, b = params
            scaled = expit(a * confidences + b)
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)

            # Convert binary targets to match confidences
            matches = (np.array(result["predictions"]) == val_labels).astype(float)
            log_likelihood = -np.mean(
                matches * np.log(scaled) + (1 - matches) * np.log(1 - scaled)
            )
            return log_likelihood

        # Optimize
        result_opt = minimize(
            neg_log_likelihood,
            x0=np.array([1.0, 0.0]),
            method="L-BFGS-B",
        )

        a, b = result_opt.x
        self.calibration_params = {"a": float(a), "b": float(b)}
        self.calibration_method = "platt"

        if verbose:
            print(f"Platt scaling: a={a:.4f}, b={b:.4f}")

        return float(a), float(b)

    def calibrate_isotonic(
        self,
        val_data: np.ndarray,
        val_labels: np.ndarray,
        n_bins: int = 10,
        verbose: bool = False,
    ) -> list[float]:
        """Calibrate using isotonic regression approach.

        Args:
            val_data: Validation data
            val_labels: Validation labels
            n_bins: Number of bins for calibration
            verbose: Whether to print details

        Returns:
            List of threshold values
        """
        # Get raw predictions
        result = self.leak_detector.predict_batch(val_data)
        confidences = np.array(result["confidences"])
        predictions = np.array(result["predictions"])

        # Compute accuracy in bins
        sorted_idx = np.argsort(confidences)
        sorted_confidences = confidences[sorted_idx]
        sorted_labels = val_labels[sorted_idx]
        sorted_predictions = predictions[sorted_idx]

        bin_size = len(sorted_confidences) // n_bins
        thresholds = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_confidences)

            bin_confidences = sorted_confidences[start_idx:end_idx]
            bin_labels = sorted_labels[start_idx:end_idx]
            bin_predictions = sorted_predictions[start_idx:end_idx]

            # Compute accuracy in bin
            accuracy = np.mean(bin_predictions == bin_labels)
            thresholds.append(accuracy)

            if verbose:
                print(f"Bin {i}: conf range [{bin_confidences[0]:.3f}, {bin_confidences[-1]:.3f}], accuracy={accuracy:.3f}")

        self.calibration_params = {"thresholds": thresholds}
        self.calibration_method = "isotonic"

        return thresholds

    def predict_calibrated(self, data: np.ndarray) -> Dict[str, Any]:
        """Make prediction with calibrated confidence.

        Args:
            data: Input data

        Returns:
            Prediction with calibrated confidence
        """
        result = self.leak_detector.predict_single(data)

        if self.calibration_method is None:
            return result

        # Apply calibration
        if self.calibration_method == "temperature":
            # Recalibrate using temperature
            # Note: simplified approach, ideally would work with logits
            calibrated_confidence = self._platt_scale(
                np.array([result["confidence"]]), (1.0, 0.0)
            )[0]
        elif self.calibration_method == "platt":
            a = self.calibration_params.get("a", 1.0)
            b = self.calibration_params.get("b", 0.0)
            calibrated_confidence = self._platt_scale(
                np.array([result["confidence"]]), (a, b)
            )[0]
        elif self.calibration_method == "isotonic":
            thresholds = self.calibration_params.get("thresholds", [0.5])
            calibrated_confidence = self._isotonic_regression_scale(
                np.array([result["confidence"]]), thresholds
            )[0]
        else:
            calibrated_confidence = result["confidence"]

        result["confidence_calibrated"] = float(calibrated_confidence)
        result["calibration_method"] = self.calibration_method

        return result

    def predict_batch_calibrated(self, data: np.ndarray) -> Dict[str, Any]:
        """Make batch predictions with calibrated confidence.

        Args:
            data: Batch data

        Returns:
            Batch predictions with calibrated confidences
        """
        result = self.leak_detector.predict_batch(data)

        if self.calibration_method is None:
            return result

        confidences = np.array(result["confidences"])

        # Apply calibration
        if self.calibration_method == "temperature":
            calibrated_confidences = self._platt_scale(confidences, (1.0, 0.0))
        elif self.calibration_method == "platt":
            a = self.calibration_params.get("a", 1.0)
            b = self.calibration_params.get("b", 0.0)
            calibrated_confidences = self._platt_scale(confidences, (a, b))
        elif self.calibration_method == "isotonic":
            thresholds = self.calibration_params.get("thresholds", [0.5])
            calibrated_confidences = self._isotonic_regression_scale(confidences, thresholds)
        else:
            calibrated_confidences = confidences

        result["confidences_calibrated"] = calibrated_confidences.tolist()
        result["calibration_method"] = self.calibration_method

        return result

    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information.

        Returns:
            Dictionary with calibration details
        """
        return {
            "method": self.calibration_method,
            "temperature": self.temperature,
            "parameters": self.calibration_params,
        }


class UncertaintyEstimator:
    """Estimate prediction uncertainty using various methods."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor: Optional[Any] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        """Initialize uncertainty estimator.

        Args:
            model_path: Path to trained model
            preprocessor: Preprocessor instance
            class_names: Dictionary mapping class indices to names
        """
        self.leak_detector = LeakDetector(model_path, preprocessor, class_names)

    def estimate_entropy(self, data: np.ndarray) -> Dict[str, Any]:
        """Estimate uncertainty using entropy.

        Args:
            data: Input data

        Returns:
            Dictionary with entropy-based uncertainty
        """
        result = self.leak_detector.predict_batch(data)
        proba = np.array(result["probabilities"])

        # Calculate entropy
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        max_entropy = np.log(proba.shape[1])
        normalized_entropy = entropy / max_entropy

        return {
            "entropy": entropy.tolist(),
            "normalized_entropy": normalized_entropy.tolist(),
            "mean_entropy": float(np.mean(entropy)),
            "predictions": result["predictions"],
        }

    def estimate_confidence_margin(self, data: np.ndarray) -> Dict[str, Any]:
        """Estimate uncertainty using confidence margin.

        Args:
            data: Input data

        Returns:
            Dictionary with margin-based uncertainty
        """
        result = self.leak_detector.predict_batch(data)
        proba = np.array(result["probabilities"])

        # Calculate margin (difference between top 2 probabilities)
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2]
        uncertainty = 1.0 - margin

        return {
            "margin": margin.tolist(),
            "uncertainty": uncertainty.tolist(),
            "mean_margin": float(np.mean(margin)),
            "predictions": result["predictions"],
        }

    def estimate_variance(self, data: np.ndarray, n_iterations: int = 10) -> Dict[str, Any]:
        """Estimate uncertainty using variance across predictions.

        Args:
            data: Input data
            n_iterations: Number of MC iterations

        Returns:
            Dictionary with variance-based uncertainty
        """
        result = self.leak_detector.predict_with_uncertainty(data, n_iterations)

        std_proba = np.array(result["std_probabilities"])
        prediction_variance = np.mean(std_proba, axis=1)

        return {
            "prediction_variance": prediction_variance.tolist(),
            "mean_variance": float(np.mean(prediction_variance)),
            "predictions": result["predictions"],
        }