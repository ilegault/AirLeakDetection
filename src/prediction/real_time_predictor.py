"""Real-time inference for streaming air leak detection data."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.prediction.predictor import LeakDetector


class RealTimePredictor:
    """Real-time predictor with sliding window approach for streaming data."""

    def __init__(
        self,
        model_path: str | Path,
        window_size: int = 1024,
        stride: int = 512,
        preprocessor: Optional[Any] = None,
        class_names: Optional[Dict[int, str]] = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize real-time predictor.

        Args:
            model_path: Path to trained model file
            window_size: Size of sliding window in samples
            stride: Number of samples to shift window
            preprocessor: Preprocessor instance
            class_names: Dictionary mapping class indices to names
            confidence_threshold: Minimum confidence for accepting predictions
        """
        self.leak_detector = LeakDetector(model_path, preprocessor, class_names)
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=window_size)
        self.prediction_history = deque(maxlen=10)  # Keep last 10 predictions
        self.callbacks: list[Callable] = []

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for predictions.

        Args:
            callback: Function to call when prediction is made
        """
        self.callbacks.append(callback)

    def _invoke_callbacks(self, prediction: Dict[str, Any]) -> None:
        """Invoke all registered callbacks.

        Args:
            prediction: Prediction result dictionary
        """
        for callback in self.callbacks:
            try:
                callback(prediction)
            except Exception as e:
                print(f"Error in callback: {e}")

    def add_sample(self, sample: np.ndarray) -> Optional[Dict[str, Any]]:
        """Add a single sample to the buffer and predict if window is full.

        Args:
            sample: Single sample data point

        Returns:
            Prediction result if window is ready, None otherwise
        """
        self.buffer.append(sample)

        # Check if we have enough samples for a prediction
        if len(self.buffer) == self.window_size:
            return self.predict_from_buffer()
        return None

    def add_samples(self, samples: np.ndarray) -> list[Dict[str, Any]]:
        """Add multiple samples and return predictions.

        Args:
            samples: Array of samples (n_samples, features)

        Returns:
            List of predictions
        """
        predictions = []
        for sample in samples:
            pred = self.add_sample(sample)
            if pred is not None:
                predictions.append(pred)
                self._invoke_callbacks(pred)
        return predictions

    def predict_from_buffer(self) -> Dict[str, Any]:
        """Make prediction from current buffer contents.

        Returns:
            Prediction result dictionary
        """
        if len(self.buffer) < self.window_size:
            raise ValueError(f"Buffer must have at least {self.window_size} samples")

        # Convert buffer to array - keep the structure of samples
        window_data = np.array(list(self.buffer))
        
        # Handle different input shapes:
        # If 1D: single feature stream, reshape to (1, window_size)
        # If 2D: (window_size, channels), use as-is but add batch dim: (1, window_size, channels)
        # If 3D or more: flatten to 2D and use first sample
        if window_data.ndim == 1:
            # Single channel stream
            window_data = window_data.reshape(1, -1)
        elif window_data.ndim == 2:
            # Already has structure, add batch dimension
            window_data = window_data.reshape(1, window_data.shape[0], window_data.shape[1])
            # For sklearn models, reshape to 2D: take mean across window
            window_data = window_data.mean(axis=1)  # Average across time window
        else:
            # Higher dimensions - flatten to 2D
            window_data = window_data.reshape(1, -1)

        # Get prediction
        pred = self.leak_detector.predict_single(window_data)
        self.prediction_history.append(pred)

        return pred

    def get_ensemble_prediction(self) -> Dict[str, Any]:
        """Get ensemble prediction from history.

        Returns:
            Ensemble prediction combining recent predictions
        """
        if not self.prediction_history:
            raise ValueError("No prediction history available")

        # Average probabilities across history
        probs_list = []
        for pred in self.prediction_history:
            probs_list.append(list(pred["probabilities"].values()))

        avg_probs = np.mean(probs_list, axis=0)
        ensemble_prediction = np.argmax(avg_probs)
        ensemble_confidence = np.max(avg_probs)

        class_names_list = list(self.leak_detector.class_names.values())

        return {
            "predicted_class": int(ensemble_prediction),
            "class_name": class_names_list[ensemble_prediction],
            "confidence": float(ensemble_confidence),
            "probabilities": {
                class_names_list[i]: float(p) for i, p in enumerate(avg_probs)
            },
            "n_predictions_averaged": len(self.prediction_history),
        }

    def reset(self) -> None:
        """Reset the predictor buffer and history."""
        self.buffer.clear()
        self.prediction_history.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the predictor.

        Returns:
            Dictionary with status information
        """
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.window_size,
            "buffer_full": len(self.buffer) == self.window_size,
            "history_size": len(self.prediction_history),
            "confidence_threshold": self.confidence_threshold,
        }


class StreamingDataProcessor:
    """Process streaming data with sliding windows."""

    def __init__(self, window_size: int = 1024, stride: int = 512, n_channels: int = 9) -> None:
        """Initialize streaming data processor.

        Args:
            window_size: Size of sliding window
            stride: Stride for sliding window
            n_channels: Number of channels in data
        """
        self.window_size = window_size
        self.stride = stride
        self.n_channels = n_channels
        self.buffer = deque(maxlen=window_size)

    def process_stream(self, data: np.ndarray) -> list[np.ndarray]:
        """Process streaming data with sliding windows.

        Args:
            data: Input data stream (n_samples, n_channels) or (n_samples,)

        Returns:
            List of windowed samples
        """
        windows = []

        for i in range(0, len(data), self.stride):
            self.buffer.extend(data[i : i + self.stride])

            if len(self.buffer) == self.window_size:
                windows.append(np.array(list(self.buffer)).copy())

        return windows

    def reset(self) -> None:
        """Reset the buffer."""
        self.buffer.clear()