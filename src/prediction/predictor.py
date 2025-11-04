"""Main inference class for air leak detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np


class LeakDetector:
    """Main class for single and batch inference on air leak data."""

    def __init__(
        self, model_path: str | Path, preprocessor: Optional[Any] = None, class_names: Optional[Dict[int, str]] = None
    ) -> None:
        """Initialize the leak detector.
        
        Args:
            model_path: Path to trained model file (.h5, .pkl, .joblib)
            preprocessor: Preprocessor instance for data preparation
            class_names: Dictionary mapping class indices to names
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        self.preprocessor = preprocessor
        self.class_names = class_names or {0: "No Leak", 1: "Leak 1/16", 2: "Leak 3/32", 3: "Leak 1/8"}

    def _load_model(self) -> Any:
        """Load model from disk.
        
        Returns:
            Loaded model object
        """
        if self.model_path.suffix == ".h5":
            try:
                import tensorflow as tf

                return tf.keras.models.load_model(str(self.model_path))
            except ImportError:
                raise RuntimeError("TensorFlow required for .h5 models")
        elif self.model_path.suffix in [".pkl", ".joblib"]:
            return joblib.load(str(self.model_path))
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data.
        
        Args:
            data: Raw input data
            
        Returns:
            Preprocessed data ready for model
        """
        if self.preprocessor is None:
            return data
        return self.preprocessor.process(data)

    def predict_single(self, data: np.ndarray) -> Dict[str, Any]:
        """Make prediction on single sample.
        
        Args:
            data: Single sample data (time series or FFT)
            
        Returns:
            Dictionary with predictions and confidence
        """
        # Add batch dimension if needed
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2 and data.shape[0] != 1:
            # Assume data is (channels, time) or (time, channels), take first sample
            data = data[0:1]

        preprocessed = self.preprocess(data)

        # Get prediction
        if hasattr(self.model, "predict_proba"):
            # sklearn model
            proba = self.model.predict_proba(preprocessed)[0]  # Get first sample
            prediction = self.model.predict(preprocessed)[0]
        elif hasattr(self.model, "predict"):
            # TensorFlow model
            proba = self.model.predict(preprocessed, verbose=0)[0]
            prediction = np.argmax(proba)
        else:
            raise RuntimeError("Model does not have predict method")

        confidence = float(np.max(proba))
        predicted_class = int(prediction)

        return {
            "predicted_class": predicted_class,
            "class_name": self.class_names.get(predicted_class, "Unknown"),
            "confidence": confidence,
            "probabilities": {self.class_names.get(i, f"Class {i}"): float(p) for i, p in enumerate(proba)},
        }

    def predict_batch(self, data: np.ndarray) -> Dict[str, Any]:
        """Make predictions on batch of samples.
        
        Args:
            data: Batch data (n_samples, features) or (n_samples, time, channels)
            
        Returns:
            Dictionary with batch predictions
        """
        if data.ndim < 2:
            raise ValueError("Batch data must be at least 2D")

        preprocessed = self.preprocess(data)

        # Get predictions
        if hasattr(self.model, "predict_proba"):
            # sklearn model
            proba = self.model.predict_proba(preprocessed)
            predictions = self.model.predict(preprocessed)
        elif hasattr(self.model, "predict"):
            # TensorFlow model
            proba = self.model.predict(preprocessed, verbose=0)
            predictions = np.argmax(proba, axis=1)
        else:
            raise RuntimeError("Model does not have predict method")

        confidences = np.max(proba, axis=1)

        return {
            "predictions": predictions.tolist(),
            "confidences": confidences.tolist(),
            "class_names": [self.class_names.get(int(p), "Unknown") for p in predictions],
            "probabilities": proba.tolist(),
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
        }

    def predict_with_uncertainty(self, data: np.ndarray, n_iterations: int = 10) -> Dict[str, Any]:
        """Make predictions with uncertainty estimation (MC Dropout).
        
        Args:
            data: Input data
            n_iterations: Number of MC iterations
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        predictions_list = []

        for _ in range(n_iterations):
            if hasattr(self.model, "predict_proba"):
                # sklearn: deterministic, so just repeat
                proba = self.model.predict_proba(data)
                predictions_list.append(proba)
            else:
                # TensorFlow: could use MC dropout if enabled
                proba = self.model.predict(data, verbose=0)
                predictions_list.append(proba)

        predictions_array = np.array(predictions_list)  # (n_iterations, n_samples, n_classes)

        mean_proba = np.mean(predictions_array, axis=0)
        std_proba = np.std(predictions_array, axis=0)
        mean_predictions = np.argmax(mean_proba, axis=1)

        return {
            "predictions": mean_predictions.tolist(),
            "mean_probabilities": mean_proba.tolist(),
            "std_probabilities": std_proba.tolist(),
            "class_names": [self.class_names.get(int(p), "Unknown") for p in mean_predictions],
        }

    def explain_prediction(self, data: np.ndarray) -> Dict[str, Any]:
        """Provide explanation for a prediction (basic feature importance for sklearn).
        
        Args:
            data: Input data (single sample)
            
        Returns:
            Dictionary with explanation details
        """
        prediction = self.predict_single(data)

        explanation = {
            "prediction": prediction,
            "feature_importance": None,
        }

        # Try to get feature importance if available
        if hasattr(self.model, "feature_importances_"):
            explanation["feature_importance"] = self.model.feature_importances_.tolist()
        elif hasattr(self.model, "coef_"):
            explanation["feature_importance"] = self.model.coef_.tolist()

        return explanation