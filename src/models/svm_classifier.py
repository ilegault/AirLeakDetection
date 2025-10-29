"""Support Vector Machine utilities for air leak detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class SVMConfig:
    """Hyperparameters for SVM classifier."""

    kernel: str
    c_value: float
    gamma: str | float
    probability: bool
    random_state: int
    class_weight: str | Dict[int, float] | None


class SVMClassifier:
    """Wrapper around scikit-learn's SVC with feature scaling and persistence helpers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        model_cfg = config.get("model", {}).get("svm", {})

        self.params = SVMConfig(
            kernel=str(model_cfg.get("kernel", "rbf")),
            c_value=float(model_cfg.get("C", 10.0)),
            gamma=model_cfg.get("gamma", "scale"),
            probability=bool(model_cfg.get("probability", True)),
            random_state=int(model_cfg.get("random_state", 42)),
            class_weight=model_cfg.get("class_weight"),
        )

        self.scaler = StandardScaler()
        self.model = SVC(
            kernel=self.params.kernel,
            C=self.params.c_value,
            gamma=self.params.gamma,
            probability=self.params.probability,
            random_state=self.params.random_state,
            class_weight=self.params.class_weight,
        )

        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the scaler and SVM model."""
        if features.ndim != 2:
            raise ValueError("`features` must be a 2D array of shape (n_samples, n_features)")

        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        scaled = self._ensure_scaled(features)
        return self.model.predict(scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities if enabled."""
        if not self.params.probability:
            raise RuntimeError("Probability estimates were disabled during initialization")
        scaled = self._ensure_scaled(features)
        return self.model.predict_proba(scaled)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return distance to decision boundary for each class."""
        scaled = self._ensure_scaled(features)
        return self.model.decision_function(scaled)

    def _ensure_scaled(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.scaler.transform(features)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist scaler and model to disk."""
        payload = {"scaler": self.scaler, "model": self.model}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> Tuple[StandardScaler, SVC]:
        """Load a persisted scaler + SVM bundle."""
        payload = joblib.load(path)
        return payload["scaler"], payload["model"]