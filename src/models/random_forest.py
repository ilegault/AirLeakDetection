"""Random Forest classifier utilities for air leak detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass
class RandomForestConfig:
    """Container for Random Forest hyperparameters."""

    n_estimators: int
    max_depth: int | None
    min_samples_split: int
    min_samples_leaf: int
    max_features: str | int | float | None
    random_state: int
    n_jobs: int


class RandomForestModel:
    """Train and persist Random Forest models for vibration features."""

    def __init__(self, config: Dict[str, Any]) -> None:
        model_cfg = config.get("model", {}).get("random_forest", {})

        self.params = RandomForestConfig(
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            max_depth=model_cfg.get("max_depth"),
            min_samples_split=int(model_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
            max_features=model_cfg.get("max_features", "sqrt"),
            random_state=int(model_cfg.get("random_state", 42)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
        )

        self.model = RandomForestClassifier(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            min_samples_split=self.params.min_samples_split,
            min_samples_leaf=self.params.min_samples_leaf,
            max_features=self.params.max_features,
            random_state=self.params.random_state,
            n_jobs=self.params.n_jobs,
        )

    # ------------------------------------------------------------------
    # Training & inference
    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the classifier using extracted features."""
        if features.ndim != 2:
            raise ValueError("`features` must be a 2D array of shape (n_samples, n_features)")
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict discrete class labels."""
        return self.model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(features)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Return (accuracy, oob_score if enabled)."""
        accuracy = float(self.model.score(features, labels))
        oob_score = float(getattr(self.model, "oob_score_", float("nan")))
        return accuracy, oob_score

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize the model to disk."""
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> RandomForestClassifier:
        """Load a previously persisted Random Forest classifier."""
        return joblib.load(path)