"""Ensemble models combining multiple classifiers for robust predictions."""

from __future__ import annotations

from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier


class EnsembleModel:
    """Ensemble classifier combining multiple base models using voting."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ensemble configuration.
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.models: List[tuple[str, Any]] = []
        self.ensemble = None
        self.is_fitted = False

    def add_model(self, name: str, model: Any) -> None:
        """Add a base model to the ensemble.
        
        Args:
            name: Unique name for the model
            model: Fitted sklearn model with predict() and predict_proba() methods
        """
        self.models.append((name, model))

    def build(self, voting: str = "soft") -> None:
        """Build the ensemble classifier.
        
        Args:
            voting: 'hard' for majority voting, 'soft' for probability averaging
        """
        if len(self.models) == 0:
            raise ValueError("No models added to ensemble. Use add_model() first.")

        self.ensemble = VotingClassifier(estimators=self.models, voting=voting)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit the ensemble on training data.
        
        Args:
            features: Training features (n_samples, n_features)
            labels: Training labels
        """
        if self.ensemble is None:
            raise RuntimeError("Ensemble not built. Call build() first.")

        if features.ndim != 2:
            raise ValueError("`features` must be a 2D array of shape (n_samples, n_features)")

        self.ensemble.fit(features, labels)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            features: Features to predict (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        return self.ensemble.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            features: Features to predict
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        return self.ensemble.predict_proba(features)

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, Dict[str, float]]:
        """Evaluate ensemble performance.
        
        Args:
            features: Test features
            labels: True labels
            
        Returns:
            Tuple of (accuracy, detailed_metrics_dict)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before evaluation")

        accuracy = float(self.ensemble.score(features, labels))
        predictions = self.predict(features)

        # Calculate per-model accuracy
        per_model_accuracy = {}
        for name, model in self.models:
            model_pred = model.predict(features)
            model_acc = np.mean(model_pred == labels)
            per_model_accuracy[name] = float(model_acc)

        metrics = {"accuracy": accuracy, "per_model_accuracy": per_model_accuracy}
        return accuracy, metrics

    def save(self, path: str) -> None:
        """Save ensemble to disk.
        
        Args:
            path: File path to save to
        """
        joblib.dump(self.ensemble, path)

    @classmethod
    def load(cls, path: str) -> VotingClassifier:
        """Load a persisted ensemble.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded VotingClassifier
        """
        return joblib.load(path)


class StackingEnsemble:
    """Stacking ensemble using a meta-learner to combine predictions."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize stacking ensemble.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.base_models: List[tuple[str, Any]] = []
        self.meta_model = None
        self.is_fitted = False

    def add_base_model(self, name: str, model: Any) -> None:
        """Add a base model.
        
        Args:
            name: Model name
            model: Model instance
        """
        self.base_models.append((name, model))

    def set_meta_model(self, meta_model: Any) -> None:
        """Set the meta-learner.
        
        Args:
            meta_model: Meta-learner model (typically simpler, like logistic regression)
        """
        self.meta_model = meta_model

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit stacking ensemble.
        
        Args:
            features: Training features
            labels: Training labels
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models added")
        if self.meta_model is None:
            raise ValueError("Meta-model not set")

        # Generate meta-features from base models
        meta_features_list = []
        for name, model in self.base_models:
            if hasattr(model, "predict_proba"):
                meta_features_list.append(model.predict_proba(features))
            else:
                meta_features_list.append(model.predict(features).reshape(-1, 1))

        meta_features = np.hstack(meta_features_list)

        # Fit meta-learner
        self.meta_model.fit(meta_features, labels)
        self.is_fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using stacking ensemble.
        
        Args:
            features: Features to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Stacking ensemble must be fitted first")

        meta_features_list = []
        for name, model in self.base_models:
            if hasattr(model, "predict_proba"):
                meta_features_list.append(model.predict_proba(features))
            else:
                meta_features_list.append(model.predict(features).reshape(-1, 1))

        meta_features = np.hstack(meta_features_list)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probabilities using stacking ensemble.
        
        Args:
            features: Features to predict
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Stacking ensemble must be fitted first")

        meta_features_list = []
        for name, model in self.base_models:
            if hasattr(model, "predict_proba"):
                meta_features_list.append(model.predict_proba(features))
            else:
                meta_features_list.append(model.predict(features).reshape(-1, 1))

        meta_features = np.hstack(meta_features_list)

        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        else:
            raise RuntimeError("Meta-model does not support predict_proba")