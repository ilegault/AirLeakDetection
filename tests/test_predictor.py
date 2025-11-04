"""Tests for prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.random_forest import RandomForestModel
from src.prediction.predictor import LeakDetector


class TestLeakDetector:
    """Test suite for LeakDetector class."""

    @pytest.fixture
    def trained_model_path(self, tmp_path: Path) -> Path:
        """Create and save a trained model."""
        import yaml

        config = {
            "model": {
                "random_forest": {
                    "n_estimators": 10,
                    "max_depth": 5,
                    "n_jobs": 1,
                }
            }
        }

        # Train a simple model
        rf_model = RandomForestModel(config)
        rng = np.random.default_rng(0)
        features = rng.normal(size=(20, 5))
        labels = rng.integers(0, 4, size=20)
        rf_model.fit(features, labels)

        # Save model
        model_path = tmp_path / "model.pkl"
        rf_model.save(model_path)

        return model_path

    @pytest.fixture
    def class_names(self) -> dict[int, str]:
        """Standard class names for air leak detection."""
        return {0: "No Leak", 1: "Leak 1/16", 2: "Leak 3/32", 3: "Leak 1/8"}

    def test_detector_initialization(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test LeakDetector initialization."""
        detector = LeakDetector(trained_model_path, class_names=class_names)
        assert detector.model is not None
        assert detector.class_names == class_names

    def test_predict_single_sample(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test single sample prediction."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        sample = np.random.randn(5).astype(np.float32)
        result = detector.predict_single(sample)

        assert "predicted_class" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "probabilities" in result

        assert 0 <= result["predicted_class"] < 4
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["probabilities"]) == 4

    def test_predict_batch(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test batch prediction."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        batch = np.random.randn(10, 5).astype(np.float32)
        result = detector.predict_batch(batch)

        assert "predictions" in result
        assert "confidences" in result
        assert "class_names" in result
        assert "probabilities" in result
        assert "mean_confidence" in result
        assert "std_confidence" in result

        assert len(result["predictions"]) == 10
        assert len(result["confidences"]) == 10
        assert len(result["class_names"]) == 10
        assert len(result["probabilities"]) == 10

    def test_predict_with_uncertainty(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test uncertainty estimation."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        sample = np.random.randn(1, 5).astype(np.float32)
        result = detector.predict_with_uncertainty(sample, n_iterations=5)

        assert "predictions" in result
        assert "mean_probabilities" in result
        assert "std_probabilities" in result
        assert "class_names" in result

        assert len(result["predictions"]) == 1

    def test_explain_prediction(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test prediction explanation."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        sample = np.random.randn(5).astype(np.float32)
        explanation = detector.explain_prediction(sample)

        assert "prediction" in explanation
        assert "feature_importance" in explanation

    def test_preprocess_2d_batch(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test preprocessing of 2D batch data."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        # Test with 2D array (n_samples, features)
        batch = np.random.randn(10, 5)
        result = detector.predict_batch(batch)

        assert len(result["predictions"]) == 10

    def test_invalid_model_format_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid model format raises error."""
        model_path = tmp_path / "model.xyz"
        model_path.touch()

        with pytest.raises(ValueError, match="Unsupported model format"):
            LeakDetector(model_path)

    def test_batch_prediction_consistency(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test that batch predictions match individual predictions."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        sample = np.random.randn(5).astype(np.float32)

        # Single prediction
        single_result = detector.predict_single(sample)

        # Batch prediction with single sample
        batch_result = detector.predict_batch(np.array([sample]))

        assert single_result["predicted_class"] == batch_result["predictions"][0]
        assert np.isclose(single_result["confidence"], batch_result["confidences"][0])

    def test_edge_case_batch_1d_input(self, trained_model_path: Path, class_names: dict[int, str]) -> None:
        """Test that batch prediction rejects 1D input."""
        detector = LeakDetector(trained_model_path, class_names=class_names)

        with pytest.raises(ValueError, match="must be at least 2D"):
            detector.predict_batch(np.random.randn(5))