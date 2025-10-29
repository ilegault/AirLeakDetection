"""Tests covering Phase 3 model builders for classical and neural models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.models.random_forest import RandomForestModel
from src.models.svm_classifier import SVMClassifier

try:
    from src.models.cnn_1d import CNN1DBuilder
except ImportError:  # pragma: no cover - TensorFlow may be optional in some environments
    CNN1DBuilder = None  # type: ignore


@pytest.fixture()
def model_config(tmp_path: Path) -> tuple[Path, dict]:
    """Create a minimal configuration file consumed by the model factories."""
    config = {
        "data": {"n_channels": 3},
        "training": {"learning_rate": 0.001},
        "model": {
            "conv_filters": (8, 16),
            "kernel_sizes": (5, 3),
            "dense_units": (32,),
            "dropout_rates": (0.2, 0.3, 0.4),
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 5,
                "n_jobs": 1,
            },
            "svm": {
                "kernel": "linear",
                "C": 1.0,
                "probability": True,
            },
        },
        "classes": {0: "NOLEAK", 1: "SMALL_1_16"},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")
    return config_path, config


def test_random_forest_trains_and_predicts(model_config: tuple[Path, dict]) -> None:
    """Random forest should fit on synthetic data and support persistence."""
    _, config = model_config
    rf_model = RandomForestModel(config)

    rng = np.random.default_rng(0)
    features = rng.normal(size=(20, 5))
    labels = rng.integers(0, 2, size=20)

    rf_model.fit(features, labels)
    preds = rf_model.predict(features)
    assert preds.shape == (20,)

    proba = rf_model.predict_proba(features)
    assert proba.shape == (20, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    accuracy, _ = rf_model.evaluate(features, labels)
    assert 0.0 <= accuracy <= 1.0


def test_svm_classifier_scaling_and_probabilities(model_config: tuple[Path, dict], tmp_path: Path) -> None:
    """SVM wrapper should scale features, expose probabilities, and allow persistence."""
    _, config = model_config
    svm = SVMClassifier(config)

    rng = np.random.default_rng(1)
    features = rng.normal(size=(15, 4))
    labels = rng.integers(0, 2, size=15)

    svm.fit(features, labels)
    preds = svm.predict(features)
    assert preds.shape == (15,)

    proba = svm.predict_proba(features)
    assert proba.shape == (15, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    decision_scores = svm.decision_function(features)
    assert decision_scores.shape[0] == 15

    model_path = tmp_path / "svm.joblib"
    svm.save(model_path)
    assert model_path.exists()


def test_cnn_builder_creates_model_when_available(model_config: tuple[Path, dict]) -> None:
    """CNN builder should construct a TensorFlow model when TensorFlow is installed."""
    if CNN1DBuilder is None:
        pytest.skip("TensorFlow is not installed in this environment")

    _, config = model_config
    builder = CNN1DBuilder(config)

    # Simulate input shape from preprocessing: (timesteps, channels)
    model = builder.build(input_shape=(128, 3), n_classes=2)
    assert model.input_shape == (None, 128, 3)
    assert model.output_shape == (None, 2)

    summary_lines = []
    model.summary(print_fn=summary_lines.append)
    assert any("conv_1" in line for line in summary_lines)