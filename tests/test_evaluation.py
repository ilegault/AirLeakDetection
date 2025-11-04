"""Tests for evaluation metrics and visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.evaluation.metrics import ModelMetrics
from src.evaluation.visualizer import ResultVisualizer


class TestModelMetrics:
    """Test suite for ModelMetrics class."""

    @pytest.fixture
    def sample_predictions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sample predictions for testing."""
        np.random.seed(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 0, 2, 0])
        y_proba = np.random.dirichlet([1, 1, 1], 10)
        return y_true, y_pred, y_proba

    def test_accuracy_calculation(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test accuracy metric."""
        y_true, y_pred, _ = sample_predictions
        metrics = ModelMetrics(y_true, y_pred)

        accuracy = metrics.accuracy()
        assert 0.0 <= accuracy <= 1.0
        assert accuracy == 0.8  # 8 correct out of 10

    def test_precision_recall_f1(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test precision, recall, and F1 scores."""
        y_true, y_pred, _ = sample_predictions
        metrics = ModelMetrics(y_true, y_pred)

        precision = metrics.precision(average="weighted")
        recall = metrics.recall(average="weighted")
        f1 = metrics.f1(average="weighted")

        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_confusion_matrix(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test confusion matrix."""
        y_true, y_pred, _ = sample_predictions
        metrics = ModelMetrics(y_true, y_pred)

        cm = metrics.confusion_matrix()
        assert cm.shape == (3, 3)
        assert np.sum(cm) == len(y_true)

    def test_per_class_metrics(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test per-class metrics."""
        y_true, y_pred, _ = sample_predictions
        metrics = ModelMetrics(y_true, y_pred)

        per_class = metrics.per_class_metrics()
        assert len(per_class) == 3
        assert all("precision" in per_class[i] for i in range(3))
        assert all("recall" in per_class[i] for i in range(3))
        assert all("f1" in per_class[i] for i in range(3))

    def test_roc_auc_with_probabilities(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test ROC-AUC calculation."""
        y_true, y_pred, y_proba = sample_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        auc = metrics.roc_auc()
        assert 0.0 <= auc <= 1.0

    def test_roc_auc_requires_probabilities(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test that ROC-AUC raises error without probabilities."""
        y_true, y_pred, _ = sample_predictions
        metrics = ModelMetrics(y_true, y_pred)

        with pytest.raises(ValueError):
            metrics.roc_auc()

    def test_summary(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test complete metrics summary."""
        y_true, y_pred, y_proba = sample_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        summary = metrics.summary()
        assert "accuracy" in summary
        assert "precision_weighted" in summary
        assert "recall_weighted" in summary
        assert "f1_weighted" in summary
        assert "per_class_metrics" in summary
        assert "roc_auc" in summary

    def test_fpr_tpr(self, sample_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Test FPR/TPR calculation."""
        y_true, y_pred, y_proba = sample_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        fpr, tpr, thresholds = metrics.fpr_tpr(class_id=1)
        assert len(fpr) == len(tpr) == len(thresholds)


class TestResultVisualizer:
    """Test suite for ResultVisualizer class."""

    @pytest.fixture(autouse=True)
    def cleanup_plots(self) -> None:
        """Clean up plots after each test."""
        yield
        plt.close("all")

    @pytest.fixture
    def sample_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Create sample predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 0, 2, 0])
        return y_true, y_pred

    def test_visualizer_initialization(self) -> None:
        """Test visualizer initialization."""
        viz = ResultVisualizer(figsize=(10, 6))
        assert viz.figsize == (10, 6)

    def test_confusion_matrix_plot(self, sample_predictions: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
        """Test confusion matrix visualization."""
        y_true, y_pred = sample_predictions
        viz = ResultVisualizer()

        fig = viz.plot_confusion_matrix(y_true, y_pred, class_names=["A", "B", "C"])
        assert fig is not None

        # Test save
        save_path = tmp_path / "confusion_matrix.png"
        fig = viz.plot_confusion_matrix(y_true, y_pred, save_path=str(save_path))
        assert save_path.exists()

    def test_roc_curves_plot(self, tmp_path: Path) -> None:
        """Test ROC curve visualization."""
        viz = ResultVisualizer()

        fpr1 = np.array([0.0, 0.2, 0.5, 1.0])
        tpr1 = np.array([0.0, 0.3, 0.7, 1.0])
        fpr2 = np.array([0.0, 0.3, 0.6, 1.0])
        tpr2 = np.array([0.0, 0.4, 0.8, 1.0])

        fig = viz.plot_roc_curves([(fpr1, tpr1, "Model 1"), (fpr2, tpr2, "Model 2")])
        assert fig is not None

        save_path = tmp_path / "roc_curves.png"
        fig = viz.plot_roc_curves([(fpr1, tpr1, "Model 1")], save_path=str(save_path))
        assert save_path.exists()

    def test_training_history_plot(self, tmp_path: Path) -> None:
        """Test training history visualization."""
        viz = ResultVisualizer()

        history = {
            "loss": [0.9, 0.7, 0.5, 0.3],
            "val_loss": [0.95, 0.75, 0.55, 0.4],
            "accuracy": [0.5, 0.6, 0.7, 0.8],
            "val_accuracy": [0.45, 0.55, 0.65, 0.75],
        }

        fig = viz.plot_training_history(history, metrics=["loss", "accuracy"])
        assert fig is not None

        save_path = tmp_path / "history.png"
        fig = viz.plot_training_history(history, save_path=str(save_path))
        assert save_path.exists()

    def test_metrics_comparison_plot(self, tmp_path: Path) -> None:
        """Test metrics comparison visualization."""
        viz = ResultVisualizer()

        models_metrics = {
            "Model A": {"accuracy": 0.85, "precision_weighted": 0.84, "recall_weighted": 0.85, "f1_weighted": 0.84},
            "Model B": {"accuracy": 0.90, "precision_weighted": 0.89, "recall_weighted": 0.90, "f1_weighted": 0.89},
        }

        fig = viz.plot_metrics_comparison(models_metrics)
        assert fig is not None

        save_path = tmp_path / "comparison.png"
        fig = viz.plot_metrics_comparison(models_metrics, save_path=str(save_path))
        assert save_path.exists()

    def test_feature_importance_plot(self, tmp_path: Path) -> None:
        """Test feature importance visualization."""
        viz = ResultVisualizer()

        importances = np.array([0.05, 0.1, 0.2, 0.15, 0.3, 0.08, 0.12])
        feature_names = [f"Feature {i}" for i in range(7)]

        fig = viz.plot_feature_importance(importances, feature_names, top_n=5)
        assert fig is not None

        save_path = tmp_path / "importance.png"
        fig = viz.plot_feature_importance(importances, feature_names, save_path=str(save_path))
        assert save_path.exists()

    def test_fft_comparison_plot(self, tmp_path: Path) -> None:
        """Test FFT comparison visualization."""
        viz = ResultVisualizer()

        frequencies = np.linspace(0, 100, 50)
        magnitude_actual = np.abs(np.sin(frequencies))
        magnitude_expected = np.abs(np.sin(frequencies + 0.1))

        fig = viz.plot_fft_comparison(frequencies, magnitude_actual, magnitude_expected)
        assert fig is not None

        save_path = tmp_path / "fft.png"
        fig = viz.plot_fft_comparison(
            frequencies, magnitude_actual, magnitude_expected, save_path=str(save_path)
        )
        assert save_path.exists()

    def test_close_all_figures(self) -> None:
        """Test closing all figures."""
        plt.figure()
        plt.figure()
        plt.figure()

        ResultVisualizer.close_all()
        assert len(plt.get_fignums()) == 0