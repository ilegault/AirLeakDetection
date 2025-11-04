"""
Comprehensive tests for Phase 5: Evaluation Suite

Tests cover:
- Metrics computation (accuracy, precision, recall, F1, ROC-AUC)
- Report generation (Markdown, HTML, LaTeX, JSON)
- Model comparison (statistical tests, rankings)
- Error analysis (patterns, confusions, uncertain samples)
- Visualization generation
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation import (
    ErrorAnalyzer,
    ModelComparator,
    ModelMetrics,
    ReportGenerator,
    ResultVisualizer,
)


class TestModelMetrics:
    """Test ModelMetrics class."""

    @pytest.fixture
    def simple_predictions(self):
        """Create simple predictions for testing."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 3])
        y_proba = np.array([
            [0.9, 0.05, 0.03, 0.02],
            [0.1, 0.8, 0.07, 0.03],
            [0.1, 0.05, 0.8, 0.05],
            [0.92, 0.04, 0.02, 0.02],
            [0.05, 0.85, 0.05, 0.05],
            [0.1, 0.75, 0.1, 0.05],
            [0.88, 0.05, 0.04, 0.03],
            [0.05, 0.9, 0.03, 0.02],
            [0.1, 0.1, 0.75, 0.05],
            [0.02, 0.03, 0.05, 0.9],
        ])
        return y_true, y_pred, y_proba

    def test_initialization(self, simple_predictions):
        """Test ModelMetrics initialization."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        assert len(metrics.y_true) == len(y_true)
        assert len(metrics.y_pred) == len(y_pred)
        assert metrics.n_classes == 4

    def test_accuracy(self, simple_predictions):
        """Test accuracy calculation."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)
        acc = metrics.accuracy()

        assert 0 <= acc <= 1
        assert isinstance(acc, float)
        expected_acc = np.sum(y_true == y_pred) / len(y_true)
        assert acc == pytest.approx(expected_acc)

    def test_precision(self, simple_predictions):
        """Test precision calculation."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        precision = metrics.precision(average="weighted")
        assert 0 <= precision <= 1
        assert isinstance(precision, float)

    def test_recall(self, simple_predictions):
        """Test recall calculation."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        recall = metrics.recall(average="weighted")
        assert 0 <= recall <= 1
        assert isinstance(recall, float)

    def test_f1(self, simple_predictions):
        """Test F1 score calculation."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        f1 = metrics.f1(average="weighted")
        assert 0 <= f1 <= 1
        assert isinstance(f1, float)

    def test_confusion_matrix(self, simple_predictions):
        """Test confusion matrix."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        cm = metrics.confusion_matrix()
        assert cm.shape == (4, 4)
        assert np.sum(cm) == len(y_true)

    def test_roc_auc_multiclass(self, simple_predictions):
        """Test ROC-AUC for multiclass."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        roc_auc = metrics.roc_auc()
        assert 0 <= roc_auc <= 1
        assert isinstance(roc_auc, float)

    def test_roc_auc_no_proba(self, simple_predictions):
        """Test ROC-AUC raises error without probabilities."""
        y_true, y_pred, _ = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba=None)

        with pytest.raises(ValueError):
            metrics.roc_auc()

    def test_per_class_metrics(self, simple_predictions):
        """Test per-class metrics."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        per_class = metrics.per_class_metrics()
        assert len(per_class) == 4
        for class_id in range(4):
            assert "precision" in per_class[class_id]
            assert "recall" in per_class[class_id]
            assert "f1" in per_class[class_id]

    def test_summary(self, simple_predictions):
        """Test metrics summary."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        summary = metrics.summary()
        assert "accuracy" in summary
        assert "precision_weighted" in summary
        assert "recall_weighted" in summary
        assert "f1_weighted" in summary
        assert "roc_auc" in summary
        assert "per_class_metrics" in summary

    def test_fpr_tpr(self, simple_predictions):
        """Test FPR/TPR calculation."""
        y_true, y_pred, y_proba = simple_predictions
        metrics = ModelMetrics(y_true, y_pred, y_proba)

        fpr, tpr, thresholds = metrics.fpr_tpr(class_id=1)
        assert len(fpr) > 0
        assert len(tpr) > 0
        assert len(thresholds) > 0


class TestReportGenerator:
    """Test ReportGenerator class."""

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics for testing."""
        return {
            "accuracy": 0.85,
            "precision_weighted": 0.84,
            "recall_weighted": 0.85,
            "f1_weighted": 0.84,
            "roc_auc": 0.89,
            "per_class_metrics": {
                0: {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                1: {"precision": 0.8, "recall": 0.85, "f1": 0.82},
                2: {"precision": 0.85, "recall": 0.9, "f1": 0.87},
            },
        }

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            assert gen.output_dir == Path(tmpdir)

    def test_markdown_report(self, sample_metrics):
        """Test Markdown report generation."""
        gen = ReportGenerator()
        report = gen.generate_markdown_report(
            sample_metrics,
            model_name="Test Model",
            additional_info={"dataset": "test", "epochs": 50}
        )

        assert "Test Model" in report
        assert "accuracy" in report.lower()
        assert "0.85" in report
        assert "Per-Class Metrics" in report

    def test_markdown_report_save(self, sample_metrics):
        """Test saving Markdown report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            save_path = Path(tmpdir) / "test_report.md"

            gen.generate_markdown_report(
                sample_metrics,
                model_name="Test",
                save_path=str(save_path)
            )

            assert save_path.exists()
            content = save_path.read_text()
            assert "Test" in content

    def test_html_report(self, sample_metrics):
        """Test HTML report generation."""
        gen = ReportGenerator()
        report = gen.generate_html_report(
            sample_metrics,
            model_name="Test Model",
            additional_info={"framework": "TensorFlow"}
        )

        assert "<!DOCTYPE html>" in report
        assert "Test Model" in report
        assert "Evaluation Report" in report

    def test_html_report_save(self, sample_metrics):
        """Test saving HTML report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            save_path = Path(tmpdir) / "test_report.html"

            gen.generate_html_report(
                sample_metrics,
                model_name="Test",
                save_path=str(save_path)
            )

            assert save_path.exists()
            content = save_path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_latex_tables(self, sample_metrics):
        """Test LaTeX table generation."""
        gen = ReportGenerator()
        latex = gen.generate_latex_tables(
            sample_metrics,
            model_name="Test Model"
        )

        assert "\\documentclass{article}" in latex
        assert "\\begin{tabular}" in latex
        assert "Test Model" in latex

    def test_latex_tables_save(self, sample_metrics):
        """Test saving LaTeX tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            save_path = Path(tmpdir) / "test_report.tex"

            gen.generate_latex_tables(
                sample_metrics,
                model_name="Test",
                save_path=str(save_path)
            )

            assert save_path.exists()
            content = save_path.read_text()
            assert "\\documentclass" in content

    def test_json_report(self, sample_metrics):
        """Test JSON report generation."""
        gen = ReportGenerator()
        json_str = gen.generate_json_report(
            sample_metrics,
            model_name="Test Model"
        )

        data = json.loads(json_str)
        assert data["model_name"] == "Test Model"
        assert "metrics" in data
        assert "accuracy" in data["metrics"]

    def test_json_report_save(self, sample_metrics):
        """Test saving JSON report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            save_path = Path(tmpdir) / "test_report.json"

            gen.generate_json_report(
                sample_metrics,
                model_name="Test",
                save_path=str(save_path)
            )

            assert save_path.exists()
            data = json.loads(save_path.read_text())
            assert data["model_name"] == "Test"


class TestModelComparator:
    """Test ModelComparator class."""

    @pytest.fixture
    def model_predictions(self):
        """Create predictions from multiple models."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3] * 2)

        y_pred_model1 = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 3] * 2)
        y_proba_model1 = np.tile(
            np.array([
                [0.9, 0.05, 0.03, 0.02],
                [0.1, 0.8, 0.07, 0.03],
                [0.1, 0.05, 0.8, 0.05],
                [0.92, 0.04, 0.02, 0.02],
                [0.05, 0.85, 0.05, 0.05],
                [0.1, 0.75, 0.1, 0.05],
                [0.88, 0.05, 0.04, 0.03],
                [0.05, 0.9, 0.03, 0.02],
                [0.1, 0.1, 0.75, 0.05],
                [0.02, 0.03, 0.05, 0.9],
            ]),
            (2, 1)
        )

        y_pred_model2 = np.array([0, 1, 2, 0, 0, 2, 0, 1, 2, 3] * 2)
        y_proba_model2 = np.tile(
            np.array([
                [0.95, 0.03, 0.01, 0.01],
                [0.1, 0.85, 0.03, 0.02],
                [0.05, 0.05, 0.85, 0.05],
                [0.88, 0.07, 0.03, 0.02],
                [0.8, 0.1, 0.05, 0.05],
                [0.05, 0.05, 0.9, 0.0],
                [0.92, 0.03, 0.03, 0.02],
                [0.08, 0.87, 0.03, 0.02],
                [0.1, 0.1, 0.75, 0.05],
                [0.01, 0.02, 0.02, 0.95],
            ]),
            (2, 1)
        )

        return {
            "y_true": y_true,
            "model1": (y_pred_model1, y_proba_model1),
            "model2": (y_pred_model2, y_proba_model2),
        }

    def test_initialization(self):
        """Test ModelComparator initialization."""
        comp = ModelComparator(confidence_level=0.95)
        assert comp.confidence_level == 0.95
        assert comp.alpha == pytest.approx(0.05, abs=1e-6)

    def test_add_model_results(self, model_predictions):
        """Test adding model results."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]
        y_pred, y_proba = model_predictions["model1"]

        comp.add_model_results("model1", y_true, y_pred, y_proba)
        assert "model1" in comp.model_results

    def test_compare_accuracy(self, model_predictions):
        """Test accuracy comparison."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        accuracies = comp.compare_accuracy()
        assert len(accuracies) == 2
        for model_name in ["model1", "model2"]:
            assert model_name in accuracies
            assert 0 <= accuracies[model_name] <= 1

    def test_compare_f1_scores(self, model_predictions):
        """Test F1 score comparison."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        f1_scores = comp.compare_f1_scores(average="weighted")
        assert len(f1_scores) == 2
        for model_name in ["model1", "model2"]:
            assert model_name in f1_scores

    def test_statistical_significance_test(self, model_predictions):
        """Test statistical significance testing."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        result = comp.statistical_significance_test("model1", "model2", metric="accuracy")
        assert "t_statistic" in result
        assert "p_value" in result
        assert "cohens_d" in result
        assert "significant" in result

    def test_mcnemar_test(self, model_predictions):
        """Test McNemar's test."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        result = comp.mcnemar_test("model1", "model2")
        assert "mcnemar_statistic" in result
        assert "p_value" in result
        assert "model1_unique_correct" in result

    def test_get_performance_table(self, model_predictions):
        """Test performance table generation."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        table = comp.get_performance_table()
        assert len(table) == 2
        for model_name in ["model1", "model2"]:
            assert model_name in table
            assert "accuracy" in table[model_name]
            assert "f1_weighted" in table[model_name]

    def test_rank_models(self, model_predictions):
        """Test model ranking."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        ranked = comp.rank_models(metric="accuracy")
        assert len(ranked) == 2
        assert ranked[0][0] in ["model1", "model2"]
        assert ranked[0][1] >= ranked[1][1]

    def test_get_best_model(self, model_predictions):
        """Test getting best model."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        best_name, best_score = comp.get_best_model(metric="accuracy")
        assert best_name in ["model1", "model2"]
        assert 0 <= best_score <= 1

    def test_pairwise_comparison(self, model_predictions):
        """Test pairwise model comparisons."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        pairwise = comp.pairwise_comparison(test_type="ttest")
        assert "model1" in pairwise
        assert "model2" in pairwise

    def test_summary_comparison(self, model_predictions):
        """Test summary comparison."""
        comp = ModelComparator()
        y_true = model_predictions["y_true"]

        for name, (y_pred, y_proba) in [
            ("model1", model_predictions["model1"]),
            ("model2", model_predictions["model2"]),
        ]:
            comp.add_model_results(name, y_true, y_pred, y_proba)

        summary = comp.summary_comparison()
        assert "num_models" in summary
        assert "best_model" in summary
        assert "worst_model" in summary
        assert "accuracy_range" in summary


class TestErrorAnalyzer:
    """Test ErrorAnalyzer class."""

    @pytest.fixture
    def predictions(self):
        """Create test predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 3] * 3)
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2, 3] * 3)
        y_proba = np.tile(
            np.array([
                [0.9, 0.05, 0.03, 0.02],
                [0.1, 0.8, 0.07, 0.03],
                [0.1, 0.05, 0.8, 0.05],
                [0.92, 0.04, 0.02, 0.02],
                [0.05, 0.85, 0.05, 0.05],
                [0.1, 0.75, 0.1, 0.05],
                [0.88, 0.05, 0.04, 0.03],
                [0.05, 0.9, 0.03, 0.02],
                [0.1, 0.1, 0.75, 0.05],
                [0.02, 0.03, 0.05, 0.9],
            ]),
            (3, 1)
        )
        return y_true, y_pred, y_proba

    def test_initialization(self, predictions):
        """Test ErrorAnalyzer initialization."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        assert len(analyzer.y_true) == len(y_true)
        assert analyzer.n_classes == 4

    def test_get_misclassified_indices(self, predictions):
        """Test getting misclassified indices."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        misclassified = analyzer.get_misclassified_indices()
        assert len(misclassified) > 0
        assert len(misclassified) < len(y_true)

    def test_get_correctly_classified_indices(self, predictions):
        """Test getting correctly classified indices."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        correct = analyzer.get_correctly_classified_indices()
        misclassified = analyzer.get_misclassified_indices()

        assert len(correct) + len(misclassified) == len(y_true)

    def test_get_misclassification_matrix(self, predictions):
        """Test misclassification matrix."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        matrix = analyzer.get_misclassification_matrix()
        assert matrix.shape == (4, 4)
        assert np.sum(matrix) == len(analyzer.get_misclassified_indices())

    def test_get_per_class_error_rates(self, predictions):
        """Test per-class error rates."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        error_rates = analyzer.get_per_class_error_rates()
        assert len(error_rates) == 4
        for class_id in range(4):
            assert 0 <= error_rates[class_id] <= 1

    def test_get_hardest_samples(self, predictions):
        """Test getting hardest samples."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        hard = analyzer.get_hardest_samples(n_samples=5)
        assert "total_hard_samples" in hard
        assert "hard_samples" in hard

    def test_get_uncertain_predictions(self, predictions):
        """Test getting uncertain predictions."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        uncertain = analyzer.get_uncertain_predictions(confidence_threshold=0.5)
        assert "total_uncertain" in uncertain
        assert "uncertain_error_rate" in uncertain
        assert "uncertain_predictions" in uncertain

    def test_get_error_patterns(self, predictions):
        """Test error pattern analysis."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        patterns = analyzer.get_error_patterns()
        assert "total_misclassifications" in patterns
        assert "misclassification_rate" in patterns
        assert "per_class_error_rates" in patterns
        assert "most_common_confusions" in patterns

    def test_get_class_confusion_details(self, predictions):
        """Test getting confusion details."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        details = analyzer.get_class_confusion_details(true_class=2, pred_class=1)
        assert "true_class" in details
        assert "predicted_class" in details
        assert "count" in details

    def test_get_easy_vs_hard_samples(self, predictions):
        """Test easy vs hard samples."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        easy_hard = analyzer.get_easy_vs_hard_samples(percentile=25)
        assert "hard_samples" in easy_hard
        assert "easy_samples" in easy_hard
        assert easy_hard["hard_samples"]["count"] + easy_hard["easy_samples"]["count"] == len(y_true)

    def test_get_confusion_summary(self, predictions):
        """Test confusion summary."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        summary = analyzer.get_confusion_summary()
        assert len(summary) == 4
        for class_id in range(4):
            assert "total_samples" in summary[class_id]
            assert "accuracy" in summary[class_id]

    def test_summary(self, predictions):
        """Test complete summary."""
        y_true, y_pred, y_proba = predictions
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)

        summary = analyzer.summary()
        assert "total_samples" in summary
        assert "error_count" in summary
        assert "accuracy" in summary
        assert "error_patterns" in summary
        assert "confusion_summary" in summary


class TestResultVisualizer:
    """Test ResultVisualizer class."""

    @pytest.fixture
    def visualizer_data(self):
        """Create test data for visualization."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 1, 2])
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.95, 0.03, 0.02],
            [0.1, 0.85, 0.05],
            [0.2, 0.7, 0.1],
            [0.88, 0.08, 0.04],
            [0.05, 0.9, 0.05],
            [0.1, 0.1, 0.8],
        ])
        return y_true, y_pred, y_proba

    def test_initialization(self):
        """Test ResultVisualizer initialization."""
        viz = ResultVisualizer(figsize=(12, 8))
        assert viz.figsize == (12, 8)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_confusion_matrix(self, mock_savefig, visualizer_data):
        """Test confusion matrix plot."""
        y_true, y_pred, _ = visualizer_data
        viz = ResultVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "cm.png"
            fig = viz.plot_confusion_matrix(y_true, y_pred, save_path=str(save_path))

            assert fig is not None
            ResultVisualizer.close_all()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_roc_curves(self, mock_savefig, visualizer_data):
        """Test ROC curve plot."""
        _, _, y_proba = visualizer_data
        viz = ResultVisualizer()

        fpr = np.array([0, 0.2, 0.5, 1])
        tpr = np.array([0, 0.4, 0.8, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "roc.png"
            fig = viz.plot_roc_curves([(fpr, tpr, "Class 1")], save_path=str(save_path))

            assert fig is not None
            ResultVisualizer.close_all()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_training_history(self, mock_savefig):
        """Test training history plot."""
        history = {
            "loss": [1.0, 0.8, 0.6, 0.4],
            "val_loss": [1.1, 0.9, 0.7, 0.5],
            "accuracy": [0.5, 0.6, 0.7, 0.8],
            "val_accuracy": [0.48, 0.58, 0.68, 0.78],
        }
        viz = ResultVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "history.png"
            fig = viz.plot_training_history(history, metrics=["loss", "accuracy"], save_path=str(save_path))

            assert fig is not None
            ResultVisualizer.close_all()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_metrics_comparison(self, mock_savefig):
        """Test metrics comparison plot."""
        models_metrics = {
            "model1": {"accuracy": 0.85, "precision_weighted": 0.84, "recall_weighted": 0.85, "f1_weighted": 0.84},
            "model2": {"accuracy": 0.80, "precision_weighted": 0.79, "recall_weighted": 0.80, "f1_weighted": 0.79},
        }
        viz = ResultVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "comparison.png"
            fig = viz.plot_metrics_comparison(models_metrics, save_path=str(save_path))

            assert fig is not None
            ResultVisualizer.close_all()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig):
        """Test feature importance plot."""
        importances = np.random.rand(20)
        feature_names = [f"Feature_{i}" for i in range(20)]
        viz = ResultVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "importance.png"
            fig = viz.plot_feature_importance(importances, feature_names, top_n=10, save_path=str(save_path))

            assert fig is not None
            ResultVisualizer.close_all()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_fft_comparison(self, mock_savefig):
        """Test FFT comparison plot."""
        frequencies = np.linspace(0, 1000, 100)
        magnitude_actual = np.random.rand(100)
        magnitude_expected = np.random.rand(100)
        viz = ResultVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "fft.png"
            fig = viz.plot_fft_comparison(
                frequencies, magnitude_actual, magnitude_expected, save_path=str(save_path)
            )

            assert fig is not None
            ResultVisualizer.close_all()


class TestIntegrationPhase5:
    """Integration tests for Phase 5."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # Generate test data
        np.random.seed(42)
        y_true = np.random.randint(0, 4, 100)
        y_pred = np.random.randint(0, 4, 100)
        y_proba = np.random.dirichlet([1] * 4, 100)

        # Test metrics
        metrics = ModelMetrics(y_true, y_pred, y_proba)
        summary = metrics.summary()
        assert "accuracy" in summary

        # Test report generation
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator(output_dir=tmpdir)
            gen.generate_markdown_report(summary, model_name="Test")
            gen.generate_html_report(summary, model_name="Test")
            gen.generate_json_report(summary, model_name="Test")

        # Test error analysis
        analyzer = ErrorAnalyzer(y_true, y_pred, y_proba)
        error_summary = analyzer.summary()
        assert "error_count" in error_summary

    def test_model_comparison_workflow(self):
        """Test model comparison workflow."""
        np.random.seed(42)
        y_true = np.random.randint(0, 4, 50)

        comp = ModelComparator()

        # Add multiple models
        for i in range(3):
            y_pred = np.random.randint(0, 4, 50)
            y_proba = np.random.dirichlet([1] * 4, 50)
            comp.add_model_results(f"model_{i}", y_true, y_pred, y_proba)

        # Compare
        summary = comp.summary_comparison()
        assert "num_models" in summary
        assert summary["num_models"] == 3

        # Rank
        ranked = comp.rank_models()
        assert len(ranked) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])