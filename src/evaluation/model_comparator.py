"""Model comparison and statistical significance testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score


class ModelComparator:
    """Compare multiple models with statistical significance tests."""

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize model comparator.

        Args:
            confidence_level: Confidence level for statistical tests (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.model_results: Dict[str, Dict[str, Any]] = {}

    def add_model_results(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add results from a model.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            metadata: Additional model metadata
        """
        self.model_results[model_name] = {
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "y_proba": np.asarray(y_proba) if y_proba is not None else None,
            "metadata": metadata or {},
        }

    def compare_accuracy(self) -> Dict[str, float]:
        """Compare accuracy across models.

        Returns:
            Dictionary mapping model names to accuracy scores
        """
        accuracies = {}
        for model_name, results in self.model_results.items():
            acc = accuracy_score(results["y_true"], results["y_pred"])
            accuracies[model_name] = float(acc)
        return accuracies

    def compare_f1_scores(self, average: str = "weighted") -> Dict[str, float]:
        """Compare F1 scores across models.

        Args:
            average: Type of averaging ('weighted', 'macro', 'micro')

        Returns:
            Dictionary mapping model names to F1 scores
        """
        f1_scores = {}
        for model_name, results in self.model_results.items():
            f1 = f1_score(results["y_true"], results["y_pred"], average=average, zero_division=0)
            f1_scores[model_name] = float(f1)
        return f1_scores

    def statistical_significance_test(
        self,
        model1_name: str,
        model2_name: str,
        metric: str = "accuracy",
    ) -> Dict[str, Any]:
        """Perform statistical significance test between two models.

        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            metric: Metric to compare ('accuracy' or 'f1')

        Returns:
            Dictionary with test results
        """
        if model1_name not in self.model_results or model2_name not in self.model_results:
            raise ValueError(f"Model not found in results")

        results1 = self.model_results[model1_name]
        results2 = self.model_results[model2_name]

        # Get predictions
        y_true = results1["y_true"]
        y_pred1 = results1["y_pred"]
        y_pred2 = results2["y_pred"]

        # Calculate per-sample metrics
        if metric == "accuracy":
            scores1 = (y_pred1 == y_true).astype(int)
            scores2 = (y_pred2 == y_true).astype(int)
        elif metric == "f1":
            # For F1, use per-class scores
            n_classes = len(np.unique(y_true))
            scores1 = np.zeros(len(y_true))
            scores2 = np.zeros(len(y_true))

            for i in range(len(y_true)):
                # Use binary F1 for each sample
                scores1[i] = 1.0 if y_pred1[i] == y_true[i] else 0.0
                scores2[i] = 1.0 if y_pred2[i] == y_true[i] else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Calculate effect size (Cohen's d)
        diff = scores1 - scores2
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < self.alpha,
            "mean_score1": float(np.mean(scores1)),
            "mean_score2": float(np.mean(scores2)),
            "score_difference": float(np.mean(scores1) - np.mean(scores2)),
        }

    def mcnemar_test(
        self,
        model1_name: str,
        model2_name: str,
    ) -> Dict[str, Any]:
        """Perform McNemar's test between two models.

        Args:
            model1_name: Name of first model
            model2_name: Name of second model

        Returns:
            Dictionary with test results
        """
        if model1_name not in self.model_results or model2_name not in self.model_results:
            raise ValueError("Model not found in results")

        results1 = self.model_results[model1_name]
        results2 = self.model_results[model2_name]

        y_true = results1["y_true"]
        y_pred1 = results1["y_pred"]
        y_pred2 = results2["y_pred"]

        # Create contingency table
        correct1 = (y_pred1 == y_true).astype(int)
        correct2 = (y_pred2 == y_true).astype(int)

        # Cases where models disagree
        # b: model1 correct, model2 incorrect
        # c: model1 incorrect, model2 correct
        b = np.sum((correct1 == 1) & (correct2 == 0))
        c = np.sum((correct1 == 0) & (correct2 == 1))

        # McNemar's statistic
        mcnemar_stat = (b - c) ** 2 / (b + c + 1e-8)
        p_value = 1.0 - stats.chi2.cdf(mcnemar_stat, df=1)

        return {
            "mcnemar_statistic": float(mcnemar_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "model1_unique_correct": int(b),
            "model2_unique_correct": int(c),
        }

    def get_performance_table(self) -> Dict[str, Dict[str, float]]:
        """Get a performance comparison table.

        Returns:
            Dictionary with model names as keys and metric dictionaries as values
        """
        table = {}
        for model_name, results in self.model_results.items():
            y_true = results["y_true"]
            y_pred = results["y_pred"]

            accuracy = accuracy_score(y_true, y_pred)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

            table[model_name] = {
                "accuracy": float(accuracy),
                "f1_weighted": float(f1_weighted),
                "f1_macro": float(f1_macro),
            }

        return table

    def rank_models(self, metric: str = "accuracy") -> List[Tuple[str, float]]:
        """Rank models by performance.

        Args:
            metric: Metric to use for ranking ('accuracy', 'f1_weighted', 'f1_macro')

        Returns:
            List of (model_name, score) tuples sorted by score descending
        """
        scores = {}

        if metric == "accuracy":
            scores = self.compare_accuracy()
        elif metric == "f1_weighted":
            scores = self.compare_f1_scores(average="weighted")
        elif metric == "f1_macro":
            scores = self.compare_f1_scores(average="macro")
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def get_best_model(self, metric: str = "accuracy") -> Tuple[str, float]:
        """Get the best performing model.

        Args:
            metric: Metric to use for selection ('accuracy', 'f1_weighted', 'f1_macro')

        Returns:
            Tuple of (model_name, score)
        """
        ranked = self.rank_models(metric=metric)
        if not ranked:
            raise ValueError("No models to compare")
        return ranked[0]

    def get_worst_model(self, metric: str = "accuracy") -> Tuple[str, float]:
        """Get the worst performing model.

        Args:
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, score)
        """
        ranked = self.rank_models(metric=metric)
        if not ranked:
            raise ValueError("No models to compare")
        return ranked[-1]

    def pairwise_comparison(
        self,
        metric: str = "accuracy",
        test_type: str = "ttest",
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Perform pairwise comparisons between all models.

        Args:
            metric: Metric to compare
            test_type: Type of test ('ttest' or 'mcnemar')

        Returns:
            Dictionary with pairwise test results
        """
        model_names = list(self.model_results.keys())
        results = {name: {} for name in model_names}

        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    results[model1][model2] = {"comparison": "same_model"}
                elif i < j:  # Only compute once
                    if test_type == "ttest":
                        test_result = self.statistical_significance_test(model1, model2, metric=metric)
                    elif test_type == "mcnemar":
                        test_result = self.mcnemar_test(model1, model2)
                    else:
                        raise ValueError(f"Unknown test type: {test_type}")

                    results[model1][model2] = test_result
                    results[model2][model1] = test_result

        return results

    def summary_comparison(self) -> Dict[str, Any]:
        """Get a summary comparison of all models.

        Returns:
            Dictionary with comparison summary
        """
        perf_table = self.get_performance_table()
        best_model, best_score = self.get_best_model(metric="accuracy")
        worst_model, worst_score = self.get_worst_model(metric="accuracy")

        return {
            "num_models": len(self.model_results),
            "model_names": list(self.model_results.keys()),
            "performance_table": perf_table,
            "best_model": best_model,
            "best_accuracy": best_score,
            "worst_model": worst_model,
            "worst_accuracy": worst_score,
            "accuracy_range": best_score - worst_score,
        }