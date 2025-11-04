"""Error analysis and misclassification investigation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from collections import Counter


class ErrorAnalyzer:
    """Analyze misclassifications and error patterns."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> None:
        """Initialize error analyzer.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None
        self.n_classes = len(np.unique(self.y_true))

        # Identify errors
        self.error_mask = self.y_true != self.y_pred
        self.correct_mask = self.y_true == self.y_pred

    def get_misclassified_indices(self) -> np.ndarray:
        """Get indices of misclassified samples.

        Returns:
            Array of indices where prediction is incorrect
        """
        return np.where(self.error_mask)[0]

    def get_correctly_classified_indices(self) -> np.ndarray:
        """Get indices of correctly classified samples.

        Returns:
            Array of indices where prediction is correct
        """
        return np.where(self.correct_mask)[0]

    def get_misclassification_matrix(self) -> np.ndarray:
        """Get misclassification matrix.

        Returns:
            Matrix where entry [i, j] is count of samples labeled i but predicted j
        """
        matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)

        for true_label, pred_label in zip(self.y_true[self.error_mask], self.y_pred[self.error_mask]):
            matrix[true_label, pred_label] += 1

        return matrix

    def get_per_class_error_rates(self) -> Dict[int, float]:
        """Get error rate per class.

        Returns:
            Dictionary mapping class ID to error rate
        """
        error_rates = {}

        for class_id in range(self.n_classes):
            mask = self.y_true == class_id
            if np.sum(mask) == 0:
                error_rates[class_id] = 0.0
            else:
                errors = np.sum(self.error_mask & mask)
                error_rates[class_id] = float(errors / np.sum(mask))

        return error_rates

    def get_hardest_samples(self, n_samples: int = 10) -> Dict[str, Any]:
        """Identify hardest samples (lowest prediction confidence for correct class).

        Args:
            n_samples: Number of hard samples to return

        Returns:
            Dictionary with hard samples information
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for this analysis")

        hard_samples = []

        for idx in self.get_misclassified_indices():
            true_label = self.y_true[idx]
            pred_label = self.y_pred[idx]
            true_prob = self.y_proba[idx, true_label]
            pred_prob = self.y_proba[idx, pred_label]

            hard_samples.append({
                "index": int(idx),
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "true_prob": float(true_prob),
                "pred_prob": float(pred_prob),
                "confidence_gap": float(pred_prob - true_prob),
            })

        # Sort by confidence gap (most problematic first)
        hard_samples.sort(key=lambda x: x["confidence_gap"], reverse=True)

        return {
            "total_hard_samples": len(hard_samples),
            "hard_samples": hard_samples[:n_samples],
        }

    def get_uncertain_predictions(self, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Identify predictions with low confidence.

        Args:
            confidence_threshold: Maximum confidence to be considered uncertain

        Returns:
            Dictionary with uncertain predictions
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for this analysis")

        max_probs = np.max(self.y_proba, axis=1)
        uncertain_mask = max_probs < confidence_threshold

        uncertain_indices = np.where(uncertain_mask)[0]
        uncertain_errors = np.sum(uncertain_mask & self.error_mask)

        return {
            "total_uncertain": int(np.sum(uncertain_mask)),
            "uncertain_error_rate": float(uncertain_errors / max(1, np.sum(uncertain_mask))),
            "uncertain_indices": uncertain_indices.tolist(),
            "uncertain_predictions": [
                {
                    "index": int(idx),
                    "true_label": int(self.y_true[idx]),
                    "pred_label": int(self.y_pred[idx]),
                    "confidence": float(max_probs[idx]),
                    "is_error": bool(self.error_mask[idx]),
                }
                for idx in uncertain_indices
            ],
        }

    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze common error patterns.

        Returns:
            Dictionary with error pattern analysis
        """
        misclass_matrix = self.get_misclassification_matrix()
        per_class_errors = self.get_per_class_error_rates()

        # Find most common confusions
        confusions = []
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                if i != j and misclass_matrix[i, j] > 0:
                    confusions.append({
                        "true_class": i,
                        "predicted_class": j,
                        "count": int(misclass_matrix[i, j]),
                        "percentage": float(misclass_matrix[i, j] / max(1, np.sum(self.error_mask)) * 100),
                    })

        # Sort by frequency
        confusions.sort(key=lambda x: x["count"], reverse=True)

        return {
            "total_misclassifications": int(np.sum(self.error_mask)),
            "misclassification_rate": float(np.sum(self.error_mask) / len(self.y_true)),
            "per_class_error_rates": per_class_errors,
            "most_common_confusions": confusions[:5],  # Top 5 confusions
            "all_confusions": confusions,
        }

    def get_class_confusion_details(self, true_class: int, pred_class: int) -> Dict[str, Any]:
        """Get details about confusion between two classes.

        Args:
            true_class: True class ID
            pred_class: Predicted class ID

        Returns:
            Dictionary with confusion details
        """
        mask = (self.y_true == true_class) & (self.y_pred == pred_class)
        indices = np.where(mask)[0]

        details = {
            "true_class": true_class,
            "predicted_class": pred_class,
            "count": int(np.sum(mask)),
            "indices": indices.tolist(),
        }

        if self.y_proba is not None and np.sum(mask) > 0:
            probs = self.y_proba[mask]
            details["avg_confidence"] = float(np.mean(np.max(probs, axis=1)))
            details["true_class_avg_prob"] = float(np.mean(probs[:, true_class]))
            details["pred_class_avg_prob"] = float(np.mean(probs[:, pred_class]))

        return details

    def get_easy_vs_hard_samples(self, percentile: float = 25.0) -> Dict[str, Any]:
        """Partition samples into easy and hard based on prediction confidence.

        Args:
            percentile: Percentile threshold for hard samples

        Returns:
            Dictionary with easy and hard sample statistics
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for this analysis")

        max_probs = np.max(self.y_proba, axis=1)
        threshold = np.percentile(max_probs, percentile)

        hard_mask = max_probs < threshold
        easy_mask = ~hard_mask

        hard_indices = np.where(hard_mask)[0]
        easy_indices = np.where(easy_mask)[0]

        hard_errors = np.sum(self.error_mask & hard_mask)
        easy_errors = np.sum(self.error_mask & easy_mask)

        return {
            "threshold_confidence": float(threshold),
            "hard_samples": {
                "count": int(np.sum(hard_mask)),
                "errors": int(hard_errors),
                "error_rate": float(hard_errors / max(1, np.sum(hard_mask))),
                "indices": hard_indices.tolist(),
            },
            "easy_samples": {
                "count": int(np.sum(easy_mask)),
                "errors": int(easy_errors),
                "error_rate": float(easy_errors / max(1, np.sum(easy_mask))),
                "indices": easy_indices.tolist(),
            },
        }

    def get_confusion_summary(self) -> Dict[str, Any]:
        """Get a summary of all confusions by class.

        Returns:
            Dictionary with confusion summary per class
        """
        summary = {}

        for class_id in range(self.n_classes):
            class_mask = self.y_true == class_id
            class_errors = self.error_mask & class_mask

            # Where are samples of this class misclassified?
            if np.sum(class_errors) > 0:
                misclassified_as = self.y_pred[class_errors]
                confusion_dist = Counter(misclassified_as)
            else:
                confusion_dist = {}

            summary[class_id] = {
                "total_samples": int(np.sum(class_mask)),
                "correct": int(np.sum(~class_errors & class_mask)),
                "incorrect": int(np.sum(class_errors)),
                "accuracy": float(np.sum(~class_errors & class_mask) / max(1, np.sum(class_mask))),
                "common_misclassifications": [
                    {"predicted_as": k, "count": v}
                    for k, v in sorted(confusion_dist.items(), key=lambda x: x[1], reverse=True)
                ],
            }

        return summary

    def summary(self) -> Dict[str, Any]:
        """Get complete error analysis summary.

        Returns:
            Dictionary with comprehensive error analysis
        """
        summary_dict = {
            "total_samples": len(self.y_true),
            "error_count": int(np.sum(self.error_mask)),
            "accuracy": float(np.sum(self.correct_mask) / len(self.y_true)),
            "error_rate": float(np.sum(self.error_mask) / len(self.y_true)),
            "per_class_error_rates": self.get_per_class_error_rates(),
            "error_patterns": self.get_error_patterns(),
            "confusion_summary": self.get_confusion_summary(),
        }

        if self.y_proba is not None:
            summary_dict["hard_samples"] = self.get_hardest_samples(n_samples=5)
            summary_dict["uncertain_predictions"] = self.get_uncertain_predictions()
            summary_dict["easy_vs_hard"] = self.get_easy_vs_hard_samples()

        return summary_dict