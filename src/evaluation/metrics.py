"""Comprehensive evaluation metrics for air leak detection models."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


class ModelMetrics:
    """Compute and aggregate performance metrics for classification models."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> None:
        """Initialize with predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (n_samples, n_classes)
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None
        self.n_classes = len(np.unique(self.y_true))

    def accuracy(self) -> float:
        """Calculate accuracy."""
        return float(accuracy_score(self.y_true, self.y_pred))

    def precision(self, average: str = "weighted") -> float:
        """Calculate precision.
        
        Args:
            average: 'micro', 'macro', 'weighted', or None (per-class)
            
        Returns:
            Precision score
        """
        return float(precision_score(self.y_true, self.y_pred, average=average, zero_division=0))

    def recall(self, average: str = "weighted") -> float:
        """Calculate recall.
        
        Args:
            average: 'micro', 'macro', 'weighted', or None
            
        Returns:
            Recall score
        """
        return float(recall_score(self.y_true, self.y_pred, average=average, zero_division=0))

    def f1(self, average: str = "weighted") -> float:
        """Calculate F1 score.
        
        Args:
            average: 'micro', 'macro', 'weighted'
            
        Returns:
            F1 score
        """
        return float(f1_score(self.y_true, self.y_pred, average=average, zero_division=0))

    def confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.
        
        Returns:
            Confusion matrix (n_classes, n_classes)
        """
        return confusion_matrix(self.y_true, self.y_pred)

    def roc_auc(self) -> float:
        """Calculate ROC-AUC score (one-vs-rest).
        
        Returns:
            Micro-averaged ROC-AUC for multiclass
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for ROC-AUC calculation")

        if self.n_classes == 2:
            return float(roc_auc_score(self.y_true, self.y_proba[:, 1]))
        else:
            return float(roc_auc_score(self.y_true, self.y_proba, multi_class="ovr", average="micro"))

    def per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get per-class precision, recall, F1.
        
        Returns:
            Dictionary mapping class_id to metrics
        """
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)

        result = {}
        for class_id in range(self.n_classes):
            result[class_id] = {
                "precision": float(precision_per_class[class_id]),
                "recall": float(recall_per_class[class_id]),
                "f1": float(f1_per_class[class_id]),
            }
        return result

    def summary(self) -> Dict[str, float | Dict]:
        """Get complete metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        summary_dict = {
            "accuracy": self.accuracy(),
            "precision_weighted": self.precision(average="weighted"),
            "recall_weighted": self.recall(average="weighted"),
            "f1_weighted": self.f1(average="weighted"),
            "per_class_metrics": self.per_class_metrics(),
        }

        if self.y_proba is not None:
            summary_dict["roc_auc"] = self.roc_auc()

        return summary_dict

    def fpr_tpr(self, class_id: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get false positive rate, true positive rate for ROC curve.
        
        Args:
            class_id: Class ID for binary or one-vs-rest
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if self.y_proba is None:
            raise ValueError("Probabilities required for ROC curve calculation")

        if self.n_classes == 2:
            fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba[:, class_id])
        else:
            # One-vs-rest for multiclass
            y_true_binary = (self.y_true == class_id).astype(int)
            fpr, tpr, thresholds = roc_curve(y_true_binary, self.y_proba[:, class_id])

        return fpr, tpr, thresholds