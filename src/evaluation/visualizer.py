"""Visualization tools for model evaluation and results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class ResultVisualizer:
    """Generate visualizations for model evaluation."""

    def __init__(self, figsize: tuple[int, int] = (12, 8), style: str = "seaborn-v0_8-darkgrid") -> None:
        """Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            # Fallback if style not available
            pass

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        n_classes = len(np.unique(y_true))

        if class_names is None:
            class_names = [f"Class {i}" for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig

    def plot_roc_curves(
        self,
        fpr_tpr_list: List[tuple[np.ndarray, np.ndarray, str]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curves for multiple classes/models.
        
        Args:
            fpr_tpr_list: List of (fpr, tpr, label) tuples
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for fpr, tpr, label in fpr_tpr_list:
            auc = np.trapz(tpr, fpr)  # Approximate AUC
            ax.plot(fpr, tpr, marker=".", label=f"{label} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        metrics: List[str] = ["loss", "accuracy"],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot training history.
        
        Args:
            history: Dictionary with metric names as keys and lists of values
            metrics: Metrics to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(history[metric], label="train")
                if f"val_{metric}" in history:
                    ax.plot(history[f"val_{metric}"], label="val")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f"Training {metric.capitalize()}")
                ax.legend()
                ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_metrics_comparison(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str] = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot metrics comparison across models.
        
        Args:
            models_metrics: Dictionary mapping model names to their metrics
            metrics_to_plot: Metrics to compare
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(models_metrics)

        for idx, (model_name, metrics) in enumerate(models_metrics.items()):
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            ax.bar(x + idx * width, values, width, label=model_name)

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x + width * (len(models_metrics) - 1) / 2)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.1])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance.
        
        Args:
            importances: Feature importance scores
            feature_names: Names of features
            top_n: Number of top features to display
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]

        # Get top features
        indices = np.argsort(importances)[-top_n:]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.barh(top_names, top_importances)
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importance")
        ax.grid(True, alpha=0.3, axis="x")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_fft_comparison(
        self,
        frequencies: np.ndarray,
        magnitude_actual: np.ndarray,
        magnitude_expected: Optional[np.ndarray] = None,
        title: str = "FFT Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot FFT magnitude comparison.
        
        Args:
            frequencies: Frequency bins
            magnitude_actual: Actual magnitude spectrum
            magnitude_expected: Expected magnitude spectrum for comparison
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(frequencies, magnitude_actual, label="Actual", linewidth=2)
        if magnitude_expected is not None:
            ax.plot(frequencies, magnitude_expected, label="Expected", linewidth=2, linestyle="--")

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def close_all() -> None:
        """Close all matplotlib figures."""
        plt.close("all")