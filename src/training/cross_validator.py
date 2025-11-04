"""
Cross-validation strategies for model evaluation.

Provides stratified k-fold, leave-one-out, and time-series split validation.
"""

import numpy as np
from typing import Tuple, List, Iterator, Optional
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    TimeSeriesSplit,
    cross_val_score
)


class CrossValidator:
    """Cross-validation orchestrator for model evaluation.
    
    Args:
        method: Validation method ('stratified_kfold', 'leave_one_out', 'time_series')
        n_splits: Number of splits for k-fold methods
    """
    
    def __init__(
        self,
        method: str = "stratified_kfold",
        n_splits: int = 5
    ):
        self.method = method
        self.n_splits = n_splits
        
        if method == "stratified_kfold":
            self.splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42
            )
        elif method == "leave_one_out":
            self.splitter = LeaveOneOut()
        elif method == "time_series":
            self.splitter = TimeSeriesSplit(n_splits=n_splits)
        else:
            raise ValueError(f"Unknown CV method: {method}")
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for cross-validation.
        
        Args:
            X: Feature array
            y: Label array
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        for train_idx, test_idx in self.splitter.split(X, y):
            yield train_idx, test_idx
    
    def get_splits(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all train/test splits."""
        return list(self.split(X, y))
    
    def evaluate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "accuracy"
    ) -> Tuple[List[float], float, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature array
            y: Label array
            scoring: Scoring metric (sklearn scoring strings)
            
        Returns:
            Tuple of (scores per fold, mean score, std score)
        """
        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.splitter,
            scoring=scoring,
            n_jobs=-1
        )
        
        return scores.tolist(), float(np.mean(scores)), float(np.std(scores))


class KFoldValidator:
    """Stratified K-fold cross-validation.
    
    Args:
        n_splits: Number of folds
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        return self.splitter.split(X, y)
    
    def get_fold_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """Get statistics about fold distribution.
        
        Returns:
            Dictionary with fold information
        """
        stats = {
            "n_folds": self.n_splits,
            "fold_info": []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X, y)):
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            fold_data = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "train_class_distribution": self._get_class_distribution(y_train),
                "test_class_distribution": self._get_class_distribution(y_test)
            }
            stats["fold_info"].append(fold_data)
        
        return stats
    
    @staticmethod
    def _get_class_distribution(y: np.ndarray) -> dict:
        """Get class distribution for a subset."""
        unique, counts = np.unique(y, return_counts=True)
        return {int(cls): int(count) for cls, count in zip(unique, counts)}


class TimeSeriesValidator:
    """Time-series cross-validation.
    
    Ensures training data is always before test data to respect temporal order.
    
    Args:
        n_splits: Number of splits
    """
    
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.splitter = TimeSeriesSplit(n_splits=n_splits)
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices respecting time order."""
        return self.splitter.split(X, y)
    
    def get_split_sizes(
        self,
        X: np.ndarray
    ) -> List[dict]:
        """Get train/test sizes for each split."""
        sizes = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            sizes.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "test_start_idx": min(test_idx),
                "test_end_idx": max(test_idx)
            })
        return sizes


class LeaveOneOutValidator:
    """Leave-one-out cross-validation for small datasets.
    
    Best for small datasets where k-fold would use too large folds.
    """
    
    def __init__(self):
        self.splitter = LeaveOneOut()
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for leave-one-out."""
        return self.splitter.split(X, y)
    
    def get_n_splits(self, X: np.ndarray) -> int:
        """Get number of splits (equals number of samples)."""
        return len(X)


class StratifiedGroupKFold:
    """K-fold that maintains group structure across folds.
    
    Args:
        n_splits: Number of folds
        random_state: Random seed
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.n_splits = n_splits
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified splits respecting group boundaries.
        
        Args:
            X: Feature array
            y: Label array
            groups: Group identifiers (same group not split across folds)
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        unique_groups = np.unique(groups)
        unique_labels = np.unique(y)
        
        # Assign groups to folds while maintaining label distribution
        np.random.seed(self.random_state)
        group_to_fold = {}
        
        for label in unique_labels:
            label_groups = unique_groups[y[np.isin(unique_groups, 
                                                     unique_groups[y == label])]
                                        .unique()]
            shuffled_groups = np.random.permutation(label_groups)
            
            for i, group in enumerate(shuffled_groups):
                group_to_fold[group] = i % self.n_splits
        
        # Generate splits
        for fold in range(self.n_splits):
            test_mask = np.array([group_to_fold.get(g, 0) == fold 
                                 for g in groups])
            train_idx = np.where(~test_mask)[0]
            test_idx = np.where(test_mask)[0]
            
            yield train_idx, test_idx


def evaluate_with_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_method: str = "stratified_kfold",
    n_splits: int = 5,
    scoring: str = "accuracy"
) -> dict:
    """
    Evaluate model using cross-validation.
    
    Args:
        model: Scikit-learn compatible model
        X: Feature array
        y: Label array
        cv_method: Cross-validation method
        n_splits: Number of splits
        scoring: Scoring metric
        
    Returns:
        Dictionary with CV results
    """
    validator = CrossValidator(method=cv_method, n_splits=n_splits)
    scores, mean_score, std_score = validator.evaluate_model(
        model, X, y, scoring=scoring
    )
    
    return {
        "method": cv_method,
        "n_splits": n_splits,
        "scores": scores,
        "mean": mean_score,
        "std": std_score,
        "scores_info": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores))
        }
    }