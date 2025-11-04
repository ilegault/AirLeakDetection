"""Create train/val/test splits with stratification and reproducibility."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


class DataSplitter:
    """Create stratified train/validation/test splits."""

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        """Initialize data splitter.
        
        Args:
            train_ratio: Fraction for training set
            val_ratio: Fraction for validation set
            test_ratio: Fraction for test set
            random_seed: Random seed for reproducibility
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split_stratified(
        self, labels: np.ndarray, file_level: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create stratified train/val/test splits.
        
        Args:
            labels: Class labels for stratification
            file_level: If True, split at file level; if False, at sample level
            
        Returns:
            train_indices, val_indices, test_indices
        """
        # First split: train + temp (val + test)
        train_idx, temp_idx = train_test_split(
            np.arange(len(labels)),
            train_size=self.train_ratio,
            test_size=self.val_ratio + self.test_ratio,
            stratify=labels,
            random_state=self.random_seed,
        )

        # Second split: val + test
        val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size,
            test_size=1.0 - val_size,
            stratify=labels[temp_idx],
            random_state=self.random_seed,
        )

        return train_idx, val_idx, test_idx

    def get_splits_by_indices(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return data splits using pre-computed indices.
        
        Args:
            signals: Input signals array
            labels: Class labels
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each mapping to (signals, labels)
        """
        return {
            "train": (signals[train_indices], labels[train_indices]),
            "val": (signals[val_indices], labels[val_indices]),
            "test": (signals[test_indices], labels[test_indices]),
        }

    def save_split_indices(self, indices_dict: Dict[str, np.ndarray], save_path: Path) -> None:
        """Save split indices for reproducibility.
        
        Args:
            indices_dict: Dictionary with 'train', 'val', 'test' keys
            save_path: Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to list for JSON serialization
        json_data = {
            key: indices.tolist() for key, indices in indices_dict.items()
        }

        with open(save_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        LOGGER.info(f"Split indices saved to {save_path}")

    def load_split_indices(self, load_path: Path) -> Dict[str, np.ndarray]:
        """Load split indices from file.
        
        Args:
            load_path: Path to JSON file with split indices
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        with open(load_path, 'r') as f:
            json_data = json.load(f)

        return {
            key: np.array(indices, dtype=np.int64)
            for key, indices in json_data.items()
        }

    def get_split_statistics(
        self,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Dict[str, Any]:
        """Get statistics about the splits.
        
        Args:
            labels: All class labels
            train_idx, val_idx, test_idx: Split indices
            
        Returns:
            Dictionary with split statistics
        """
        def class_distribution(indices: np.ndarray) -> Dict[int, int]:
            unique, counts = np.unique(labels[indices], return_counts=True)
            return {int(c): int(count) for c, count in zip(unique, counts)}

        return {
            "train": {
                "count": len(train_idx),
                "class_distribution": class_distribution(train_idx),
            },
            "val": {
                "count": len(val_idx),
                "class_distribution": class_distribution(val_idx),
            },
            "test": {
                "count": len(test_idx),
                "class_distribution": class_distribution(test_idx),
            },
        }