"""Create TensorFlow/PyTorch datasets with batching and augmentation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate batches for training/evaluation."""

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
        random_seed: int = 42,
    ) -> None:
        """Initialize dataset generator.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
            random_seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def get_batches(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        infinite: bool = False,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generator yielding batches of data.
        
        Args:
            signals: Input signals (n_samples, ...)
            labels: Target labels (n_samples,)
            infinite: If True, cycle through data indefinitely
            
        Returns:
            Iterator yielding (signal_batch, label_batch) tuples
        """
        n_samples = len(signals)
        indices = np.arange(n_samples)

        while True:
            if self.shuffle:
                np.random.shuffle(indices)

            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_signals = signals[batch_indices]
                batch_labels = labels[batch_indices]

                yield batch_signals, batch_labels

            if not infinite:
                break

    def get_tf_dataset(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        infinite: bool = False,
    ) -> Any:
        """Create TensorFlow dataset from signals and labels.
        
        Args:
            signals: Input signals
            labels: Target labels
            infinite: Whether to repeat indefinitely
            
        Returns:
            tf.data.Dataset object
        """
        try:
            import tensorflow as tf
        except ImportError:
            LOGGER.error("TensorFlow not installed")
            raise

        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((signals, labels))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(signals))

        dataset = dataset.batch(self.batch_size)

        if infinite:
            dataset = dataset.repeat()

        return dataset

    def get_torch_dataloader(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        infinite: bool = False,
    ) -> Any:
        """Create PyTorch DataLoader from signals and labels.
        
        Args:
            signals: Input signals
            labels: Target labels
            infinite: Whether to repeat indefinitely
            
        Returns:
            torch.utils.data.DataLoader object
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
        except ImportError:
            LOGGER.error("PyTorch not installed")
            raise

        # Convert to tensors
        signal_tensor = torch.from_numpy(signals).float()
        label_tensor = torch.from_numpy(labels).long()

        dataset = TensorDataset(signal_tensor, label_tensor)

        if self.shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
        )

        return dataloader

    def get_num_batches(self, n_samples: int) -> int:
        """Calculate number of batches.
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Number of batches
        """
        return (n_samples + self.batch_size - 1) // self.batch_size

    def stratified_batch_generator(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        class_weights: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate batches with stratification for imbalanced data.
        
        Args:
            signals: Input signals
            labels: Target labels
            class_weights: Optional class weights for sampling
            
        Returns:
            Iterator yielding balanced batches
        """
        unique_classes = np.unique(labels)
        class_indices = {c: np.where(labels == c)[0] for c in unique_classes}

        samples_per_class = self.batch_size // len(unique_classes)

        while True:
            batch_signals = []
            batch_labels = []

            for class_id in unique_classes:
                indices = class_indices[class_id]
                selected = np.random.choice(indices, size=samples_per_class, replace=True)
                batch_signals.append(signals[selected])
                batch_labels.append(labels[selected])

            batch_signals_array = np.vstack(batch_signals)
            batch_labels_array = np.hstack(batch_labels)

            yield batch_signals_array, batch_labels_array