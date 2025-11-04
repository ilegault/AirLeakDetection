"""Data augmentation for training robustness."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

LOGGER = logging.getLogger(__name__)


class DataAugmentor:
    """Apply various data augmentation techniques."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize augmentor.
        
        Args:
            config: Configuration dictionary with augmentation parameters
        """
        aug_cfg = config.get("augmentation", {})
        self.noise_factor = float(aug_cfg.get("noise_factor", 0.005))
        self.time_shift_range = float(aug_cfg.get("time_shift_range", 0.1))
        self.amplitude_range = tuple(aug_cfg.get("amplitude_range", [0.9, 1.1]))
        self.freq_mask_width = int(aug_cfg.get("freq_mask_width", 50))
        self.random_seed = int(aug_cfg.get("random_seed", 42))

    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to signal.
        
        Args:
            signal: Input signal
            
        Returns:
            Signal with added noise
        """
        noise = np.random.normal(0, self.noise_factor, signal.shape)
        return signal + noise

    def time_shift(self, signal: np.ndarray) -> np.ndarray:
        """Apply random time shifting.
        
        Args:
            signal: Input signal of shape (timesteps, channels)
            
        Returns:
            Time-shifted signal
        """
        shift_amount = int(signal.shape[0] * self.time_shift_range * np.random.uniform(-1, 1))
        if shift_amount == 0:
            return signal
        return np.roll(signal, shift_amount, axis=0)

    def amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        """Scale signal amplitude randomly.
        
        Args:
            signal: Input signal
            
        Returns:
            Amplitude-scaled signal
        """
        scale_factor = np.random.uniform(self.amplitude_range[0], self.amplitude_range[1])
        return signal * scale_factor

    def frequency_mask(self, fft_spectrum: np.ndarray) -> np.ndarray:
        """Apply frequency masking (zero out frequency bands).
        
        Args:
            fft_spectrum: FFT magnitude spectrum of shape (freq_bins, channels)
            
        Returns:
            Frequency-masked spectrum
        """
        masked = fft_spectrum.copy()
        n_freqs = fft_spectrum.shape[0]
        
        # Randomly select frequency band to mask
        mask_start = np.random.randint(0, max(1, n_freqs - self.freq_mask_width))
        mask_end = min(mask_start + self.freq_mask_width, n_freqs)
        
        masked[mask_start:mask_end, :] = 0
        return masked

    def augment_batch(
        self,
        signals: np.ndarray,
        augmentation_type: str = "all",
    ) -> np.ndarray:
        """Apply augmentations to a batch of signals.
        
        Args:
            signals: Batch of signals with shape (n_samples, timesteps, channels)
            augmentation_type: Type of augmentation - 'noise', 'shift', 'scale', 'all'
            
        Returns:
            Augmented signals
        """
        augmented = signals.copy()

        if augmentation_type in ["noise", "all"]:
            augmented = self.add_noise(augmented)

        if augmentation_type in ["shift", "all"]:
            # Apply time shift to each sample
            for i in range(augmented.shape[0]):
                augmented[i] = self.time_shift(augmented[i])

        if augmentation_type in ["scale", "all"]:
            for i in range(augmented.shape[0]):
                augmented[i] = self.amplitude_scaling(augmented[i])

        return augmented

    def create_augmented_dataset(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        multiplier: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create augmented copies of dataset.
        
        Args:
            signals: Original signals
            labels: Original labels
            multiplier: How many times to augment each sample
            
        Returns:
            Augmented signals and labels
        """
        augmented_signals = [signals]
        augmented_labels = [labels]

        for _ in range(multiplier - 1):
            aug_signals = self.augment_batch(signals, augmentation_type="all")
            augmented_signals.append(aug_signals)
            augmented_labels.append(labels)

        return np.vstack(augmented_signals), np.hstack(augmented_labels)