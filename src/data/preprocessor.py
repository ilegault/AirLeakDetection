import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from typing import Tuple, Optional
class SignalPreprocessor:
    """FFT and signal preprocessing operations"""
    
    def __init__(self, config: dict):
        self.sample_rate = config['data']['sample_rate']
        self.fft_size = config['preprocessing']['fft_size']
        self.window = config['preprocessing']['window']
        self.freq_max = config['preprocessing']['freq_max']
        
    def compute_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of multi-channel signal
        Args:
            signal_data: shape (n_samples, n_channels)
        Returns:
            frequencies, fft_magnitude
        """
        n_samples, n_channels = signal_data.shape
        
        # Apply window
        if self.window == 'hanning':
            window = np.hanning(n_samples)
        elif self.window == 'hamming':
            window = np.hamming(n_samples)
        else:
            window = np.ones(n_samples)
        
        # Initialize FFT results
        fft_results = []
        
        for ch in range(n_channels):
            # Apply window and compute FFT
            windowed = signal_data[:, ch] * window
            fft_vals = rfft(windowed, n=self.fft_size)
            fft_mag = np.abs(fft_vals)
            fft_results.append(fft_mag)
        
        # Get frequency bins
        freqs = rfftfreq(self.fft_size, 1/self.sample_rate)
        
        # Limit to max frequency
        freq_mask = freqs <= self.freq_max
        freqs = freqs[freq_mask]
        fft_results = np.array(fft_results)[:, freq_mask]
        
        return freqs, fft_results.T  # Return (n_freqs, n_channels)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        # Use log scale for better visualization
        features = np.log1p(features)
        
        # Normalize
        features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        
        return features