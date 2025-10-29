import numpy as np
from scipy import stats
from typing import Dict

class FeatureExtractor:
    """Extract time and frequency domain features"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def extract_time_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract time-domain features from signal"""
        features = {}
        
        # RMS (Root Mean Square)
        features['rms'] = np.sqrt(np.mean(signal**2))
        
        # Peak-to-peak
        features['peak_to_peak'] = np.max(signal) - np.min(signal)
        
        # Statistical moments
        features['kurtosis'] = stats.kurtosis(signal)
        features['skewness'] = stats.skew(signal)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        
        return features
    
    def extract_frequency_features(self, freqs: np.ndarray, 
                                  fft_mag: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features"""
        features = {}
        
        # Peak frequency
        peak_idx = np.argmax(fft_mag)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_magnitude'] = fft_mag[peak_idx]
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs * fft_mag) / np.sum(fft_mag)
        
        # Band power in different frequency ranges
        bands = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
        for i, (low, high) in enumerate(bands):
            band_mask = (freqs >= low) & (freqs < high)
            features[f'band_power_{i}'] = np.sum(fft_mag[band_mask])
        
        return features