"""Mathematical utilities for Air Leak Detection system."""

import numpy as np
from typing import Tuple, Optional
from scipy import signal
import logging


logger = logging.getLogger(__name__)


class MathUtils:
    """Mathematical operations and signal processing helpers."""
    
    @staticmethod
    def bandpass_filter(
        data: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.
        
        Args:
            data: Input signal
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if low <= 0 or high >= 1:
            logger.warning(f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, nyquist={nyquist}")
            return data
        
        try:
            sos = signal.butter(order, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, data)
            logger.debug(f"Bandpass filter applied: {lowcut}-{highcut} Hz")
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filter failed: {e}")
            return data
    
    @staticmethod
    def normalize_minmax(
        data: np.ndarray,
        min_val: float = 0,
        max_val: float = 1
    ) -> Tuple[np.ndarray, float, float]:
        """
        Min-max normalization.
        
        Args:
            data: Input data
            min_val: Minimum value for output
            max_val: Maximum value for output
            
        Returns:
            Normalized data, original min, original max
        """
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            normalized = np.ones_like(data) * (min_val + max_val) / 2
        else:
            normalized = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
        
        logger.debug(f"Min-max normalization applied: [{data_min}, {data_max}] -> [{min_val}, {max_val}]")
        return normalized, data_min, data_max
    
    @staticmethod
    def normalize_zscore(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Z-score normalization.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data, mean, std
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - mean) / std
        
        logger.debug(f"Z-score normalization applied: mean={mean:.4f}, std={std:.4f}")
        return normalized, mean, std
    
    @staticmethod
    def normalize_robust(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Robust normalization using median and IQR.
        
        Args:
            data: Input data
            
        Returns:
            Normalized data, median, IQR
        """
        median = np.median(data)
        q75 = np.percentile(data, 75)
        q25 = np.percentile(data, 25)
        iqr = q75 - q25
        
        if iqr == 0:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - median) / iqr
        
        logger.debug(f"Robust normalization applied: median={median:.4f}, IQR={iqr:.4f}")
        return normalized, median, iqr
    
    @staticmethod
    def apply_window(
        data: np.ndarray,
        window_type: str = "hanning"
    ) -> np.ndarray:
        """
        Apply window function to data.
        
        Args:
            data: Input signal
            window_type: Window type ('hanning', 'hann', 'hamming', 'blackman', 'bartlett')
            
        Returns:
            Windowed signal
        """
        if len(data) == 0:
            return data
        
        try:
            # Handle 'hanning' -> 'hann' mapping for scipy compatibility
            if window_type == "hanning":
                window_type = "hann"
            
            window = signal.get_window(window_type, len(data))
            windowed = data * window
            logger.debug(f"Window '{window_type}' applied")
            return windowed
        except Exception as e:
            logger.error(f"Failed to apply window: {e}")
            return data
    
    @staticmethod
    def compute_fft(
        data: np.ndarray,
        fft_size: int = 2048,
        window: Optional[str] = "hanning"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of signal.
        
        Args:
            data: Input signal
            fft_size: FFT size
            window: Window function type
            
        Returns:
            FFT magnitude, frequencies
        """
        if len(data) < fft_size:
            # Pad data
            data = np.pad(data, (0, fft_size - len(data)), mode='constant')
        else:
            data = data[:fft_size]
        
        if window:
            data = MathUtils.apply_window(data, window)
        
        # Compute FFT
        fft_values = np.fft.fft(data)
        fft_magnitude = np.abs(fft_values[:fft_size // 2])
        
        # Compute frequencies
        frequencies = np.fft.fftfreq(fft_size)[:fft_size // 2]
        
        logger.debug(f"FFT computed: size={fft_size}, magnitude shape={fft_magnitude.shape}")
        return fft_magnitude, frequencies
    
    @staticmethod
    def rms(data: np.ndarray) -> float:
        """
        Compute Root Mean Square.
        
        Args:
            data: Input signal
            
        Returns:
            RMS value
        """
        rms_val = float(np.sqrt(np.mean(data ** 2)))
        return rms_val
    
    @staticmethod
    def peak_to_peak(data: np.ndarray) -> float:
        """
        Compute peak-to-peak value.
        
        Args:
            data: Input signal
            
        Returns:
            Peak-to-peak value
        """
        ptp = float(np.max(data) - np.min(data))
        return ptp
    
    @staticmethod
    def crest_factor(data: np.ndarray) -> float:
        """
        Compute crest factor (peak / RMS).
        
        Args:
            data: Input signal
            
        Returns:
            Crest factor
        """
        peak = np.max(np.abs(data))
        rms_val = MathUtils.rms(data)
        
        if rms_val == 0:
            return 0
        
        cf = float(peak / rms_val)
        return cf
    
    @staticmethod
    def spectral_centroid(
        magnitude: np.ndarray,
        frequencies: np.ndarray
    ) -> float:
        """
        Compute spectral centroid.
        
        Args:
            magnitude: FFT magnitude
            frequencies: Frequency bins
            
        Returns:
            Spectral centroid (Hz)
        """
        if len(magnitude) == 0 or len(frequencies) == 0:
            return 0
        
        centroid = float(np.sum(magnitude * frequencies) / np.sum(magnitude))
        return centroid
    
    @staticmethod
    def zero_crossing_rate(data: np.ndarray) -> float:
        """
        Compute zero crossing rate.
        
        Args:
            data: Input signal
            
        Returns:
            Zero crossing rate (0 to 1)
        """
        if len(data) < 2:
            return 0
        
        crossings = np.sum(np.abs(np.diff(np.sign(data)))) / 2
        zcr = float(crossings / len(data))
        return zcr
    
    @staticmethod
    def kurtosis(data: np.ndarray) -> float:
        """
        Compute kurtosis.
        
        Args:
            data: Input signal
            
        Returns:
            Kurtosis value
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        kurt = float(np.mean(((data - mean) / std) ** 4) - 3)
        return kurt
    
    @staticmethod
    def skewness(data: np.ndarray) -> float:
        """
        Compute skewness.
        
        Args:
            data: Input signal
            
        Returns:
            Skewness value
        """
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skew = float(np.mean(((data - mean) / std) ** 3))
        return skew
    
    @staticmethod
    def compute_band_power(
        magnitude: np.ndarray,
        frequencies: np.ndarray,
        freq_range: Tuple[float, float]
    ) -> float:
        """
        Compute power in frequency band.
        
        Args:
            magnitude: FFT magnitude
            frequencies: Frequency bins
            freq_range: (min_freq, max_freq) tuple
            
        Returns:
            Power in band
        """
        min_freq, max_freq = freq_range
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        
        if np.sum(mask) == 0:
            return 0
        
        power = float(np.sum(magnitude[mask] ** 2))
        return power
    
    @staticmethod
    def correlation(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient.
        
        Args:
            data1: First signal
            data2: Second signal
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(data1) == 0 or len(data2) == 0:
            return 0
        
        # Flatten arrays
        data1 = data1.flatten()
        data2 = data2.flatten()
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        corr = float(np.corrcoef(data1, data2)[0, 1])
        return corr
    
    @staticmethod
    def mean_squared_error(data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Compute mean squared error.
        
        Args:
            data1: First signal
            data2: Second signal
            
        Returns:
            MSE value
        """
        if len(data1) == 0 or len(data2) == 0:
            return np.inf
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1.flatten()[:min_len]
        data2 = data2.flatten()[:min_len]
        
        mse = float(np.mean((data1 - data2) ** 2))
        return mse