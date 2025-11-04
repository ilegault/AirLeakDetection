"""Flexible FFT processing supporting multiple methods (MATLAB, scipy, NumPy)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy import signal as scipy_signal

LOGGER = logging.getLogger(__name__)


class FlexibleFFTProcessor:
    """Support multiple FFT computation methods for flexibility and comparison."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize FFT processor with configuration.
        
        Args:
            config: Configuration dictionary with 'data' and 'preprocessing' sections
        """
        data_cfg = config.get("data", {})
        prep_cfg = config.get("preprocessing", {})

        self.sample_rate: int = int(data_cfg.get("sample_rate", 10000))
        self.fft_size: int = int(prep_cfg.get("fft_size", 2048))
        self.window_type: str = str(prep_cfg.get("window", "hanning"))
        self.freq_min: float = float(prep_cfg.get("freq_min", 30.0))
        self.freq_max: float = float(prep_cfg.get("freq_max", 2000.0))

    def load_matlab_fft(self, mat_file_path: Path) -> np.ndarray:
        """Load FFT from MATLAB .mat file.
        
        Args:
            mat_file_path: Path to .mat file
            
        Returns:
            FFT magnitude array
        """
        try:
            import scipy.io as sio
            mat_data = sio.loadmat(mat_file_path)
            
            # Look for common MATLAB variable names
            for key in ['fft', 'FFT', 'spectrum', 'magnitude']:
                if key in mat_data:
                    return np.asarray(mat_data[key], dtype=np.float32)
            
            # If no known key, return first non-meta variable
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    return np.asarray(value, dtype=np.float32)
                    
            raise ValueError("No suitable FFT data found in .mat file")
        except ImportError:
            LOGGER.error("scipy.io not available for loading .mat files")
            raise

    def compute_matlab_style_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Replicate MATLAB approach: average 3 sensors then compute FFT.
        
        Args:
            signal_data: Shape (timesteps, 9) for 3 accelerometers x 3 axes
            
        Returns:
            frequencies, magnitude_spectrum
        """
        if signal_data.shape[1] != 9:
            LOGGER.warning("Expected 9 channels (3 sensors x 3 axes), got %d", signal_data.shape[1])

        # Average across the 3 sensors (each sensor has 3 axes)
        averaged = np.mean([signal_data[:, i*3:(i+1)*3] for i in range(3)], axis=0)  # (timesteps, 3)
        averaged = np.mean(averaged, axis=1)  # (timesteps,)

        # Apply Hanning window
        window = np.hanning(len(averaged))
        windowed = averaged * window

        # Compute FFT
        fft_result = rfft(windowed, n=self.fft_size)
        magnitude = np.abs(fft_result)
        frequencies = rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        return frequencies[mask], magnitude[mask]

    def compute_scipy_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT using scipy with Hanning window.
        
        Args:
            signal_data: Shape (timesteps, channels)
            
        Returns:
            frequencies, magnitude_spectrum (averaged across channels)
        """
        # Average across channels
        if signal_data.ndim == 2:
            averaged = np.mean(signal_data, axis=1)
        else:
            averaged = signal_data

        # Apply window
        window = scipy_signal.windows.hann(len(averaged))
        windowed = averaged * window

        # Compute FFT
        fft_result = rfft(windowed, n=self.fft_size)
        magnitude = np.abs(fft_result)
        frequencies = rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        return frequencies[mask], magnitude[mask]

    def compute_numpy_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple NumPy FFT computation (for MATLAB compatibility).
        
        Args:
            signal_data: Shape (timesteps, channels)
            
        Returns:
            frequencies, magnitude_spectrum (averaged across channels)
        """
        # Average across channels
        if signal_data.ndim == 2:
            averaged = np.mean(signal_data, axis=1)
        else:
            averaged = signal_data

        # Apply window
        window = np.hanning(len(averaged))
        windowed = averaged * window

        # Compute FFT (using real FFT)
        fft_result = np.fft.rfft(windowed, n=self.fft_size)
        magnitude = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        return frequencies[mask], magnitude[mask]

    def compare_methods(
        self, signal_data: np.ndarray
    ) -> Dict[str, Any]:
        """Compare FFT methods and return correlation & MSE metrics.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary with comparison metrics
        """
        freq_scipy, mag_scipy = self.compute_scipy_fft(signal_data)
        freq_numpy, mag_numpy = self.compute_numpy_fft(signal_data)
        freq_matlab, mag_matlab = self.compute_matlab_style_fft(signal_data)

        # Interpolate to common frequency grid
        common_freqs = freq_scipy  # Use scipy as reference
        
        mag_numpy_interp = np.interp(common_freqs, freq_numpy, mag_numpy)
        mag_matlab_interp = np.interp(common_freqs, freq_matlab, mag_matlab)

        # Compute correlation
        corr_numpy = float(np.corrcoef(mag_scipy, mag_numpy_interp)[0, 1])
        corr_matlab = float(np.corrcoef(mag_scipy, mag_matlab_interp)[0, 1])

        # Compute MSE
        mse_numpy = float(np.mean((mag_scipy - mag_numpy_interp) ** 2))
        mse_matlab = float(np.mean((mag_scipy - mag_matlab_interp) ** 2))

        return {
            "correlation": {"numpy": corr_numpy, "matlab": corr_matlab},
            "mse": {"numpy": mse_numpy, "matlab": mse_matlab},
            "frequencies": common_freqs,
            "magnitude_scipy": mag_scipy,
            "magnitude_numpy": mag_numpy_interp,
            "magnitude_matlab": mag_matlab_interp,
        }