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
        """Compute FFT for each accelerometer separately (not averaged).
        
        Args:
            signal_data: Shape (timesteps, 3) for 3 separate accelerometers
            
        Returns:
            frequencies: Shape (n_freqs,)
            magnitude_spectrum: Shape (n_freqs, 3) - FFT for each accelerometer
        """
        if signal_data.shape[1] != 3:
            LOGGER.warning("Expected 3 accelerometers, got %d", signal_data.shape[1])

        # Compute FFT for EACH accelerometer separately
        magnitude_list = []
        for i in range(signal_data.shape[1]):
            # Extract signal for this accelerometer
            single_channel = signal_data[:, i]
            
            # Apply Hanning window
            window = np.hanning(len(single_channel))
            windowed = single_channel * window
            
            # Compute FFT
            fft_result = rfft(windowed, n=self.fft_size)
            magnitude = np.abs(fft_result)
            magnitude_list.append(magnitude)
        
        # Stack magnitudes: shape (n_freqs, 3)
        magnitude_stacked = np.column_stack(magnitude_list)
        frequencies = rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        return frequencies[mask], magnitude_stacked[mask, :]

    def compute_scipy_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT for each accelerometer separately using scipy with Hanning window.
        
        Args:
            signal_data: Shape (timesteps, 3) for 3 separate accelerometers
            
        Returns:
            frequencies: Shape (n_freqs,)
            magnitude_spectrum: Shape (n_freqs, 3) - FFT for each accelerometer
        """
        if signal_data.ndim == 1:
            # Single channel
            signal_data = signal_data.reshape(-1, 1)
        
        # Compute FFT for EACH accelerometer separately
        magnitude_list = []
        for i in range(signal_data.shape[1]):
            # Extract signal for this accelerometer
            single_channel = signal_data[:, i]
            
            # Apply window
            window = scipy_signal.windows.hann(len(single_channel))
            windowed = single_channel * window
            
            # Compute FFT
            fft_result = rfft(windowed, n=self.fft_size)
            magnitude = np.abs(fft_result)
            magnitude_list.append(magnitude)
        
        # Stack magnitudes: shape (n_freqs, n_channels)
        magnitude_stacked = np.column_stack(magnitude_list)
        frequencies = rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        return frequencies[mask], magnitude_stacked[mask, :]

    def compute_numpy_fft(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple NumPy FFT computation for each accelerometer (for MATLAB compatibility).
        
        Args:
            signal_data: Shape (timesteps, 3) for 3 separate accelerometers
            
        Returns:
            frequencies: Shape (n_freqs,)
            magnitude_spectrum: Shape (n_freqs, 3) - FFT for each accelerometer
        """
        if signal_data.ndim == 1:
            # Single channel
            signal_data = signal_data.reshape(-1, 1)
        
        # Compute FFT for EACH accelerometer separately
        magnitude_list = []
        for i in range(signal_data.shape[1]):
            # Extract signal for this accelerometer
            single_channel = signal_data[:, i]
            
            # Apply window
            window = np.hanning(len(single_channel))
            windowed = single_channel * window
            
            # Compute FFT (using real FFT)
            fft_result = np.fft.rfft(windowed, n=self.fft_size)
            magnitude = np.abs(fft_result)
            magnitude_list.append(magnitude)
        
        # Stack magnitudes: shape (n_freqs, n_channels)
        magnitude_stacked = np.column_stack(magnitude_list)
        frequencies = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sample_rate)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        return frequencies[mask], magnitude_stacked[mask, :]

    def compare_methods(
        self, signal_data: np.ndarray
    ) -> Dict[str, Any]:
        """Compare FFT methods and return correlation & MSE metrics per accelerometer.
        
        Args:
            signal_data: Input signal
            
        Returns:
            Dictionary with comparison metrics for each accelerometer
        """
        freq_scipy, mag_scipy = self.compute_scipy_fft(signal_data)
        freq_numpy, mag_numpy = self.compute_numpy_fft(signal_data)
        freq_matlab, mag_matlab = self.compute_matlab_style_fft(signal_data)

        # Interpolate to common frequency grid
        common_freqs = freq_scipy  # Use scipy as reference
        
        # Interpolate each accelerometer separately
        n_accel = mag_scipy.shape[1] if mag_scipy.ndim == 2 else 1
        corr_numpy_list = []
        corr_matlab_list = []
        mse_numpy_list = []
        mse_matlab_list = []
        
        for i in range(n_accel):
            if mag_scipy.ndim == 2:
                scipy_accel = mag_scipy[:, i]
                numpy_accel = mag_numpy[:, i]
                matlab_accel = mag_matlab[:, i]
            else:
                scipy_accel = mag_scipy
                numpy_accel = mag_numpy
                matlab_accel = mag_matlab
            
            numpy_interp = np.interp(common_freqs, freq_numpy, numpy_accel)
            matlab_interp = np.interp(common_freqs, freq_matlab, matlab_accel)
            
            # Compute correlation
            corr_numpy = float(np.corrcoef(scipy_accel, numpy_interp)[0, 1])
            corr_matlab = float(np.corrcoef(scipy_accel, matlab_interp)[0, 1])
            corr_numpy_list.append(corr_numpy)
            corr_matlab_list.append(corr_matlab)
            
            # Compute MSE
            mse_numpy = float(np.mean((scipy_accel - numpy_interp) ** 2))
            mse_matlab = float(np.mean((scipy_accel - matlab_interp) ** 2))
            mse_numpy_list.append(mse_numpy)
            mse_matlab_list.append(mse_matlab)

        return {
            "correlation": {"numpy": corr_numpy_list, "matlab": corr_matlab_list},
            "mse": {"numpy": mse_numpy_list, "matlab": mse_matlab_list},
            "frequencies": common_freqs,
            "magnitude_scipy": mag_scipy,
            "magnitude_numpy": mag_numpy,
            "magnitude_matlab": mag_matlab,
        }