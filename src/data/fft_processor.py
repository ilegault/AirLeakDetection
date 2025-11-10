"""Flexible FFT processing supporting multiple methods (MATLAB, scipy, NumPy, Welch)."""

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

    def compute_welch_psd(
        self,
        signal_data: np.ndarray,
        num_segments: int = 16,
        window_type: str = 'hamming'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density using Welch's method for each accelerometer.

        This implementation follows the professor's specific parameters:
        - Segment length: floor(Nx / (numSegments/2 + 0.5))
        - Hamming window (better spectral leakage reduction)
        - 50% overlap between segments
        - Zero-padding for better frequency resolution

        Args:
            signal_data: Shape (timesteps, 3) for 3 separate accelerometers
            num_segments: Number of segments for Welch's method (default: 16)
            window_type: Window type, default 'hamming' (not 'hanning')

        Returns:
            frequencies: Shape (n_freqs,)
            psd: Shape (n_freqs, 3) - PSD for each accelerometer
        """
        if signal_data.ndim == 1:
            # Single channel
            signal_data = signal_data.reshape(-1, 1)

        if signal_data.shape[1] != 3:
            LOGGER.warning("Expected 3 accelerometers, got %d", signal_data.shape[1])

        # Calculate segment length using professor's formula
        Nx = signal_data.shape[0]
        segment_length = int(np.floor(Nx / (num_segments / 2 + 0.5)))

        # 50% overlap
        num_overlap = segment_length // 2

        # FFT points: max(256, 2^nextpow2(segmentLength))
        nextpow2 = int(np.ceil(np.log2(segment_length)))
        nfft = max(256, 2**nextpow2)

        LOGGER.info(
            f"Welch's method parameters: Nx={Nx}, segments={num_segments}, "
            f"segment_length={segment_length}, overlap={num_overlap}, nfft={nfft}"
        )

        # Compute Welch PSD for EACH accelerometer separately
        psd_list = []
        for i in range(signal_data.shape[1]):
            # Extract signal for this accelerometer
            single_channel = signal_data[:, i]

            # Compute Welch's PSD using scipy
            frequencies, psd = scipy_signal.welch(
                single_channel,
                fs=self.sample_rate,
                window=window_type,
                nperseg=segment_length,
                noverlap=num_overlap,
                nfft=nfft,
                scaling='density',
                detrend='constant'
            )

            psd_list.append(psd)

        # Stack PSDs: shape (n_freqs, 3)
        psd_stacked = np.column_stack(psd_list)

        # Limit to frequency range
        mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)

        LOGGER.info(
            f"Welch PSD computed: frequency range [{frequencies[mask][0]:.2f}, "
            f"{frequencies[mask][-1]:.2f}] Hz, {np.sum(mask)} frequency bins"
        )

        return frequencies[mask], psd_stacked[mask, :]

    def compute_bandpower_welch(
        self,
        signal_data: np.ndarray,
        freq_range: Tuple[float, float] = (50.0, 4000.0),
        num_segments: int = 16
    ) -> np.ndarray:
        """Calculate band power in specific frequency range using Welch's PSD.

        This provides a single metric per accelerometer for ML classification.

        Args:
            signal_data: Shape (timesteps, 3) for 3 separate accelerometers
            freq_range: Frequency range tuple (min_freq, max_freq) in Hz
            num_segments: Number of segments for Welch's method

        Returns:
            bandpower: Shape (3,) - one value per accelerometer
        """
        # Compute Welch PSD
        frequencies, psd = self.compute_welch_psd(signal_data, num_segments=num_segments)

        # Calculate band power for each accelerometer
        min_freq, max_freq = freq_range
        freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)

        if np.sum(freq_mask) == 0:
            LOGGER.warning(
                f"No frequencies in range [{min_freq}, {max_freq}] Hz. "
                f"Available range: [{frequencies[0]:.2f}, {frequencies[-1]:.2f}] Hz"
            )
            return np.zeros(signal_data.shape[1])

        # Integrate PSD over frequency band (using trapezoidal rule)
        # Power = integral of PSD over frequency
        bandpower_list = []
        for i in range(psd.shape[1]):
            # Get PSD for this accelerometer in the frequency band
            psd_band = psd[freq_mask, i]
            freqs_band = frequencies[freq_mask]

            # Integrate using trapezoidal rule
            power = np.trapz(psd_band, freqs_band)
            bandpower_list.append(power)

        bandpower = np.array(bandpower_list)

        LOGGER.info(
            f"Band power ({min_freq}-{max_freq} Hz): "
            f"Accel0={bandpower[0]:.6e}, Accel1={bandpower[1]:.6e}, Accel2={bandpower[2]:.6e}"
        )

        return bandpower

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