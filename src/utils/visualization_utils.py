"""Visualization utilities for Air Leak Detection system."""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Union
import logging


logger = logging.getLogger(__name__)


class VisualizationUtils:
    """Common plotting and visualization functions."""
    
    _matplotlib_available = False
    _plt = None
    
    try:
        import matplotlib.pyplot as plt
        _matplotlib_available = True
        _plt = plt
    except ImportError:
        logger.warning("matplotlib not available - visualization disabled")
    
    @classmethod
    def plot_fft(
        cls,
        fft_data: np.ndarray,
        frequencies: Optional[np.ndarray] = None,
        title: str = "FFT Spectrum",
        xlabel: str = "Frequency (Hz)",
        ylabel: str = "Magnitude",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot FFT data.
        
        Args:
            fft_data: FFT magnitude data
            frequencies: Frequency bins (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save figure
        """
        if not cls._matplotlib_available:
            raise ImportError("matplotlib is required for visualization")
        
        if fft_data.ndim > 1:
            fft_data = fft_data.flatten()
        
        if frequencies is None:
            frequencies = np.arange(len(fft_data))
        
        fig, ax = cls._plt.subplots(figsize=figsize)
        ax.plot(frequencies, fft_data, linewidth=1.5)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cls._plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"FFT plot saved: {save_path}")
        
        cls._plt.close(fig)
    
    @classmethod
    def plot_time_series(
        cls,
        data: np.ndarray,
        time: Optional[np.ndarray] = None,
        title: str = "Time Series",
        xlabel: str = "Time (s)",
        ylabel: str = "Acceleration (g)",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot time series data.
        
        Args:
            data: Time series data
            time: Time axis (optional)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save figure
        """
        if not cls._matplotlib_available:
            raise ImportError("matplotlib is required for visualization")
        
        if data.ndim > 1:
            data = data.flatten()
        
        if time is None:
            time = np.arange(len(data))
        
        fig, ax = cls._plt.subplots(figsize=figsize)
        ax.plot(time, data, linewidth=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cls._plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Time series plot saved: {save_path}")
        
        cls._plt.close(fig)
    
    @classmethod
    def plot_spectrogram(
        cls,
        data: np.ndarray,
        sample_rate: float = 10000,
        title: str = "Spectrogram",
        figsize: Tuple[int, int] = (12, 6),
        cmap: str = "viridis",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot spectrogram of time series data.
        
        Args:
            data: Time series data
            sample_rate: Sampling rate in Hz
            title: Plot title
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
        """
        if not cls._matplotlib_available:
            raise ImportError("matplotlib is required for visualization")
        
        if data.ndim > 1:
            data = data.flatten()
        
        fig, ax = cls._plt.subplots(figsize=figsize)
        
        # Compute spectrogram
        Pxx, freqs, bins, im = ax.specgram(
            data,
            Fs=sample_rate,
            cmap=cmap,
            scale='dB'
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Frequency (Hz)", fontsize=12)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power (dB)", fontsize=12)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cls._plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Spectrogram saved: {save_path}")
        
        cls._plt.close(fig)
    
    @classmethod
    def plot_multi_channel(
        cls,
        data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        title: str = "Multi-Channel Data",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot multiple channels of data.
        
        Args:
            data: Data array (n_samples, n_channels) or (n_channels, n_samples)
            channel_names: Names for each channel
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        """
        if not cls._matplotlib_available:
            raise ImportError("matplotlib is required for visualization")
        
        # Handle data shape
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")
        
        # Ensure data is (n_channels, n_samples)
        if data.shape[0] > data.shape[1]:
            data = data.T
        
        n_channels = data.shape[0]
        
        if channel_names is None:
            channel_names = [f"Channel {i}" for i in range(n_channels)]
        
        fig, axes = cls._plt.subplots(n_channels, 1, figsize=figsize)
        if n_channels == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, channel_names)):
            ax.plot(data[i, :], linewidth=1)
            ax.set_ylabel(name, fontsize=10)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel("Sample", fontsize=12)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cls._plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Multi-channel plot saved: {save_path}")
        
        cls._plt.close(fig)
    
    @classmethod
    def plot_histogram(
        cls,
        data: np.ndarray,
        bins: int = 30,
        title: str = "Histogram",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot histogram of data.
        
        Args:
            data: Data to plot
            bins: Number of bins
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            figsize: Figure size
            save_path: Path to save figure
        """
        if not cls._matplotlib_available:
            raise ImportError("matplotlib is required for visualization")
        
        if data.ndim > 1:
            data = data.flatten()
        
        fig, ax = cls._plt.subplots(figsize=figsize)
        ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cls._plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Histogram saved: {save_path}")
        
        cls._plt.close(fig)
    
    @classmethod
    def plot_comparison(
        cls,
        data1: np.ndarray,
        data2: np.ndarray,
        label1: str = "Data 1",
        label2: str = "Data 2",
        title: str = "Comparison",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot two data series for comparison.
        
        Args:
            data1: First data series
            data2: Second data series
            label1: Label for first series
            label2: Label for second series
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        """
        if not cls._matplotlib_available:
            raise ImportError("matplotlib is required for visualization")
        
        if data1.ndim > 1:
            data1 = data1.flatten()
        if data2.ndim > 1:
            data2 = data2.flatten()
        
        fig, ax = cls._plt.subplots(figsize=figsize)
        
        # Normalize for comparison
        data1_norm = data1 / (np.max(np.abs(data1)) + 1e-10)
        data2_norm = data2 / (np.max(np.abs(data2)) + 1e-10)
        
        ax.plot(data1_norm, label=label1, linewidth=2, alpha=0.8)
        ax.plot(data2_norm, label=label2, linewidth=2, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cls._plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved: {save_path}")
        
        cls._plt.close(fig)