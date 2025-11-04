"""MATLAB integration utilities for Air Leak Detection system."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging


logger = logging.getLogger(__name__)


class MATLABBridge:
    """Handle MATLAB file operations and data format conversions."""
    
    # Try to import scipy.io for .mat file support
    _scipy_available = False
    _sio = None
    
    try:
        import scipy.io as sio
        _scipy_available = True
        _sio = sio
    except ImportError:
        logger.warning("scipy not available - MATLAB file operations disabled")
    
    @classmethod
    def load_mat(cls, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load MATLAB .mat file.
        
        Args:
            file_path: Path to .mat file
            
        Returns:
            Dictionary with .mat file contents
        """
        if not cls._scipy_available:
            raise ImportError("scipy is required to load .mat files")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            mat_data = cls._sio.loadmat(file_path)
            logger.info(f"MATLAB file loaded: {file_path}")
            
            # Remove metadata keys
            mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            
            return mat_data
        except Exception as e:
            logger.error(f"Failed to load MATLAB file {file_path}: {e}")
            raise
    
    @classmethod
    def save_mat(cls, data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save data to MATLAB .mat file.
        
        Args:
            data: Dictionary of arrays to save
            file_path: Path to save .mat file
        """
        if not cls._scipy_available:
            raise ImportError("scipy is required to save .mat files")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cls._sio.savemat(str(file_path), data)
            logger.info(f"MATLAB file saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save MATLAB file {file_path}: {e}")
            raise
    
    @staticmethod
    def extract_fft(mat_data: Dict[str, Any], key: str = "fft") -> Optional[np.ndarray]:
        """
        Extract FFT data from MATLAB file.
        
        Args:
            mat_data: Dictionary from loaded .mat file
            key: Key containing FFT data (default: 'fft')
            
        Returns:
            FFT array or None if not found
        """
        if key not in mat_data:
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            logger.warning(f"Key '{key}' not found in MATLAB data. Available: {available_keys}")
            return None
        
        fft_data = mat_data[key]
        
        # MATLAB returns transposed arrays
        if isinstance(fft_data, np.ndarray) and fft_data.ndim == 2:
            fft_data = fft_data.T
        
        logger.debug(f"Extracted FFT data shape: {fft_data.shape}")
        return fft_data
    
    @staticmethod
    def extract_labels(mat_data: Dict[str, Any], key: str = "labels") -> Optional[np.ndarray]:
        """
        Extract labels from MATLAB file.
        
        Args:
            mat_data: Dictionary from loaded .mat file
            key: Key containing labels
            
        Returns:
            Labels array or None if not found
        """
        if key not in mat_data:
            logger.warning(f"Key '{key}' not found in MATLAB data")
            return None
        
        labels = mat_data[key].flatten()
        logger.debug(f"Extracted labels shape: {labels.shape}")
        return labels
    
    @staticmethod
    def convert_to_numpy(
        mat_data: Dict[str, Any],
        remove_metadata: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Convert MATLAB data to NumPy arrays.
        
        Args:
            mat_data: Dictionary from loaded .mat file
            remove_metadata: Remove MATLAB metadata keys (__*)
            
        Returns:
            Dictionary with NumPy arrays
        """
        converted = {}
        
        for key, value in mat_data.items():
            if remove_metadata and key.startswith('__'):
                continue
            
            if isinstance(value, np.ndarray):
                # Convert MATLAB's transposed matrices back
                if value.ndim == 2 and value.shape[0] == 1:
                    value = value.flatten()
                converted[key] = value
            else:
                converted[key] = np.array(value)
        
        logger.debug(f"Converted {len(converted)} MATLAB variables to NumPy")
        return converted
    
    @staticmethod
    def compare_fft_with_matlab(
        python_fft: np.ndarray,
        matlab_fft: np.ndarray,
        return_correlation: bool = True
    ) -> Dict[str, float]:
        """
        Compare Python FFT with MATLAB FFT.
        
        Args:
            python_fft: Python-computed FFT
            matlab_fft: MATLAB-computed FFT
            return_correlation: Include correlation coefficient
            
        Returns:
            Dictionary with comparison metrics
        """
        # Ensure same shape
        if python_fft.shape != matlab_fft.shape:
            logger.warning(f"FFT shapes differ: Python {python_fft.shape} vs MATLAB {matlab_fft.shape}")
            # Try to align shapes
            min_len = min(python_fft.shape[0] if python_fft.ndim > 0 else 1,
                         matlab_fft.shape[0] if matlab_fft.ndim > 0 else 1)
            python_fft = python_fft[:min_len]
            matlab_fft = matlab_fft[:min_len]
        
        # Normalize for comparison
        python_norm = python_fft / (np.max(np.abs(python_fft)) + 1e-10)
        matlab_norm = matlab_fft / (np.max(np.abs(matlab_fft)) + 1e-10)
        
        # Compute metrics
        mse = np.mean((python_norm - matlab_norm) ** 2)
        mae = np.mean(np.abs(python_norm - matlab_norm))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'max_diff': float(np.max(np.abs(python_norm - matlab_norm)))
        }
        
        if return_correlation:
            correlation = float(np.corrcoef(
                python_norm.flatten(),
                matlab_norm.flatten()
            )[0, 1])
            metrics['correlation'] = correlation
        
        logger.debug(f"FFT comparison - MSE: {mse:.6f}, Correlation: {metrics.get('correlation', 'N/A')}")
        
        return metrics
    
    @staticmethod
    def load_structure_array(
        mat_data: Dict[str, Any],
        key: str
    ) -> Dict[str, np.ndarray]:
        """
        Load MATLAB struct array.
        
        Args:
            mat_data: Dictionary from loaded .mat file
            key: Key containing struct array
            
        Returns:
            Dictionary with struct fields
        """
        if key not in mat_data:
            raise KeyError(f"Key '{key}' not found in MATLAB data")
        
        struct_array = mat_data[key]
        
        # MATLAB struct arrays are stored as object arrays
        if struct_array.dtype == object and struct_array.size > 0:
            result = {}
            struct = struct_array[0, 0]
            
            if hasattr(struct, '_fieldnames'):
                for field in struct._fieldnames:
                    result[field] = getattr(struct, field)
            
            logger.debug(f"Loaded struct '{key}' with fields: {list(result.keys())}")
            return result
        
        logger.warning(f"Could not parse struct array '{key}'")
        return {}
    
    @staticmethod
    def save_structure(
        data: Dict[str, Any],
        file_path: Union[str, Path],
        struct_name: str = "data"
    ) -> None:
        """
        Save data as MATLAB struct.
        
        Args:
            data: Dictionary to save as struct
            file_path: Path to save .mat file
            struct_name: Name of struct in MATLAB
        """
        if not MATLABBridge._scipy_available:
            raise ImportError("scipy is required to save MATLAB files")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to MATLAB-compatible format
        matlab_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                matlab_data[key] = value
            else:
                matlab_data[key] = np.array(value)
        
        try:
            MATLABBridge._sio.savemat(str(file_path), {struct_name: matlab_data})
            logger.info(f"MATLAB struct saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save MATLAB struct: {e}")
            raise