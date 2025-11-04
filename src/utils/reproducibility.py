"""Reproducibility utilities for Air Leak Detection system."""

import os
import random
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """Manage reproducibility and deterministic operations."""
    
    _seed_value = None
    _version_info = {}
    
    @classmethod
    def set_seed(cls, seed: int = 42) -> None:
        """
        Set random seed for all libraries.
        
        Args:
            seed: Random seed value
        """
        cls._seed_value = seed
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # OS random
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"Random seed set to {seed}")
        
        # Try to set TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            logger.debug("TensorFlow seed set")
        except (ImportError, AttributeError):
            pass
        
        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            logger.debug("PyTorch seed set")
        except ImportError:
            pass
    
    @classmethod
    def enable_deterministic(cls) -> None:
        """Enable deterministic mode for reproducibility."""
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Try to enable TensorFlow deterministic mode
        try:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
            logger.debug("TensorFlow deterministic mode enabled")
        except (ImportError, AttributeError):
            pass
        
        # Try to enable PyTorch deterministic mode
        try:
            import torch
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.debug("PyTorch deterministic mode enabled")
        except (ImportError, AttributeError):
            pass
    
    @classmethod
    def record_versions(cls) -> Dict[str, str]:
        """
        Record version information of key libraries.
        
        Returns:
            Dictionary of library versions
        """
        versions = {}
        
        # Python version
        import sys
        versions['python'] = sys.version.split()[0]
        
        # NumPy
        versions['numpy'] = np.__version__
        
        # SciPy
        try:
            import scipy
            versions['scipy'] = scipy.__version__
        except ImportError:
            pass
        
        # scikit-learn
        try:
            import sklearn
            versions['sklearn'] = sklearn.__version__
        except ImportError:
            pass
        
        # TensorFlow
        try:
            import tensorflow as tf
            versions['tensorflow'] = tf.__version__
        except ImportError:
            pass
        
        # PyTorch
        try:
            import torch
            versions['torch'] = torch.__version__
        except ImportError:
            pass
        
        # Pandas
        try:
            import pandas as pd
            versions['pandas'] = pd.__version__
        except ImportError:
            pass
        
        # Matplotlib
        try:
            import matplotlib
            versions['matplotlib'] = matplotlib.__version__
        except ImportError:
            pass
        
        cls._version_info = versions
        logger.debug(f"Recorded versions: {versions}")
        
        return versions
    
    @classmethod
    def get_versions(cls) -> Dict[str, str]:
        """
        Get recorded version information.
        
        Returns:
            Dictionary of library versions
        """
        if not cls._version_info:
            return cls.record_versions()
        return cls._version_info.copy()
    
    @classmethod
    def save_reproducibility_info(
        cls,
        output_path: Union[str, Path],
        seed: Optional[int] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save reproducibility information to file.
        
        Args:
            output_path: Path to save info
            seed: Random seed used
            additional_info: Additional reproducibility info
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        info = {
            'seed': seed or cls._seed_value,
            'versions': cls.get_versions(),
            'additional': additional_info or {}
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2)
            logger.info(f"Reproducibility info saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save reproducibility info: {e}")
            raise
    
    @classmethod
    def compute_data_hash(
        cls,
        data: np.ndarray,
        hash_algorithm: str = "sha256"
    ) -> str:
        """
        Compute hash of data for integrity checking.
        
        Args:
            data: Data to hash
            hash_algorithm: Algorithm ('md5', 'sha256', 'sha512')
            
        Returns:
            Hex digest of hash
        """
        data_bytes = data.tobytes()
        
        if hash_algorithm == "md5":
            hasher = hashlib.md5()
        elif hash_algorithm == "sha256":
            hasher = hashlib.sha256()
        elif hash_algorithm == "sha512":
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unknown hash algorithm: {hash_algorithm}")
        
        hasher.update(data_bytes)
        return hasher.hexdigest()
    
    @classmethod
    def verify_data_integrity(
        cls,
        data: np.ndarray,
        expected_hash: str,
        hash_algorithm: str = "sha256"
    ) -> bool:
        """
        Verify data integrity using hash.
        
        Args:
            data: Data to verify
            expected_hash: Expected hash value
            hash_algorithm: Algorithm used
            
        Returns:
            True if hash matches
        """
        actual_hash = cls.compute_data_hash(data, hash_algorithm)
        
        if actual_hash == expected_hash:
            logger.debug("Data integrity verified")
            return True
        else:
            logger.warning(f"Data integrity check failed: {actual_hash} != {expected_hash}")
            return False


# Convenience functions
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    ReproducibilityManager.set_seed(seed)


def enable_deterministic() -> None:
    """Enable deterministic mode."""
    ReproducibilityManager.enable_deterministic()


def record_versions() -> Dict[str, str]:
    """Record library versions."""
    return ReproducibilityManager.record_versions()


def get_versions() -> Dict[str, str]:
    """Get recorded versions."""
    return ReproducibilityManager.get_versions()


def compute_data_hash(data: np.ndarray, hash_algorithm: str = "sha256") -> str:
    """Compute hash of data."""
    return ReproducibilityManager.compute_data_hash(data, hash_algorithm)


def verify_data_integrity(
    data: np.ndarray,
    expected_hash: str,
    hash_algorithm: str = "sha256"
) -> bool:
    """Verify data integrity."""
    return ReproducibilityManager.verify_data_integrity(data, expected_hash, hash_algorithm)