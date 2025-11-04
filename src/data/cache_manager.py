"""Cache processed data for faster loading in subsequent runs."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class CacheManager:
    """Manage caching of preprocessed data with version control."""

    def __init__(self, cache_dir: Path | str = "data/cache", version: str = "1.0") -> None:
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory for storing cached data
            version: Cache version for invalidation
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.metadata_file = self.cache_dir / "cache_metadata.json"

    def _get_cache_key(self, data_identifier: str) -> str:
        """Generate cache key from identifier.
        
        Args:
            data_identifier: Unique identifier for data (e.g., file path)
            
        Returns:
            Cache key hash
        """
        identifier_bytes = data_identifier.encode('utf-8')
        return hashlib.md5(identifier_bytes).hexdigest()

    def save_preprocessed_fft(
        self,
        frequencies: np.ndarray,
        magnitudes: np.ndarray,
        labels: Optional[np.ndarray] = None,
        data_id: Optional[str] = None,
    ) -> Path:
        """Save preprocessed FFT data to cache.
        
        Args:
            frequencies: Frequency array
            magnitudes: FFT magnitude array (n_samples, n_freqs) or (n_samples, n_freqs, n_channels)
            labels: Optional class labels
            data_id: Optional unique identifier for this data
            
        Returns:
            Path to saved cache file
        """
        if data_id is None:
            data_id = f"fft_{len(magnitudes)}samples"

        cache_key = self._get_cache_key(data_id)
        cache_file = self.cache_dir / f"{cache_key}.npz"

        # Save data
        if labels is not None:
            np.savez(cache_file, frequencies=frequencies, magnitudes=magnitudes, labels=labels)
        else:
            np.savez(cache_file, frequencies=frequencies, magnitudes=magnitudes)

        # Update metadata
        self._update_metadata(cache_key, data_id, cache_file)

        LOGGER.info(f"Cached FFT data to {cache_file}")
        return cache_file

    def load_preprocessed_fft(self, data_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load preprocessed FFT data from cache.
        
        Args:
            data_id: Unique identifier for data
            
        Returns:
            Dictionary with 'frequencies', 'magnitudes', and optionally 'labels'
        """
        cache_key = self._get_cache_key(data_id)
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if not cache_file.exists():
            LOGGER.info(f"Cache miss for {data_id}")
            return None

        try:
            cached_data = np.load(cache_file, allow_pickle=False)
            result = {
                "frequencies": cached_data["frequencies"],
                "magnitudes": cached_data["magnitudes"],
            }
            if "labels" in cached_data:
                result["labels"] = cached_data["labels"]

            LOGGER.info(f"Loaded FFT data from cache: {cache_file}")
            return result
        except Exception as e:
            LOGGER.error(f"Failed to load cache file {cache_file}: {e}")
            return None

    def save_memory_mapped(
        self,
        data: np.ndarray,
        data_id: str,
        dtype: str = "float32",
    ) -> Path:
        """Save data as memory-mapped array for efficient access.
        
        Args:
            data: Input array
            data_id: Unique identifier
            dtype: Data type
            
        Returns:
            Path to saved memory-mapped file
        """
        cache_key = self._get_cache_key(data_id)
        cache_file = self.cache_dir / f"{cache_key}.dat"

        # Create memory-mapped array
        mmap_array = np.memmap(cache_file, dtype=dtype, mode='w+', shape=data.shape)
        mmap_array[:] = data[:]
        mmap_array.flush()

        LOGGER.info(f"Saved memory-mapped data to {cache_file}")
        return cache_file

    def load_memory_mapped(
        self,
        data_id: str,
        shape: tuple,
        dtype: str = "float32",
    ) -> Optional[np.ndarray]:
        """Load memory-mapped array.
        
        Args:
            data_id: Unique identifier
            shape: Expected shape of array
            dtype: Data type
            
        Returns:
            Memory-mapped array or None if not found
        """
        cache_key = self._get_cache_key(data_id)
        cache_file = self.cache_dir / f"{cache_key}.dat"

        if not cache_file.exists():
            LOGGER.info(f"Memory-mapped file not found: {cache_file}")
            return None

        try:
            mmap_array = np.memmap(cache_file, dtype=dtype, mode='r', shape=shape)
            LOGGER.info(f"Loaded memory-mapped data from {cache_file}")
            return mmap_array
        except Exception as e:
            LOGGER.error(f"Failed to load memory-mapped file {cache_file}: {e}")
            return None

    def clear_cache(self, data_id: Optional[str] = None) -> None:
        """Clear cache files.
        
        Args:
            data_id: If specified, only clear this ID; if None, clear all
        """
        if data_id is not None:
            cache_key = self._get_cache_key(data_id)
            cache_file = self.cache_dir / f"{cache_key}.npz"
            dat_file = self.cache_dir / f"{cache_key}.dat"

            for f in [cache_file, dat_file]:
                if f.exists():
                    f.unlink()
                    LOGGER.info(f"Deleted cache file: {f}")
        else:
            # Clear all cache
            for f in self.cache_dir.glob("*.npz"):
                f.unlink()
            for f in self.cache_dir.glob("*.dat"):
                f.unlink()
            LOGGER.info("Cleared all cache files")

    def _update_metadata(self, cache_key: str, data_id: str, cache_file: Path) -> None:
        """Update cache metadata file.
        
        Args:
            cache_key: Cache key
            data_id: Data identifier
            cache_file: Path to cache file
        """
        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

        metadata[cache_key] = {
            "data_id": data_id,
            "cache_file": str(cache_file),
            "version": self.version,
        }

        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def is_cache_valid(self, data_id: str, required_version: str) -> bool:
        """Check if cached data is valid for current version.
        
        Args:
            data_id: Data identifier
            required_version: Required cache version
            
        Returns:
            True if cache exists and version matches
        """
        if not self.metadata_file.exists():
            return False

        cache_key = self._get_cache_key(data_id)

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            if cache_key in metadata:
                cached_version = metadata[cache_key].get("version")
                return cached_version == required_version

            return False
        except Exception as e:
            LOGGER.error(f"Failed to check cache validity: {e}")
            return False

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data.
        
        Returns:
            Dictionary with cache statistics
        """
        npz_files = list(self.cache_dir.glob("*.npz"))
        dat_files = list(self.cache_dir.glob("*.dat"))

        total_size = sum(f.stat().st_size for f in npz_files + dat_files)

        return {
            "cache_dir": str(self.cache_dir),
            "version": self.version,
            "n_npz_files": len(npz_files),
            "n_dat_files": len(dat_files),
            "total_size_mb": total_size / (1024 * 1024),
        }