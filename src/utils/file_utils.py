"""File operations utilities for Air Leak Detection system."""

import os
import shutil
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union
import logging


logger = logging.getLogger(__name__)


class FileUtils:
    """Utilities for file operations."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Create directory if it doesn't exist.
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
        return path
    
    @staticmethod
    def safe_save_file(
        data: Any,
        file_path: Union[str, Path],
        format: str = "pickle",
        backup: bool = True
    ) -> None:
        """
        Safely save data to file with optional backup.
        
        Args:
            data: Data to save
            file_path: Path to save file
            format: Format ('pickle', 'json', 'txt')
            backup: Create backup if file exists
        """
        file_path = Path(file_path)
        
        # Create directory if needed
        FileUtils.ensure_directory(file_path.parent)
        
        # Backup existing file
        if backup and file_path.exists():
            backup_path = file_path.with_stem(file_path.stem + ".backup")
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Backup created: {backup_path}")
        
        # Save with temporary file first
        temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        
        try:
            if format == "pickle":
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
            elif format == "json":
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif format == "txt":
                with open(temp_path, 'w') as f:
                    f.write(str(data))
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Move temp file to final location
            shutil.move(str(temp_path), str(file_path))
            logger.info(f"File saved: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save file {file_path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    @staticmethod
    def safe_load_file(
        file_path: Union[str, Path],
        format: str = "pickle",
        default: Any = None
    ) -> Any:
        """
        Safely load data from file.
        
        Args:
            file_path: Path to load file from
            format: Format ('pickle', 'json', 'txt')
            default: Default value if file not found
            
        Returns:
            Loaded data or default
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return default
        
        try:
            if format == "pickle":
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            elif format == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif format == "txt":
                with open(file_path, 'r') as f:
                    data = f.read()
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"File loaded: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            if default is not None:
                return default
            raise
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    @staticmethod
    def get_human_readable_size(size_bytes: int) -> str:
        """
        Convert bytes to human readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Human readable size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    @staticmethod
    def list_files(
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False
    ) -> list:
        """
        List files in directory matching pattern.
        
        Args:
            directory: Directory path
            pattern: File pattern (e.g., '*.csv')
            recursive: Search recursively
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))
        
        logger.debug(f"Found {len(files)} files in {directory}")
        return sorted(files)
    
    @staticmethod
    def remove_file(file_path: Union[str, Path]) -> bool:
        """
        Safely remove file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if successful
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        try:
            file_path.unlink()
            logger.debug(f"File removed: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove file {file_path}: {e}")
            raise
    
    @staticmethod
    def remove_directory(directory: Union[str, Path], recursive: bool = True) -> bool:
        """
        Safely remove directory.
        
        Args:
            directory: Directory path
            recursive: Remove recursively
            
        Returns:
            True if successful
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return False
        
        try:
            if recursive:
                shutil.rmtree(directory)
            else:
                directory.rmdir()
            logger.debug(f"Directory removed: {directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove directory {directory}: {e}")
            raise
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        Copy file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        src = Path(src)
        dst = Path(dst)
        
        FileUtils.ensure_directory(dst.parent)
        
        try:
            shutil.copy2(src, dst)
            logger.debug(f"File copied: {src} -> {dst}")
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            raise
    
    @staticmethod
    def get_absolute_path(path: Union[str, Path]) -> Path:
        """
        Get absolute path.
        
        Args:
            path: Path (relative or absolute)
            
        Returns:
            Absolute path
        """
        return Path(path).resolve()
    
    @staticmethod
    def get_relative_path(
        path: Union[str, Path],
        start: Union[str, Path] = "."
    ) -> Path:
        """
        Get relative path.
        
        Args:
            path: Path
            start: Start directory
            
        Returns:
            Relative path
        """
        return Path(path).relative_to(Path(start))