"""Logging setup module for Air Leak Detection system."""

import logging
import logging.handlers
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        if record.extra:
            log_obj.update(record.extra)
        
        return json.dumps(log_obj)


class LoggerSetup:
    """Configure logging for the application."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    @classmethod
    def setup(
        cls,
        log_dir: str = "logs",
        file_level: int = logging.DEBUG,
        console_level: int = logging.INFO,
        json_format: bool = False,
        module_levels: Optional[Dict[str, int]] = None
    ) -> logging.Logger:
        """
        Setup logging for the application.
        
        Args:
            log_dir: Directory to save log files
            file_level: Logging level for file handler
            console_level: Logging level for console handler
            json_format: Use JSON format for file logs
            module_levels: Dict of module names to logging levels
            
        Returns:
            Root logger
        """
        root_logger = logging.getLogger()
        
        if cls._initialized:
            return root_logger
        
        cls._initialized = True
        
        # Set root logger level to lowest to allow handlers to filter
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = Path(log_dir) / f"air_leak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(file_level)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter(
            '%(levelname)-8s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Set module-specific levels
        if module_levels:
            for module_name, level in module_levels.items():
                logging.getLogger(module_name).setLevel(level)
        
        return root_logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get logger for a specific module.
        
        Args:
            name: Module name
            
        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: int, module_name: str = None) -> None:
        """
        Set logging level.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            module_name: Specific module or None for root logger
        """
        if module_name:
            logging.getLogger(module_name).setLevel(level)
        else:
            logging.getLogger().setLevel(level)
    
    @classmethod
    def add_file_handler(
        cls,
        log_file: str,
        level: int = logging.DEBUG,
        json_format: bool = False
    ) -> logging.FileHandler:
        """
        Add additional file handler to root logger.
        
        Args:
            log_file: Path to log file
            level: Logging level
            json_format: Use JSON format
            
        Returns:
            File handler instance
        """
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(log_file)
        handler.setLevel(level)
        
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        
        return handler


# Convenience function
def setup_logging(
    log_dir: str = "logs",
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    json_format: bool = False,
    module_levels: Optional[Dict[str, int]] = None
) -> logging.Logger:
    """
    Setup logging for application (convenience wrapper).
    
    Args:
        log_dir: Directory to save log files
        file_level: Logging level for file handler
        console_level: Logging level for console handler
        json_format: Use JSON format for file logs
        module_levels: Dict of module names to logging levels
        
    Returns:
        Root logger
    """
    return LoggerSetup.setup(
        log_dir=log_dir,
        file_level=file_level,
        console_level=console_level,
        json_format=json_format,
        module_levels=module_levels
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger for a specific module (convenience function)."""
    return LoggerSetup.get_logger(name)