"""Configuration management module for Air Leak Detection system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading, validation, and environment override."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to YAML config file. Defaults to 'config.yaml'
        """
        self.config_path = config_path or "config.yaml"
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}, using empty config")
            self.config = {}
            return
        
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                self.config = loaded_config
                logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports nested keys like 'data.raw_data_path')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports nested keys like 'data.raw_data_path')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def override_from_env(self, prefix: str = "ALD_") -> None:
        """
        Override configuration from environment variables.
        
        Args:
            prefix: Environment variable prefix (e.g., 'ALD_')
                   Variables like ALD_DATA__RAW_DATA_PATH will be parsed
        """
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Remove prefix and convert to config key
            config_key = env_key[len(prefix):].lower().replace('__', '.')
            
            # Try to parse as YAML to handle types
            try:
                # Try parsing as number or boolean
                if env_value.lower() in ('true', 'false'):
                    value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    value = int(env_value)
                elif self._is_float(env_value):
                    value = float(env_value)
                else:
                    value = env_value
            except (ValueError, AttributeError):
                value = env_value
            
            self.set(config_key, value)
            logger.debug(f"Configuration overridden from env: {config_key} = {value}")
    
    def validate(self, required_keys: list = None) -> bool:
        """
        Validate configuration contains required keys.
        
        Args:
            required_keys: List of required keys
            
        Returns:
            True if valid, raises exception otherwise
        """
        if required_keys is None:
            required_keys = []
        
        for key in required_keys:
            if self.get(key) is None:
                raise ValueError(f"Required configuration key missing: {key}")
        
        return True
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self.config.copy()
    
    def save(self, output_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def merge(self, other_config: Dict[str, Any]) -> None:
        """
        Merge another configuration into current config.
        
        Args:
            other_config: Configuration dictionary to merge
        """
        def _merge_dicts(base: dict, override: dict) -> dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    _merge_dicts(base[key], value)
                else:
                    base[key] = value
            return base
        
        _merge_dicts(self.config, other_config)
        logger.debug("Configuration merged successfully")
    
    @staticmethod
    def _is_float(value: str) -> bool:
        """Check if string can be parsed as float."""
        try:
            float(value)
            return True
        except ValueError:
            return False