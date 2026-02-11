"""
Configuration Manager for PumpIQ
Handles loading, validation, and merging of user configurations
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from jsonschema import validate, ValidationError
from dataclasses import dataclass


@dataclass
class ConfigPaths:
    """Configuration file paths"""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    DEFAULT_CONFIG = CONFIG_DIR / "default_config.json"
    CONFIG_SCHEMA = CONFIG_DIR / "config_schema.json"


class ConfigurationManager:
    """
    Manages system and user configurations with validation
    """
    
    def __init__(self):
        self.default_config = self._load_default_config()
        self.schema = self._load_schema()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from file"""
        try:
            with open(ConfigPaths.DEFAULT_CONFIG, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Default config not found at {ConfigPaths.DEFAULT_CONFIG}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in default config: {e}")
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load configuration JSON schema"""
        try:
            with open(ConfigPaths.CONFIG_SCHEMA, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Config schema not found at {ConfigPaths.CONFIG_SCHEMA}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in config schema: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against JSON schema
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate(instance=config, schema=self.schema)
            
            # Additional custom validations
            self._validate_mode_exclusivity(config)
            self._validate_hybrid_weights(config)
            
            return True
        except ValidationError as e:
            raise ValidationError(f"Configuration validation failed: {e.message}")
    
    def _validate_mode_exclusivity(self, config: Dict[str, Any]) -> None:
        """Ensure only one mode is enabled at a time"""
        modes = config.get('modes', {})
        
        # Count enabled single modes
        single_modes = ['news_only_mode', 'onchain_only_mode', 
                       'technical_only_mode', 'social_only_mode']
        enabled_single_modes = sum(
            1 for mode in single_modes 
            if modes.get(mode, {}).get('enabled', False)
        )
        
        hybrid_enabled = modes.get('hybrid_mode', {}).get('enabled', False)
        
        # Either one single mode OR hybrid mode, not both
        if enabled_single_modes > 1:
            raise ValidationError("Only one single mode can be enabled at a time")
        
        if enabled_single_modes > 0 and hybrid_enabled:
            raise ValidationError("Cannot enable single mode and hybrid mode simultaneously")
        
        if enabled_single_modes == 0 and not hybrid_enabled:
            raise ValidationError("At least one mode must be enabled")
    
    def _validate_hybrid_weights(self, config: Dict[str, Any]) -> None:
        """Validate hybrid mode weights sum to 100"""
        modes = config.get('modes', {})
        hybrid_mode = modes.get('hybrid_mode', {})
        
        if hybrid_mode.get('enabled', False):
            weights = hybrid_mode.get('weights', {})
            total_weight = sum(weights.values())
            
            if abs(total_weight - 100) > 0.01:  # Allow for floating point precision
                raise ValidationError(
                    f"Hybrid mode weights must sum to 100, got {total_weight}"
                )
            
            # At least 2 sources must have weight > 0
            active_sources = sum(1 for w in weights.values() if w > 0)
            if active_sources < 2:
                raise ValidationError(
                    "Hybrid mode requires at least 2 data sources with weight > 0"
                )
    
    def get_tier_defaults(self, tier: str) -> Dict[str, Any]:
        """
        Get default configuration for a subscription tier
        
        Args:
            tier: Subscription tier (free, basic, premium, pro)
            
        Returns:
            Dict with tier-specific defaults
        """
        tier_configs = {
            'free': {
                'api_settings': {
                    'rate_limits': {
                        'requests_per_minute': 60,
                        'requests_per_hour': 1000,
                        'requests_per_day': 10000
                    }
                },
                'user_preferences': {
                    'max_recommendations': 5
                }
            },
            'basic': {
                'api_settings': {
                    'rate_limits': {
                        'requests_per_minute': 120,
                        'requests_per_hour': 5000,
                        'requests_per_day': 50000
                    }
                },
                'user_preferences': {
                    'max_recommendations': 10
                }
            },
            'premium': {
                'api_settings': {
                    'rate_limits': {
                        'requests_per_minute': 300,
                        'requests_per_hour': 20000,
                        'requests_per_day': 200000
                    }
                },
                'user_preferences': {
                    'max_recommendations': 20
                }
            },
            'pro': {
                'api_settings': {
                    'rate_limits': {
                        'requests_per_minute': 600,
                        'requests_per_hour': 50000,
                        'requests_per_day': 500000
                    }
                },
                'user_preferences': {
                    'max_recommendations': 50
                }
            }
        }
        
        return tier_configs.get(tier, tier_configs['free'])
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge multiple configuration dictionaries
        Later configs override earlier ones
        
        Args:
            *configs: Variable number of config dictionaries
            
        Returns:
            Merged configuration dictionary
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Recursively merge two dictionaries"""
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        result = {}
        for config in configs:
            result = deep_merge(result, config)
        
        return result
    
    def get_user_config(self, user_config: Optional[Dict[str, Any]] = None, 
                       tier: str = 'free') -> Dict[str, Any]:
        """
        Get final configuration for a user
        Merges: default -> tier defaults -> user overrides
        
        Args:
            user_config: User-specific configuration overrides
            tier: User's subscription tier
            
        Returns:
            Final merged and validated configuration
        """
        # Start with default config
        config = self.default_config.copy()
        
        # Apply tier defaults
        tier_defaults = self.get_tier_defaults(tier)
        config = self.merge_configs(config, tier_defaults)
        
        # Apply user overrides if provided
        if user_config:
            config = self.merge_configs(config, user_config)
        
        # Validate final configuration
        self.validate_config(config)
        
        return config
    
    def get_active_sources(self, config: Dict[str, Any]) -> tuple[list[str], Dict[str, float]]:
        """
        Determine which data sources are active and their weights
        
        Args:
            config: User configuration
            
        Returns:
            Tuple of (active_sources list, weights dict)
        """
        modes = config.get('modes', {})
        
        # Check single modes first
        single_mode_map = {
            'news_only_mode': 'news',
            'onchain_only_mode': 'onchain',
            'technical_only_mode': 'technical',
            'social_only_mode': 'social'
        }
        
        for mode_key, source_name in single_mode_map.items():
            if modes.get(mode_key, {}).get('enabled', False):
                return [source_name], {source_name: 100.0}
        
        # Default to hybrid mode
        if modes.get('hybrid_mode', {}).get('enabled', False):
            weights = modes['hybrid_mode']['weights']
            active_sources = [
                source for source, weight in weights.items() 
                if weight > 0
            ]
            return active_sources, weights
        
        # Fallback (should not reach here if validation passed)
        return [], {}
    
    def get_refresh_interval(self, config: Dict[str, Any], source: str) -> int:
        """
        Get refresh interval for a specific data source
        
        Args:
            config: User configuration
            source: Data source name (news, onchain, technical, social)
            
        Returns:
            Refresh interval in seconds
        """
        intervals = config.get('data_refresh_intervals', {})
        key = f'{source}_collector'
        return intervals.get(key, 300)  # Default 5 minutes
    
    def get_cache_ttl(self, config: Dict[str, Any], source: str) -> int:
        """
        Get cache TTL for a specific data source
        
        Args:
            config: User configuration
            source: Data source name
            
        Returns:
            Cache TTL in seconds
        """
        cache_settings = config.get('cache_settings', {})
        key = f'{source}_cache_ttl'
        return cache_settings.get(key, 600)  # Default 10 minutes


# Singleton instance
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """Get singleton ConfigurationManager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


if __name__ == "__main__":
    # Example usage
    manager = get_config_manager()
    
    # Get config for a premium user
    user_overrides = {
        "modes": {
            "hybrid_mode": {
                "enabled": True,
                "weights": {
                    "news": 20,
                    "onchain": 40,
                    "technical": 30,
                    "social": 10
                }
            }
        },
        "user_preferences": {
            "risk_tolerance": "aggressive",
            "investment_timeframe": "day_trading"
        }
    }
    
    config = manager.get_user_config(user_overrides, tier='premium')
    active_sources, weights = manager.get_active_sources(config)
    
    print(f"Active sources: {active_sources}")
    print(f"Weights: {weights}")
    print(f"Risk tolerance: {config['user_preferences']['risk_tolerance']}")
