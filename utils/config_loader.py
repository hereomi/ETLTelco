"""
Secure Configuration Loader with Environment Variable Support

This module provides secure configuration loading with environment variable
substitution and validation.

Usage:
    from utils.config_loader import load_config
    
    config = load_config('config.yml')
    db_uri = config['server']['oracle']  # Automatically loads from env vars
"""
import yaml
import os
import re
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.
    
    Supports syntax: ${VAR_NAME} or ${VAR_NAME:default_value}
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ''
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replacer, value)
    
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    
    return value


def load_config(config_path: str = 'config.yml', strict: bool = False) -> Dict[str, Any]:
    """
    Load configuration file with environment variable substitution.
    
    Args:
        config_path: Path to YAML configuration file
        strict: If True, raise error if environment variables are missing
    
    Returns:
        Dictionary with configuration values
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If strict=True and required env vars are missing
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    # Validate in strict mode
    if strict:
        _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate that all required configuration values are present.
    
    Raises:
        ValueError: If required values are missing
    """
    missing = []
    
    # Check database URIs
    for db_type in ['mysql', 'mssql', 'postgre', 'oracle']:
        uri = config.get('server', {}).get(db_type, '')
        if not uri or 'YOUR_PASSWORD' in uri or 'PASSWORD_HERE' in uri:
            missing.append(f"server.{db_type}")
    
    # Check Telegram token
    token = config.get('telegram', {}).get('token', '')
    if not token or 'YOUR_BOT_TOKEN' in token:
        missing.append("telegram.token")
    
    if missing:
        raise ValueError(
            f"Missing or invalid configuration values: {', '.join(missing)}. "
            "Please check your .env file and ensure all required variables are set."
        )


def get_db_uri(config: Dict[str, Any], db_type: str = None) -> str:
    """
    Get database URI for specified type or active server.
    
    Args:
        config: Configuration dictionary
        db_type: Database type (mysql, mssql, postgre, oracle, sqlite)
                 If None, uses active_server from config
    
    Returns:
        Database connection URI
    """
    if db_type is None:
        db_type = config.get('active_server', 'oracle')
    
    uri = config.get('server', {}).get(db_type)
    
    if not uri:
        raise ValueError(f"No database URI configured for: {db_type}")
    
    return uri


# Example usage
if __name__ == "__main__":
    try:
        config = load_config('config.yml', strict=True)
        print("✓ Configuration loaded successfully")
        print(f"Active server: {config['active_server']}")
        
        # Get database URI (passwords will be masked in logs)
        db_uri = get_db_uri(config)
        print(f"Database URI: {db_uri[:20]}...{db_uri[-20:]}")  # Masked
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
