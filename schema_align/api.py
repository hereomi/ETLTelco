"""External API for DataAligner with configuration and decorator support."""

import yaml
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
import pandas as pd
from sqlalchemy import Engine, Connection

from .core import DataAligner
from .config import AlignmentConfig, ValidationMode, OutlierMethod

# Load default configuration
_config_path = Path(__file__).parent / "aligner.yml"
with open(_config_path, 'r') as f:
    _default_config = yaml.safe_load(f)

def _merge_config(base_config: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge configuration dictionaries."""
    if not overrides:
        return base_config.copy()
    merged = base_config.copy()
    merged.update(overrides)
    return merged

def _create_alignment_config(config_dict: Dict[str, Any]) -> AlignmentConfig:
    """Create AlignmentConfig from dictionary."""
    methods = config_dict.get("outlier_methods", [OutlierMethod.ISOLATION_FOREST])
    norm_methods = []
    for m in methods:
        try:
            norm_methods.append(OutlierMethod(m))
        except Exception:
            norm_methods.append(m)
    return AlignmentConfig(
        validation_mode=ValidationMode(config_dict.get("validation_mode", "balanced")),
        failure_threshold=config_dict.get("failure_threshold", 0.1),
        outlier_detection=config_dict.get("outlier_detection", True),
        outlier_methods=norm_methods,
        parallel_processing=config_dict.get("parallel_processing", False),
        cache_enabled=config_dict.get("enable_caching", config_dict.get("cache_enabled", True)),
        max_sample_size=config_dict.get("max_sample_size", 10000),
        chunk_size=config_dict.get("chunk_size", 10000),
        collect_diagnostics=config_dict.get("collect_diagnostics", True),
        enable_metrics=config_dict.get("enable_metrics", False),
        max_null_increase=config_dict.get("max_null_increase", 0.1)
    )

def aligner(engine: Union[Engine, Connection], dataframe: pd.DataFrame, tablename: str, config_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Align DataFrame to SQL table schema with automatic resource management.
    
    Args:
        engine: SQLAlchemy Engine or Connection
        dataframe: Input DataFrame to align
        tablename: Target table name
        config_dict: Configuration overrides (optional)
    
    Returns:
        Aligned DataFrame ready for SQL insertion
    """
    # Merge configuration
    final_config = _merge_config(_default_config["defaults"], config_dict)
    alignment_config = _create_alignment_config(final_config)
    
    # Initialize aligner
    data_aligner = DataAligner(config=alignment_config)
    
    try:
        # Perform alignment
        if isinstance(engine, Engine):
            with engine.connect() as conn:
                return data_aligner.align(conn, dataframe, tablename)
        else:
            return data_aligner.align(engine, dataframe, tablename)
    finally:
        # Cleanup resources
        data_aligner.cleanup()

def align_to_sql(tablename: str, config_dict: Optional[Dict[str, Any]] = None, preset: Optional[str] = None):
    """
    Decorator for automatic DataFrame-to-SQL alignment.
    
    Args:
        tablename: Target table name
        config_dict: Configuration overrides
        preset: Use preset configuration (strict, fast, safe)
    
    Usage:
        @align_to_sql("users", {"failure_threshold": 0.05})
        def process_users(engine, df):
            return df.to_sql("users", engine, if_exists="append", index=False)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(engine: Union[Engine, Connection], dataframe: pd.DataFrame, *args, **kwargs):
            # Use preset if specified
            if preset and preset in _default_config["presets"]:
                base_config = _default_config["presets"][preset].copy()
            else:
                base_config = _default_config["defaults"].copy()
            
            # Merge with custom config
            final_config = _merge_config(base_config, config_dict)
            
            # Align DataFrame
            aligned_df = aligner(engine, dataframe, tablename, final_config)
            
            # Call original function with aligned DataFrame
            return func(engine, aligned_df, *args, **kwargs)
        return wrapper
    return decorator

# Convenience functions for presets
def strict_aligner(engine: Union[Engine, Connection], dataframe: pd.DataFrame, tablename: str, config_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Strict alignment with minimal tolerance for errors."""
    preset_config = _default_config["presets"]["strict"].copy()
    final_config = _merge_config(preset_config, config_dict)
    return aligner(engine, dataframe, tablename, final_config)

def fast_aligner(engine: Union[Engine, Connection], dataframe: pd.DataFrame, tablename: str, config_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fast alignment optimized for performance."""
    preset_config = _default_config["presets"]["fast"].copy()
    final_config = _merge_config(preset_config, config_dict)
    return aligner(engine, dataframe, tablename, final_config)

def safe_aligner(engine: Union[Engine, Connection], dataframe: pd.DataFrame, tablename: str, config_dict: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Safe alignment with schema evolution enabled."""
    preset_config = _default_config["presets"]["safe"].copy()
    final_config = _merge_config(preset_config, config_dict)
    return aligner(engine, dataframe, tablename, final_config)