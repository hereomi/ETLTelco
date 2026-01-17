"""Utility functions for memory management, sampling, and helper operations."""

import re
from typing import Any, Optional
import pandas as pd
import numpy as np

def smart_sample(series: pd.Series, max_size: int, strategy: str = "stratified") -> pd.Series:
    """Intelligently sample series for type inference."""
    clean = series.dropna()
    if len(clean) <= max_size:
        return clean
    if strategy == "stratified":
        n_chunks = min(5, len(clean) // 100)
        if n_chunks <= 1:
            return clean.sample(max_size, random_state=42)
        chunk_size = len(clean) // n_chunks
        samples = [clean.iloc[i*chunk_size:(i+1)*chunk_size].sample(min(max_size//n_chunks, chunk_size), random_state=42) for i in range(n_chunks)]
        return pd.concat(samples)
    return clean.sample(max_size, random_state=42)

def validate_identifier(name: Optional[str]) -> Optional[str]:
    """Validate and sanitize SQL identifier names."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError(f"Identifier must be a string: {name!r}")
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        return name
    raise ValueError(f"Invalid identifier name: {name}")

def quote_identifier(name: str, dialect: str) -> str:
    """Quote identifier based on database dialect."""
    safe_name = validate_identifier(name)
    if dialect in ('mysql', 'mariadb'):
        return f'`{safe_name}`'
    elif dialect == 'mssql':
        return f'[{safe_name}]'
    else:
        return f'"{safe_name}"'

def estimate_memory_usage(df: pd.DataFrame) -> float:
    """Estimate DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1024 / 1024

def should_chunk_processing(df: pd.DataFrame, chunk_size: int) -> bool:
    """Determine if DataFrame should be processed in chunks."""
    return len(df) > chunk_size or estimate_memory_usage(df) > 500

def validate_conversion_quality(original: pd.Series, converted: pd.Series, max_null_increase: float) -> bool:
    """Check if conversion maintains acceptable data quality."""
    if len(original) == 0:
        return True
    null_before = original.isna().sum()
    null_after = converted.isna().sum()
    null_increase = (null_after - null_before) / len(original)
    return null_increase <= max_null_increase

def get_sample_failures(series: pd.Series, failure_mask: pd.Series, max_samples: int = 5) -> list:
    """Extract sample values that failed validation."""
    failed_values = series[failure_mask]
    if len(failed_values) == 0:
        return []
    return failed_values.head(max_samples).tolist()

def normalize_dtype_name(dtype_name: str) -> str:
    """Normalize pandas dtype names for consistent handling."""
    dtype_lower = dtype_name.lower()
    if 'int' in dtype_lower:
        return 'integer'
    elif 'float' in dtype_lower:
        return 'float'
    elif 'bool' in dtype_lower:
        return 'boolean'
    elif 'datetime' in dtype_lower:
        return 'datetime'
    elif 'object' in dtype_lower or 'string' in dtype_lower:
        return 'string'
    return 'object'

def safe_astype(series: pd.Series, target_dtype: str, errors: str = 'coerce') -> pd.Series:
    """Safely convert series to target dtype with error handling."""
    try:
        if target_dtype == 'Int64':
            return pd.to_numeric(series, errors=errors).astype('Int64')
        elif target_dtype == 'Float64':
            return pd.to_numeric(series, errors=errors).astype('Float64')
        elif target_dtype == 'boolean':
            return series.astype('boolean')
        elif target_dtype == 'datetime64[ns]':
            return pd.to_datetime(series, errors=errors)
        else:
            return series.astype(target_dtype)
    except Exception:
        return series