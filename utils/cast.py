"""
casting_dataframe.py
--------------------
Enhanced helpers to discover & cast pandas-compatible tabular objects.
Integrates transform_dtypes from sqlutils.py and ML-like classification.

Public API
----------
parse_datetime(obj)           -> scalar / list / Series (tolerant ISO)
split_series_by_percent       -> (head, tail) of a Series
datetime_column_name(df)      -> DataFrame (optimisation, returns copy)
transform_dtypes(df, dtype)   -> DataFrame (column name hints + currency handling)
auto_convert_dtypes_ml(df)    -> DataFrame (ML-like classification)
cast_df(obj, dtype, options)  -> DataFrame (new copy, auto-typed with multiple strategies)

The `cast_df` entry point accepts:
• pandas.DataFrame
• mapping (dict)            → 1-row or n-row frame
• list / tuple of mappings  → n-row frame

Enhancements:
• Column name hints for datetime detection (date, time, timestamp, _at, etc.)
• Currency symbol handling ($, €, £, ¥, ₹) in numeric columns
• Enhanced boolean detection (true/false, yes/no, on/off, 1/0, etc.)
• ML-like classification with feature extraction
• Multiple casting strategies: transform_dtypes (default), ML classification, original
"""

from __future__ import annotations
import logging, math, re
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import pandas as pd
import numpy as np
from collections import Counter

logging.basicConfig(level=logging.WARNING)
_LOG = logging.getLogger(__name__)

# ────────────────────────────────────────────────── #
# Enhanced datetime parsing with multiple strategies #
# ────────────────────────────────────────────────── #
try:
    from dateutil import parser as _du
except ImportError:                                    # pragma: no cover
    _du = None

_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$"), "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$"),       "%Y-%m-%d %H:%M"),
    (re.compile(r"\d{14}$"),                                 "%Y%m%d%H%M%S"),
    (re.compile(r"\d{12}$"),                                 "%Y%m%d%H%M"),
    (re.compile(r"\d{8}$"),                                  "%Y%m%d"),
    (re.compile(r"\d{4}-\d{2}-\d{2}$"),                      "%Y-%m-%d"),
    (re.compile(r"\d{4}/\d{2}/\d{2}$"),                      "%Y/%m/%d"),
    (re.compile(r"\d{4}\.\d{2}\.\d{2}$"),                    "%Y.%m.%d"),
    (re.compile(r"\d{2}/\d{2}/\d{4}$"),                      "%d/%m/%Y"),
    (re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}$"),        "%Y/%m/%d %H:%M"),
    (re.compile(r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}$"),        "%d/%m/%Y %H:%M"),
)

def _parse_one(val: Any) -> Optional[str]:
    if val is None: return None
    s = str(val).strip()
    if not s:       return None
    for rgx, fmt in _PATTERNS:
        if rgx.fullmatch(s):
            try: return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OverflowError):
                break
    if _du:
        try: return _du.parse(s).strftime("%Y-%m-%d %H:%M")
        except (ValueError, OverflowError):
            pass
    else:
        _LOG.debug("dateutil not available; falling back to None for parsing: %s", s)
    return None

def parse_datetime(obj: Any) -> Any:
    if isinstance(obj, (str, bytes)) or not isinstance(obj, Iterable):
        return _parse_one(obj)
    if isinstance(obj, pd.Series):
        return obj.apply(_parse_one)
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_parse_one(x) for x in obj)
    return None

# ────────────────────────────────────────────────── #
# misc helpers                                       #
# ────────────────────────────────────────────────── #
def split_series_by_percent(s: pd.Series, pct: float = .05) -> tuple[pd.Series, pd.Series]:
    k = max(1, math.ceil(len(s) * pct))
    return s.head(k), s.tail(k)

def datetime_column_name(df: pd.DataFrame, nan_threshold: float = 0.30) -> pd.DataFrame:
    df = df.copy()  # Avoid in-place modification
    if len(df) == 0:
        return df
    keys = ("time", "date", "_at")
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        if not any(k in col.lower() for k in keys): continue
        src       = df[col]
        na_before = src.isna().sum()
        parsed    = src.apply(_parse_one)
        tgt       = pd.to_datetime(parsed, errors="coerce", format="%Y-%m-%d %H:%M") \
                    if parsed.notna().any() else pd.to_datetime(src, errors="coerce")
        na_after  = tgt.isna().sum()
        if max(0, na_after - na_before) / len(df) <= nan_threshold:
            df[col] = tgt
    return df
    
# ────────────────────────────────────────────────── #
# Outlier Detection & Replacement Logic              #
# ────────────────────────────────────────────────── #

def detect_outliers(df: pd.DataFrame, columns: list[str] | None = None, method: str = 'iqr', threshold: float = 1.5, z_thresh: float = 3.0, drop_nans: bool = True) -> pd.Series:
    """
    Detect outliers in specified columns. Returns a boolean mask where True means 'not an outlier'.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    dfc = df.copy()
    if columns is None:
        columns = dfc.select_dtypes(include=[np.number]).columns.tolist()
    
    if not columns:
        return pd.Series(True, index=dfc.index)
    
    mask = pd.Series(True, index=dfc.index)
    
    for col in columns:
        x = dfc[col].dropna() if drop_nans else dfc[col]
        if len(x) == 0:
            continue
        
        if method == 'iqr':
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            col_mask = dfc[col].between(lower, upper, inclusive='both')
            if not drop_nans:
                col_mask = col_mask.fillna(False)
            mask &= col_mask
        elif method == 'zscore':
            z = (x - x.mean()) / x.std(ddof=1)
            col_mask = pd.Series(abs(z) < z_thresh, index=x.index)
            if drop_nans:
                full_mask = pd.Series(True, index=dfc.index)
                full_mask.loc[x.index] = col_mask
                col_mask = full_mask
            else:
                col_mask = col_mask.reindex(dfc.index).fillna(False)
            mask &= col_mask
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'.")
    
    return mask

def replace_outliers_with_zero_safe(df: pd.DataFrame, columns: list[str] | None = None, method: str = 'iqr', threshold: float = 1.5, z_thresh: float = 3.0, drop_nans: bool = True, replace_nans_with_zero: bool = True) -> pd.DataFrame:
    """
    Replace outliers in numeric columns with 0. 
    Can also replace NaNs with 0 if specified.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    dfc = df.copy()
    if columns is None:
        numeric_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
        object_cols = []
        for col in dfc.select_dtypes(include=['object']).columns:
            try:
                temp = pd.to_numeric(dfc[col], errors='coerce').notna().sum()
                if temp > 0:
                    object_cols.append(col)
            except:
                pass
        columns = numeric_cols + object_cols
    
    if not columns:
        return dfc
    
    for col in columns:
        x_numeric = pd.to_numeric(dfc[col], errors='coerce')
        non_numeric_mask = x_numeric.isna()
        x_for_calc = x_numeric.dropna() if drop_nans else x_numeric
        if len(x_for_calc) == 0:
            continue
        if method == 'iqr':
            q1, q3 = x_for_calc.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
            outlier_mask = ~x_numeric.between(lower, upper, inclusive='both')
        elif method == 'zscore':
            z = (x_for_calc - x_for_calc.mean()) / x_for_calc.std(ddof=1)
            outlier_mask = pd.Series(abs(z) >= z_thresh, index=x_for_calc.index)
            if drop_nans:
                full_outlier = pd.Series(False, index=dfc.index)
                full_outlier.loc[x_for_calc.index] = outlier_mask
                outlier_mask = full_outlier
            else:
                outlier_mask = outlier_mask.reindex(dfc.index).fillna(True)
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'.")
        
        to_replace = outlier_mask & ~non_numeric_mask
        dfc.loc[to_replace, col] = 0
        
        if replace_nans_with_zero:
            nan_mask = x_numeric.isna() & dfc[col].isna()
            dfc.loc[nan_mask, col] = 0
            
    return dfc

# ────────────────────────────────────────────────── #
# casting engine                                     #
# ────────────────────────────────────────────────── #
_PATTERN_DICT: Mapping[str, Sequence[str]] = {
    "timestamp": [r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}",
                  r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}",
                  r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}"],
    "date":      [r"\d{4}-\d{2}-\d{2}$", r"\d{2}/\d{2}/\d{4}$",
                  r"\d{4}/\d{2}/\d{2}$", r"\d{2}-\d{2}-\d{4}$"],
    "time":      [r"\d{2}:\d{2}:\d{2}$", r"\d{2}:\d{2}:\d{2}\.\d+$", r"\d{2}:\d{2}$"],
    "duration":  [r"\d+:\d{2}:\d{2}$",
                  r"\d+\s*(hours?|hrs?|h)\s*\d*\s*(minutes?|mins?|m)?",
                  r"\d+\s*days?\s*\d*:\d{2}:\d{2}"],
    "boolean":   [r"(true|false)$", r"(yes|no)$", r"(y|n)$", r"(t|f)$"],
    "binary":    [r"[01]$"],
    "integer":   [r"-?\d+$"],
    "float":     [r"-?\d*\.?\d+$", r"-?\d*\.?\d*[eE][+-]?\d+$"],
}

_BOOL_MAP = {
    "true": True, "false": False, "yes": True, "no": False,
    "y": True, "n": False, "t": True, "f": False, "1": True, "0": False,
    "on": True, "off": False
}

# Enhanced patterns for better type detection
_CURRENCY_PATTERN = re.compile(r'[\$€£¥₹,\s]')
_INT_PATTERN = re.compile(r'^-?\d+$')
_FLOAT_PATTERN = re.compile(r'^-?\d*\.?\d+([eE][-+]?\d+)?$')

def _infer_one(v: str) -> str:
    for name, pats in _PATTERN_DICT.items():
        if any(re.fullmatch(p, v, re.IGNORECASE) for p in pats):
            return name
    return "string"

def _infer(sample: Sequence[str], threshold: float = 0.9) -> str:
    if not sample: return "string"
    types = [_infer_one(v) for v in sample]
    most_common = Counter(types).most_common(1)[0]
    if most_common[1] / len(sample) >= threshold and most_common[0] != "string":
        return most_common[0]
    return "string"

def _warn_on_coerce(original_series: pd.Series, coerced_series: pd.Series, type_name: str):
    """Logs a warning if a coercion operation introduces a high percentage of nulls."""
    na_before = original_series.isna().sum()
    na_after = coerced_series.isna().sum()
    if len(original_series) > 0:
        null_increase_pct = (na_after - na_before) * 100 / len(original_series)
        if null_increase_pct > 10.0:  # 10% threshold
            _LOG.warning(
                "Coercing to %s introduced %.1f%% nulls. Data quality may be poor.",
                type_name, null_increase_pct
            )

def _convert_series(s: pd.Series, target: str) -> pd.Series:
    """Converts a Series to a target type, with warnings for high null counts."""
    try:
        if target in ("timestamp", "date"):
            converted = pd.to_datetime(s, errors="coerce")
            _warn_on_coerce(s, converted, target)
            return converted.dt.date if target == "date" else converted
        
        if target == "time":
            # Stricter format, less risk of incorrect coercion
            return pd.to_datetime(s, format="%H:%M:%S", errors="coerce").dt.time
            
        if target == "duration":
            converted = pd.to_timedelta(s, errors="coerce")
            _warn_on_coerce(s, converted, target)
            return converted
            
        if target == "boolean":
            converted = s.astype(str).str.lower().str.strip().map(_BOOL_MAP)
            _warn_on_coerce(s, converted, target)
            return converted
            
        if target in ("binary", "integer", "float"):
            converted = pd.to_numeric(s, errors="coerce")
            _warn_on_coerce(s, converted, target)
            if target in ("binary", "integer"):
                return converted.astype("Int64")
            return converted.astype("float64")
            
    except Exception as e:
        _LOG.debug("Casting column '%s' to '%s' failed: %s", s.name, target, e)
    return s

def _explicit_cast(s: pd.Series, dt: str) -> pd.Series:
    """Explicitly casts a Series to a specified dtype, with warnings for nulls."""
    try:
        dt_lower = dt.lower()
        
        # DateTime types
        if "datetime" in dt or "date" in dt:
            converted = pd.to_datetime(s, errors="coerce")
            _warn_on_coerce(s, converted, dt)
            return converted
        
        # Timedelta types
        if "timedelta" in dt:
            converted = pd.to_timedelta(s, errors="coerce")
            _warn_on_coerce(s, converted, dt)
            return converted
        
        # Numeric types
        if "int" in dt_lower or "float" in dt_lower:
            converted = pd.to_numeric(s, errors="coerce")
            _warn_on_coerce(s, converted, dt)
            # Use try-except for the final astype, as it can fail (e.g., float to int)
            try:
                return converted.astype(dt)
            except (ValueError, TypeError):
                return converted
        
        # Boolean types
        if "bool" in dt_lower:
            return s.astype(bool)
        
        # String/Object types
        if "str" in dt_lower or "object" in dt_lower:
            return s.astype(str)
        
        # Default fallback
        return s.astype(dt, errors="ignore")
    except Exception as e:
        _LOG.debug("Explicit conversion of '%s' to %s failed: %s", s.name, dt, e)
        return s

def _to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):           return obj.copy()
    if isinstance(obj, Mapping):
        try:
            return pd.DataFrame(obj)
        except ValueError:
            return pd.DataFrame([obj])
        except Exception as e:
            raise ValueError(f"Cannot convert mapping to DataFrame: {e}") from None
    if isinstance(obj, (list, tuple)) and all(isinstance(x, Mapping) for x in obj):
        try:    return pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(f"Cannot convert list of mappings to DataFrame: {e}") from None
    raise ValueError("Unsupported input type for casting")

def transform_dtypes(df: pd.DataFrame, dtype_map: Mapping[str, str] | None = None, nan_threshold: float = 0.30) -> pd.DataFrame:
    """
    Transform DataFrame dtypes with column name hints for date/time detection.
    Enhanced version from sqlutils.py with better datetime handling.
    """
    dtype_map = dtype_map or {}
    df = df.copy()
    
    # Apply explicit dtype mappings first
    for col, override_type in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(override_type) if override_type else df[col]
            except (ValueError, TypeError) as e:
                _LOG.debug(f"Failed to convert {col} to {override_type}: {e}")
    
    # Process object columns with column name hints
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if col in dtype_map:  # Skip already processed columns
            continue
            
        # Enhanced datetime detection with column name hints (per-column)
        if any(x in col.lower() for x in ['date', 'time', 'timestamp', '_at', 'created', 'updated','occur', 'start', 'end', 'period']):
            src = df[col]
            na_before = src.isna().sum()
            parsed = src.apply(_parse_one)
            tgt = pd.to_datetime(parsed, errors="coerce", format="%Y-%m-%d %H:%M") \
                if parsed.notna().any() else pd.to_datetime(src, errors="coerce")
            na_after = tgt.isna().sum()
            # Removed debug print: print(col, na_before, na_after)
            if len(df) > 0 and max(0, na_after - na_before) / len(df) <= nan_threshold:
                df[col] = tgt
            continue
                
        # Enhanced numeric detection with currency handling
        elif _is_numeric_column(df[col]):
            cleaned_col = df[col].astype(str).str.replace(_CURRENCY_PATTERN, '', regex=True).str.strip()
            df[col] = pd.to_numeric(cleaned_col, errors='coerce')
            
        # Enhanced boolean detection
        elif _is_boolean_column(df[col]):
            df[col] = df[col].astype(str).str.lower().str.strip().map(_BOOL_MAP)
            
        else:
            df[col] = df[col].astype(str)
    
    return df

def cast_df(obj: Any, dtype: Mapping[str, str] | None = None, use_transform: bool = True, use_ml: bool = False, infer_threshold: float = 0.9, nan_threshold: float = 0.30, clean_outliers: bool = False, outlier_method: str = 'iqr', outlier_threshold: float = 1.5, return_dtype_meta: bool = False) -> pd.DataFrame:
    """
    Heuristically cast columns to richer dtypes.
    Accepts DataFrame / dict / list-of-dicts.
    
    Args:
        obj: Input data (DataFrame, dict, or list of dicts)
        dtype: Explicit dtype mappings
        use_transform: Whether to use enhanced transform_dtypes (default: True)
        use_ml: Whether to use ML-like classification (default: False)
        clean_outliers: If True, replaces numeric outliers with 0 before/during casting.
        outlier_threshold: Threshold for outlier detection (default: 1.5 for IQR)
    """
    df = _to_dataframe(obj)
    if df.empty:
        _LOG.warning("DataFrame is empty – nothing to cast")
        return (df, {}) if return_dtype_meta else df

    input_dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

    if clean_outliers:
        df = replace_outliers_with_zero_safe(df, method=outlier_method, threshold=outlier_threshold)

    # Use ML-like classification if requested
    if use_ml:
        result = auto_convert_dtypes_ml(df)
        # Apply explicit dtype mappings after ML conversion
        if dtype:
            for col, dt in dtype.items():
                if col in result.columns:
                    result[col] = _explicit_cast(result[col], dt)
    
    # Use enhanced transform_dtypes by default
    elif use_transform:
        result = transform_dtypes(df, dtype, nan_threshold=nan_threshold)
    
    else:
        # Original casting logic as fallback
        dtype = dtype or {}
        out = df.copy()

        for col in out.columns:
            series = out[col]

            # explicit mapping wins
            if col in dtype:
                out[col] = _explicit_cast(series, dtype[col])
                continue

            # only inspect likely text columns
            if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
                continue

            clean = series.dropna().astype(str).str.strip()
            clean = clean[clean != ""]
            if clean.empty:
                continue

            # Updated sampling for performance
            k = min(100, len(clean) // 2)
            head, tail = clean.head(k), clean.tail(k)
            samples = pd.concat([head, tail]).tolist()
            inferred_type = _infer(samples, threshold=infer_threshold)

            if inferred_type != "string":
                out[col] = _convert_series(series, inferred_type)
                _LOG.debug("Column '%s' cast to %s", col, inferred_type)
            else:
                # Fallback to flexible coercion if ambiguous
                num_coerced = pd.to_numeric(series, errors='coerce')
                if num_coerced.notna().mean() >= infer_threshold:
                    out[col] = num_coerced
                else:
                    dt_coerced = pd.to_datetime(series, errors='coerce')
                    if dt_coerced.notna().mean() >= infer_threshold:
                        out[col] = dt_coerced
                _LOG.debug("Column '%s' ambiguous – attempted coercion", col)

        result = out

    output_dtypes = {col: str(dt) for col, dt in result.dtypes.items()}
    dtype_meta = {col: {"input_dtype": input_dtypes.get(col), "output_dtype": output_dtypes.get(col)} for col in result.columns}

    return (result, dtype_meta) if return_dtype_meta else result

def _is_numeric_column(series: pd.Series, threshold: float = 0.9) -> bool:
    """Check if series contains numeric values (including currency)."""
    sample = series.dropna().astype(str)
    if len(sample) == 0:
        return False
    
    cleaned = sample.str.replace(_CURRENCY_PATTERN, '', regex=True).str.strip()
    return cleaned.apply(lambda x: bool(_INT_PATTERN.match(x) or _FLOAT_PATTERN.match(x)) or x == '').mean() > threshold

def _is_boolean_column(series: pd.Series, threshold: float = 0.9) -> bool:
    """Check if series contains boolean-like values."""
    sample = series.dropna().astype(str).str.lower().str.strip()
    if len(sample) == 0:
        return False
    
    mapped = sample.map(_BOOL_MAP)
    return mapped.notna().mean() >= threshold

def auto_convert_dtypes_ml(df: pd.DataFrame, datetime_format: str = '%Y-%m-%d %H:%M') -> pd.DataFrame:
    """
    Convert DataFrame columns using rule-based classifier (ML-replacement).
    Enhanced version from trans_dtype.py with comprehensive type detection.
    """
    
    class DataTypeClassifier:
        def _extract_features(self, value):
            raw_value = str(value).strip()
            cleaned = _CURRENCY_PATTERN.sub('', raw_value.lower())  # Clean currency and lower for 'e'
            cleaned_no_exp = re.sub(r'[eE][+-]?\d+', '', cleaned)  # Remove exp parts more precisely
            features = [
                len(cleaned),
                sum(c.isdigit() for c in cleaned_no_exp) / max(len(cleaned_no_exp), 1),
                sum(c.isalpha() for c in cleaned_no_exp) / max(len(cleaned_no_exp), 1),
                sum(c in '/-:' for c in cleaned) / max(len(cleaned), 1),
                sum(c == '.' for c in cleaned) / max(len(cleaned), 1),
                bool(re.match(r'^-?\d+$', cleaned)),
                bool(re.match(r'^-?\d*\.?\d+([e][-+]?\d+)?$', cleaned)),
                bool(re.match(r'^\d{4}-\d{2}-\d{2}', raw_value)),
                raw_value.lower() in list(_BOOL_MAP.keys())
            ]
            return features

        def _predict_single(self, features):
            len_val, digit_ratio, alpha_ratio, date_sep_ratio, decimal_ratio, is_int, is_float, is_date, is_bool = features
            if is_bool:
                return 'boolean'
            if is_date:
                return 'datetime'
            if is_float:
                return 'float'
            if is_int:
                return 'integer'
            if decimal_ratio > 0 and digit_ratio > 0.8:
                return 'float'
            if digit_ratio > 0.8 and alpha_ratio < 0.2:
                return 'integer'
            if date_sep_ratio > 0.1 and digit_ratio > 0.5 and alpha_ratio < 0.2:
                return 'datetime'
            return 'object'

        def predict_dtype(self, series):
            if len(series) == 0:
                return 'object'
            sample = series.dropna().astype(str).sample(min(200, len(series)), random_state=42) if len(series) > 200 else series.dropna().astype(str)
            if len(sample) == 0:
                return 'object'
            features_list = [self._extract_features(val) for val in sample]
            predictions = [self._predict_single(f) for f in features_list]
            return Counter(predictions).most_common(1)[0][0]
    
    classifier = DataTypeClassifier()
    conversions = {}
    
    for column in df.columns:
        predicted_type = classifier.predict_dtype(df[column])
        try:
            if predicted_type == 'integer':
                conversions[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
            elif predicted_type == 'float':
                conversions[column] = pd.to_numeric(df[column], errors='coerce').astype('Float64')
            elif predicted_type == 'boolean':
                conversions[column] = df[column].astype(str).str.lower().str.strip().map(_BOOL_MAP).astype('boolean')
            elif predicted_type == 'datetime':
                parsed = df[column].apply(_parse_one)
                if parsed.notna().any():
                    conversions[column] = pd.to_datetime(parsed, errors='coerce', format=datetime_format)
                else:
                    conversions[column] = pd.to_datetime(df[column], errors='coerce')
            else:
                conversions[column] = df[column]
        except (ValueError, TypeError) as e:
            _LOG.debug(f"ML conversion failed for {column}: {e}")
            conversions[column] = df[column]
    
    return pd.DataFrame(conversions)
