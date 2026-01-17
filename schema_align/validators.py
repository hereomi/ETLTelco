"""Enhanced type validation with strict checking and outlier detection."""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from .config import AlignmentConfig, OutlierMethod
from .diagnostics import DiagnosticsCollector
from .utils import validate_conversion_quality

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from dateutil import parser as dateutil_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

class TypeValidator:
    """Enhanced type validation with outlier detection and strict checking."""
    
    DEFAULT_TRUE = {"1", "true", "yes", "y", "on", "t"}
    DEFAULT_FALSE = {"0", "false", "no", "n", "off", "f"}
    
    def __init__(self, config: AlignmentConfig, diagnostics: DiagnosticsCollector):
        self.config = config
        self.diagnostics = diagnostics
    
    def validate_and_convert(self, series: pd.Series, column_info: Any, target_type: str) -> pd.Series:
        """Main validation and conversion entry point."""
        if target_type == 'integer':
            return self._strict_int(series, column_info)
        elif target_type == 'float':
            return self._strict_float(series, column_info)
        elif target_type == 'string':
            return self._strict_string(series, column_info)
        elif target_type == 'boolean':
            return self._strict_bool(series, column_info)
        elif target_type == 'datetime':
            return self._strict_datetime(series, column_info)
        elif target_type == 'json':
            return self._strict_json(series, column_info)
        elif target_type == 'binary':
            return self._strict_binary(series, column_info)
        return series
    
    def _strict_int(self, s: pd.Series, info: Any) -> pd.Series:
        """Strict integer validation with range checking."""
        s_num = pd.to_numeric(s, errors='coerce')
        mask_invalid_str = s.notna() & s_num.isna()
        has_remainder = (s_num.fillna(0) % 1).abs() > 1e-9
        mask_float_loss = s_num.notna() & has_remainder
        mask_range_error = pd.Series(False, index=s.index)
        if hasattr(info, 'precision') and info.precision:
            if info.precision <= 32:
                mask_range_error = (s_num < -2147483648) | (s_num > 2147483647)
            elif info.precision <= 64:
                mask_range_error = (s_num < -9223372036854775808) | (s_num > 9223372036854775807)
        total_failures = mask_invalid_str | mask_float_loss | mask_range_error
        if total_failures.any():
            self._handle_validation_failures(total_failures, info.name, "strict integer check", s_num)
            s_num[total_failures] = np.nan
        return s_num.astype('Int64')
    
    def _strict_float(self, s: pd.Series, info: Any) -> pd.Series:
        """Strict float validation."""
        s_num = pd.to_numeric(s, errors='coerce')
        mask_fail = s.notna() & s_num.isna()
        if mask_fail.any():
            self._handle_validation_failures(mask_fail, info.name, "numeric conversion", s_num)
        return s_num.astype('Float64')
    
    def _strict_string(self, s: pd.Series, info: Any) -> pd.Series:
        """Enhanced string validation with length and encoding checks."""
        s_str = s.astype(str).where(s.notna())
        length_violations = pd.Series(False, index=s.index)
        if hasattr(info, 'length') and info.length:
            length_violations = s_str.str.len() > info.length
        encoding_violations = pd.Series(False, index=s.index)
        try:
            encoding_violations = s_str.apply(lambda x: not self._is_valid_utf8(x) if pd.notna(x) else False)
        except:
            pass
        total_violations = length_violations | encoding_violations
        if total_violations.any():
            error_msg = f"string validation (max_len={getattr(info, 'length', 'unlimited')})"
            self._handle_validation_failures(total_violations, info.name, error_msg, s_str)
            s_str.loc[total_violations] = None
        return s_str
    
    def _strict_bool(self, s: pd.Series, info: Any) -> pd.Series:
        """Strict boolean validation."""
        s_norm = s.astype(str).str.lower().str.strip()
        mask_na = s.isna()
        s_norm = s_norm.where(~mask_na, np.nan)
        mask_true = s_norm.isin(self.DEFAULT_TRUE)
        mask_false = s_norm.isin(self.DEFAULT_FALSE)
        mask_fail = s.notna() & (~mask_true) & (~mask_false)
        if mask_fail.any():
            self._handle_validation_failures(mask_fail, info.name, "boolean constraint", s)
        out = pd.Series(index=s.index, dtype='object')
        out[mask_true] = True
        out[mask_false] = False
        out[mask_fail] = None
        return out.astype('boolean')
    
    def _strict_datetime(self, s: pd.Series, info: Any) -> pd.Series:
        """Enhanced datetime validation with multiple parsing strategies."""
        try:
            s_dt = pd.to_datetime(s, errors='coerce', format='mixed')
        except Exception:
            s_dt = pd.Series(pd.NaT, index=s.index)
        if DATEUTIL_AVAILABLE:
            mask_try = s_dt.isna() & s.notna()
            if mask_try.any():
                def try_dateutil_parse(v):
                    try:
                        return dateutil_parser.parse(str(v))
                    except:
                        return pd.NaT
                parsed = s.loc[mask_try].map(try_dateutil_parse)
                s_dt.loc[mask_try] = pd.to_datetime(parsed, errors='coerce')
        mask_fail = s.notna() & s_dt.isna()
        if mask_fail.any():
            self._handle_validation_failures(mask_fail, info.name, "datetime parsing", s)
        return s_dt
    
    def _strict_json(self, s: pd.Series, info: Any) -> pd.Series:
        """Strict JSON validation."""
        def validate_json(val):
            if pd.isna(val):
                return None
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            try:
                if isinstance(val, str):
                    parsed = json.loads(val)
                    return json.dumps(parsed)
            except:
                pass
            return None
        res = s.apply(validate_json)
        mask_fail = s.notna() & res.isna()
        if mask_fail.any():
            self._handle_validation_failures(mask_fail, info.name, "valid JSON", s)
        return res
    
    def _strict_binary(self, s: pd.Series, info: Any) -> pd.Series:
        """Strict binary validation."""
        is_binary = s.map(lambda x: pd.isna(x) or isinstance(x, (bytes, bytearray, memoryview)))
        mask_fail = ~is_binary
        if mask_fail.any():
            self._handle_validation_failures(mask_fail, info.name, "binary content (bytes)", s)
        out = s.copy()
        out[mask_fail] = None
        return out
    
    def _handle_validation_failures(self, failure_mask: pd.Series, col_name: str, error_type: str, original_series: pd.Series):
        """Handle validation failures with outlier detection and diagnostics."""
        fail_count = int(failure_mask.sum())
        if fail_count == 0:
            return
        total = len(failure_mask)
        rate = fail_count / total
        outliers_detected = 0
        outlier_method = "none"
        if rate > self.config.failure_threshold and self.config.outlier_detection and SKLEARN_AVAILABLE:
            if pd.api.types.is_numeric_dtype(original_series):
                outlier_mask, method = self._detect_outliers_multi_algorithm(original_series)
                systematic_failures = failure_mask & (~outlier_mask)
                sys_count = int(systematic_failures.sum())
                sys_rate = sys_count / total
                outliers_detected = fail_count - sys_count
                outlier_method = method
                if sys_rate <= self.config.failure_threshold:
                    self.diagnostics.add_warning(f"[{col_name}] Outlier detection ({method}) reduced failure rate from {rate:.1%} to {sys_rate:.1%}")
                    rate = sys_rate
                    fail_count = sys_count
        action_taken = "coerced_to_null"
        if rate > self.config.failure_threshold:
            if self.config.validation_mode.value == "strict":
                action_taken = "raised_error"
                raise ValueError(f"[{col_name}] Critical: {fail_count}/{total} ({rate:.1%}) rows failed {error_type}. Exceeds threshold {self.config.failure_threshold:.1%}")
            else:
                self.diagnostics.add_warning(f"[{col_name}] {fail_count}/{total} ({rate:.1%}) failures exceed threshold but coercing to NULL")
        self.diagnostics.record_validation_result(
            column=col_name,
            total_rows=total,
            failed_rows=fail_count,
            error_type=error_type,
            original_series=original_series,
            failure_mask=failure_mask,
            action_taken=action_taken,
            outliers_detected=outliers_detected,
            outlier_method=outlier_method
        )
    
    def _detect_outliers_multi_algorithm(self, series: pd.Series) -> tuple[pd.Series, str]:
        """Try multiple outlier detection algorithms and return the best result."""
        s_clean = series.dropna()
        if len(s_clean) < 10:
            return pd.Series(False, index=series.index), "insufficient_data"
        best_method = "none"
        best_outliers = pd.Series(False, index=series.index)
        for method in self.config.outlier_methods:
            try:
                if method == OutlierMethod.ISOLATION_FOREST:
                    outliers = self._detect_outliers_isolation_forest(s_clean, series.index)
                elif method == OutlierMethod.LOCAL_OUTLIER_FACTOR:
                    outliers = self._detect_outliers_lof(s_clean, series.index)
                elif method == OutlierMethod.Z_SCORE:
                    outliers = self._detect_outliers_zscore(s_clean, series.index)
                elif method == OutlierMethod.IQR:
                    outliers = self._detect_outliers_iqr(s_clean, series.index)
                else:
                    continue
                if outliers.sum() > 0:
                    best_outliers = outliers
                    best_method = method.value
                    break
            except Exception:
                continue
        return best_outliers, best_method
    
    def _detect_outliers_isolation_forest(self, s_clean: pd.Series, original_index: pd.Index) -> pd.Series:
        """Isolation Forest outlier detection."""
        X = s_clean.to_numpy().reshape(-1, 1)
        clf = IsolationForest(random_state=42, contamination='auto')
        preds = clf.fit_predict(X)
        outlier_mask = pd.Series(False, index=original_index)
        outlier_mask.loc[s_clean.index] = (preds == -1)
        return outlier_mask
    
    def _detect_outliers_lof(self, s_clean: pd.Series, original_index: pd.Index, n_neighbors: int = 20) -> pd.Series:
        """Local Outlier Factor detection."""
        X = s_clean.to_numpy().reshape(-1, 1)
        n_neighbors = min(n_neighbors, len(X) - 1)
        if n_neighbors < 1:
            return pd.Series(False, index=original_index)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        preds = lof.fit_predict(X)
        outlier_mask = pd.Series(False, index=original_index)
        outlier_mask.loc[s_clean.index] = (preds == -1)
        return outlier_mask
    
    def _detect_outliers_zscore(self, s_clean: pd.Series, original_index: pd.Index, threshold: float = 3.0) -> pd.Series:
        """Z-score based outlier detection."""
        z_scores = np.abs((s_clean - s_clean.mean()) / s_clean.std())
        outlier_mask = pd.Series(False, index=original_index)
        outlier_mask.loc[s_clean.index] = z_scores > threshold
        return outlier_mask
    
    def _detect_outliers_iqr(self, s_clean: pd.Series, original_index: pd.Index, multiplier: float = 1.5) -> pd.Series:
        """IQR based outlier detection."""
        q1, q3 = s_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=original_index)
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        outlier_mask = pd.Series(False, index=original_index)
        outlier_mask.loc[s_clean.index] = (s_clean < lower_bound) | (s_clean > upper_bound)
        return outlier_mask
    
    def _is_valid_utf8(self, text: str) -> bool:
        """Check if text is valid UTF-8."""
        try:
            text.encode('utf-8').decode('utf-8')
            return True
        except:
            return False