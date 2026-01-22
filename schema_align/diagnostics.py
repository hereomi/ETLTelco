"""Diagnostics and error reporting for alignment operations."""

import logging
from typing import List, Dict, Any
from .config import ValidationResult, AlignmentConfig
from .utils import get_sample_failures

logger = logging.getLogger("DataAligner")

class DiagnosticsCollector:
    """Collects and manages validation diagnostics and error reporting."""
    
    def __init__(self, config: AlignmentConfig):
        self.config = config
        self.validation_results: List[ValidationResult] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.metadata: Dict[str, Any] = {}  # Issue 2/3/4: store surfaced validation metadata
    
    def add_metadata(self, key: str, value: Any):
        """Attach observable metadata for downstream callers."""
        existing = self.metadata.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = existing.copy()
            merged.update(value)
            self.metadata[key] = merged
        elif isinstance(existing, list) and isinstance(value, list):
            self.metadata[key] = existing + value
        elif existing is not None and isinstance(existing, list):
            self.metadata[key] = existing + [value]
        elif existing is not None and not isinstance(existing, list):
            self.metadata[key] = [existing, value]
        else:
            self.metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retrieve collected metadata."""
        return self.metadata
    
    def record_validation_result(self, column: str, total_rows: int, failed_rows: int, error_type: str, original_series=None, failure_mask=None, action_taken: str = "coerced_to_null", outliers_detected: int = 0, outlier_method: str = "none"):
        """Record detailed validation result for a column."""
        if not self.config.collect_diagnostics:
            return
        failure_rate = failed_rows / total_rows if total_rows > 0 else 0.0
        sample_failures = get_sample_failures(original_series, failure_mask) if original_series is not None and failure_mask is not None else []
        result = ValidationResult(
            column=column,
            total_rows=total_rows,
            failed_rows=failed_rows,
            failure_rate=failure_rate,
            error_type=error_type,
            sample_failures=sample_failures,
            action_taken=action_taken,
            outliers_detected=outliers_detected,
            outlier_method=outlier_method
        )
        self.validation_results.append(result)
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {"total_columns": 0, "total_failures": 0, "avg_failure_rate": 0.0}
        total_failures = sum(r.failed_rows for r in self.validation_results)
        total_rows = sum(r.total_rows for r in self.validation_results)
        avg_failure_rate = total_failures / total_rows if total_rows > 0 else 0.0
        return {
            "total_columns": len(self.validation_results),
            "total_failures": total_failures,
            "total_rows": total_rows,
            "avg_failure_rate": avg_failure_rate,
            "columns_with_failures": len([r for r in self.validation_results if r.failed_rows > 0]),
            "outliers_detected": sum(r.outliers_detected for r in self.validation_results)
        }
    
    def get_failed_columns(self) -> List[ValidationResult]:
        """Get validation results for columns that had failures."""
        return [r for r in self.validation_results if r.failed_rows > 0]
    
    def get_high_failure_columns(self, threshold: float = 0.1) -> List[ValidationResult]:
        """Get columns with failure rates above threshold."""
        return [r for r in self.validation_results if r.failure_rate > threshold]
    
    def print_summary_report(self):
        """Print a formatted summary report."""
        summary = self.get_validation_summary()
        print(f"\n=== Alignment Summary ===")
        print(f"Columns processed: {summary['total_columns']}")
        print(f"Total rows: {summary['total_rows']}")
        print(f"Total failures: {summary['total_failures']}")
        print(f"Average failure rate: {summary['avg_failure_rate']:.2%}")
        print(f"Columns with failures: {summary['columns_with_failures']}")
        print(f"Outliers detected: {summary['outliers_detected']}")
        failed_cols = self.get_failed_columns()
        if failed_cols:
            print(f"\n=== Failed Columns ===")
            for result in failed_cols:
                print(f"{result.column}: {result.failed_rows}/{result.total_rows} ({result.failure_rate:.2%}) - {result.error_type}")
        if self.warnings:
            print(f"\n=== Warnings ({len(self.warnings)}) ===")
            for warning in self.warnings[-5:]:
                print(f"⚠️  {warning}")
        if self.errors:
            print(f"\n=== Errors ({len(self.errors)}) ===")
            for error in self.errors[-5:]:
                print(f"❌ {error}")
    
    def clear_results(self):
        """Clear all collected results."""
        self.validation_results.clear()
        self.warnings.clear()
        self.errors.clear()