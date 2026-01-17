"""Configuration classes for alignment behavior and validation settings."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

class ValidationMode(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"

class OutlierMethod(Enum):
    """Outlier detection algorithms."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    Z_SCORE = "zscore"
    IQR = "iqr"

@dataclass
class ValidationResult:
    """Detailed validation result for a single column."""
    column: str
    total_rows: int
    failed_rows: int
    failure_rate: float
    error_type: str
    sample_failures: List[Any] = field(default_factory=list)
    action_taken: str = "unknown"
    outliers_detected: int = 0
    outlier_method: str = "none"

@dataclass
class AlignmentConfig:
    """Configuration for DataFrame alignment behavior."""
    validation_mode: ValidationMode = ValidationMode.BALANCED
    failure_threshold: float = 0.1
    outlier_detection: bool = True
    outlier_methods: List[OutlierMethod] = field(default_factory=lambda: [OutlierMethod.ISOLATION_FOREST])
    parallel_processing: bool = False
    cache_enabled: bool = True
    max_sample_size: int = 10000
    chunk_size: int = 50000
    collect_diagnostics: bool = False
    enable_metrics: bool = False
    max_null_increase: float = 0.1
    
    @classmethod
    def strict(cls) -> 'AlignmentConfig':
        """Create strict validation configuration."""
        return cls(
            validation_mode=ValidationMode.STRICT,
            failure_threshold=0.01,
            outlier_detection=True,
            parallel_processing=False,
            collect_diagnostics=True,
            max_null_increase=0.05
        )
    
    @classmethod
    def fast(cls) -> 'AlignmentConfig':
        """Create performance-optimized configuration."""
        return cls(
            validation_mode=ValidationMode.PERMISSIVE,
            failure_threshold=0.2,
            outlier_detection=False,
            parallel_processing=True,
            cache_enabled=True,
            max_sample_size=1000,
            collect_diagnostics=False
        )
    
    @classmethod
    def balanced(cls) -> 'AlignmentConfig':
        """Create balanced configuration (default)."""
        return cls()

@dataclass
class PerformanceMetrics:
    """Performance metrics for alignment operations."""
    total_time: float = 0.0
    schema_introspection_time: float = 0.0
    type_coercion_time: float = 0.0
    validation_time: float = 0.0
    rows_processed: int = 0
    columns_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0