"""Enhanced DataFrame-to-SQL alignment with strict validation and performance optimizations."""

from .core import DataAligner
from .config import AlignmentConfig, ValidationMode, ValidationResult
from .validators import TypeValidator
from .performance import PerformanceOptimizer
from .diagnostics import DiagnosticsCollector
try:
    from .schema_analyzer import SchemaAnalyzer, ColumnInfo, ConstraintInfo
except ImportError:
    SchemaAnalyzer = ColumnInfo = ConstraintInfo = None
from .api import aligner, align_to_sql, strict_aligner, fast_aligner, safe_aligner

def strict_align(conn, df, table, **kwargs):
    """Strict alignment with minimal tolerance for failures."""
    config = AlignmentConfig.strict()
    return DataAligner(config=config).align(conn, df, table, **kwargs)

def fast_align(conn, df, table, **kwargs):
    """Fast alignment optimized for performance over strict validation."""
    config = AlignmentConfig.fast()
    return DataAligner(config=config).align(conn, df, table, **kwargs)

def safe_align(conn, df, table, **kwargs):
    """Balanced alignment with good performance and validation."""
    config = AlignmentConfig.balanced()
    return DataAligner(config=config).align(conn, df, table, **kwargs)

__all__ = ['DataAligner', 'AlignmentConfig', 'ValidationMode', 'ValidationResult', 'TypeValidator', 'PerformanceOptimizer', 'DiagnosticsCollector', 'SchemaAnalyzer', 'ColumnInfo', 'ConstraintInfo', 'aligner', 'align_to_sql', 'strict_aligner', 'fast_aligner', 'safe_aligner', 'strict_align', 'fast_align', 'safe_align']