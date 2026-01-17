# Schema Align Review and Comparison

This document provides a detailed comparison between the legacy monolithic script `sql_corrector.py` and the new, modular `schema_align` package.

**Date:** 2026-01-17
**Scope:** Review of `E:\PyCode\_WebProject\telcoetl\etl_2\Tetl\schema_align` vs `sql_corrector.py`

## Executive Summary

The `schema_align` package is a complete, production-grade refactoring of the proof-of-concept `sql_corrector.py`. It strictly improves upon the monolith in every dimension: maintainability, type safety, extensibility, and clarity.

**Verdict:** The monolithic `sql_corrector.py` should be considered deprecated. New development should exclusively use `schema_align`.

## 1. Architectural Comparison

| Feature | `sql_corrector.py` (Monolith) | `schema_align` (Modular) |
| :--- | :--- | :--- |
| **Structure** | Single file (~950 lines) mixing introspection, validation, coercion, and reporting. | Split into 5 focused modules (`core`, `config`, `validators`, `diagnostics`, `schema_analyzer`). |
| **Entry Point** | `SchemaAligner` class instantiated directly. | `api.py` provides high-level functions (`aligner`, `strict_aligner`) and decorators. |
| **Separation of Concerns** | Low. Schema analysis is tightly coupled with validation logic. | High. `SchemaAnalyzer` only inspects; `TypeValidator` only validates; `DataAligner` orchestrates. |
| **Dependencies** | Hard imports of `sklearn` (optional but integrated). | Modular imports; `sklearn` logic isolated in `validators.py`. |

## 2. Functional Improvements in `schema_align`

### 2.1. Enhanced Configuration (`config.py`)

- **Old:** Relied on passing miscellaneous kwargs (`on_error`, `failure_threshold`) to methods.
- **New:** Uses a `AlignmentConfig` dataclass and `ValidationMode` enums.
- **Benefit:** Auto-completion support in IDEs, type safety, and centralized default management via `aligner.yml`.

### 2.2. Robust Validation (`validators.py`)

- **Old:** Validation logic scattered in `_strict_*` methods within the main class.
- **New:** Dedicated `TypeValidator` class.
- **New Feature:** **Outlier Detection**. The new validator explicitly supports multiple outlier detection algorithms (`IsolationForest`, `LocalOutlierFactor`, `Z-Score`, `IQR`) to distinguish between systematic data issues and true outliers.
- **New Feature:** **Encoding Checks**. `_strict_string` now includes checks for valid UTF-8 encoding.

### 2.3. Detailed Diagnostics (`diagnostics.py`)

- **Old:** Simple `logging` calls mixed with business logic.
- **New:** `DiagnosticsCollector` class that accumulates `ValidationResult` objects.
- **Benefit:** Can generate structured summary reports programmatically (`get_validation_summary`) or tailored CLI output (`print_summary_report`). It tracks not just that a failure occurred, but *why* (e.g., "outlier detected" vs "systematic failure").

### 2.4. Performance (`performance.py`)

- **Old:** No built-in performance tracking.
- **New:** `PerformanceOptimizer` allows measuring time spent in introspection vs validation and tracks cache hits (memoization of schema analysis).

## 3. Workflow Comparison

### Scenario: Aligning a strict financial table

**Legacy (`sql_corrector.py`):**

```python
from sql_corrector import SchemaAligner
aligner = SchemaAligner(engine, on_error='raise', failure_threshold=0.01)
try:
    df_clean = aligner.align(df, "finance_data")
except ValueError as e:
    print(e)
```

**Modern (`schema_align`):**

```python
from schema_align import strict_aligner

# Pre-configured "strict" preset (1% threshold, raise error, strict int checks)
try:
    df_clean = strict_aligner(engine, df, "finance_data")
except ValueError:
    # Diagnostics contain the full report
    from schema_align.core import DataAligner
    # ... access diagnostics if needed
    pass
```

### Scenario: ETL Pipeline Integration

**Legacy:** Manual instantiation and calls inside every ETL function.

**Modern:** Decorator-driven.

```python
from schema_align import align_to_sql

@align_to_sql("target_table", preset="fast")
def extract_and_load(engine, raw_data):
    # Logic is decoupled from validation
    return raw_data.to_sql(...)
```

## 4. Gap Analysis (Resolved)

Previous feature gaps between `sql_corrector.py` and `schema_align` have been analyzed and **resolved** as of this review.

### 4.1. Dialect-Specific Type Introspection

* **Resolution:** `schema_align.core` now explicitly imports and checks for dialect-specific types (e.g., Oracle `NUMBER` with `scale=0`). It matches the robust logic of the legacy script.

### 4.2. Driver Compatibility (`_finalize_types`)

* **Resolution:** A `_finalize_types` pass has been implemented in `schema_align` to convert Pandas extension types (`Int64`, `Float64`) to native Python types, preventing `DPY-3002` errors.

## 5. Migration Guide

1. **Imports:** Change `from sql_corrector import SchemaAligner` to `from schema_align import aligner`.
2. **Configuration:** Instead of passing raw kwargs to `align()`, simpler use cases can just use `aligner(..., config_dict={...})`. For complex setups, instantiate `AlignmentConfig`.
3. **Outlier Detection:** If you were manually handling outliers, you can now enable `outlier_detection=True` in the config to have it handled automatically during type coercion.
