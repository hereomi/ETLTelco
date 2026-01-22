---
description: Repository Information Overview
alwaysApply: true
---

# Schema Align Information

## Summary
Python-based DataAligner toolkit that aligns pandas DataFrames to relational database schemas using SQLAlchemy connections. Provides strict type validation, optional outlier detection, schema introspection, and optional schema evolution to auto-add missing columns. Includes convenience entry points (decorator and preset helpers) plus a YAML-driven default configuration (`aligner.yml`) and an extensive usage guide (`aligner_usage.md`). Focuses on operational data quality with diagnostics, performance metrics, and cross-dialect support (SQLite, Postgres, MySQL/MariaDB, MSSQL, Oracle when SQLAlchemy dialects are installed).

## Structure
- `api.py`: Public interface exposing `aligner`, decorator `align_to_sql`, and preset helpers (`strict_aligner`, `fast_aligner`, `safe_aligner`); loads defaults from `aligner.yml` and merges runtime overrides.
- `core.py`: Core `DataAligner` implementation handling dialect detection, schema analysis, type coercion, constraint validation, and optional schema evolution; integrates diagnostics and performance tracking.
- `config.py`: Configuration dataclasses and enums (`AlignmentConfig`, `ValidationMode`, `OutlierMethod`, `ValidationResult`, `PerformanceMetrics`).
- `validators.py`: Type validation and outlier handling (pandas/numpy, optional scikit-learn and python-dateutil); strict conversion paths for integer/float/string/bool/datetime/json/binary.
- `performance.py`, `diagnostics.py`, `utils.py`: Helpers for timing, diagnostics aggregation, identifier quoting/validation, chunking, and conversion quality checks.
- `schema_analyzer.py`, `schema_manager.py`, `schema_inspector.py`: Schema introspection utilities building column and constraint metadata across SQLAlchemy dialects.
- `aligner.yml`: Default configuration and presets (strict/fast/safe) controlling validation behavior, schema evolution, performance tuning, and diagnostics.
- `aligner_usage.md`: Detailed usage guide with installation instructions, code samples, preset descriptions, and troubleshooting tips.
- Sample datasets/configs: `table_link.json/txt`, `sqlite_test_verify_schema.json`, `test_quota*.json/txt` used for example or validation scenarios.
- Meta directories: `.github/instructions`, `.zencoder/workflows`, `.zenflow/workflows` (workflow/policy metadata); `__pycache__` for compiled Python artifacts.

## Language & Runtime
**Language**: Python (modules target Python 3; compiled artifacts present for 3.10 and 3.11).  
**Version**: Not pinned in repo; runtime inferred from `.pyc` cache variants (3.10/3.11).  
**Build System**: None declared (no `pyproject.toml`/`setup.py`).  
**Package Manager**: Pip expected (manual installation required as no lock/requirement files are present).

## Dependencies
**Main Dependencies** (imported in code or documented in `aligner_usage.md`):
- `pandas`: DataFrame handling, type conversion, chunking.
- `sqlalchemy`: Database connectivity, dialect typing (sqlite, postgres, mysql/mariadb, mssql, oracle when installed), schema inspection, DDL execution.
- `numpy`: Numeric operations and nullable dtypes.
- `scikit-learn` (optional): IsolationForest and LocalOutlierFactor for outlier detection; gracefully bypassed if unavailable.
- `pyyaml`/`yaml`: Loads default configuration from `aligner.yml`.
- `python-dateutil` (optional): Fallback datetime parsing in validators.

**Development/Tooling Dependencies**: Not defined in repo. No explicit lint/test/tool configs found.

## Build & Installation
No package metadata is present; install dependencies manually (per `aligner_usage.md`). Example environment bootstrap:
```bash
pip install pandas sqlalchemy numpy scikit-learn pyyaml python-dateutil
```
Import module directly from repo path or add to `PYTHONPATH`; entrypoints are plain Python modules rather than a packaged distribution.

## Main Files & Entry Points
- Primary import surface: `__init__.py` exports `DataAligner`, configs, validators, diagnostics/performance helpers, schema analyzer types, and preset aligner helpers plus convenience wrappers (`strict_align`, `fast_align`, `safe_align`).
- Default config loader: `aligner.yml` consumed by `api.py` to build `AlignmentConfig` with merged overrides and presets.
- Operational workflow: `aligner(engine, df, tablename, config_dict)` wraps `DataAligner.align`, handling engine connections and cleanup. Decorator `align_to_sql(tablename, config_dict=None, preset=None)` enables automatic alignment inside pipeline functions. Preset functions simplify common modes (strict/fast/safe).
- Schema introspection/management: `schema_analyzer.py` builds column/constraint metadata via SQLAlchemy inspect APIs; `core.DataAligner` uses these reports for validation and optional `ALTER TABLE` operations.
- Validation pipeline: `validators.TypeValidator` enforces strict integer/float/string/bool/datetime/json/binary rules, length/encoding checks, range validation, and optional outlier filtering; diagnostics recorded via `DiagnosticsCollector`.
- Performance tracking: `performance.PerformanceOptimizer` measures total, schema introspection, type coercion, and validation timings; metrics stored in `PerformanceMetrics`.

## Usage & Operations
- Typical flow (from `aligner_usage.md`): create SQLAlchemy engine, build pandas DataFrame, call `aligner(engine, df, "table")`, then write with `to_sql`. Decorator usage wraps ingestion functions for automatic alignment.
- Configuration surface (see `aligner.yml` and examples): validation mode, failure thresholds, error handling (`coerce`/`raise`/`skip`), schema evolution toggles (`add_missing_cols`, `allow_schema_evolution`), constraint validation, caching, chunk sizing, parallelism, outlier detection methods/thresholds, diagnostics verbosity.
- Dialect support: auto-detects dialect from SQLAlchemy connection; custom mappings for Oracle, SQLite, Postgres, MySQL/MariaDB, MSSQL (imports guarded to allow partial installations). Schema evolution adjusts SQL syntax per dialect (e.g., `ADD COLUMN` vs `ADD`).

## Project Structure Notes
- No testing framework or test suite present; no `pytest`, `unittest`, or CI configs for tests detected.
- No Docker or containerization assets present.
- Repository functions as a single Python library module set; no monorepo or multiple subprojects identified.
