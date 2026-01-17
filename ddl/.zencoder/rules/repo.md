---
description: Repository Information Overview
alwaysApply: true
---

# DDL Utilities Information

## Summary
Small Python utility set for generating database DDL from pandas DataFrames. The code infers column metadata, sanitizes identifiers, and produces CREATE TABLE statements plus schema metadata for MySQL, PostgreSQL, and Oracle. It also offers helper builders for primary/unique/foreign key constraints, sequences, and triggers. No packaging metadata, tests, or docker assets are present; usage is via direct import of the modules in this directory.

## Structure
- `ddl_common.py`: Shared helpers for validation, identifier sanitization, object inspection (string/binary/JSON/mixed detection), deduping column names, dataframe normalization (JSON stringification, binary coercion, timedelta conversion), and base schema construction metadata with SQL snippets.
- `ddl_func.py`: Constraint/sequence/trigger builders that sanitize identifiers and validate referenced columns, producing SQL strings and metadata for PK, unique, FK, sequence (Oracle/PostgreSQL), and trigger (Oracle) generation.
- `ddl_mysql.py`: MySQL-specific DDL generator; maps pandas dtypes to MySQL types (boolean, integer sizing, double, datetime, bigint for timedeltas, varchar/longtext, json, varbinary/blob), enforces reserved words and 64-char identifier limits, dedupes/sanitizes column names, tracks warnings/decisions, and emits CREATE TABLE SQL plus normalized dataframe and schema metadata.
- `ddl_postgre.py`: PostgreSQL-specific DDL generator; maps pandas dtypes to boolean/bigint/double precision/timestamp/double for timedeltas/jsonb/bytea/varchar/text, applies reserved words and 63-char limits, dedupes/sanitizes names, normalizes dataframe, and returns CREATE TABLE SQL and schema metadata.
- `ddl_oracle.py`: Oracle-specific DDL generator; maps pandas dtypes to NUMBER with inferred precision, BINARY_FLOAT/BINARY_DOUBLE, TIMESTAMP, NUMBER for timedeltas, VARCHAR2/CLOB, RAW/BLOB, optional JSON type (21c+), enforces reserved words and 30-char limits, dedupes/sanitizes names, normalizes dataframe, and returns CREATE TABLE SQL and schema metadata.

## Language & Runtime
**Language**: Python  
**Version**: Not specified  
**Build System**: None (standalone scripts)  
**Package Manager**: None declared (install dependencies manually)

## Dependencies
**Main Dependencies**:
- `pandas` (DataFrame handling, dtype introspection, sampling, NA detection)
- `numpy` (numeric type checks, NaN handling, absolute value and log operations)
- Python standard library: `json`, `re`, `dataclasses`, `typing`, `math`

**Development Dependencies**:
- Not declared; no testing or tooling configs are present.

## Build & Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy
```

## Usage & Operations
- Import the desired dialect module (`ddl_mysql`, `ddl_postgre`, `ddl_oracle`) and call `generate_ddl(df, table_name, options)` with a pandas DataFrame. Returns `(processed_df, create_table_sql, base_schema_dict)`.
- `processed_df` normalizes JSON/object/binary/timedelta columns for database compatibility; use it when loading data after DDL creation.
- `create_table_sql` provides the CREATE TABLE statement with optional NOT NULL constraints based on `include_not_null`.
- `base_schema_dict` captures dialect, table/schema names, column mappings (original vs sanitized names, pandas dtype, semantic object type, SQL type, nullability), warnings, decisions, and SQL snippets. It is suitable for downstream documentation or migrations.
- Identifier sanitization enforces allowed characters, trims length, uppercases for Oracle/MySQL naming logic then lowercases for emitted identifiers where appropriate, and dedupes collisions by suffixing counters.
- Object inspection samples up to `object_sample_size` rows (default 5000) to classify object/strings as JSON, binary, or mixed, influencing type mapping and normalization.

## Main APIs & Options
- `ddl_mysql.generate_ddl(df, table_name, options)`: Options include `schema` (string), `include_not_null` (bool), `object_sample_size` (int, default 5000), `json_text_threshold` (float, default 0.85). Produces MySQL CREATE TABLE SQL with varchar sizing (text fallback) and JSON/varbinary/blob handling; warns on renamed columns and ambiguous object data.
- `ddl_postgre.generate_ddl(df, table_name, options)`: Options include `schema`, `include_not_null`, `varchar_limit` (default 10485760), `object_sample_size`, `json_text_threshold`. Maps object data to jsonb, bytea, varchar/text; notes renames and ambiguous/mixed content.
- `ddl_oracle.generate_ddl(df, table_name, options)`: Options include `schema`, `include_not_null`, `varchar2_limit` (default 4000), `raw_limit` (default 2000), `object_sample_size`, `json_text_threshold`, `prefer_json_datatype` (default True). Infers NUMBER precision for integers, supports RAW/BLOB for binary, VARCHAR2/CLOB for text, JSON type if allowed, and warns about Oracle 21c+ requirement when JSON is used.
- `ddl_func.build_pk_constraint(...)`, `build_unique_constraint(...)`, `build_fk_constraint(...)`: Validate identifiers/columns against sanitized set, return constraint SQL and metadata for the target dialect context.
- `ddl_func.build_sequence(...)`: Supports Oracle and PostgreSQL; accepts start/increment/cache parameters and returns CREATE SEQUENCE SQL plus metadata.
- `ddl_func.build_trigger(...)`: Oracle-only trigger for populating a column from a sequence on insert; returns SQL and metadata.

## Data Handling & Type Mapping Notes
- Nullability is inferred per column via `pandas.Series.isna()`. `include_not_null` toggles NOT NULL emission; otherwise columns are nullable in emitted DDL.
- Object/JSON detection checks for JSON-like strings (balanced braces/brackets and `json.loads` success) and actual dict/list types; ambiguous mixes fall back to text/CLOB equivalents with warnings.
- Binary handling converts strings/bytearray/memoryview to bytes, preserving None/NaN; length-driven type choices vary by dialect (varbinary/blob for MySQL, bytea for PostgreSQL, RAW/BLOB for Oracle).
- Timedelta columns are normalized to floating seconds (PostgreSQL: double precision), bigint microseconds (MySQL), or NUMBER seconds (Oracle) with warnings recorded.
- Identifier sanitation removes invalid characters, collapses underscores, prefixes digits, adjusts reserved word clashes, uppercases then truncates to dialect limits, and dedupes with numeric suffixes to avoid collisions.
