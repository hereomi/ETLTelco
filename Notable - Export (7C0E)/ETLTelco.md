---
tags: [sql]
title: ETLTelco
created: '2026-01-18T15:45:29.402Z'
modified: '2026-01-18T20:48:52.510Z'
---

# ETLTelco

Data_profiler exists to feed ETL decision-making: profile_dataframe scans each column for nulls/uniqueness/categorical/datetime heuristics (dropping object-to-datetime parsing errors quietly and logging high-null columns), sample_dispatcher filters rows, applies optional sorting, and returns head/tail slices plus metadata to help judge data health, and get_pk leverages those profiles to award a unique column or build composites (preferring db‑resolvable datetime and high uniqueness before iterating through candidates) and logs every PK detection step so ETL.auto_etl or gen_config can auto-assign upserts.
Failure modes: empty inputs yield empty profiles or immediate ValueErrors in _validate_inputs, sampling can drop every row when null filtering removes >20 % without aborting so callers may operate on empty sets, datetime inference swallows ValueErrors so bad strings simply aren’t flagged, candidate composites stop when null rows outnumber data (warnings but still proceed), and get_pk’s fallbacks may return non-unique keys after max_composite_size is reached or when dfinfo lacks required columns, leaving downstream ETL to base joins on potentially duplicated identifiers while only logging warnings.

## Clearing.py
CleaningConfig defaults normalize/repair unicode/smart punctuation, clean_string short-circuits non-strings, applies normalization, custom_map, optional control-char and special-char removal, trim; clean_series casts to string, runs custom replacements, optional normalization and control filtering via apply, regex special-char strip, trim; clean_headers sanitizes names with clean_string, lowercasing, non-alphanumerics to underscores, duplicate collapse; robust_clean runs headers then applies clean_series to specified or all object/string columns, quick_clean presets config. Failure modes: clean_series coerces all columns to string so numeric dtypes vanish and NaNs become literal "nan", handle_na hides missingness, remove_special_chars can strip legitimate DB characters, per-row .apply for control cleaning slows large Series. Alternative: pyjanitor


## DDL: 
Control flow: df_ddl validates engine, df, and table_name, instantiates DDLConfig (merging user options), and routes to the dialect-specific generator, where each module sanitizes and deduplicates column identifiers against dialect-specific reserved words, inspects object columns by sampling up to 5k values to classify as string/binary/json/mixed, maps pandas dtypes to SQL types (with timestamp, timedelta, numeric branching), records warnings, normalizes JSON/binary/timedelta columns to consistent string/byte/float forms, assembles the CREATE TABLE DDL plus optional PK/constraint statements via ddl_func, builds a schema metadata dict, and returns the renamed/normalized DataFrame together with SQL/constraints/SQLAlchemy type mappings. Failure modes: identifier sanitization truncates and renames columns to meet max-length and reserved-word guards, so supplied column/PK names can be replaced or even collide once dedupe suffixes are added; object inspection works on limited samples and falls back silently to text/longtext (or JSON-as-text) for mixed or unclassifiable data, hiding rare types and misinterpreting non-JSON strings as JSON; normalization coerces JSON, binary, and timedelta content to strings/bytes/floats, so non-serializable values or user expectations about original dtypes are lost, and the constraint builders depend on sanitized names that differ from the deduped ones unless aliases propagate perfectly, leading to ValueErrors when PK columns with identical sanitized bases get renamed with suffixes. Alternative: use dbt (github.com/dbt-labs/dbt), which is a production-grade, well-maintained open-source framework for declaring schema models, generating DDL for multiple databases, and managing migrations via YAML and SQL that can replace bespoke DDL inference logic.

## Align:
DataAligner.align begins by ensuring a dialect is known (detects from SQLAlchemy connection if unset), then inspects the target table via SchemaAnalyzer to capture columns/constraints; if schema evolution is enabled it compares DataFrame columns (after optional col mapping) to existing columns and issues ALTER TABLE DDL for any new ones before re-reading metadata. It then maps DataFrame columns case-insensitively to SQL names, drops extras (logging warnings), coerces each column via TypeValidator (integer/float/string/bool/datetime/json/binary flows that log failures, run optional outlier detection when scikit-learn is available, and raise errors when strict thresholds are breached), enforces NOT NULL constraints (raising in strict mode or warning and coercing otherwise), fills missing required columns with null, validates primary key and unique constraints (duplicates trigger diagnostic errors), records metrics if enabled, finishes by converting pandas nullable/NumPy numeric types to native Python scalars to keep strict drivers happy, and returns the aligned frame.

Failure modes include missing tables (stops with NoSuchTableError before any alignment), schema evolution ALTER TABLE statements failing (errors are caught and recorded but alignment proceeds with diagnostics), outlier detection/date parsing falling back silently when dependencies are absent so failure tracking is still recorded but no extra remediation occurs, strict validation mode raising ValueError when failure thresholds are exceeded while permissive modes only warn and coerce to null, extra columns being dropped without user intervention, NOT NULL violations emitting warnings/raising depending on mode, constraint duplicates logged with diagnostics but not auto-resolved, and PerformanceOptimizer’s cache stats being inaccurate because cached_type_check always increments hits and never records misses, so reported cache efficiency may not reflect reality.


## Cast: 
Core Problem: Provide resilient, dialect-aware bulk upsert/insert/update utilities for ETL ingestion by normalizing inputs, validating schema constraints, and falling back to per-row execution when bulk writes fail.

Execution Flow Narrative: Inputs are normalized into pure Python records, schema-aligned (auto_* helpers call DataAligner to match DB tables and add missing columns), and then dispatched to dialect-specific upsert/insert routines that chunk the data, validate columns/constraints, optionally trace SQL, and rely on _execute_with_row_isolation to perform bulk batches with lazy per-row retries and tolerance limits when errors occur; updates use build_update with aligned where clauses and case-insensitive field resolution before executing via SQLAlchemy text.

Steps:

_normalize_data converts DataFrame/dict/sequence into list of dicts while coercing pandas timestamp/NaN variants to native Python values.
Dialect upsert/insert functions iterate chunks via _chunk_iter, resolve the target table/columns, validate the supplied constraint (except MySQL), build a dialect-specific MERGE/ON CONFLICT/ON DUPLICATE KEY statement, and optionally dump SQL for tracing.
_execute_with_row_isolation attempts the bulk chunk, catches any exception, runs row-by-row up to tolerance, logs aggregated errors, aborts when failures exceed tolerance, and returns statistics.
auto_upsert/auto_insert align incoming data with DataAligner, detect the engine dialect, and call the appropriate specialized routine.
auto_update ensures where clauses match actual column names, builds SQL via build_update, and executes with traced SQL when requested.
Implicit Contracts: SQLAlchemy Engine/Connection available, target tables reflectable, input data convertible to DataFrame/dict, constrain matches a unique/PK (except MySQL ignores it), logging and SchemaAligner dependencies present, and chunk/tolerance defaults (10k/5) suit dataset size.

Failure Modes: Bulk insert/upsert rejects rows silently until row-by-row fallback, so partial success may hide errors if warnings suppressed; _validate_constrain_unique trusts caller after reflection failure, risking non-unique conflict detection; MySQL upsert ignores constrain, so wrong key may be used; row-fallback aborts once tolerance reached, leaving remaining rows untouched but reported only via stats; auto_* relies on SchemaAligner – if alignment fails silently (e.g., pandas missing) operations may run against misaligned schema.

Obscure Choices: _execute_with_row_isolation compiles detailed per-row error messages but only raises when no rows succeed; _write_sql_to_file emits literal-binded SQL for tracing; auto_update lowercases BOM-stripped where fields to match case-sensitive schemas; Oracle upsert builds raw MERGE SQL against DUAL to avoid SQLAlchemy merge bugs; _normalize_data aggressively converts pandas NA/NaT to None to avoid Oracle/MSSQL binding issues.

## Data Profiler:
data_profiler.py profiles pandas DataFrames, samples them, and locates a primary key: profile_dataframe walks each column to tally nulls, unique counts, and infers categorical or datetime flags (with a best-effort object-to-datetime conversion and suppressed ValueErrors) while logging high-null columns; sample_dispatcher validates inputs, optionally drops null rows, sorts by a key or index, and concatenates head/tail slices to deliver a small representative frame along with metadata/warnings; get_pk relies on the dfinfo output to compute uniqueness ratios, pick a unique column (or compose one via _build_composite_pk, adding datetime candidates up to max_composite_size), and drops rows with null components before testing uniqueness, logging both info and suppressed json logging failures.

Failure modes include: profile_dataframe returns empty for empty inputs and continues despite ValueErrors, so malformed datetimes quietly bypass detection; sample_dispatcher silently returns an empty sample when filtering removes all rows or when percent is invalid (raised) but other heuristics just log warnings, so high-null filtering can bias results without halting; get_pk raises if dfinfo lacks required columns, but when no unique candidates exist it falls back to the first column or a composite that may still not be unique, emits warnings, and stores non-unique pk values without enforcement, while dropping null rows before checking uniqueness can hide duplicated null-heavy rows.

Replacement option: use ydata-profiling (formerly pandas-profiling), a well-maintained production-grade library that generates detailed profiling reports and can drive sampling/uniqueness analysis with richer diagnosti

## Harness

I have built a crud.py module with functions for generating insert, upsert and update sql query and param from dataframe. Now I want to test whether those function can really clean a DataFrame and whether they can produce the correct CRUD syntax from it by executing generated statement with real sql database.  

I want to create a utility module that will hand me three DataFrames prepared for INSERT, UPSERT and UPDATE, all derived from one original DataFrame. 

Assume the original DataFrame has 10 rows. 
- For INSERT df: I pick the rows 0-5
- For UPSERT df: I pick rows 3-7
- For UPDATE df: I take row-1 and row-7 and row-8

so the total data added rows will be 0-7 while row-8 update oprtation could not do any modification.
please generate other clear assumption

### Logic:

Nothing has to be changed in the INSERT frame.

For the UPSERT frame, however, we must simulate real overlap: rows 3 and 4 already exist, so I will alter the last two columns of 3 and 4 no rows value with another respecting the series datatypes.primary-key column we will simply skip it while changing values. we will also prepare a SELECT statement (through a separate helper method) that pulls the same rows out of the database. user get upsert testing ready dataframe and select query for extrac exact these 2 rows what we build using pk. we have to implement upsert-validation method that have access original df, df prepared for upsert and input dataframe for comparisong return return compared result.

Similarly, for INSERT testing, we will alter a few values in the two rows we chose (never touching any constraint columns).
This means:
 
To obtain the UPSERT test frame and select query the caller must supply the primary-key column name(s).
To obtain the UPDATE test frame the caller must supply the constraint column name(s).
In both cases we will also return the matching SELECT statement(s).

Our ultimate goal is to verify that the SQL-generation modules we wrote really emit the correct SQL and our execution manager executes which reflection takes places.




