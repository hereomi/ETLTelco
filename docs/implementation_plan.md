# Refactor main.py into a Reusable Function (Updated)

Refactor the ETL logic into a modular `run_etl` function that supports both configuration-driven and manual overrides.

## User Review Required

> [!IMPORTANT]
> **CSV Rewriting**: When `rename_column=True` and sanitization occurs, the source CSV file will be overwritten with the sanitized DataFrame to ensure future runs are consistent.
>
## Proposed Strategy: CRUD v2 Integration

We are transitioning to `crud_v2.py`. This module uses SQLAlchemy's robust dialect-specific MERGE implementations (especially for Oracle and MSSQL) via `oracle.dml.OracleMerge` and `mssql.dml.Merge`.

We will:

1. **Integrate SchemaAligner**: Add full alignment support to `v2` for inserts, upserts, and updates.
2. **Ultra-Safe Normalization**: Port our high-robustness null cleaning to `v2` to prevent Oracle `DPY-4004` errors.
3. **Unified API**: Ensure `etl.py` can switch seamlessly to this more advanced CRUD layer.

## Proposed Changes

### [main.py](file:///e:/PyCode/_WebProject/telcoetl/etl_2/fn2/main.py)

1. **Define `run_etl` Function**:
    - **Arguments**:
        - `csv_path: Path`
        - `table: str`
        - `db_uri: str`
        - `drop_if_exists: bool = True`
        - `rename_column: bool = True`
        - `schema_dir: Optional[Path] = None`
        - `logging_enabled: bool = True`
    - **Key Logic**:
        - **Dialect Derivation**: Automatically detect the server type (oracle, postgres, etc.) from the SQLAlchemy engine.
        - **CSV Sync**: If `rename_column` is active and the column names change, save the updated DataFrame back to the `csv_path`.
        - **Logging Control**: Use the `logging_enabled` flag to toggle the global logging state.
        - **Clean State**: Use the `drop_if_exists` flag to manage table recreation.

2. **Update `main` block**:
    - Simplified to a single `run_etl` call which internally instantiates the `ETL` class and uses the "smart" `auto_etl` dispatcher.
    - This provides a clean interface while maintaining the full power of the class-based engine.

### [etl.py](file:///e:/PyCode/_WebProject/telcoetl/etl_2/fn2/etl.py) (Updated)

Transition from a functional `run_etl` approach to a stateful `ETL` class with **Schema-Guided Casting** (using DB types to inform `cast_df` for existing tables):

1. **`ETL.__init__`**:
    - Initializes engine and detects dialect.
    - Stores common configuration (`logging_enabled`, `schema_dir`).
    - **Specific Defaults**:
        - `pk`: 'verify'
        - `chunksize`: 10000
        - `add_missing_cols`: False
        - `failure_threshold`: 1
        - `rename_column`: False
    - **Cast Support**: Accepts `**cast_kwargs` to pass through to `cast_df`.
    - **Schema Awareness**: New flag `schema_aware_casting: bool = True`.
2. **`ETL.create_load()`**:
    - Equivalent to the current full ETL flow.
    - Handles full lifecycle: sanitize, generate DDL, drop/create table, and insert.
3. **`ETL.append()`**:
    - A streamlined method for adding data to existing tables.
    - Skips DDL generation and table management.
    - Maintains data sanitization and batch insertion.
4. **`ETL.upsert()`**:
    - Updates existing records or inserts new ones based on a PK/constraint.
    - Uses `crud_v1.upsert` for record-level persistence.
5. **`ETL.auto_etl()`**:
    - A "smart" dispatcher:
        - If table exists and PK provided: runs **`upsert`**.
        - If table exists and no PK: runs **`append`**.
        - If table missing: runs **`create_load`**.

### [crud_v1.py](file:///e:/PyCode/_WebProject/telcoetl/etl_2/fn2/crud_v1.py) [NEW]

1. **Update `insert`/`upsert`/`update`**:
    - Accept `semantic_meta: Dict[str, str]` argument.
    - Pass `semantic_meta` to `SchemaAligner.align()`.

This design improves state management and provides clearer intent for different ingestion scenarios. Private helpers are moved inside the class or kept as utility functions as appropriate.

## Verification Plan

### Automated Tests

- **Sanitization Sync**: Run a test with a "dirty" CSV (e.g., colons in headers), verify the CSV is updated with clean headers after the run.
- **Dialect Detection**: Verify the function correctly identifies "oracle" from the URI.

### Manual Verification

- Ensure `log/` files are only generated when `logging_enabled=True`.
- Check that DDL text/JSON are saved to the specified `schema_dir`.
