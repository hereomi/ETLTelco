---
description: How to run the ETL schema evolution test suite
---

# ETL Test Workflow

This workflow describes how to test the ETL engine's schema evolution and database operations using `test_etl_evolution.py`.

## Prerequisites

1. **Database Connection**: Ensure your Oracle/PostgreSQL/MySQL database is running
2. **Configuration**: Update `config.yml` with your database URI or pass it via `--db-uri`
3. **Test Data**: Place your CSV file in the `csv/` directory

## Quick Start

```bash
# Run with defaults (csv/quota.csv)
python test_etl_evolution.py

# Run all tests
python test_etl_evolution.py --all-tests
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `csv_path` | Path to CSV file | `csv/quota.csv` |
| `table_name` | Target table name | Derived from CSV filename |
| `--pk` | Primary key column(s) | `derive` (auto-detect) |
| `--subset-cols` | Columns for Phase 1 | `3` |
| `--subset-rows` | Rows for Phase 1 | `5` |
| `--db-uri` | Database connection URI | From `config.yml` |
| `--schema-dir` | DDL output directory | `schema` |
| `--all-tests` | Run extended test suite | `False` |

## Test Cases

### Core Tests (Default)

1. **create_load with Subset Schema**
   - **Logic**: Simulates an "old" version of the file or a partial initial load.
   - **Action**: Loads only the first *N* columns (default 3) and *M* rows (default 5).
   - **Validates**: DDL generation for a small schema, table creation, and initial data existence.

2. **auto_etl Upsert (add_missing_cols=False)**
   - **Logic**: Simulates receiving the full file (all rows/cols) but with schema evolution **disabled**.
   - **Action**: Attempts to upsert all data. Extra columns in source are dropped/ignored.
   - **Validates**: Robustness against schema drift when not configured to adapt. Existing rows are updated; new rows are inserted (with missing columns null/default).

3. **auto_etl Upsert (add_missing_cols=True)**
   - **Logic**: Simulates enabling schema evolution to "catch up" with the source.
   - **Action**: Detects missing columns, runs `ALTER TABLE` to add them, and re-upserts data.
   - **Validates**: Dynamic schema expansion. All columns for all rows should now be populated.

### Extended Tests (--all-tests)

1. **append New Data**
   - Appends new rows with offset PKs
   - Validates: Row count increases, no PK conflicts

2. **update Specific Records**
   - Updates single record via WHERE clause
   - Validates: Targeted modification works

3. **create_load Full Replace**
   - Drops and recreates with full schema
   - Validates: Clean replacement works

## Example Usage

```bash
# Test a specific CSV file
python test_etl_evolution.py csv/customers.csv

# Test with custom table name
python test_etl_evolution.py csv/orders.csv order_history

# Specify primary key columns
python test_etl_evolution.py csv/products.csv --pk=product_id,variant_id

# Use more columns in Phase 1
python test_etl_evolution.py csv/sales.csv --subset-cols=5 --subset-rows=10

# Connect to different database
python test_etl_evolution.py --db-uri="postgresql://USER:PASSWORD@localhost/mydb"

# Run full test suite
python test_etl_evolution.py csv/quota.csv --all-tests
```

## Test Flow Diagram

```mermaid
flowchart TD
    A[Load CSV] --> B[Phase 1: create_load]
    B --> C{Table Created?}
    C -->|Yes| D[Phase 2: Upsert no evolution]
    C -->|No| X[FAIL]
    D --> E[Phase 3: Upsert with evolution]
    E --> F{--all-tests?}
    F -->|Yes| G[Phase 4: Append]
    F -->|No| I[Verify]
    G --> H[Phase 5: Update]
    H --> J[Phase 6: Full Replace]
    J --> I
    I --> K[Summary]
```

## Expected Output

```
============================================================
 Test Summary
============================================================
  Tests Passed: 6/6
  [PASS] - create_load with Subset Schema
  [PASS] - auto_etl Upsert (add_missing_cols=False)
  [PASS] - auto_etl Upsert (add_missing_cols=True)
  [PASS] - append New Data
  [PASS] - update Specific Records
  [PASS] - create_load Full Replace
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ORA-00942: table or view does not exist` | Run Phase 1 first to create table |
| `ORA-00001: unique constraint violated` | PK detection failed; specify `--pk` manually |
| `ORA-38104: Columns in ON Clause cannot be updated` | Ensure PK columns are excluded from UPDATE |
| `UnicodeEncodeError` | Apply the ASCII fix in test_etl_evolution.py |
| `constrain column not found` | Case mismatch; ensure crud_v2.py lowercases keys |
| `DPY-3002: Python value of type "int64"`| Ensure type normalization is applied |

## Integration with CI/CD

```yaml
# Example GitHub Actions step
- name: Run ETL Tests
  run: |
    python test_etl_evolution.py csv/test_data.csv --all-tests
  env:
    DB_URI: ${{ secrets.DATABASE_URI }}
```
