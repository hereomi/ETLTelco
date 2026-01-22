# ETLTelco CRUD Utilities

Utilities for building parameterized SQL statements (SELECT/UPDATE/INSERT) with DataFrame-aware column alignment and safe identifier escaping.

## Installation

```bash
pip install pandas
```

## Quick Start

```python
import pandas as pd
from crud import query_builder

row = pd.DataFrame([{"id": 1, "name": "Alice", "age": 30}])

# SELECT
sql, params = query_builder.build_select(
    data=row,
    table="public.users",
    where=[("id", "=", 1)],
    dialect="postgres"
)
# sql: SELECT * FROM "public"."users" WHERE "id" = :id_1
# params: {"id_1": 1}

# UPDATE
sql, params = query_builder.build_update(
    data=row,
    table="public.users",
    where=[("id", "=", 1)],
    dialect="postgres",
    update_cols=["name", "age"]
)
# sql: UPDATE "public"."users" SET "name" = :u_name, "age" = :u_age WHERE "id" = :id_1
# params: {"u_name": "Alice", "u_age": 30, "id_1": 1}

# INSERT
sql, params = query_builder.build_insert(row, table="public.users", dialect="postgres")
# sql: INSERT INTO "public"."users" ("id", "name", "age") VALUES (:i_id, :i_name, :i_age)
# params: {"i_id": 1, "i_name": "Alice", "i_age": 30}
```

## WHERE Clause Helpers

- Use tuples `(field, operator, value)` or SQL-like strings (e.g., `"age >= 21"`).
- Supported operators: `=`, `!=`, `>`, `>=`, `<`, `<=`, `LIKE`, `IN (...)`, `BETWEEN ... AND ...`.
- Placeholders: use `?` as value to pull the value from the current row.
- Expression: pass `expression` (e.g., `"1 AND (2 OR 3)"`) to combine multiple conditions; otherwise conditions are joined with `AND`.

### Example with placeholders and expression

```python
row = pd.DataFrame([{"id": 1, "name": "Alice", "status": "active"}])
sql, params = query_builder.build_select(
    data=row,
    table="public.users",
    where=[("id", "=", "?"), "status = 'active'"],
    expression="1 AND 2",
    dialect="postgres"
)
# sql: SELECT * FROM "public"."users" WHERE "id" = :id_1 AND "status" = :status_2
# params: {"id_1": 1, "status_2": "active"}
```

## Column Alignment

- Column matching is case-insensitive; names are aligned to DataFrame columns.
- By default SELECT requires WHERE columns to exist in the data; UPDATE allows missing WHERE columns (set `allow_missing_where_cols` to control).

## Dialects

Supported `dialect` values: `sqlite`, `postgres`/`postgresql`, `oracle`, `mysql`, `mssql`. Identifiers are escaped per dialect.

## Logging

Debug logging writes to `where_build.log` when `DEBUG_LOG_ENABLED` is `True` inside `query_builder.py`.

## Error Handling

- `ValueError` for unsupported dialects or invalid conditions
- `TypeError` if input data is not DataFrame/list[dict]/dict
- `ValueError` when required WHERE/UPDATE columns are missing

## Auto Upsert/Insert/Update (with schema alignment)

```python
from sqlalchemy import create_engine
from crud import auto_upsert, auto_insert, auto_update

engine = create_engine("postgresql+psycopg2://user:pwd@host/db")
data = [{"id": 1, "name": "Alice", "status": "active"}]

# Upsert rows using dialect-specific bulk SQL
result = auto_upsert(engine, data, table="public.users", constrain=["id"], chunk=5000, tolerance=3, trace_sql=True)

# Bulk insert with chunking and diagnostics
result = auto_insert(engine, data, table="public.users", chunk_size=5000, tolerance=2, trace_sql=False)

# Single-row update with aligned WHERE conditions
result = auto_update(
    engine=engine,
    table="public.users",
    data={"status": "inactive"},
    where=[("id", "=", 1)],
    expression=None,
    trace_sql=True,
)
```

- `constrain` must reference a primary/unique key; validation is strict by default.
- `add_missing_cols` auto-aligns incoming data to the table using `schema_align` before execution.
- `trace_sql=True` writes compiled SQL to `<func>_<table>.txt` for inspection.

## Dialect Helpers

Per-dialect functions expose bulk insert/upsert primitives on top of SQLAlchemy:

- PostgreSQL: `postgres_upsert`, `postgres_insert`
- Oracle: `oracle_upsert`, `oracle_insert`
- MySQL: `mysql_upsert`, `mysql_insert`
- SQLite: `sqlite_upsert`, `sqlite_insert`
- SQL Server: `mssql_upsert`, `mssql_insert`

Each helper accepts `engine`, `data`, `table`, chunk sizing, `tolerance`, `trace_sql`, and `strict` flags. Use these when you want direct control without auto alignment.

## Data Handling and Safety

- `_normalize_data` converts dict/list/DataFrame inputs into clean `list[dict]`, replacing NaN/NaT with `None` and converting timestamps to Python datetime.
- `_ensure_data_columns_in_table` validates payload columns against reflected table metadata before execution.
- `_collect_not_null_violations` captures rows missing NOT NULL columns; raise immediately in `strict` mode.
- `_execute_with_row_isolation` attempts bulk execution first, then falls back to row-by-row with failure `tolerance`, recording diagnostics.
