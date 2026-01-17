# ETL API Reference

This document provides a detailed list of classes, functions, and arguments available in the TelcoETL pipeline.

---

### 1. Class `ETL`

The main orchestrator for database operations.

#### `__init__(self, db_uri, schema_dir=None, logging_enabled=True, pk='verify', chunksize=10000, add_missing_cols=False, rename_column=False, schema_aware_casting=True, **cast_kwargs)`

Used to set instance-level defaults.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `db_uri` | `str` | *None* | **Required**. SQLAlchemy connection string (e.g. `sqlite:///data.db`). |
| `schema_dir` | `str/Path`| `None` | Path to save DDL and metadata history. |
| `logging_enabled`| `bool` | `True` | Toggles console and file logging. |
| `pk` | `str/list`| `'verify'` | Default PK strategy for future operations. |
| `chunksize` | `int` | `10000` | Default number of rows per batch. |
| `add_missing_cols`| `bool` | `False` | Default for auto-expanding tables with new columns. |
| `rename_column` | `bool` | `False` | Default for auto-sanitizing column headers. |
| `schema_aware_casting`| `bool` | `True` | Whether to use DB schema to guide type casting. |
| `**cast_kwargs` | `Any` | - | Any argument from `CastConfig` (see Section 3). |

---

#### `create_load(source, table, drop_if_exists=True, pk=None, failure_threshold=None, **kwargs)`

Performs a full table creation and load.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `Any` | *None* | File path (CSV/Excel), DataFrame, or list of dicts. |
| `table` | `str` | *None* | Target database table name. |
| `drop_if_exists` | `bool` | `True` | If `True`, drops the table before recreating it. |
| `pk` | `str/list`| `None` | Override the default PK strategy for this run. |
| `failure_threshold`| `int` | `None` | Override casting failure tolerance (0.0 to 1.0). |
| `**kwargs` | `Any` | - | Overrides for `CastConfig`. |

---

#### `append(source, table, failure_threshold=None, **kwargs)`

Adds new rows to an existing table.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `Any` | *None* | Data source (File/DF/List). |
| `table` | `str` | *None* | Target table name. |
| `failure_threshold`| `int` | `None` | Casting tolerance. |
| `**kwargs` | `Any` | - | Overrides for `CastConfig`. |

---

#### `upsert(source, table, constrain, failure_threshold=None, **kwargs)`

Updates records if they exist (based on `constrain`), otherwise inserts them.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `Any` | *None* | Data source (File/DF/List). |
| `table` | `str` | *None* | Target table name. |
| **`constrain`** | `list` | *None* | **Required**. Column(s) used as unique keys for matching. |
| `failure_threshold`| `int` | `None` | Casting tolerance. |
| `**kwargs` | `Any` | - | Overrides for `CastConfig`. |

---

#### `update(source, table, where, expression=None, rename_column=None, add_missing_cols=None, trace_sql=False, **kwargs)`

Updates specific records based on filter conditions.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `Any` | *None* | Data source containing values to update. |
| `table` | `str` | *None* | Target table name. |
| `where` | `list` | *None* | List of filter tuples, e.g., `[('ID', '=', 101)]`. |
| `expression` | `str` | `None` | Raw SQL string for WHERE clause (overrides `where`). |
| `trace_sql` | `bool` | `False` | Writes the final SQL statement to a text file for review. |
| `**kwargs` | `Any` | - | Overrides for `CastConfig`. |

---

#### `auto_etl(source, table, pk='derive', add_missing_cols=None, **kwargs)`

Intelligently decides between `create_load` or `upsert` based on table existence.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `Any` | *None* | Data source. |
| `table` | `str` | *None* | Table name. |
| `pk` | `str/list`| `'derive'` | Strategy for identifying unique records. |
| `add_missing_cols`| `bool` | `None` | Enable/disable schema evolution for this operation. |

---

### 2. Helper Class `CleaningConfig`

Used to control the `data_cleaner.py` logic.

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `strip_bom` | `bool` | `True` | Strips the hidden `\ufeff` from strings. |
| `normalize_whitespace`| `bool` | `True` | Fixes non-breaking spaces and removes zero-width joiners. |
| `standardize_smart_chars`| `bool` | `True` | Converts smart quotes and em-dashes to ASCII. |
| `normalize_unicode` | `str` | `'NFKC'` | Resolves visual homoglyphs across alphabets. |
| `remove_control_chars`| `bool` | `True` | Removes non-printable characters. |
| `remove_special_chars` | `bool` | `False` | Removes all non-alphanumeric chars (Aggressive). |
| `lowercase_headers` | `bool` | `False` | Forces all columns to lowercase. |

---

### 3. Helper Class `CastConfig`

Underlying settings for the casting engine (`utils/casting.py`).

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `infer_threshold` | `float` | `0.9` | Ratio of rows required to confirm a data type. |
| `nan_threshold` | `float` | `0.3` | Null ratio that stops aggressive casting to stay as 'string'. |
| `max_null_increase` | `float` | `0.1` | Safety tripwire: rejects casting if too much data is lost. |
| `max_sample_size` | `int` | `1000` | Max rows to scan for type inference. |
| `parallel` | `bool` | `False` | Multi-threaded processing for performance. |
| **`dtype`** | `dict` | `None` | **Critical**: Force specific types, e.g., `{"zip": "string"}`. |

---

> [!NOTE]
> All parameters marked as `**kwargs` in `ETL` methods are passed directly to `CastConfig`. This means you can call `etl.upsert(..., infer_threshold=0.95, dtype={'id': 'string'})` on the fly.
