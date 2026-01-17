# ETL Toolkit Usage Guide

Production-ready ETL modules for DataFrame processing, type casting, schema analysis, and database operations.

---

## Quick Start

```python
import pandas as pd
from casting import cast_df
from ddl_create import df_ddl, df_ddl_create
from data_profiler import profile_dataframe, get_pk
from sql_corrector import SchemaAligner
from crud_v1 import insert, upsert, update

# Load data
df = pd.read_csv('data.csv')

# Cast types with metadata
df_clean, dtype_meta = cast_df(df, return_dtype_meta=True)

# Generate DDL
ddl, schema = df_ddl(df_clean, 'my_table', server='oracle', pk='id')

# Insert to database
from sqlalchemy import create_engine
engine = create_engine('oracle://USER:PASSWORD@host/db')
rows_inserted = insert(engine, 'my_table', df_clean)
```

---

## Module 1: Type Casting (`casting.py`)

### Basic Usage

```python
from casting import cast_df, CastConfig

# Auto-cast with defaults
df_typed = cast_df(df)

# With metadata tracking
df_typed, meta = cast_df(df, return_dtype_meta=True)
print(meta['column_name'])  # {'input_dtype': 'object', 'output_dtype': 'int64'}
```

### Advanced Configuration

```python
config = CastConfig(
    use_transform=True,           # Use enhanced datetime/numeric detection
    validate_conversions=True,    # Reject high-null conversions
    max_null_increase=0.1,        # Max 10% null increase allowed
    infer_threshold=0.9,          # 90% confidence for type inference
    parallel=True                 # Parallel processing for large DataFrames
)

df_typed = cast_df(df, config=config)
```

### Explicit Type Mapping

```python
dtype_map = {
    'user_id': 'int64',
    'created_at': 'datetime64',
    'is_active': 'boolean'
}

df_typed = cast_df(df, dtype=dtype_map)
```

### Semantic Object Types

```python
df_typed, meta = cast_df(df, return_dtype_meta=True)

# Check object column semantics
for col, info in meta.items():
    if 'object_semantic_type' in info:
        print(f"{col}: {info['object_semantic_type']}")
        # STRING_OBJECT, STRUCTURED_OBJECT, or TRUE_OBJECT
```

---

## Module 2: DDL Generation (`ddl_create.py`)

### Generate DDL for Single Database

```python
from ddl_create import df_ddl

ddl, metadata = df_ddl(
    df,
    table='users',
    server='oracle',           # oracle, postgres, mysql, mssql, sqlite
    schema='hr',
    pk='user_id',              # Single or list of columns
    fk=[('dept_id', 'departments', 'id')],
    unique=['email'],
    autoincrement=('user_id', 1),
    varchar_sizes={'name': 100, 'email': 255}
)

print(ddl)  # CREATE TABLE statement
print(metadata['columns'])  # Column metadata
```

### Multi-Database DDL + Data Load

```python
from ddl_create import df_ddl_create

conn_dict = {
    'oracle': 'oracle://USER:PASSWORD@host/db',
    'postgres': 'postgresql://USER:PASSWORD@host/db',
    'mysql': 'mysql://USER:PASSWORD@host/db'
}

results = df_ddl_create(
    conn_dict=conn_dict,
    df=df,
    table='users',
    schema='public',
    pk='user_id',
    cast=True,              # Auto-cast before DDL
    sanitize=True           # Clean column names
)

# Check results per database
for db, result in results['servers'].items():
    print(f"{db}: {result['status']}")
    if result['status'] == 'COMPLETED':
        print(f"  DDL: {result['ddl_file']}")
```

---

## Module 3: Data Profiling (`data_profiler.py`)

### Profile DataFrame

```python
from data_profiler import profile_dataframe, get_pk

# Generate column statistics
profile = profile_dataframe(df)
print(profile[['Column', 'Data Type', 'Is Categorical', '% of None']])
```

### Auto-Detect Primary Key

```python
df_with_pk, pk_name, pk_meta = get_pk(
    df,
    dfinfo=profile_dataframe(df),
    uniqueness_threshold=0.8,
    max_composite_size=3
)

print(f"Primary Key: {pk_name}")
print(f"Components: {pk_meta['components']}")
print(f"Is Unique: {pk_meta['is_unique']}")
```

### Smart Sampling

```python
from data_profiler import sample_dispatcher

sampled_df, meta = sample_dispatcher(
    df,
    percent=10,              # Sample 10% (5% head + 5% tail)
    sort=True,
    filter_none=True,
    sort_key='created_at'
)

print(f"Sampled {meta['sampled_rows']} rows ({meta['reduction_pct']:.1f}% reduction)")
```

---

## Module 4: Schema Analysis (`schema_analyzer.py`)

### Analyze Table Schema

```python
from schema_analyzer import SchemaAnalyzer
from sqlalchemy import create_engine

engine = create_engine('oracle://USER:PASSWORD@host/db')
analyzer = SchemaAnalyzer(engine)

report = analyzer.analyze_table(
    schema='hr',
    table_name='employees',
    df=df,                    # Optional: validate DataFrame against schema
    run_fk_checks=True        # Validate foreign key integrity
)

# Check results
print(f"Table exists: {report.table_exists}")
print(f"Columns: {list(report.columns.keys())}")
print(f"Primary Key: {report.constraints.primary_key}")

if report.validation:
    print(f"NOT NULL OK: {report.validation.not_null_ok}")
    print(f"Issues: {report.validation.issues}")
    print(f"Suggestions: {report.validation.suggestions}")
```

### Dialect-Specific Checks

```python
if report.dialect_checks:
    print(f"Dialect: {report.dialect_checks.dialect}")
    for issue in report.dialect_checks.issues:
        print(f"  ⚠️ {issue}")
    for suggestion in report.dialect_checks.suggestions:
        print(f"  💡 {suggestion}")
```

---

## Module 5: Schema Alignment (`sql_corrector.py`)

### Strict Type Enforcement

```python
from sql_corrector import SchemaAligner

aligner = SchemaAligner(
    conn=engine,
    on_error='coerce',           # 'coerce' or 'raise'
    failure_threshold=0.03,      # Max 3% failures allowed
    validate_fk=True,
    add_missing_cols=True        # Auto-add missing columns
)

df_aligned = aligner.align(
    df,
    table='employees',
    schema='hr',
    semantic_type_meta={         # From casting module
        'notes': 'STRING_OBJECT',
        'metadata': 'STRUCTURED_OBJECT'
    }
)
```

### Column Mapping

```python
col_map = {
    'user_name': 'username',     # DataFrame col -> SQL col
    'user_email': 'email'
}

df_aligned = aligner.align(df, 'users', col_map=col_map)
```

---

## Module 6: CRUD Operations (`crud_v1.py`)

### Insert

```python
from crud_v1 import insert

rows = insert(
    engine,
    table_name='users',
    data=df,                     # DataFrame, dict, or list of dicts
    chunksize=1000
)
print(f"Inserted {rows} rows")
```

### Upsert (Insert or Update)

```python
from crud_v1 import upsert

rows = upsert(
    engine,
    dialect='oracle',            # oracle, postgres, mysql, mssql, sqlite
    table_name='users',
    data=df,
    constraint=['user_id'],      # Unique key for conflict detection
    chunksize=1000
)
print(f"Upserted {rows} rows")
```

### Update with Conditions

```python
from crud_v1 import update

# Simple update
rows = update(
    engine,
    table='users',
    data={'status': 'active'},
    where=[('user_id', '=', 123)]
)

# Complex conditions
rows = update(
    engine,
    table='users',
    data={'status': 'inactive'},
    where=[
        ('last_login', '<', '2023-01-01'),
        ('status', '=', 'active')
    ],
    expression='AND'             # 'AND', 'OR', or indexed pattern
)
```

---

## Complete ETL Pipeline Example

```python
import pandas as pd
from sqlalchemy import create_engine
from casting import cast_df
from data_profiler import profile_dataframe, get_pk
from ddl_create import df_ddl
from sql_corrector import SchemaAligner
from crud_v1 import upsert

# 1. Load raw data
df_raw = pd.read_csv('raw_data.csv')

# 2. Cast types with metadata
df_typed, dtype_meta = cast_df(df_raw, return_dtype_meta=True)

# 3. Profile and detect primary key
profile = profile_dataframe(df_typed)
df_with_pk, pk_name, pk_meta = get_pk(df_typed, profile)
pk_cols = pk_meta['components']

# 4. Generate DDL (if table doesn't exist)
ddl, schema = df_ddl(
    df_typed,
    table='staging_data',
    server='oracle',
    pk=pk_cols,
    cast=False  # Already casted
)
print(ddl)

# 5. Connect to database
engine = create_engine('oracle://USER:PASSWORD@host/db')

# 6. Align DataFrame to schema
aligner = SchemaAligner(engine, failure_threshold=0.05)
df_aligned = aligner.align(
    df_typed,
    table='staging_data',
    semantic_type_meta={col: meta.get('object_semantic_type') 
                        for col, meta in dtype_meta.items() 
                        if 'object_semantic_type' in meta}
)

# 7. Upsert data
rows = upsert(
    engine,
    dialect='oracle',
    table_name='staging_data',
    data=df_aligned,
    constraint=pk_cols,
    chunksize=5000
)

print(f"✅ ETL Complete: {rows} rows processed")
```

---

## Best Practices

### 1. Always Cast Before DDL

```python
df_typed = cast_df(df)  # Cast first
ddl, _ = df_ddl(df_typed, 'table', cast=False)  # Don't re-cast
```

### 2. Use Metadata for Debugging

```python
df_typed, meta = cast_df(df, return_dtype_meta=True)
for col, info in meta.items():
    if info['input_dtype'] != info['output_dtype']:
        print(f"{col}: {info['input_dtype']} → {info['output_dtype']}")
```

### 3. Validate Before Load

```python
from schema_analyzer import SchemaAnalyzer
analyzer = SchemaAnalyzer(engine)
report = analyzer.analyze_table('schema', 'table', df=df)
if report.validation and report.validation.issues:
    print("⚠️ Validation issues:", report.validation.issues)
```

### 4. Handle Failures Gracefully

```python
try:
    aligner = SchemaAligner(engine, on_error='raise', failure_threshold=0.01)
    df_aligned = aligner.align(df, 'table')
except ValueError as e:
    print(f"Alignment failed: {e}")
    # Fallback: use coerce mode
    aligner = SchemaAligner(engine, on_error='coerce')
    df_aligned = aligner.align(df, 'table')
```

### 5. Chunk Large DataFrames

```python
config = CastConfig(chunk_size=50000)
df_typed = cast_df(large_df, config=config)

rows = insert(engine, 'table', df_typed, chunksize=10000)
```

---

## Troubleshooting

### Issue: High null increase during casting

```python
# Solution: Adjust threshold or inspect data
config = CastConfig(max_null_increase=0.2, validate_conversions=True)
df_typed = cast_df(df, config=config)
```

### Issue: Oracle BLOB for string columns

```python
# Solution: Use semantic type metadata
df_typed, meta = cast_df(df, return_dtype_meta=True)
semantic_meta = {col: info.get('object_semantic_type') 
                 for col, info in meta.items() 
                 if 'object_semantic_type' in info}

aligner = SchemaAligner(engine)
df_aligned = aligner.align(df_typed, 'table', semantic_type_meta=semantic_meta)
```

### Issue: Primary key not unique

```python
# Solution: Check PK metadata and deduplicate
df_with_pk, pk_name, pk_meta = get_pk(df, profile_dataframe(df))
if not pk_meta['is_unique']:
    print(f"⚠️ PK not unique: {pk_meta['components']}")
    df_dedup = df.drop_duplicates(subset=pk_meta['components'])
```

---

## Supported Databases

| Database | DDL | CRUD | Schema Analysis | Notes |
|----------|-----|------|-----------------|-------|
| Oracle | ✅ | ✅ | ✅ | Full support including MERGE |
| PostgreSQL | ✅ | ✅ | ✅ | JSONB, arrays supported |
| MySQL | ✅ | ✅ | ✅ | ON DUPLICATE KEY UPDATE |
| MSSQL | ✅ | ✅ | ✅ | MERGE with OUTPUT |
| SQLite | ✅ | ✅ | ✅ | Limited ALTER TABLE |

---

## Dependencies

```bash
pip install pandas numpy sqlalchemy python-dateutil scikit-learn
```

Optional:

- `cx_Oracle` for Oracle
- `psycopg2` for PostgreSQL
- `pymysql` for MySQL
- `pyodbc` for MSSQL
