# DataAligner Usage Guide

## Quick Start

```python
from sqlalchemy import create_engine
from aligner import aligner
import pandas as pd

engine = create_engine("sqlite:///mydb.db")
df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

# Basic alignment
aligned_df = aligner(engine, df, "users")
aligned_df.to_sql("users", engine, if_exists="append", index=False)
```

## Installation

```bash
pip install pandas sqlalchemy numpy scikit-learn pyyaml
```

## API Methods

### 1. Direct Function Call

```python
from aligner import aligner

# Basic usage
aligned_df = aligner(engine, df, "table_name")

# With custom configuration
config = {
    "failure_threshold": 0.05,
    "add_missing_cols": True,
    "outlier_detection": False
}
aligned_df = aligner(engine, df, "table_name", config)
```

### 2. Preset Functions

```python
from aligner import strict_aligner, fast_aligner, safe_aligner

# Strict validation (1% failure tolerance)
aligned_df = strict_aligner(engine, df, "users")

# Fast processing (20% tolerance, caching enabled)
aligned_df = fast_aligner(engine, df, "users")

# Safe mode (schema evolution enabled)
aligned_df = safe_aligner(engine, df, "users")
```

### 3. Decorator Usage

```python
from aligner import align_to_sql

# Basic decorator
@align_to_sql("users")
def insert_users(engine, df):
    return df.to_sql("users", engine, if_exists="append", index=False)

# With custom config
@align_to_sql("products", {"failure_threshold": 0.02, "add_missing_cols": True})
def insert_products(engine, df):
    return df.to_sql("products", engine, if_exists="append", index=False)

# Using presets
@align_to_sql("orders", preset="strict")
def insert_orders(engine, df):
    return df.to_sql("orders", engine, if_exists="append", index=False)

# Call decorated functions
result = insert_users(engine, user_df)
```

## Configuration Options

### Core Settings

```python
config = {
    # Validation behavior
    "validation_mode": "balanced",     # strict, balanced, fast
    "on_error": "coerce",             # coerce, raise, skip
    "failure_threshold": 0.1,         # 10% max failure rate
    
    # Schema operations
    "add_missing_cols": False,        # Auto-add new columns
    "validate_constraints": True,     # Check PK/FK/unique
    "allow_schema_evolution": False,  # Enable ALTER TABLE
    
    # Performance
    "enable_caching": True,
    "parallel_processing": True,
    "chunk_size": 10000,             # Rows per chunk
    "max_workers": 4,
    
    # Outlier detection
    "outlier_detection": True,
    "outlier_methods": ["isolation_forest", "z_score"],
    "outlier_threshold": 0.05,
    
    # Diagnostics
    "collect_diagnostics": True,
    "print_summary": False
}
```

### Preset Configurations

```yaml
# aligner.yml presets
strict:
  validation_mode: "strict"
  failure_threshold: 0.01
  strict_integers: true
  outlier_detection: true

fast:
  validation_mode: "fast"
  failure_threshold: 0.2
  parallel_processing: true
  collect_diagnostics: false

safe:
  validation_mode: "balanced"
  add_missing_cols: true
  allow_schema_evolution: true
  print_summary: true
```

## Type Validation Examples

### Integer Validation

```python
# Strict mode rejects fractional values
df = pd.DataFrame({"score": [100, 95.5, 87]})  # 95.5 → NULL
aligned = strict_aligner(engine, df, "scores")

# Fast mode allows coercion
aligned = fast_aligner(engine, df, "scores")   # 95.5 → 96
```

### String Length Validation

```python
# VARCHAR(50) column
df = pd.DataFrame({"name": ["Alice", "Very Long Name That Exceeds Limit"]})
aligned = aligner(engine, df, "users")  # Long name → NULL
```

### DateTime Validation

```python
df = pd.DataFrame({
    "created_at": ["2023-01-01", "invalid-date", "2023-12-31T10:30:00"]
})
aligned = aligner(engine, df, "events")  # invalid-date → NULL
```

## Schema Evolution

```python
# DataFrame has new column not in database
df = pd.DataFrame({
    "id": [1, 2],
    "name": ["Alice", "Bob"],
    "new_field": ["value1", "value2"]  # Missing in DB
})

config = {"add_missing_cols": True, "allow_schema_evolution": True}
aligned = aligner(engine, df, "users", config)
# Executes: ALTER TABLE users ADD COLUMN "new_field" VARCHAR(255)
```

## Outlier Detection

```python
# 15% of rows have invalid data
df = pd.DataFrame({
    "age": [25, 30, 999, 35, -50, 28, 888, 32]  # 999, -50, 888 are outliers
})

config = {
    "outlier_detection": True,
    "outlier_methods": ["isolation_forest", "z_score"],
    "outlier_threshold": 0.1  # 10% outlier tolerance
}
aligned = aligner(engine, df, "users", config)
# Outliers coerced to NULL instead of failing entire batch
```

## Error Handling

### Coerce Mode (Default)

```python
config = {"on_error": "coerce", "failure_threshold": 0.1}
aligned = aligner(engine, df, "users", config)
# Invalid values → NULL, continues if <10% failures
```

### Raise Mode

```python
config = {"on_error": "raise", "failure_threshold": 0.05}
try:
    aligned = aligner(engine, df, "users", config)
except ValueError as e:
    print(f"Alignment failed: {e}")
```

### Skip Mode

```python
config = {"on_error": "skip"}
aligned = aligner(engine, df, "users", config)
# Drops rows with validation errors
```

## Performance Optimization

### Large DataFrames

```python
config = {
    "chunk_size": 50000,        # Process in 50K row chunks
    "parallel_processing": True,
    "max_workers": 8,
    "enable_caching": True
}
aligned = aligner(engine, large_df, "big_table", config)
```

### Caching

```python
# First call builds cache
aligned1 = aligner(engine, df1, "users")

# Subsequent calls use cached schema
aligned2 = aligner(engine, df2, "users")  # Faster
```

## Diagnostics and Monitoring

### Enable Detailed Reporting

```python
config = {
    "collect_diagnostics": True,
    "print_summary": True,
    "verbose_logging": True
}
aligned = aligner(engine, df, "users", config)

# Output:
# === Alignment Summary ===
# Columns processed: 5
# Total failures: 23
# Average failure rate: 2.3%
# Outliers detected: 12
```

### Access Diagnostics Programmatically

```python
from aligner import DataAligner, AlignmentConfig

config = AlignmentConfig(collect_diagnostics=True)
data_aligner = DataAligner(config)
aligned = data_aligner.align(engine, df, "users")

# Get diagnostics
diagnostics = data_aligner.diagnostics
summary = diagnostics.get_validation_summary()
failed_columns = diagnostics.get_failed_columns()

for result in failed_columns:
    print(f"{result.column}: {result.failure_rate:.2%} failure rate")
    print(f"Sample failures: {result.sample_failures[:3]}")
```

## Database Support

### SQLite

```python
engine = create_engine("sqlite:///mydb.db")
aligned = aligner(engine, df, "users")
```

### PostgreSQL

```python
engine = create_engine("postgresql://USER:PASSWORD@localhost/mydb")
aligned = aligner(engine, df, "users", {"validate_constraints": True})
```

### MySQL

```python
engine = create_engine("mysql+pymysql://USER:PASSWORD@localhost/mydb")
aligned = aligner(engine, df, "users")
```

## Advanced Usage

### Column Mapping

```python
# DataFrame columns don't match DB columns
df = pd.DataFrame({
    "user_name": ["Alice"],     # Maps to "name"
    "user_score": [95]          # Maps to "score"
})

# Use DataAligner directly for column mapping
from aligner import DataAligner, AlignmentConfig
config = AlignmentConfig()
data_aligner = DataAligner(config)
aligned = data_aligner.align(engine, df, "users", 
                           col_map={"user_name": "name", "user_score": "score"})
```

### Batch Processing

```python
def process_batch(batch_df):
    return aligner(engine, batch_df, "events", {"chunk_size": 10000})

# Process multiple files
for file in data_files:
    df = pd.read_csv(file)
    aligned = process_batch(df)
    aligned.to_sql("events", engine, if_exists="append", index=False)
```

### Custom Validation Pipeline

```python
@align_to_sql("users", {"failure_threshold": 0.01})
def validate_and_insert(engine, df):
    # Custom pre-processing
    df = df.dropna(subset=["email"])
    df["created_at"] = pd.Timestamp.now()
    
    # Insert aligned data
    return df.to_sql("users", engine, if_exists="append", index=False)

result = validate_and_insert(engine, user_df)
```

## Best Practices

1. **Start with safe_aligner** for new projects
2. **Use strict_aligner** for critical data pipelines
3. **Enable diagnostics** during development
4. **Set appropriate failure_threshold** based on data quality
5. **Use chunking** for DataFrames >100K rows
6. **Enable caching** for repeated operations
7. **Test schema evolution** in development first
8. **Monitor outlier detection** results

## Troubleshooting

### High Failure Rates

```python
# Enable detailed diagnostics
config = {"collect_diagnostics": True, "print_summary": True}
aligned = aligner(engine, df, "users", config)
# Check output for specific column issues
```

### Performance Issues

```python
# Optimize for large datasets
config = {
    "chunk_size": 25000,
    "parallel_processing": True,
    "collect_diagnostics": False  # Disable for speed
}
```

### Schema Conflicts

```python
# Allow automatic schema updates
config = {
    "add_missing_cols": True,
    "allow_schema_evolution": True,
    "on_error": "coerce"
}
```
