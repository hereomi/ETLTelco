# ETLTelco

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/yourusername/ETLTelco.svg?style=social)](https://github.com/yourusername/ETLTelco)

**ETLTelco** is a high-performance, SQL-intelligent ETL engine designed to bridge the gap between messy real-world data and structured database systems. It doesn't just move data; it sanitizes, heals, and evolves your database schema automatically.

Whether you are dealing with legacy Oracle systems, massive Postgres clusters, or agile SQLite files, ETLTelco ensures your data lands perfectly every time.

---

## Key Features

### Robust Data Sanitization

Stop hidden characters from breaking your pipeline. ETLTelco includes a specialized cleaning layer that targets:

- **Zero-Width Villains**: Strips BOM (`\ufeff`), zero-width spaces, and non-breaking chars.
- **Visual Homoglyphs**: Automatically converts look-alike characters from mixed alphabets (e.g., Cyrillic 'А' ➔ Latin 'A').
- **Smart Typography**: Standardizes "curly" quotes and em-dashes into DB-safe ASCII.

### Intelligent Type Inference & Casting

Forget manual schema definitions. Our casting engine uses:

- **Confidence Scoring**: Analyzes data distributions to guess types (Date, UUID, Int, JSON) with customizable thresholds.
- **Safety Tripwires**: Rejects conversions that would result in excessive data loss (null increase check).
- **Schema-Guided Casting**: Re-uses existing database types to guide the ingestion of new data.

### Auto-DDL & Schema Evolution

- **Dialect-Aware**: Generates optimized DDL for **Oracle, PostgreSQL, SQLite, MySQL, and MSSQL**.
- **Self-Healing Schema**: New columns in your source file? ETLTelco automatically executes `ALTER TABLE` to keep your database in sync.
- **Primary Key Intelligence**: Automatically derives or verifies PK constraints across all supported dialects.

### SQL Intelligence (Where-Build)

Power your updates and upserts with an advanced WHERE clause builder:

- Supports complex tuples, literal expressions, and case-insensitive column alignment.
- Strips artifacts (like BOMs) from column metadata on the fly.

### Performance & Logging

- **Trace-SQL**: Inspect the exact SQL generated for every transaction.
- **Extensive Logging**: Detailed JSON-based logs for performance metrics, type transitions, and alignment diagnostics.
- **Chunked Processing**: Vectorized operations and chunked database inserts for massive datasets.

---

## Quick Start

```python
from etl import ETL

# 1. Initialize with Multi-DB support
etl = ETL(
    db_uri="postgresql://user:PASSWORD@localhost:5432/db",
    # Example URIs for other databases:
    # mysql: "mysql+pymysql://root:PASSWORD@127.0.0.1:3306/akomi"
    # mssql: "mssql+pyodbc://sa:PASSWORD@127.0.0.1:1433/master?driver=ODBC+Driver+17+for+SQL+Server"
    # postgre: "postgresql+psycopg2://postgres:PASSWORD@127.0.0.1:5432/akomi"
    # oracle: "oracle+oracledb://akomi:PASSWORD@localhost:1521/?service_name=XEPDB1"
    schema_dir="./history",
    rename_column=True,   # Auto-sanitize messy headers
    add_missing_cols=True # Enable Schema Evolution
)

# 2. Intelligent Load (Creates table if missing, otherwise Upserts)
etl.auto_etl(
    source="data/complex_telecom_data.csv",
    table="network_metrics",
    pk="verify"
)

# 3. Targeted Updates with Where-Build
etl.update(
    source=[{"status": "Active", "user_id": 101}],
    table="users",
    where=[("user_id", "=", 101)],
    trace_sql=True # See the magic happen
)
```

---

## Project Architecture

```mermaid
graph TD
    A[Source: CSV/Excel/DF] --> B[DataCleaner: Hidden Char Removal]
    B --> C[CastingEngine: Type Inference]
    C --> D[DataAligner: Schema Matching]
    D --> E{Action?}
    E -->|New Table| F[DDLGenerator: Dialect Specific]
    E -->|Existing| G[SchemaEvolution: Add Missing Cols]
    F --> H[DML: Bulk Insert]
    G --> H
    H --> I[Database: Oracle/Postgres/SQLite]
```

---

## Smart Dispatcher (AutoETL) Logic

The `auto_etl` method acts as a smart controller that decides the best ingestion strategy based on the current database state and data profile:

```mermaid
flowchart TD
    Start([Start auto_etl]) --> Input[Receive Source Data & Table Name]
    Input --> CheckExists{Table Exists in DB?}
    
    CheckExists -- No --> CreateLoad[Redirect to create_load]
    CreateLoad --> End([Finish])

    CheckExists -- Yes --> DetectPK[Run _load_and_sanitize_data]
    DetectPK --> PKFound{PK Columns Detected?}

    PKFound -- Yes --> Upsert[Redirect to upsert]
    Upsert --> End

    PKFound -- No / Error --> Append[Redirect to append]
    Append --> End

    style CreateLoad fill:#f9f,stroke:#333,stroke-width:2px
    style Upsert fill:#bbf,stroke:#333,stroke-width:2px
    style Append fill:#bfb,stroke:#333,stroke-width:2px
```

---

## Multi-Database Compatibility

ETLTelco supports the following dialects out of the box:

- **Oracle**: Optimized `MERGE` and `VARCHAR2` handling.
- **PostgreSQL**: Native `JSONB` support and BigInt optimization.
- **SQLite**: Robust Primary Key handling for serverless environments.
- **SQL Server (MSSQL)**: T-SQL specific DDL and identifier quoting.
- **MySQL/MariaDB**: Performance-tuned bulk operations.

---

## Documentation

For a deep dive into every method and configuration argument, please refer to:
[Methods API Reference](methods.md)

---

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

