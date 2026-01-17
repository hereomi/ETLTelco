```python
from datetime import date
import pandas as pd
from sqlalchemy import create_engine, text
from ddl import create_ddl

records = [
    {"id": 1, "amount": 100.5, "description": "alpha", "event_date": date(2025, 1, 1)},
    {"id": 2, "amount": 200.75, "description": "beta", "event_date": date(2025, 1, 2)},
]
df = pd.DataFrame(records)

engines = {
    "oracle": create_engine("oracle+cx_oracle://USER:PASSWORD@host/service"),
    "postgresql": create_engine("postgresql+psycopg://USER:PASSWORD@host/db"),
    "mysql": create_engine("mysql+pymysql://USER:PASSWORD@host/db"),
    "sqlite": create_engine("sqlite:///local.db"),
    "mssql": create_engine("mssql+pyodbc://USER:PASSWORD@host/db?driver=ODBC Driver 18 for SQL Server"),
}
```

```python
for dialect, engine in engines.items():
    processed_df, create_sql, constraint_sql, schema, sqlalchemy_schema = create_ddl.df_ddl(
        engine,
        df,
        f"etl_data_{dialect}",
        schema="analytics",
        include_not_null=True,
        object_sample_size=2000,
        json_text_threshold=0.9,
    )
    with engine.begin() as conn:
        conn.execute(text(create_sql))
        for stmt in constraint_sql:
            conn.execute(text(stmt))
```

```python
example_engine = engines["postgresql"]
processed_df, _, _, _, sqlalchemy_schema = create_ddl.df_ddl(
    example_engine,
    df,
    "etl_data_postgresql",
    schema="analytics",
    include_not_null=True,
)
with example_engine.begin() as conn:
    processed_df.to_sql(
        "etl_data_postgresql",
        conn,
        schema="analytics",
        if_exists="append",
        index=False,
        dtype=sqlalchemy_schema,
        method="multi",
    )
```
