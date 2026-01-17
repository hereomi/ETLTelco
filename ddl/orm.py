from __future__ import annotations

import pandas as pd
from typing import Optional, Union, List, Dict, Any
from .ddl_common import sanitize_identifier, dedupe_identifiers

def generate_sqlalchemy_model(
    df: pd.DataFrame,
    table_name: str,
    class_name: Optional[str] = None,
    sanitize: bool = True,
    pk: Optional[Union[str, List[str]]] = None,
    autoincrement: Optional[str] = None,
) -> str:
    """
    Generate a Python script defining a SQLAlchemy declarative model for the given DataFrame.
    
    Args:
        df: The pandas DataFrame.
        table_name: The name of the table in the database.
        class_name: Optional name for the Python class. Defaults to CamelCase of table_name.
        sanitize: Whether to sanitize column names.
        pk: Primary key column(s).
        autoincrement: Column name to mark as autoincrement.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    work_df = df.copy()

    # Reuse sanitization logic from ddl_common for consistency
    # Note: We use a generic set of reserved words or just rely on the fact that these are python identifiers
    # For simplicity in ORM generation, we primarily care about valid python identifiers.
    
    if sanitize:
        original_cols = [str(c) for c in work_df.columns]
        # We usually want slightly different rules for Python attributes vs SQL columns, 
        # but ddl_common.sanitize_identifier is a good baseline (alphanumeric + underscore).
        # We'll use a dummy reserved set for now or empty.
        sanitized = [sanitize_identifier(c, 255, set()) for c in original_cols]
        sanitized = dedupe_identifiers(sanitized, 255)
        
        # We need to map back to handle pk/autoincrement matching
        mapping = dict(zip(original_cols, sanitized))
        work_df.columns = sanitized
        
        if pk:
            if isinstance(pk, str):
                pk = mapping.get(pk, pk)
            else:
                pk = [mapping.get(c, c) for c in pk]
        if autoincrement:
            autoincrement = mapping.get(autoincrement, autoincrement)

    if not class_name:
        # Convert table_name (snake_case) to CamelCase
        class_name = "".join(x.title() for x in table_name.replace("-", "_").replace(" ", "_").split('_'))

    pk_cols = [pk] if isinstance(pk, str) else (pk or [])

    lines = [
        "from sqlalchemy import Column, BigInteger, Float, Boolean, String, DateTime, Date, Time, Interval, JSON, LargeBinary, Text",
        "from sqlalchemy.orm import declarative_base",
        "",
        "Base = declarative_base()",
        "",
        f"class {class_name}(Base):",
        f"    __tablename__ = '{table_name}'",
        ""
    ]

    for col in work_df.columns:
        dtype = str(work_df[col].dtype).lower()
        
        # Mapping logic similar to ddl_create.py but slightly refined
        if 'int' in dtype:
            sa_type = "BigInteger"
        elif 'float' in dtype or 'double' in dtype:
            sa_type = "Float"
        elif 'bool' in dtype:
            sa_type = "Boolean"
        elif 'datetime' in dtype:
            sa_type = "DateTime"
        elif dtype == 'date':
            sa_type = "Date"
        elif dtype == 'time':
            sa_type = "Time"
        elif 'timedelta' in dtype or 'interval' in dtype:
            sa_type = "Interval"
        else:
            # Default fallback
            sa_type = "String(255)"

        attrs = []
        if col in pk_cols:
            attrs.append("primary_key=True")
        if col == autoincrement:
            attrs.append("autoincrement=True")
        
        # If it's a PK, it's implicitly not null. Otherwise check data.
        if (col in pk_cols) or (not work_df[col].isna().any()):
            attrs.append("nullable=False")


        attr_str = ", ".join(attrs)
        if attr_str:
            lines.append(f"    {col} = Column({sa_type}, {attr_str})")
        else:
            lines.append(f"    {col} = Column({sa_type})")

    return "\n".join(lines)
