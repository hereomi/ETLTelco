"""
SQL Query Generator for Multiple Database Dialects

This module provides pure SQL generation without database engine dependencies.
Supports PostgreSQL, Oracle, SQLite, MySQL, and MSSQL.

Usage:
    from evocsv.sql_generator import lazy_insert_sql, lazy_upsert_sql
    
    data = [{"id": 1, "name": "Alice", "city": "Paris"}]
    queries = lazy_insert_sql(data, "users", "postgresql")
    for q in queries:
        print(q.sql, q.params)
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Mapping, Sequence

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

__all__ = [
    'SQLQuery',
    'lazy_insert_sql',
    'lazy_upsert_sql',
    'lazy_select_sql',
    'lazy_update_sql',
    '_normalize_data',
]

DataItem = Dict[str, Any]
DataLike = Any  # To avoid complex Union issues with pd.DataFrame
ALLOWED_OPERATORS = {'=', '!=', '<', '>', '<=', '>=', 'LIKE', 'IN', 'IS', 'IS NOT'}


class SQLQuery:
    """Container for SQL query and parameters."""
    def __init__(self, sql: str, params: Optional[Dict[str, Any]] = None):
        self.sql = sql
        self.params = params or {}
    
    def __str__(self):
        return self.sql
    
    def __repr__(self):
        return f"SQLQuery(sql={self.sql!r}, params={self.params!r})"


def _normalize_data(data: DataLike) -> List[Dict[str, Any]]:
    """
    Normalize input data with strict null handling.
    
    Converts DataFrame, dict, or list to list of dicts with proper NULL handling.
    """
    if pd is not None and isinstance(data, pd.DataFrame):
        data_clean = data.astype(object).where(pd.notnull(data), None)
        records = data_clean.to_dict(orient="records")
    elif isinstance(data, Mapping):
        records = [dict(data)]
    elif isinstance(data, Sequence):
        records = [dict(row) for row in data]
    else:
        raise TypeError("Unsupported data type; expected dict, list[dict], or DataFrame")
        
    # Clean pandas/numpy artifacts
    for r in records:
        for k, v in r.items():
            if hasattr(v, 'to_pydatetime'):
                r[k] = v.to_pydatetime()
            elif pd is not None and (v is pd.NaT or pd.isna(v)):
                r[k] = None
            elif np is not None and isinstance(v, (float, np.floating)) and np.isnan(v):
                r[k] = None
                
    return records


# ============================================================================
# Oracle SQL Generation
# ============================================================================

def oracle_upsert_sql(table_name: str, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate Oracle MERGE statement."""
    src_cols = list(sample_row.keys())
    select_parts = [f":{c} AS {c}" for c in src_cols]
    src_sql = f"SELECT {', '.join(select_parts)} FROM DUAL"
    
    on_parts = [f"tgt.{c} = src.{c}" for c in key_cols]
    on_sql = " AND ".join(on_parts)
    
    key_cols_lower = {k.lower() for k in key_cols}
    update_cols = [c for c in src_cols if c.lower() not in key_cols_lower]
    
    merge_sql = f"MERGE INTO {table_name} tgt USING ({src_sql}) src ON ({on_sql})"
    
    if update_cols:
        update_sql = ", ".join(f"tgt.{c} = src.{c}" for c in update_cols)
        merge_sql += f" WHEN MATCHED THEN UPDATE SET {update_sql}"
    
    insert_cols_sql = ", ".join(src_cols)
    insert_vals_sql = ", ".join(f"src.{c}" for c in src_cols)
    merge_sql += f" WHEN NOT MATCHED THEN INSERT ({insert_cols_sql}) VALUES ({insert_vals_sql})"
    
    return SQLQuery(merge_sql, sample_row)


def oracle_insert_sql(table_name: str, sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate Oracle INSERT statement."""
    cols = list(sample_row.keys())
    placeholders = [f":{c}" for c in cols]
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    return SQLQuery(sql, sample_row)


# ============================================================================
# PostgreSQL SQL Generation
# ============================================================================

def postgres_upsert_sql(table_name: str, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate PostgreSQL INSERT ON CONFLICT statement."""
    cols = list(sample_row.keys())
    placeholders = [f"%({c})s" for c in cols]
    
    key_cols_lower = {k.lower() for k in key_cols}
    update_cols = [c for c in cols if c.lower() not in key_cols_lower]
    
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    conflict_cols = ', '.join(key_cols)
    if update_cols:
        sql += f" ON CONFLICT ({conflict_cols}) DO UPDATE SET "
        sql += ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    else:
        sql += f" ON CONFLICT ({conflict_cols}) DO NOTHING"
    
    return SQLQuery(sql, sample_row)


def postgres_insert_sql(table_name: str, sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate PostgreSQL INSERT statement."""
    cols = list(sample_row.keys())
    placeholders = [f"%({c})s" for c in cols]
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    return SQLQuery(sql, sample_row)


# ============================================================================
# SQLite SQL Generation
# ============================================================================

def sqlite_upsert_sql(table_name: str, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate SQLite INSERT ON CONFLICT statement."""
    cols = list(sample_row.keys())
    placeholders = [f":{c}" for c in cols]
    
    key_cols_lower = {k.lower() for k in key_cols}
    update_cols = [c for c in cols if c.lower() not in key_cols_lower]
    
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    conflict_cols = ', '.join(key_cols)
    if update_cols:
        sql += f" ON CONFLICT ({conflict_cols}) DO UPDATE SET "
        sql += ", ".join(f"{c} = excluded.{c}" for c in update_cols)
    else:
        sql += f" ON CONFLICT ({conflict_cols}) DO NOTHING"
    
    return SQLQuery(sql, sample_row)


def sqlite_insert_sql(table_name: str, sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate SQLite INSERT statement."""
    cols = list(sample_row.keys())
    placeholders = [f":{c}" for c in cols]
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    return SQLQuery(sql, sample_row)


# ============================================================================
# MySQL SQL Generation
# ============================================================================

def mysql_upsert_sql(table_name: str, sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate MySQL INSERT ON DUPLICATE KEY UPDATE statement."""
    cols = list(sample_row.keys())
    placeholders = [f"%({c})s" for c in cols]
    
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    sql += " ON DUPLICATE KEY UPDATE "
    sql += ", ".join(f"{c} = VALUES({c})" for c in cols)
    
    return SQLQuery(sql, sample_row)


def mysql_insert_sql(table_name: str, sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate MySQL INSERT statement."""
    cols = list(sample_row.keys())
    placeholders = [f"%({c})s" for c in cols]
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    return SQLQuery(sql, sample_row)


# ============================================================================
# MSSQL SQL Generation
# ============================================================================

def mssql_upsert_sql(table_name: str, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate MSSQL MERGE statement."""
    cols = list(sample_row.keys())
    
    # Source values
    src_values = ", ".join(f"@{c}" for c in cols)
    src_cols = ", ".join(cols)
    
    # ON clause
    on_parts = [f"tgt.{c} = src.{c}" for c in key_cols]
    on_sql = " AND ".join(on_parts)
    
    # Update clause
    key_cols_lower = {k.lower() for k in key_cols}
    update_cols = [c for c in cols if c.lower() not in key_cols_lower]
    
    sql = f"MERGE {table_name} AS tgt USING (SELECT {src_values}) AS src ({src_cols}) ON ({on_sql})"
    
    if update_cols:
        update_sql = ", ".join(f"tgt.{c} = src.{c}" for c in update_cols)
        sql += f" WHEN MATCHED THEN UPDATE SET {update_sql}"
    
    sql += f" WHEN NOT MATCHED THEN INSERT ({src_cols}) VALUES ({', '.join(f'src.{c}' for c in cols)});"
    
    # Convert params to @param format
    params = {f"@{k}": v for k, v in sample_row.items()}
    return SQLQuery(sql, params)


def mssql_insert_sql(table_name: str, sample_row: Dict[str, Any]) -> SQLQuery:
    """Generate MSSQL INSERT statement."""
    cols = list(sample_row.keys())
    placeholders = [f"@{c}" for c in cols]
    sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(placeholders)})"
    params = {f"@{k}": v for k, v in sample_row.items()}
    return SQLQuery(sql, params)


# ============================================================================
# Select and Update SQL Generation
# ============================================================================

def select_sql(table_name: str, where_conditions: List[Tuple[str, str, Any]], dialect: str = "postgresql", columns: Optional[List[str]] = None) -> SQLQuery:
    """Generate SELECT statement with WHERE clause."""
    params, where_parts = {}, []
    cols_sql = '*' if not columns else ', '.join(columns)
    
    for i, (col, op, val) in enumerate(where_conditions):
        if op.upper() not in ALLOWED_OPERATORS:
            raise ValueError(f"Invalid operator: {op}")
        param_name = f"w_{i}"
        prefix = '%' if dialect in ["postgresql", "mysql"] else ':' if dialect in ["oracle", "sqlite"] else '@'
        fmt = f"{prefix}({param_name})s" if prefix == '%' else f"{prefix}{param_name}"
        where_parts.append(f"{col} {op} {fmt}")
        params[param_name] = val
    
    sql = f"SELECT {cols_sql} FROM {table_name}"
    if where_parts:
        sql += f" WHERE {' AND '.join(where_parts)}"
    return SQLQuery(sql, params)


def update_sql(table_name: str, data: Dict[str, Any], where_conditions: List[Tuple[str, str, Any]], dialect: str = "postgresql") -> SQLQuery:
    """Generate UPDATE statement with WHERE clause."""
    if not data:
        raise ValueError("No data provided for update")
    set_parts, params, where_parts = [], {}, []
    prefix = '%' if dialect in ["postgresql", "mysql"] else ':' if dialect in ["oracle", "sqlite"] else '@'

    for i, (col, val) in enumerate(data.items()):
        p_name = f"s_{i}"
        fmt = f"{prefix}({p_name})s" if prefix == '%' else f"{prefix}{p_name}"
        set_parts.append(f"{col} = {fmt}")
        params[p_name] = val
    
    for i, (col, op, val) in enumerate(where_conditions):
        if op.upper() not in ALLOWED_OPERATORS:
            raise ValueError(f"Invalid operator: {op}")
        p_name = f"w_{i}"
        fmt = f"{prefix}({p_name})s" if prefix == '%' else f"{prefix}{p_name}"
        where_parts.append(f"{col} {op} {fmt}")
        params[p_name] = val
    
    sql = f"UPDATE {table_name} SET {', '.join(set_parts)}"
    if where_parts:
        sql += f" WHERE {' AND '.join(where_parts)}"
    return SQLQuery(sql, params)


# ============================================================================
# High-Level API Functions
# ============================================================================

def lazy_upsert_sql(data: DataLike, table: str, constrain: List[str], dialect: str = "postgresql", already_normalized: bool = False) -> List[SQLQuery]:
    """
    Generate upsert SQL without engine or validation.
    
    Args:
        data: DataFrame, dict, or list of dicts
        table: Table name
        constrain: Primary key or unique constraint columns
        dialect: Database dialect (postgresql, oracle, mysql, sqlite, mssql)
        already_normalized: If True, skip normalization step
    
    Returns:
        List of SQLQuery objects
    """
    rows = data if already_normalized else _normalize_data(data)
    if not isinstance(rows, list) or not rows:
        return []
    
    key_cols = tuple(constrain)
    queries = []
    
    for row in rows:
        if dialect == "postgresql":
            queries.append(postgres_upsert_sql(table, key_cols, row))
        elif dialect == "oracle":
            queries.append(oracle_upsert_sql(table, key_cols, row))
        elif dialect == "mysql":
            queries.append(mysql_upsert_sql(table, row))
        elif dialect == "sqlite":
            queries.append(sqlite_upsert_sql(table, key_cols, row))
        elif dialect in ["mssql", "sqlserver"]:
            queries.append(mssql_upsert_sql(table, key_cols, row))
        else:
            raise ValueError(f"Unsupported dialect: {dialect}")
    
    return queries


def lazy_insert_sql(data: DataLike, table: str, dialect: str = "postgresql", already_normalized: bool = False) -> List[SQLQuery]:
    """
    Generate insert SQL without engine or validation.
    
    Args:
        data: DataFrame, dict, or list of dicts
        table: Table name
        dialect: Database dialect (postgresql, oracle, mysql, sqlite, mssql)
        already_normalized: If True, skip normalization step
    
    Returns:
        List of SQLQuery objects
    """
    rows = data if already_normalized else _normalize_data(data)
    if not isinstance(rows, list) or not rows:
        return []
    
    queries = []
    for row in rows:
        if dialect == "postgresql":
            queries.append(postgres_insert_sql(table, row))
        elif dialect == "oracle":
            queries.append(oracle_insert_sql(table, row))
        elif dialect == "mysql":
            queries.append(mysql_insert_sql(table, row))
        elif dialect == "sqlite":
            queries.append(sqlite_insert_sql(table, row))
        elif dialect in ["mssql", "sqlserver"]:
            queries.append(mssql_insert_sql(table, row))
        else:
            raise ValueError(f"Unsupported dialect: {dialect}")
    
    return queries


def lazy_select_sql(where: List[Tuple[str, str, Any]], table: str, dialect: str = "postgresql", columns: Optional[List[str]] = None) -> SQLQuery:
    """
    Generate select SQL without engine or validation.
    
    Args:
        where: List of (column, operator, value) tuples
        table: Table name
        dialect: Database dialect
        columns: Optional list of columns to select (default: *)
    
    Returns:
        SQLQuery object
    """
    return select_sql(table, where, dialect, columns)


def lazy_update_sql(data: DataLike, table: str, where: List[Tuple[str, str, Any]], dialect: str = "postgresql", constrain: Optional[List[str]] = None, already_normalized: bool = False) -> SQLQuery:
    """
    Generate update SQL without engine or validation.
    
    Args:
        data: DataFrame, dict, or list of dicts (only first row used)
        table: Table name
        where: List of (column, operator, value) tuples for WHERE clause
        dialect: Database dialect
        constrain: Optional list of columns to update (if None, update all)
        already_normalized: If True, skip normalization step
    
    Returns:
        SQLQuery object
    """
    rows = data if already_normalized else _normalize_data(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError("No data provided for update")
    
    row = rows[0]  # normalize_data always returns a list of dicts
    
    # If constrain provided, filter update data to only those columns
    if constrain:
        constrain_lower = {c.lower() for c in constrain}
        row = {k: v for k, v in row.items() if str(k).lower() in constrain_lower}
        if not row:
            raise ValueError(f"No update columns match constrain: {constrain}")
    
    return update_sql(table, row, where, dialect)
