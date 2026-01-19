from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy.engine import Engine, Connection

try:
    import pandas as pd
except ImportError:
    pd = None

from .crud_v3 import (
    lazy_insert_sql, lazy_upsert_sql, lazy_update_sql, lazy_select_sql,
    _normalize_data, SQLQuery
)

logger = logging.getLogger(__name__)

DataLike = Union[List[Dict[str, Any]], Dict[str, Any], Any] # Actually pd.DataFrame if available


@lru_cache(maxsize=128)
def _get_table_metadata(engine: Engine, table_name: str) -> sa.Table:
    """Get table metadata with caching."""
    try: return sa.Table(table_name, sa.MetaData(), autoload_with=engine)
    except Exception as e: raise ValueError(f"Table '{table_name}' not found: {e}")


def _validate_columns(rows: List[Dict[str, Any]], table: sa.Table) -> Dict[str, str]:
    """Validate columns exist and return case mapping."""
    table_cols = {c.name.lower(): c.name for c in table.columns}
    col_mapping = {}
    invalid = set()
    
    for r in rows:
        for k in r.keys():
            if k.lower() in table_cols: col_mapping[k] = table_cols[k.lower()]
            else: invalid.add(k)
    
    if invalid: raise ValueError(f"Columns not found in table '{table.name}': {sorted(invalid)}")
    return col_mapping


def _get_unique_constraints(engine: Engine, table: sa.Table) -> List[Tuple[str, ...]]:
    """Get unique constraints with error handling."""
    try:
        inspector = sa.inspect(engine)
        pk = inspector.get_pk_constraint(table.name)
        pk_cols = tuple(pk.get("constrained_columns") or [])
        unique_sets = [pk_cols] if pk_cols else []
        
        for uq in inspector.get_unique_constraints(table.name):
            if cols := tuple(uq.get("column_names") or []): unique_sets.append(cols)
        return unique_sets
    except Exception: return []


def _validate_constraint(engine: Engine, table: sa.Table, constrain: List[str]) -> Tuple[str, ...]:
    """Validate constraint exists and return normalized column names."""
    if not constrain: raise ValueError("constrain must contain at least one column")
    
    table_cols = {c.name.lower(): c.name for c in table.columns}
    resolved = []
    
    for c in constrain:
        if c.lower() not in table_cols:
            raise ValueError(f"Constraint column '{c}' not found in table '{table.name}'")
        resolved.append(table_cols[c.lower()])
    
    unique_sets = _get_unique_constraints(engine, table)
    if unique_sets:
        resolved_set = tuple(sorted(c.lower() for c in resolved))
        for u in unique_sets:
            if tuple(sorted(c.lower() for c in u)) == resolved_set: return tuple(resolved)
        raise ValueError(f"Constraint {constrain} does not match any unique key. Available: {unique_sets}")
    
    return tuple(resolved)


def _chunk_data(data: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    """Split data into chunks."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def _execute_chunk_bulk(conn: Connection, queries: List[SQLQuery], chunk_data: List[Dict[str, Any]]) -> bool:
    """Try to execute chunk as bulk operation using executemany."""
    if not queries: return True
    try:
        first_sql = queries[0].sql
        if all(q.sql == first_sql for q in queries):
            # Use parameters from queries relative to mapped columns, not raw chunk_data
            conn.execute(sa.text(first_sql), [q.params for q in queries])
            return True
        return False
    except Exception as e:
        logger.warning(f"Bulk execution failed: {e}")
        return False


def _execute_lazy_fallback(conn: Connection, queries: List[SQLQuery], chunk_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute queries one by one with error tracking."""
    success_count, failed_count, errors = 0, 0, []
    
    for i, (query, _) in enumerate(zip(queries, chunk_data)):
        try:
            conn.execute(sa.text(query.sql), query.params)
            success_count += 1
        except Exception as e:
            failed_count += 1
            errors.append(f"Row {i}: {type(e).__name__}: {e}")
            logger.error(f"Row {i} failed: {e}")
    
    return {"success": success_count, "failed": failed_count, "errors": errors[:5]}


# Validated API Functions (With Engine Validation)
def generate_upsert_sql(engine: Engine, data: DataLike, table: str, constrain: List[str], rows: Optional[List[Dict[str, Any]]] = None) -> List[SQLQuery]:
    """Generate upsert SQL queries with validation."""
    if not (rows := rows or _normalize_data(data)): return []
    
    table_obj = _get_table_metadata(engine, table)
    col_mapping = _validate_columns(rows, table_obj)
    key_cols = _validate_constraint(engine, table_obj, constrain)
    
    normalized_rows = [{col_mapping[k]: v for k, v in row.items()} for row in rows]
    return lazy_upsert_sql(normalized_rows, table, list(key_cols), engine.dialect.name.lower(), already_normalized=True)


def generate_insert_sql(engine: Engine, data: DataLike, table: str, rows: Optional[List[Dict[str, Any]]] = None) -> List[SQLQuery]:
    """Generate insert SQL queries with validation."""
    if not (rows := rows or _normalize_data(data)): return []
    
    col_mapping = _validate_columns(rows, _get_table_metadata(engine, table))
    normalized_rows = [{col_mapping[k]: v for k, v in row.items()} for row in rows]
    return lazy_insert_sql(normalized_rows, table, engine.dialect.name.lower(), already_normalized=True)


def generate_select_sql(engine: Engine, where: List[Tuple[str, str, Any]], table: str, columns: Optional[List[str]] = None) -> SQLQuery:
    """Generate select SQL query with validation."""
    table_obj = _get_table_metadata(engine, table)
    table_cols = {c.name.lower(): c.name for c in table_obj.columns}
    
    normalized_where = []
    for col, op, val in where:
        if col.lower() in table_cols: normalized_where.append((table_cols[col.lower()], op, val))
        else: raise ValueError(f"WHERE column '{col}' not found in table '{table}'")
    
    if columns:
        missing = []
        normalized_columns: List[str] = []
        for col in columns:
            key = col.lower()
            if key in table_cols:
                normalized_columns.append(table_cols[key])
            else:
                missing.append(col)
        if missing:
            raise ValueError(f"Some SELECT columns not found in '{table}': {sorted(missing)}")
        columns = normalized_columns

    return lazy_select_sql(normalized_where, table, engine.dialect.name.lower(), columns)


def generate_update_sql(engine: Engine, data: DataLike, table: str, where: List[Tuple[str, str, Any]], constrain: Optional[List[str]] = None, rows: Optional[List[Dict[str, Any]]] = None) -> SQLQuery:
    """Generate update SQL query with validation."""
    if not (rows := rows or _normalize_data(data)): raise ValueError("No data provided for update")
    
    table_obj = _get_table_metadata(engine, table)
    col_mapping = _validate_columns(rows, table_obj)
    normalized_row = {col_mapping[k]: v for k, v in rows[0].items()}
    
    constrain_mapped = None
    if constrain:
        table_cols = {c.name.lower(): c.name for c in table_obj.columns}
        constrain_mapped = [table_cols[c.lower()] for c in constrain if c.lower() in table_cols]
        if len(constrain_mapped) != len(constrain): raise ValueError(f"Some constrain columns not found in '{table}'")
        
        normalized_row = {k: v for k, v in normalized_row.items() if k in constrain_mapped}
        if not normalized_row: raise ValueError(f"No update columns match constrain: {constrain}")

    table_cols = {c.name.lower(): c.name for c in table_obj.columns}
    normalized_where = []
    for col, op, val in where:
        if col.lower() in table_cols: normalized_where.append((table_cols[col.lower()], op, val))
        else: raise ValueError(f"WHERE column '{col}' not found in table '{table}'")
    
    return lazy_update_sql([normalized_row], table, normalized_where, engine.dialect.name.lower(), constrain_mapped, already_normalized=True)


def execute_crud_operation(
    engine: Engine, data: DataLike, table: str, operation: str = "insert",
    constrain: Optional[List[str]] = None, where: Optional[List[Tuple[str, str, Any]]] = None,
    chunk_size: int = 1000, add_missing_cols: bool = True, failure_threshold: float = 0.1
) -> Dict[str, Any]:
    """Execute CRUD operation with data alignment, chunking, and lazy fallback."""
    try:
        from schema_align.core import DataAligner
        from schema_align.config import AlignmentConfig
        config = AlignmentConfig(failure_threshold=failure_threshold)
        aligner = DataAligner(db_type=engine.dialect.name, config=config)
        if pd is not None and not (hasattr(data, 'iloc')): data = pd.DataFrame(data)
        data = aligner.align(conn=engine, df=data, table=table, add_missing_cols=add_missing_cols) # type: ignore
    except ImportError: logger.warning("schema_align not available, skipping data alignment")
    
    if not (rows := _normalize_data(data)): return {"total": 0, "success": 0, "failed": 0, "method": "empty"}
    
    if operation == "upsert" and not constrain: raise ValueError("constrain required for upsert operation")
    if operation == "update" and not where: raise ValueError("where conditions required for update operation")
    
    chunks = _chunk_data(rows, chunk_size)
    total_success, total_failed, chunk_stats = 0, 0, []
    
    with engine.begin() as conn:
        for chunk_idx, chunk in enumerate(chunks):
            try:
                queries = []
                if operation == "insert": queries = generate_insert_sql(engine, chunk, table, rows=chunk)
                elif operation == "upsert": queries = generate_upsert_sql(engine, chunk, table, constrain, rows=chunk) # type: ignore
                elif operation == "update":
                    queries = [generate_update_sql(engine, chunk[:1], table, where, constrain, rows=chunk[:1])] # type: ignore
                    chunk = chunk[:1]
                else: raise ValueError(f"Unsupported operation: {operation}")
                
                if _execute_chunk_bulk(conn, queries, chunk):
                    stats = {"chunk": chunk_idx, "total": len(chunk), "success": len(chunk), "failed": 0, "method": "bulk"}
                    total_success += len(chunk)
                else:
                    lazy = _execute_lazy_fallback(conn, queries, chunk)
                    stats = {"chunk": chunk_idx, "total": len(chunk), "success": lazy["success"], "failed": lazy["failed"], "method": "lazy", "errors": lazy["errors"]}
                    total_success += lazy["success"]
                    total_failed += lazy["failed"]
                chunk_stats.append(stats)
            except Exception as e:
                logger.error(f"Chunk {chunk_idx} completely failed: {e}")
                chunk_stats.append({"chunk": chunk_idx, "total": len(chunk), "success": 0, "failed": len(chunk), "method": "failed", "error": str(e)})
                total_failed += len(chunk)
    
    return {"operation": operation, "total": len(rows), "success": total_success, "failed": total_failed, "chunks_processed": len(chunks), "chunk_stats": chunk_stats}


# Convenience functions
def insert_data(engine: Engine, data: DataLike, table: str, **kwargs) -> Dict[str, Any]:
    """Insert data with chunking and lazy fallback."""
    return execute_crud_operation(engine, data, table, "insert", **kwargs)


def upsert_data(engine: Engine, data: DataLike, table: str, constrain: List[str], **kwargs) -> Dict[str, Any]:
    """Upsert data with chunking and lazy fallback."""
    return execute_crud_operation(engine, data, table, "upsert", constrain=constrain, **kwargs)


def update_data(engine: Engine, data: DataLike, table: str, where: List[Tuple[str, str, Any]], **kwargs) -> Dict[str, Any]:
    """Update data with chunking and lazy fallback."""
    return execute_crud_operation(engine, data, table, "update", where=where, **kwargs)


if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table
    
    engine = create_engine("sqlite:///:memory:")
    meta = MetaData()
    users = Table("users", meta,
        Column("id", Integer, primary_key=True),
        Column("username", String(50), unique=True),
        Column("city", String(50))
    )
    meta.create_all(engine)
    
    # Test data
    data = [
        {"id": 1, "username": "alice", "city": "Paris"},
        {"id": 2, "username": "bob", "city": "London"},
        {"id": 3, "username": "charlie", "city": "Berlin"}
    ]
    
    # Insert with chunking
    result = insert_data(engine, data, "users", chunk_size=2)
    print(f"Insert result: {result}")
    
    # Upsert with chunking
    upsert_data_list = [
        {"id": 1, "username": "alice", "city": "Lyon"},
        {"id": 4, "username": "david", "city": "Madrid"}
    ]
    result = upsert_data(engine, upsert_data_list, "users", ["id"], chunk_size=1)
    print(f"Upsert result: {result}")
    
    # Update with chunking
    update_data_list = [{"city": "New York"}]
    result = update_data(engine, update_data_list, "users", [("id", "=", 1)])
    print(f"Update result: {result}")
    
    # Test validated functions
    validated_insert = generate_insert_sql(engine, data[:1], "users")
    print(f"Validated insert: {validated_insert[0].sql}")
    
    validated_select = generate_select_sql(engine, [("id", "=", 1)], "users", ["username"])
    print(f"Validated select: {validated_select.sql}")