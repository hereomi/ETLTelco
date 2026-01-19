from typing import List, Dict, Any, Union, Tuple
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.dialects import mysql

from ..common import _normalize_data, _chunk_iter, _ensure_connection, _get_table, DataLike, _write_sql_to_file
from ..core import _ensure_data_columns_in_table, _validate_constrain_unique, _execute_with_row_isolation

def _mysql_upsert_stmt(table: sa.Table, sample_row: Dict[str, Any]) -> sa.sql.dml.Insert:
    """
    Build MySQL INSERT .. ON DUPLICATE KEY UPDATE.
    """
    ins = mysql.insert(table)
    update_cols = {c: ins.inserted[c] for c in sample_row.keys()}
    return ins.on_duplicate_key_update(**update_cols)


def mysql_upsert(engine: Engine, data: DataLike, table: Union[str, sa.Table], constrain: List[str], chunk: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> Dict[str, Any]:
    """
    MySQL upsert using ON DUPLICATE KEY UPDATE with row-level fallback.
    """
    rows = _normalize_data(data)
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none"}
    stats = {"total": len(rows), "success": 0, "failed": 0, "chunks": []}
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        _ = constrain  # unused
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            stmt = _mysql_upsert_stmt(tbl, part[0])
            if trace_sql:
                _write_sql_to_file("mysql_upsert", tbl.name, stmt, engine)
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            stmt = _mysql_upsert_stmt(tbl, row)
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk):
            chunk_stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            stats["chunks"].append(chunk_stats)
            stats["success"] += chunk_stats["success"]
            stats["failed"] += chunk_stats["failed"]
    return stats


def mysql_insert(engine: Engine, data: DataLike, table: Union[str, sa.Table], chunk_size: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> int:
    """MySQL bulk insert with row-level fallback and error attribution."""
    rows = _normalize_data(data)
    if not rows:
        return 0
    total_success = 0
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        stmt = tbl.insert()
        if trace_sql:
            _write_sql_to_file("mysql_insert", tbl.name, stmt, engine)
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk_size):
            stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            total_success += stats["success"]
    return total_success
