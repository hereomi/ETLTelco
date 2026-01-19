from typing import List, Dict, Any, Union, Tuple
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.dialects import postgresql

from ..common import _normalize_data, _chunk_iter, _ensure_connection, _get_table, DataLike, _write_sql_to_file
from ..core import _ensure_data_columns_in_table, _validate_constrain_unique, _execute_with_row_isolation

def _postgres_upsert_stmt(table: sa.Table, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> sa.sql.dml.Insert:
    """Build PostgreSQL INSERT .. ON CONFLICT .. DO UPDATE."""
    ins = postgresql.insert(table)
    update_cols = {c: ins.excluded[c] for c in sample_row.keys() if c not in key_cols}
    return ins.on_conflict_do_update(index_elements=list(key_cols), set_=update_cols)


def postgres_upsert(engine: Engine, data: DataLike, table: Union[str, sa.Table], constrain: List[str], chunk: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> Dict[str, Any]:
    """PostgreSQL upsert with row-level fallback and explicit conflict target validation."""
    rows = _normalize_data(data)
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none"}
    
    stats = {"total": len(rows), "success": 0, "failed": 0, "chunks": []}
    
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        key_cols = _validate_constrain_unique(conn, tbl, constrain)
        
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            stmt = _postgres_upsert_stmt(tbl, key_cols, part[0])
            if trace_sql:
                _write_sql_to_file("postgres_upsert", tbl.name, stmt, engine)
            conn.execute(stmt, part)
        
        def exec_row(row: Dict[str, Any]) -> None:
            stmt = _postgres_upsert_stmt(tbl, key_cols, row)
            conn.execute(stmt, [row])
        
        for part in _chunk_iter(rows, chunk):
            chunk_stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            stats["chunks"].append(chunk_stats)
            stats["success"] += chunk_stats["success"]
            stats["failed"] += chunk_stats["failed"]
    
    return stats


def postgres_insert(engine: Engine, data: DataLike, table: Union[str, sa.Table], chunk_size: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> int:
    """PostgreSQL bulk insert with row-level fallback and error attribution."""
    rows = _normalize_data(data)
    if not rows:
        return 0
    total_success = 0
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        stmt = tbl.insert()
        if trace_sql:
            _write_sql_to_file("postgres_insert", tbl.name, stmt, engine)
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk_size):
            stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            total_success += stats["success"]
    return total_success
