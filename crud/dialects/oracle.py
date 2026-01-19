from typing import List, Dict, Any, Union, Tuple
import sqlalchemy as sa
from sqlalchemy.engine import Engine

from ..common import _normalize_data, _chunk_iter, _ensure_connection, _get_table, DataLike, _write_sql_to_file
from ..core import _ensure_data_columns_in_table, _validate_constrain_unique, _execute_with_row_isolation

def _oracle_merge_sql(table_name: str, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> str:
    """Build Oracle MERGE statement as raw SQL text."""
    src_cols = list(sample_row.keys())
    
    # Build source SELECT from DUAL
    select_parts = [f":{c} AS {c}" for c in src_cols]
    src_sql = f"SELECT {', '.join(select_parts)} FROM DUAL"
    
    # ON clause
    on_parts = [f"tgt.{c} = src.{c}" for c in key_cols]
    on_sql = " AND ".join(on_parts)
    
    # UPDATE SET clause (non-key columns, case-insensitive comparison)
    key_cols_lower = {k.lower() for k in key_cols}
    update_cols = [c for c in src_cols if c.lower() not in key_cols_lower]
    update_sql = ", ".join(f"tgt.{c} = src.{c}" for c in update_cols) if update_cols else None
    
    # INSERT clause
    insert_cols_sql = ", ".join(src_cols)
    insert_vals_sql = ", ".join(f"src.{c}" for c in src_cols)
    
    # Build full MERGE
    merge_sql = f"MERGE INTO {table_name} tgt USING ({src_sql}) src ON ({on_sql})"
    if update_sql:
        merge_sql += f" WHEN MATCHED THEN UPDATE SET {update_sql}"
    merge_sql += f" WHEN NOT MATCHED THEN INSERT ({insert_cols_sql}) VALUES ({insert_vals_sql})"
    
    return merge_sql


def oracle_upsert(engine: Engine, data: DataLike, table: Union[str, sa.Table], constrain: List[str], chunk: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> Dict[str, Any]:
    """Oracle upsert using MERGE with row-level fallback and error attribution."""
    rows = _normalize_data(data)
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none"}
    
    stats = {"total": len(rows), "success": 0, "failed": 0, "chunks": []}
    
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        key_cols = _validate_constrain_unique(conn, tbl, constrain)
        
        # Build raw SQL MERGE statement
        merge_sql = _oracle_merge_sql(tbl.name, key_cols, rows[0])
        if trace_sql:
            _write_sql_to_file("oracle_upsert", tbl.name, merge_sql, engine)
        
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            for row in part:
                conn.execute(sa.text(merge_sql), row)
        
        def exec_row(row: Dict[str, Any]) -> None:
            conn.execute(sa.text(merge_sql), row)
        
        # NOTE: Oracle connection execute() often doesn't support list of dicts for RAW SQL with all drivers the same way textual methods do.
        # But crud_v2 implementation iterates row by row inside 'exec_chunk' anyway? 
        # Checking crud_v2 code: 
        # def exec_chunk(part): 
        #   for row in part: conn.execute(sa.text(merge_sql), row)
        # So YES, it iterates even in chunk method for Oracle currently. 
        # Ideally using executemany with text() depends on driver support (cx_Oracle supports it). 
        # I will keep exact logic from crud_v2 for safety.
        
        for part in _chunk_iter(rows, chunk):
            chunk_stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            stats["chunks"].append(chunk_stats)
            stats["success"] += chunk_stats["success"]
            stats["failed"] += chunk_stats["failed"]
    
    return stats


def oracle_insert(engine: Engine, data: DataLike, table: Union[str, sa.Table], chunk_size: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> int:
    """Oracle bulk insert with row-level fallback and error attribution."""
    rows = _normalize_data(data)
    if not rows:
        return 0
    total_success = 0
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        stmt = tbl.insert()
        if trace_sql:
            _write_sql_to_file("oracle_insert", tbl.name, stmt, engine)
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk_size):
            stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            total_success += stats["success"]
    return total_success
