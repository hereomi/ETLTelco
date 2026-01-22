from typing import List, Dict, Any, Union, Tuple
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.dialects import sqlite

from ..common import _normalize_data, _chunk_iter, _ensure_connection, _get_table, DataLike, _write_sql_to_file
from ..core import _ensure_data_columns_in_table, _validate_constrain_unique, _execute_with_row_isolation, _collect_not_null_violations

def _sqlite_upsert_stmt(table: sa.Table, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> sa.sql.dml.Insert:
    """Build SQLite INSERT .. ON CONFLICT .. DO UPDATE."""
    ins = sqlite.insert(table)
    update_cols = {c: ins.excluded[c] for c in sample_row.keys() if c not in key_cols}
    return ins.on_conflict_do_update(index_elements=list(key_cols), set_=update_cols)


def sqlite_upsert(engine: Engine, data: DataLike, table: Union[str, sa.Table], constrain: List[str], chunk: int = 10_000, tolerance: int = 5, trace_sql: bool = False, strict: bool = True) -> Dict[str, Any]:
    rows = _normalize_data(data)
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none", "diagnostics": {"mode": "strict" if strict else "non_strict", "fallback_used": False, "row_errors": []}}
    stats = {"total": len(rows), "success": 0, "failed": 0, "chunks": [], "diagnostics": {"mode": "strict" if strict else "non_strict"}}
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        try:
            key_cols = _validate_constrain_unique(conn, tbl, constrain)
            stats["diagnostics"]["constraints_validated"] = True
        except Exception as e:
            stats["diagnostics"]["constraint_error"] = str(e)
            if strict:
                raise
            key_cols = constrain
        nn_issues = _collect_not_null_violations(rows, tbl)
        if nn_issues:
            stats["diagnostics"]["not_null_violations"] = nn_issues
            if strict:
                raise ValueError(f"NOT NULL violation for rows: {nn_issues}")
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            stmt = _sqlite_upsert_stmt(tbl, key_cols, part[0])
            if trace_sql:
                _write_sql_to_file("sqlite_upsert", tbl.name, stmt, engine)
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            stmt = _sqlite_upsert_stmt(tbl, key_cols, row)
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk):
            chunk_stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance, strict)
            stats["chunks"].append(chunk_stats)
            stats["success"] += chunk_stats["success"]
            stats["failed"] += chunk_stats["failed"]
            if chunk_stats.get("diagnostics"):
                stats["diagnostics"].setdefault("chunks", []).append(chunk_stats["diagnostics"])
    return stats


def sqlite_insert(engine: Engine, data: DataLike, table: Union[str, sa.Table], chunk_size: int = 10_000, tolerance: int = 5, trace_sql: bool = False, strict: bool = True) -> Dict[str, Any]:
    rows = _normalize_data(data)
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none", "diagnostics": {"mode": "strict" if strict else "non_strict", "fallback_used": False, "row_errors": []}}
    stats = {"total": len(rows), "success": 0, "failed": 0, "chunks": [], "diagnostics": {"mode": "strict" if strict else "non_strict"}}
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        nn_issues = _collect_not_null_violations(rows, tbl)
        if nn_issues:
            stats["diagnostics"]["not_null_violations"] = nn_issues
            if strict:
                raise ValueError(f"NOT NULL violation for rows: {nn_issues}")
        stmt = tbl.insert()
        if trace_sql:
            _write_sql_to_file("sqlite_insert", tbl.name, stmt, engine)
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk_size):
            chunk_stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance, strict)
            stats["chunks"].append(chunk_stats)
            stats["success"] += chunk_stats["success"]
            stats["failed"] += chunk_stats["failed"]
            if chunk_stats.get("diagnostics"):
                stats["diagnostics"].setdefault("chunks", []).append(chunk_stats["diagnostics"])
    return stats
