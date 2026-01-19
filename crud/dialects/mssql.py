from typing import List, Dict, Any, Union, Tuple
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.dialects import mssql

from ..common import _normalize_data, _chunk_iter, _ensure_connection, _get_table, DataLike, _write_sql_to_file
from ..core import _ensure_data_columns_in_table, _validate_constrain_unique, _execute_with_row_isolation

# Handle missing MSSQL Merge support in older SQLAlchemy versions
if hasattr(mssql, 'dml'):
    Merge = mssql.dml.Merge
else:
    Merge = None

def _mssql_merge_statement(table: sa.Table, key_cols: Tuple[str, ...], sample_row: Dict[str, Any]) -> Any:
    if Merge is None:
        raise NotImplementedError("MSSQL Merge (upsert) not supported in this SQLAlchemy version.")
        
    src_cols = list(sample_row.keys())
    src_alias = sa.alias(
        sa.select(*[sa.bindparam(c, key=None, unique=True) for c in src_cols]).subquery(),
        name="src",
    )
    merge = Merge(table, src_alias)
    on_expr = sa.and_(*[table.c[c] == src_alias.c[c] for c in key_cols])
    merge = merge.on(on_expr)
    update_cols = {c: src_alias.c[c] for c in src_cols if c not in key_cols}
    if update_cols:
        merge = merge.when_matched_then_update(set_=update_cols)
    insert_cols = {c: src_alias.c[c] for c in src_cols}
    merge = merge.when_not_matched_then_insert(values=insert_cols)
    return merge


def mssql_upsert(engine: Engine, data: DataLike, table: Union[str, sa.Table], constrain: List[str], chunk: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> Dict[str, Any]:
    """MS SQL Server upsert using MERGE with row-level fallback."""
    if Merge is None:
        raise NotImplementedError("MSSQL Merge not supported.")

    rows = _normalize_data(data)
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none"}
    stats = {"total": len(rows), "success": 0, "failed": 0, "chunks": []}
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        key_cols = _validate_constrain_unique(conn, tbl, constrain)
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            stmt = _mssql_merge_statement(tbl, key_cols, part[0])
            if trace_sql:
                _write_sql_to_file("mssql_upsert", tbl.name, stmt, engine)
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            stmt = _mssql_merge_statement(tbl, key_cols, row)
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk):
            chunk_stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            stats["chunks"].append(chunk_stats)
            stats["success"] += chunk_stats["success"]
            stats["failed"] += chunk_stats["failed"]
    return stats


def mssql_insert(engine: Engine, data: DataLike, table: Union[str, sa.Table], chunk_size: int = 10_000, tolerance: int = 5, trace_sql: bool = False) -> int:
    """MS SQL Server bulk insert with row-level fallback."""
    rows = _normalize_data(data)
    if not rows:
        return 0
    total_success = 0
    with _ensure_connection(engine) as conn:
        tbl = _get_table(conn, table)
        _ensure_data_columns_in_table(rows, tbl)
        stmt = tbl.insert()
        if trace_sql:
            _write_sql_to_file("mssql_insert", tbl.name, stmt, engine)
        def exec_chunk(part: List[Dict[str, Any]]) -> None:
            conn.execute(stmt, part)
        def exec_row(row: Dict[str, Any]) -> None:
            conn.execute(stmt, [row])
        for part in _chunk_iter(rows, chunk_size):
            stats = _execute_with_row_isolation(conn, part, exec_chunk, exec_row, tolerance)
            total_success += stats["success"]
    return total_success
