from __future__ import annotations

from typing import List, Dict, Any, Tuple, Callable
import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector

from .common import _ensure_connection

def _ensure_data_columns_in_table(rows: List[Dict[str, Any]], table: sa.Table) -> None:
    """Ensure all keys in rows exist in table columns; raise if not."""
    table_cols = {c.name for c in table.columns}
    invalid = set()
    for r in rows:
        invalid.update(k for k in r.keys() if k not in table_cols)
    if invalid:
        raise ValueError(f"Columns not found in table '{table.name}': {sorted(invalid)}")


def _reflect_unique_sets(inspector: Inspector, table: sa.Table) -> List[Tuple[str, ...]]:
    """Return list of unique/PK column name tuples for table."""
    pk = inspector.get_pk_constraint(table.name)
    pk_cols = tuple(pk.get("constrained_columns") or [])
    unique_sets: List[Tuple[str, ...]] = []
    if pk_cols:
        unique_sets.append(pk_cols)
    for uq in inspector.get_unique_constraints(table.name):
        cols = tuple(uq.get("column_names") or [])
        if cols:
            unique_sets.append(cols)
    return unique_sets


def _validate_constrain_unique(conn: Connection, table: sa.Table, constrain: List[str]) -> Tuple[str, ...]:
    """
    Validate that `constrain` corresponds to a known PK/unique constraint if possible.
    """
    # Case-insensitive column matching
    cols_map = {c.name.lower(): c.name for c in table.columns}
    resolved_constrain = []
    for c in constrain:
        if c.lower() not in cols_map:
            raise ValueError(f"constrain column '{c}' not found in table '{table.name}'. Available: {list(cols_map.values())}")
        resolved_constrain.append(cols_map[c.lower()])
    
    if not resolved_constrain:
        raise ValueError("constrain must contain at least one column name")
    
    inspector = sa.inspect(conn)
    try:
        unique_sets = _reflect_unique_sets(inspector, table)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Constraint reflection failed: {e}. Trusting caller.")
        return tuple(resolved_constrain)
    
    # Case-insensitive comparison for unique sets
    cset = tuple(sorted(c.lower() for c in resolved_constrain))
    for u in unique_sets:
        if tuple(sorted(c.lower() for c in u)) == cset:
            return tuple(resolved_constrain)
            
    # Not strictly unique according to catalog; warn via exception
    raise ValueError(
        f"constrain={constrain} does not match any primary/unique key on '{table.name}'. "
        f"Found unique sets: {unique_sets}. "
        "Upsert requires a unique constraint to be safe."
    )


def _execute_with_row_isolation(
    conn: Connection,
    rows: List[Dict[str, Any]],
    bulk_exec: Callable[[List[Dict[str, Any]]], None],
    row_exec: Callable[[Dict[str, Any]], None],
    tolerance: int,
) -> Dict[str, Any]:
    """
    Execute in two phases: try bulk; if fails, retry row-by-row with attribution.
    Returns execution statistics.
    """
    if not rows:
        return {"total": 0, "success": 0, "failed": 0, "method": "none"}
    
    try:
        bulk_exec(rows)
        return {"total": len(rows), "success": len(rows), "failed": 0, "method": "bulk"}
    except Exception as bulk_err:
        # Fallback to lazy
        success_count = 0
        bad_rows: List[Tuple[int, Dict[str, Any], Exception]] = []
        failures_in_row = 0
        idx = -1
        
        for idx, r in enumerate(rows):
            try:
                row_exec(r)
                success_count += 1
            except Exception as row_err:
                bad_rows.append((idx, r, row_err))
                failures_in_row += 1
                if failures_in_row >= tolerance:
                    break
        
        stats = {
            "total": len(rows),
            "success": success_count,
            "failed": len(bad_rows),
            "method": "lazy_fallback",
            "bulk_error": str(bulk_err)
        }
        
        if failures_in_row >= tolerance:
            stats["aborted"] = True
            stats["unprocessed"] = len(rows) - (idx + 1)
        
        if bad_rows:
            messages = []
            for idx_inner, r, err in bad_rows:
                messages.append(
                    f"row_index={idx_inner}, row_keys={sorted(r.keys())}, error={type(err).__name__}: {err}"
                )
            error_msg = (
                "Bulk operation failed, lazy fallback partially succeeded.\n"
                f"Bulk error: {type(bulk_err).__name__}: {bulk_err}\n"
                f"Stats: {success_count}/{len(rows)} succeeded\n"
            )
            if failures_in_row >= tolerance:
                 error_msg += f"\n[!] Processing aborted after {tolerance} failures. {len(rows) - idx - 1} rows not attempted."

            error_msg += "Failing rows (first subset):\n" + "\n".join(messages)
            if success_count == 0:  # Total failure
                raise RuntimeError(error_msg) from bulk_err
            # Partial success - log warning
            import logging
            logging.getLogger(__name__).warning(error_msg)
        
        return stats
