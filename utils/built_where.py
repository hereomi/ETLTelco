"""
Built WHERE clause generator with native Oracle support and enhanced SELECT/UPDATE operations.
Supports multiple database dialects and correctly handles row-level condition preparation.
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Tuple, Union, Generator, Optional
import pandas as pd
from .sanitization import escape_identifier

# Logging Configuration
DEBUG_LOG_ENABLED = True
LOG_FILE = "built_where.log"

def _log(func_name: str, msg: Any):
    if not DEBUG_LOG_ENABLED: return
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{func_name}] {msg}\n")
    except: pass

_WD = r"[A-Za-z_][\w$]*"
_OP = r"BETWEEN|IN|LIKE|<=|>=|!=|=|>|<"
_rx_plain = re.compile(rf"^(?P<field>{_WD})\s*(?P<operator>{_OP})\s*(?P<values>.+)$", re.I)
_rx_where = re.compile(rf"^(?P<field>{_WD})\s*(?P<op>{_OP})\s*(?P<val>.+)$", re.I)

def _escape_like(val: str, dialect: str) -> Tuple[str, str]:
    """Escape %, _ and \\ so LIKE never over-matches; Oracle & PG need explicit ESCAPE."""
    esc = val.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")
    needs_escape = dialect in ('postgres', 'postgresql', 'oracle', 'mysql', 'mssql')
    return esc, (" ESCAPE '\\'" if needs_escape else '')

def _process_values(op: str, raw: str) -> List[str] | str:
    op = op.lower()
    if op == 'in':
        import csv, io
        if not (raw.startswith('(') and raw.endswith(')')):
            raise ValueError("IN expects parentheses e.g. col IN ('a','b')")
        inner = raw[1:-1].strip()
        if not inner: return []
        reader = csv.reader(io.StringIO(inner), quotechar="'", skipinitialspace=True)
        try:
            return next(reader)
        except StopIteration:
            return []
    if op == 'between':
        parts = re.split(r'\band\b', raw, flags=re.I)
        if len(parts) != 2:
            raise ValueError("BETWEEN needs exactly two values")
        return [p.strip().strip("'").strip('"') for p in parts]
    return raw.strip().strip("'").strip('"')

def parse_sql_condition(cond: str) -> Dict[str, Any]:
    m = _rx_plain.match(cond.strip())
    if not m:
        raise ValueError(f"Invalid SQL condition format: {cond}")
    field, op, values = m.group('field'), m.group('operator').upper(), m.group('values')
    return {'field': field, 'operator': op, 'value': _process_values(op, values)}

def _param_name(base: str, c_id: int, idx: int | None = None) -> str:
    return f"{base}_{c_id}" if idx is None else f"{base}_{c_id}_{idx}"

def _build_condition(cd: Dict[str, Any], params: Dict[str, Any], dialect: str) -> Tuple[str, Dict[str, Any]]:
    op, fld, val, cid = cd['operator'], cd['field'], cd['value'], cd['id']
    p_fld = escape_identifier(fld, dialect)

    if op == 'BETWEEN':
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            val = [val, val] # Fallback
        p1, p2 = _param_name(fld, cid, 0), _param_name(fld, cid, 1)
        params[p1], params[p2] = val[0], val[1]
        return f"{p_fld} BETWEEN :{p1} AND :{p2}", params
    
    if op == 'IN':
        if not isinstance(val, (list, tuple)): val = [val]
        ph = []
        for idx, v in enumerate(val):
            pn = _param_name(fld, cid, idx)
            ph.append(f":{pn}")
            params[pn] = v
        return f"{p_fld} IN ({', '.join(ph)})", params
    
    if op == 'LIKE':
        lit, esc = _escape_like(str(val), dialect)
        pn = _param_name(fld, cid)
        params[pn] = f"%{lit}%"
        return f"{p_fld} LIKE :{pn}{esc}", params
    
    if op in ('=', '!=', '>', '>=', '<', '<='):
        pn = _param_name(fld, cid)
        params[pn] = val
        return f"{p_fld} {op} :{pn}", params
    
    raise ValueError(f"Unsupported operator: {op}")

def _handle_single(condition: Any, idx: int):
    if isinstance(condition, str):
        d = parse_sql_condition(condition)
        d['id'] = idx
        return d
    if isinstance(condition, dict):
        # ensure id is set
        res = dict(condition)
        res['id'] = idx
        return res
    if isinstance(condition, tuple):
        if len(condition) != 3: raise ValueError("Tuple condition must be (field, op, value)")
        fld, op, v = condition
        return {'field': fld, 'operator': op, 'value': v, 'id': idx}
    if isinstance(condition, list):
        return [_handle_single(c, idx) for c in condition]
    raise TypeError(f"Unsupported condition type: {type(condition)}")

def _flatten(conds: Any):
    # Bug fix: prevent iteration on single objects
    if isinstance(conds, (str, dict, tuple)):
        conds = [conds]
    elif not hasattr(conds, '__iter__'):
        conds = [conds]
        
    res = []
    for i, c in enumerate(conds, 1):
        norm = _handle_single(c, i)
        if isinstance(norm, list):
            res.extend(norm)
        else:
            res.append(norm)
    return res

def _single_list_query(conds: list, dialect: str):
    parsed = _flatten(conds)
    q_map, params = {}, {}
    for cd in parsed:
        q, params = _build_condition(cd, params, dialect)
        q_map[cd['id']] = q
    return q_map, params

def sql_where(conditions: Any, expression: str | None = None, dialect: str = 'sqlite') -> Tuple[str, Dict[str, Any]]:
    """Return (SQL, params) for multiple dialects."""
    _log("sql_where", f"expr={expression}, dialect={dialect}")
    if dialect not in ('sqlite', 'postgres', 'postgresql', 'oracle', 'mysql', 'mssql'):
        raise ValueError(f"Unsupported dialect: {dialect}")
    
    q_map, params = _single_list_query(conditions, dialect)
    if not expression:
        return " AND ".join(q_map.values()), params

    def _repl(m):
        idx_str = m.group(0)
        idx = int(idx_str)
        if idx not in q_map:
            return idx_str # Bug fix: return as-is for non-indices
        return q_map[idx]

    res_sql = re.sub(r"\b\d+\b", _repl, expression)
    _log("sql_where", f"result_sql={res_sql}")
    return res_sql, params

# ─────────────────────────────── query build ────────────────────────────────

def _ensure_df(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame): return data.copy()
    if isinstance(data, list): return pd.DataFrame(data)
    if isinstance(data, dict): return pd.DataFrame([data])
    raise TypeError("data must be DataFrame / list[dict] / dict")

def _parse_where_item(item: Any, df_cols: set[str]) -> Dict[str, Any] | None:
    """Convert template into sql_where dict. Handles ? placeholders."""
    if isinstance(item, tuple):
        field, op, raw_val = item
    else:
        m = _rx_where.match(str(item).strip())
        if not m: raise ValueError(f"Invalid where clause: {item}")
        field, op, raw_val = m.group('field'), m.group('op'), m.group('val')

    # Case-insensitive column matching
    df_cols_upper = {c.upper(): c for c in df_cols}
    if field.upper() not in df_cols_upper:
        # If not in data, only allow if it's NOT a placeholder
        if isinstance(raw_val, str) and raw_val.strip() == '?':
            return None
        return {'field': field, 'operator': op.upper() if isinstance(op, str) else op, 'value': raw_val}
    
    actual_field = df_cols_upper[field.upper()]
    if isinstance(raw_val, str):
        value = '?' if raw_val.strip() == '?' else raw_val.strip().strip("'").strip('"')
    else:
        value = raw_val
        
    return {'field': actual_field, 'operator': op.upper() if isinstance(op, str) else op, 'value': value}

def _row_conditions(row: pd.Series, where: List[Any]) -> List[Dict[str, Any]]:
    conds = []
    df_cols = set(row.index)
    for w in where:
        d = _parse_where_item(w, df_cols)
        if not d: continue
        if d['value'] == '?':
            d['value'] = row[d['field']]
        conds.append(d)
    return conds

# ────────────────────────────── stmt builders ───────────────────────────────

def _select_stmt(row: pd.Series, table: str, where: List[Any], expr: str, dialect: str, columns: Optional[List[str]] = None):
    sql_w, param_w = sql_where(_row_conditions(row, where), expr, dialect)
    tbl_esc = ".".join(escape_identifier(p, dialect) for p in table.split('.'))
    cols_sql = '*'
    if columns:
        cols_sql = ", ".join(escape_identifier(c, dialect) for c in columns)
    return f"SELECT {cols_sql} FROM {tbl_esc} WHERE {sql_w}", param_w

def _update_stmt(row: pd.Series, table: str, where: List[Any], expr: str, dialect: str, update_cols: Optional[List[str]] = None):
    sql_w, param_w = sql_where(_row_conditions(row, where), expr, dialect)
    tbl_esc = ".".join(escape_identifier(p, dialect) for p in table.split('.'))
    
    cols_to_use = update_cols if update_cols else list(row.index)
    set_parts = []
    param_u = {}
    for c in cols_to_use:
        c_esc = escape_identifier(str(c), dialect)
        set_parts.append(f"{c_esc} = :u_{c}")
        param_u[f"u_{c}"] = row[c]
    
    param_u.update(param_w)
    return f"UPDATE {tbl_esc} SET {', '.join(set_parts)} WHERE {sql_w}", param_u

def _insert_stmt(row: pd.Series, table: str, dialect: str = 'sqlite'):
    tbl_esc = ".".join(escape_identifier(p, dialect) for p in table.split('.'))
    cols = list(row.index)
    cols_esc = [escape_identifier(str(c), dialect) for c in cols]
    ph = [f":i_{c}" for c in cols]
    return f"INSERT INTO {tbl_esc} ({', '.join(cols_esc)}) VALUES ({', '.join(ph)})", {f"i_{c}": row[c] for c in cols}

# ───────────────────────────── generators API ───────────────────────────────

def select_gen(df: pd.DataFrame, table: str, dialect: str, where: List[Any], expr: str, columns: Optional[List[str]] = None):
    for _, r in df.iterrows(): yield _select_stmt(r, table, where, expr, dialect, columns)

def update_gen(df: pd.DataFrame, table: str, dialect: str, where: List[Any], expr: str, update_cols: Optional[List[str]] = None):
    for _, r in df.iterrows(): yield _update_stmt(r, table, where, expr, dialect, update_cols)

def insert_gen(df: pd.DataFrame, table: str, dialect: str = 'sqlite'):
    for _, r in df.iterrows(): yield _insert_stmt(r, table, dialect)

# ───────────────────────────── master convenience ───────────────────────────

def build_update(data: Any, table: str, where: list[Any], dialect: str = 'sqlite', expression: str | None = None, update_cols: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    df = _ensure_df(data)
    if len(df) != 1: raise ValueError(f"Expected 1 row, got {len(df)}")
    row = df.iloc[0]
    expr = expression or ' AND '.join(str(i) for i in range(1, len(where) + 1))
    return _update_stmt(row, table, where, expr, dialect, update_cols)

def build_select(data: Any, table: str, where: list[Any], dialect: str = 'sqlite', expression: str | None = None, columns: list[str] | None = None) -> tuple[str, dict[str, Any]]:
    df = _ensure_df(data)
    if len(df) != 1: raise ValueError(f"Expected 1 row, got {len(df)}")
    row = df.iloc[0]
    expr = expression or ' AND '.join(str(i) for i in range(1, len(where) + 1))
    return _select_stmt(row, table, where, expr, dialect, columns)

def upsert_custom(data, table: str, dialect: str, where: List[Any], expression: str = '1 AND 2') -> pd.DataFrame:
    df = _ensure_df(data)
    sel_rows = list(select_gen(df, table, dialect, where, expression))
    upd_rows = list(update_gen(df, table, dialect, where, expression))
    ins_rows = list(insert_gen(df, table, dialect))
    
    sel_q, sel_p = zip(*sel_rows) if sel_rows else ([], [])
    upd_q, upd_p = zip(*upd_rows) if upd_rows else ([], [])
    ins_q, ins_p = zip(*ins_rows) if ins_rows else ([], [])
    
    return pd.DataFrame({
        'select_query': sel_q, 'select_params': sel_p,
        'update_query': upd_q, 'update_params': upd_p,
        'insert_query': ins_q, 'insert_params': ins_p
    })

if __name__ == "__main__":
    # Test complex custom queries
    conds = [
        "cell_id = '00123'",
        ("vendor", "IN", ["GP", "BL"]),
        {"field": "tech", "operator": "LIKE", "value": "2G"},
        "dt BETWEEN '2023-01-01' and '2023-01-31'"
    ]
    sql, binds = sql_where(conds, "1 AND (2 OR 3) AND 4", dialect="oracle")
    print(sql)
    print(binds)