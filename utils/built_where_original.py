"""
Adds native Oracle support (dialect='oracle').
Only change → accept new keyword and make LIKE ESCAPE clause Oracle-compatible.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import pandas as pd

_WD = r"[A-Za-z_][\w$]*"
_OP = r"BETWEEN|IN|LIKE|<=|>=|!=|=|>|<"
_rx_plain = re.compile(rf"^(?P<field>{_WD})\s*(?P<operator>{_OP})\s*(?P<values>.+)$", re.I)


def _escape_like(val: str, dialect: str) -> Tuple[str, str]:
    """Escape %, _ and \ so LIKE never over-matches; Oracle & PG need explicit ESCAPE."""
    esc = val.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")
    needs_escape = dialect in ('postgres', 'oracle', 'mysql', 'mssql')
    return esc, (" ESCAPE '\\'" if needs_escape else '')


def _process_values(op: str, raw: str) -> List[str] | str:
    op = op.lower()
    if op == 'in':
        import csv
        import io
        if not (raw.startswith('(') and raw.endswith(')')):
            raise ValueError("IN expects parentheses e.g. col IN ('a','b')")
        inner = raw[1:-1].strip()
        if not inner:
            return []
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
    return raw.strip().strip("'").strip('"')  # no numeric coercion


def parse_sql_condition(cond: str) -> Dict[str, Any]:
    m = _rx_plain.match(cond.strip())
    if not m:
        raise ValueError("Invalid SQL condition format")
    field, op, values = m.group('field'), m.group('operator').upper(), m.group('values')
    return {'field': field, 'operator': op, 'value': _process_values(op, values)}


def _param_name(base: str, c_id: int, idx: int | None = None) -> str:
    return f"{base}_{c_id}" if idx is None else f"{base}_{c_id}_{idx}"


def _build_condition(cd: Dict[str, Any], params: Dict[str, Any], dialect: str) -> Tuple[str, Dict[str, Any]]:
    op, fld, val, cid = cd['operator'], cd['field'], cd['value'], cd['id']
    if op == 'BETWEEN':
        p1, p2 = _param_name(fld, cid, 0), _param_name(fld, cid, 1)
        params[p1], params[p2] = val[0], val[1]
        return f"{fld} BETWEEN :{p1} AND :{p2}", params
    if op == 'IN':
        ph = []
        for idx, v in enumerate(val):
            pn = _param_name(fld, cid, idx)
            ph.append(f":{pn}")
            params[pn] = v
        return f"{fld} IN ({', '.join(ph)})", params
    if op == 'LIKE':
        lit, esc = _escape_like(val, dialect)
        pn = _param_name(fld, cid)
        params[pn] = f"%{lit}%"
        return f"{fld} LIKE :{pn}{esc}", params
    if op in ('=', '!=', '>', '>=', '<', '<='):
        pn = _param_name(fld, cid)
        params[pn] = val
        return f"{fld} {op} :{pn}", params
    raise ValueError(f"Unsupported operator: {op}")


def _handle_single(condition: Any, idx: int):
    if isinstance(condition, str):
        d = parse_sql_condition(condition)
        d['id'] = idx
        return d
    if isinstance(condition, dict):
        condition['id'] = idx
        return condition
    if isinstance(condition, tuple):
        fld, op, v = condition
        return {'field': fld, 'operator': op, 'value': v, 'id': idx}
    if isinstance(condition, list):
        return [_handle_single(c, idx) for c in condition]
    raise TypeError("Unsupported condition type")


def _flatten(conds: List[Any]):
    res = []
    for i, c in enumerate(conds, 1):
        norm = _handle_single(c, i)
        res.extend(norm) if isinstance(norm, list) else res.append(norm)
    return res


def _single_list_query(conds: list, dialect: str):
    parsed = _flatten(conds)
    q_map, params = {}, {}
    for cd in parsed:
        q, params = _build_condition(cd, params, dialect)
        q_map[cd['id']] = q
    return q_map, params


def sql_where(conditions, expression: str | None = None, dialect: str = 'sqlite') -> Tuple[str, Dict[str, Any]]:
    """Return (SQL, params) for sqlite | postgres | oracle | mysql | mssql."""
    if dialect not in ('sqlite', 'postgres', 'oracle', 'mysql', 'mssql'):
        raise ValueError("dialect must be 'sqlite', 'postgres', 'oracle', 'mysql' or 'mssql'")
    q_map, params = _single_list_query(conditions, dialect)
    if not expression:
        return " AND ".join(q_map.values()), params

    def _repl(m):
        idx = int(m.group(0))
        if idx not in q_map:
            raise ValueError(f"Unknown condition {idx} in expression")
        return q_map[idx]

    return re.sub(r"\b\d+\b", _repl, expression), params

# ─────────────────────────────── query build ────────────────────────────────

_rx_qmark = re.compile(rf"^\s*(?P<field>{_WD})\s*(?P<op>{_OP})\s*\?\s*$", re.I)
_WD = r"[A-Za-z_][\w$]*"
_OP = r"BETWEEN|IN|LIKE|<=|>=|!=|=|>|<"
_rx_where = re.compile(                                  # generic “fld op value” parser
    rf"^(?P<field>{_WD})\s*(?P<op>{_OP})\s*(?P<val>.+)$",
    re.I
)


def _ensure_df(data) -> pd.DataFrame:  # noqa: ANN001
    """Coerce list/dict/df into DataFrame with stable column order."""
    if isinstance(data, pd.DataFrame): return data.copy()
    if isinstance(data, list): return pd.DataFrame(data)
    if isinstance(data, dict): return pd.DataFrame([data])
    raise TypeError("data must be DataFrame / list[dict] / dict")

# Aliases for backward compatibility
parse_plain_sql = parse_sql_condition

def single_list_query(conds: list, dialect: str = 'sqlite'):
    q_map, params = _single_list_query(conds, dialect)
    return {'alchemy_query': q_map, 'alchemy_params': params}

def multi_list_query(*args, **kwargs):
    raise NotImplementedError("multi_list_query is not implemented in this version")

def _parse_where_item(item, df_cols: set[str]) -> Dict[str, Any] | None:
    """Convert one string/tuple template into sql_where-ready dict; return None if skipped."""
    if isinstance(item, tuple):
        field, op, raw_val = item
    else:  # string path
        m = _rx_where.match(item.strip())
        if not m: raise ValueError(f"Invalid where clause: {item}")
        field, op, raw_val = m.group('field'), m.group('op'), m.group('val')

    if field not in df_cols:                            # skip unknown columns
        return None

    raw_val = raw_val.strip()
    value = '?' if raw_val == '?' else raw_val.strip("'").strip('"')
    return {'field': field, 'operator': op.upper(), 'value': value}


def _row_conditions(row: pd.Series, where: List[Any]) -> List[Dict[str, Any]]:
    """Build per-row condition list, replacing ? with row values or using static."""
    conds: List[Dict[str, Any]] = []
    df_cols = set(row.index)
    for w in where:
        d = _parse_where_item(w, df_cols)
        if not d: continue                               # template skipped
        if d['value'] == '?': d['value'] = row[d['field']]
        conds.append(d)
    return conds


# ────────────────────────────── stmt builders ───────────────────────────────
def _select_stmt(row: pd.Series, table: str, where: List[Any], expr: str, dialect: str):
    sql, param = sql_where(_row_conditions(row, where), expr, dialect)
    return f"SELECT 1 FROM {table} WHERE {sql}", param


def _update_stmt(row: pd.Series, table: str, where: List[Any], expr: str, dialect: str):
    sql_w, param_w = sql_where(_row_conditions(row, where), expr, dialect)
    set_cols = [f"{c} = :u_{c}" for c in row.index]
    param_u = {f"u_{c}": row[c] for c in row.index} | param_w
    return f"UPDATE {table} SET {', '.join(set_cols)} WHERE {sql_w}", param_u


def _insert_stmt(row: pd.Series, table: str):
    cols = list(row.index)
    ph = [f":i_{c}" for c in cols]
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(ph)})", {f"i_{c}": row[c] for c in cols}


# ───────────────────────────── generators API ───────────────────────────────
def select_gen(df: pd.DataFrame, table: str, dialect: str, where: List[Any], expr: str):
    for _, r in df.iterrows(): yield _select_stmt(r, table, where, expr, dialect)


def update_gen(df: pd.DataFrame, table: str, dialect: str, where: List[Any], expr: str):
    for _, r in df.iterrows(): yield _update_stmt(r, table, where, expr, dialect)


def insert_gen(df: pd.DataFrame, table: str):
    for _, r in df.iterrows(): yield _insert_stmt(r, table)


# ───────────────────────────── master convenience ───────────────────────────
def upsert_custom(
    data, table: str, dialect: str,
    where: List[Any], expression: str = '1 AND 2'
) -> pd.DataFrame:
    """Return DataFrame with select/update/insert sql+params per row."""
    df = _ensure_df(data)
    sel_q, sel_p = zip(*select_gen(df, table, dialect, where, expression))
    upd_q, upd_p = zip(*update_gen(df, table, dialect, where, expression))
    ins_q, ins_p = zip(*insert_gen(df, table))
    return pd.DataFrame({
        'select_query': sel_q, 'select_params': sel_p,
        'update_query': upd_q, 'update_params': upd_p,
        'insert_query': ins_q, 'insert_params': ins_p
    })


# ────────────────────────────────── demo ────────────────────────────────────
if __name__ == "__main__":
    # sample data
    rows = [
        {'sitecode': '001', 'eventdate': '2023-08-01', 'kpi': 95.2},
        {'sitecode': '002', 'eventdate': '2023-08-02', 'kpi': 97.5},
    ]
    # hybrid WHERE list – tuple, placeholder, static literal
    where_tpl = [
        ("sitecode", "=", "?"),          # per-row bind
        "eventdate > '2023-01-01'",      # static literal
        "unknown_col = ?"                # silently ignored (not in df)
    ]
    df_q = upsert_custom(rows, "network_kpi", "postgres", where_tpl, expression="1 AND 2")
    print(df_q.head(1).to_string(index=False))
    
    # ──────────────── demo ────────────────
    conds = [
        "cell_id = '00123'",
        ("vendor", "IN", ["GP", "BL"]),
        {"field": "tech", "operator": "LIKE", "value": "2G"},
        "dt BETWEEN '2023-01-01' and '2023-01-31'"
    ]
    sql, binds = sql_where(conds, "1 AND (2 OR 3) AND 4", dialect="oracle")
    print(sql)
    print(binds)

