import re
from contextlib import contextmanager
from math import ceil
from sqlalchemy import text
import pandas as pd
import numpy as np
from logger import log_call, log_json
from sql_corrector import SchemaAligner

# --- PRIVATE CORE: SQL GENERATION ---

def _prepare_insert_parts(data):
    insert_cols = ", ".join(data)
    insert_vals = ", ".join(f":{c}" for c in data)
    return insert_cols, insert_vals

def _split_key_update_cols(data, constraint):
    key_cols = [c for c in constraint if c in data]
    update_cols = [c for c in data if c not in key_cols]
    return key_cols, update_cols

def _get_upsert(dialect, table_name, data, constraint):
    key_cols, update_cols = _split_key_update_cols(data, constraint)
    params = dict(data)
    insert_cols, insert_vals = _prepare_insert_parts(data)
    if dialect == "oracle":
        on_clause = " AND ".join(f"tgt.{c} = :{c}" for c in key_cols)
        set_clause = ", ".join(f"tgt.{c} = :{c}" for c in update_cols)
        sql = f"MERGE INTO {table_name} tgt USING (SELECT 1 FROM dual) src ON ({on_clause}) WHEN MATCHED THEN UPDATE SET {set_clause} WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
    elif dialect == "mssql":
        on_clause = " AND ".join(f"tgt.{c} = :{c}" for c in key_cols)
        set_clause = ", ".join(f"tgt.{c} = :{c}" for c in update_cols)
        sql = f"MERGE {table_name} AS tgt USING (SELECT 1 AS dummy) AS src ON ({on_clause}) WHEN MATCHED THEN UPDATE SET {set_clause} WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals});"
    elif dialect == "mysql":
        set_clause = ", ".join(f"{c} = VALUES({c})" for c in update_cols)
        sql = f"INSERT INTO {table_name} ({insert_cols}) VALUES ({insert_vals}) ON DUPLICATE KEY UPDATE {set_clause}"
    elif dialect in ("postgresql", "sqlite"):
        conflict_cols = ", ".join(key_cols)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        sql = f"INSERT INTO {table_name} ({insert_cols}) VALUES ({insert_vals}) ON CONFLICT ({conflict_cols}) DO UPDATE SET {set_clause}"
    else:
        raise ValueError(f"Unsupported dialect: {dialect}")
    return sql, params

def _get_insert(table_name, data):
    insert_cols, insert_vals = _prepare_insert_parts(data)
    sql = f"INSERT INTO {table_name} ({insert_cols}) VALUES ({insert_vals})"
    return sql, dict(data)

def _expand_indexed_pattern(where, pattern):
    def repl(m):
        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(where)):
            raise IndexError(f"Condition index {idx + 1} out of range")
        return f"({where[idx]})"
    return re.sub(r'\b(\d+)\b', repl, pattern)

def _build_where_clause(where, expression=None):
    if not where:
        return "1=1", {}
    if expression is not None and re.search(r'\b\d+\b', str(expression)):
        expanded = _expand_indexed_pattern(where, expression)
        return expanded, {}
    parts, params, i = [], {}, 0
    def bind(v):
        nonlocal i
        if isinstance(v, str) and any(c in v for c in "('\")"):
            return v
        k = f"p{i}"
        i += 1
        params[k] = v
        return f":{k}"
    for cond in where:
        if isinstance(cond, str):
            parts.append(cond)
            continue
        col, op, val = cond
        op = op.upper()
        if op in ("IN", "NOT IN"):
            ph = ", ".join(bind(v) for v in val)
            parts.append(f"{col} {op} ({ph})")
        elif op in ("LIKE", "NOT LIKE"):
            parts.append(f"{col} {op} {bind(val)}")
        elif op in ("BETWEEN", "NOT BETWEEN"):
            low, high = val
            parts.append(f"{col} {op} {bind(low)} AND {bind(high)}")
        else:
            parts.append(f"{col} {op} {bind(val)}")
    join_op = f" {expression} " if isinstance(expression, str) and not re.search(r'\b\d+\b', expression) else " AND "
    where_sql = join_op.join(f"({p})" for p in parts)
    return where_sql, params

def _get_update(table, data, where, expression=None):
    set_clause = ", ".join(f"{col} = :{col}" for col in data)
    where_sql, where_params = _build_where_clause(where, expression)
    sql = f"UPDATE {table} SET {set_clause} WHERE {where_sql}"
    return sql, {**dict(data), **where_params}

# --- BULK UPSERT HELPERS ---

def _bulk_upsert_pglike(dialect, table, chunk, constraint):
    cols = list(chunk[0].keys())
    key_cols = [c for c in constraint if c in cols]
    update_cols = [c for c in cols if c not in key_cols]
    values_rows, params = [], {}
    for i, rec in enumerate(chunk):
        row_vals = []
        for col in cols:
            k = f"v_{i}_{col}"
            row_vals.append(f":{k}")
            params[k] = rec[col]
        values_rows.append(f"({', '.join(row_vals)})")
    values_sql = ", ".join(values_rows)
    insert_cols = ", ".join(cols)
    if dialect == "mysql":
        set_clause = ", ".join(f"{c} = VALUES({c})" for c in update_cols)
        sql = f"INSERT INTO {table} ({insert_cols}) VALUES {values_sql} ON DUPLICATE KEY UPDATE {set_clause}"
    else:  # postgresql, sqlite
        conflict_cols = ", ".join(key_cols)
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        sql = f"INSERT INTO {table} ({insert_cols}) VALUES {values_sql} ON CONFLICT ({conflict_cols}) DO UPDATE SET {set_clause}"
    return sql, params

def _bulk_upsert_merge(dialect, table, chunk, constraint):
    cols = list(chunk[0].keys())
    key_cols = [c for c in constraint if c in cols]
    update_cols = [c for c in cols if c not in key_cols]
    values_rows, params = [], {}
    for i, rec in enumerate(chunk):
        row_vals = []
        for col in cols:
            k = f"v_{i}_{col}"
            row_vals.append(f":{k}")
            params[k] = rec[col]
        values_rows.append(f"({', '.join(row_vals)})")
    values_sql = ", ".join(values_rows)
    col_aliases = ", ".join(cols)
    if dialect == "oracle":
        # Oracle (pre-23c) doesn't support VALUES in FROM. Use UNION ALL SELECT FROM DUAL.
        select_rows = []
        for i, rec in enumerate(chunk):
            p_parts = []
            for col in cols:
                k = f"v_{i}_{col}"
                # AS is usually optional in SELECT but good for clarity
                p_parts.append(f"BIND_VAL_FORCE_TYPING(:{k}) AS {col}" if i == 0 else f":{k}")
                params[k] = rec[col]
            
            # Note: Oracle 12c+ supports a cleaner syntax but UNION ALL is universal
            if i == 0:
                select_rows.append(f"SELECT {', '.join(p_parts)} FROM DUAL")
            else:
                select_rows.append(f"SELECT {', '.join(p_parts)} FROM DUAL")
        
        # We need to wrap it to ensure column names are preserved across the UNION
        src = f"({' UNION ALL '.join(select_rows)})"
        # Since BIND_VAL_FORCE_TYPING is pseudo-code, let's use a simpler approach:
        # Just use aliases on the first SELECT.
        rows = []
        for i, rec in enumerate(chunk):
            p_parts = []
            for col in cols:
                k = f"v_{i}_{col}"
                params[k] = rec[col]
                p_parts.append(f":{k}" + (f" AS {col}" if i == 0 else ""))
            rows.append(f"SELECT {', '.join(p_parts)} FROM DUAL")
        src = f"({' UNION ALL '.join(rows)})"

        on_clause = " AND ".join(f"tgt.{c} = src.{c}" for c in key_cols)
        set_clause = ", ".join(f"tgt.{c} = src.{c}" for c in update_cols)
        sql = (
            f"MERGE INTO {table} tgt "
            f"USING {src} src ON ({on_clause}) "
            f"WHEN MATCHED THEN UPDATE SET {set_clause} "
            f"WHEN NOT MATCHED THEN INSERT ({', '.join(cols)}) VALUES ({', '.join('src.' + c for c in cols)})"
        )
    elif dialect == "mssql":
        # MSSQL supports VALUES constructor
        values_rows = []
        for i, rec in enumerate(chunk):
            row_vals = []
            for col in cols:
                k = f"v_{i}_{col}"
                row_vals.append(f":{k}")
                params[k] = rec[col]
            values_rows.append(f"({', '.join(row_vals)})")
        values_sql = ", ".join(values_rows)
        src = f"(VALUES {values_sql}) AS src({col_aliases})"
        on_clause = " AND ".join(f"tgt.{c} = src.{c}" for c in key_cols)
        set_clause = ", ".join(f"tgt.{c} = src.{c}" for c in update_cols)
        sql = (
            f"MERGE {table} AS tgt "
            f"USING {src} AS src ON ({on_clause}) "
            f"WHEN MATCHED THEN UPDATE SET {set_clause} "
            f"WHEN NOT MATCHED THEN INSERT ({', '.join(cols)}) VALUES ({', '.join('src.' + c for c in cols)});"
        )
    else:
        raise ValueError("Only oracle/mssql use this path")
    return sql, params

# --- UTILS ---

def _normalize_data(data):
    if isinstance(data, dict):
        data = [data]
    
    if isinstance(data, pd.DataFrame):
        # Stage 1: Vectorized cleaning for speed
        # Convert all variations of NaN/NA to None
        data_clean = data.astype(object).where(pd.notnull(data), None)
        recs = data_clean.to_dict('records')
        # Stage 2: Dictionary-level sweep to force pure Python None
        # This is the "Ultra-Safe" path for Oracle/MSSQL drivers
        return [{k: (v if pd.notnull(v) else None) for k, v in r.items()} for r in recs]
        
    if isinstance(data, list):
        # Stage 2 only for already-listed data
        return [{k: (v if pd.notnull(v) else None) for k, v in r.items()} if isinstance(r, dict) else r for r in data]
        
    raise TypeError("data must be dict, list of dict, or pandas DataFrame")

def _chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

@contextmanager
def get_connection(engine):
    conn = engine.connect()
    trans = conn.begin()
    try:
        yield conn
        trans.commit()
    except Exception:
        trans.rollback()
        raise
    finally:
        conn.close()

def _execute_chunk(conn, sql, params):
    return conn.execute(text(sql), params).rowcount

# --- CHUNK HANDLERS ---

def _insert_chunk(conn, table, chunk):
    try:
        if not chunk:
            return 0
        
        # Diagnostic print for the first row of a problematic load
        try:
            sample = chunk[0]
            print(f"DEBUG: First record sample (keys={list(sample.keys())[:3]}...): {sample}")
        except Exception:
            pass

        cols = list(chunk[0].keys())
        # Use standard SQLAlchemy executemany for bulk insert
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(':' + c for c in cols)})"
        return conn.execute(text(sql), chunk).rowcount
    except Exception as e:
        try:
            log_json(f"chunk_failure_{table}", {"table": table, "chunk_size": len(chunk), "error": str(e), "fallback": "row_by_row"})
        except Exception:
            pass
        total = 0
        for row in chunk:
            sql, params = _get_insert(table, row)
            total += _execute_chunk(conn, sql, params)
        return total

def _upsert_chunk_fallback(conn, dialect, table, chunk, constraint):
    total = 0
    for row in chunk:
        sql, params = _get_upsert(dialect, table, row, constraint)
        total += _execute_chunk(conn, sql, params)
    return total

def _upsert_chunk(conn, dialect, table, chunk, constraint):
    n = len(chunk)
    try:
        if dialect in ("mysql", "postgresql", "sqlite"):
            sql, params = _bulk_upsert_pglike(dialect, table, chunk, constraint)
        elif dialect in ("oracle", "mssql") and n <= 100:
            sql, params = _bulk_upsert_merge(dialect, table, chunk, constraint)
        else:
            raise Exception("Skip bulk")
        return _execute_chunk(conn, sql, params)
    except Exception:
        return _upsert_chunk_fallback(conn, dialect, table, chunk, constraint)

# --- PUBLIC API ---

@log_call
def insert(engine, table_name, data, chunksize=1000, add_missing_cols=True, failure_threshold=3, semantic_meta=None):
    data = SchemaAligner(conn = engine, failure_threshold=failure_threshold, add_missing_cols=add_missing_cols).align(data, table_name, semantic_type_meta=semantic_meta)
    records = _normalize_data(data)
    if not records:
        return 0
    total = 0
    total_chunks = ceil(len(records) / chunksize)
    with get_connection(engine) as conn:
        for chunk in _chunked(records, chunksize):
            total += _insert_chunk(conn, table_name, chunk)
    try:
        log_json(f"insert_{table_name}", {"table": table_name, "rows_inserted": total, "chunksize": chunksize, "chunks": total_chunks})
    except Exception:
        pass
    return total

@log_call
def upsert(engine, dialect, table_name, data, constraint, chunksize=1000, add_missing_cols=True, failure_threshold=3, semantic_meta=None):
    data = SchemaAligner(conn = engine, failure_threshold=failure_threshold, add_missing_cols=add_missing_cols).align(data, table_name, semantic_type_meta=semantic_meta)
    records = _normalize_data(data)
    if not records:
        return 0
    total = 0
    total_chunks = ceil(len(records) / chunksize)
    with get_connection(engine) as conn:
        for chunk in _chunked(records, chunksize):
            total += _upsert_chunk(conn, dialect, table_name, chunk, constraint)
    try:
        log_json(f"upsert_{table_name}", {"table": table_name, "dialect": dialect, "rows_upserted": total, "constraint": constraint, "chunks": total_chunks, "chunksize": chunksize})
    except Exception:
        pass
    return total

@log_call
def update(engine, table, data, where, expression=None, add_missing_cols=True, failure_threshold=3, semantic_meta=None):
    data = SchemaAligner(conn = engine, failure_threshold=failure_threshold, add_missing_cols=add_missing_cols).align(data, table, semantic_type_meta=semantic_meta)
    records = _normalize_data(data)
    if not records:
        return 0
    record = records[0]
    sql, params = _get_update(table, record, where, expression)
    with get_connection(engine) as conn:
        result = conn.execute(text(sql), params)
    try:
        log_json(f"update_{table}", {"table": table, "rows_updated": result.rowcount, "where_conditions": len(where) if where else 0, "expression": expression})
    except Exception:
        pass
    return result.rowcount
    
