from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import types as satypes

from .ddl_common import ColumnMeta, SEMANTIC_BINARY_OBJECT, SEMANTIC_JSON_OBJECT, SEMANTIC_MIXED_OBJECT, SEMANTIC_STRING_OBJECT, SEMANTIC_UNKNOWN_OBJECT
from .ddl_common import build_base_schema, dedupe_identifiers, inspect_object_series, normalize_df, sanitize_identifier, validate_dataframe, validate_table_name
from .ddl_func import build_pk_constraint, build_unique_constraint, build_fk_constraint, build_sequence, build_trigger


ORACLE_MAX_IDENT = 30

ORACLE_RESERVED = {
    "ACCESS", "ADD", "ALL", "ALTER", "AND", "ANY", "AS", "ASC", "AUDIT",
    "BETWEEN", "BY", "CHAR", "CHECK", "CLUSTER", "COLUMN", "COMMENT", "COMPRESS",
    "CONNECT", "CREATE", "CURRENT", "DATE", "DECIMAL", "DEFAULT", "DELETE", "DESC",
    "DISTINCT", "DROP", "ELSE", "EXCLUSIVE", "EXISTS", "FILE", "FLOAT", "FOR",
    "FROM", "GRANT", "GROUP", "HAVING", "IDENTIFIED", "IMMEDIATE", "IN",
    "INCREMENT", "INDEX", "INITIAL", "INSERT", "INTEGER", "INTERSECT", "INTO",
    "IS", "LEVEL", "LIKE", "LOCK", "LONG", "MAXEXTENTS", "MINUS", "MLSLABEL",
    "MODE", "MODIFY", "NOAUDIT", "NOCOMPRESS", "NOT", "NOWAIT", "NULL", "NUMBER",
    "OF", "OFFLINE", "ON", "ONLINE", "OPTION", "OR", "ORDER", "PCTFREE", "PRIOR",
    "PRIVILEGES", "PUBLIC", "RAW", "RENAME", "RESOURCE", "REVOKE", "ROW", "ROWID",
    "ROWNUM", "ROWS", "SELECT", "SESSION", "SET", "SHARE", "SIZE", "SMALLINT",
    "START", "SUCCESSFUL", "SYNONYM", "SYSDATE", "TABLE", "THEN", "TO",
    "TRIGGER", "UID", "UNION", "UNIQUE", "UPDATE", "USER", "VALIDATE", "VALUES",
    "VARCHAR", "VARCHAR2", "VIEW", "WHENEVER", "WHERE", "WITH",
}


def _san(name: str) -> str:
    return sanitize_identifier(name, ORACLE_MAX_IDENT, ORACLE_RESERVED)


def _nullable(s: pd.Series) -> bool:
    return bool(s.isna().any())


def _varchar_or_clob(max_len: int, limit: int) -> str:
    if max_len <= 0:
        return "VARCHAR2(1 CHAR)"
    if max_len <= limit:
        return f"VARCHAR2({max_len} CHAR)"
    return "CLOB"


def _raw_or_blob(max_len: int, limit: int) -> str:
    if max_len <= 0:
        return "RAW(1)"
    if max_len <= limit:
        return f"RAW({max_len})"
    return "BLOB"


def _int_precision(s: pd.Series) -> tuple[int, list[str]]:
    warnings: list[str] = []
    non_null = s.dropna()
    if non_null.empty:
        return 38, ["Empty integer column defaulted to NUMBER(38,0)"]
    try:
        vals = non_null.astype("int64", errors="ignore")
        if not np.issubdtype(vals.dtype, np.integer):
            return 38, ["Non-integer values found; defaulted to NUMBER(38,0)"]
        max_abs = int(np.max(np.abs(vals.to_numpy())))
        if max_abs == 0:
            return 1, warnings
        p = int(math.floor(math.log10(max_abs))) + 1
        if p < 1:
            p = 1
        if p > 38:
            p = 38
        return p, warnings
    except Exception:
        return 38, ["Failed precision inference; defaulted to NUMBER(38,0)"]


def _map_type(df: pd.DataFrame, col: str, options: dict[str, Any]) -> tuple[str, str, list[str]]:
    s = df[col]
    warnings: list[str] = []
    semantic = SEMANTIC_UNKNOWN_OBJECT

    varchar2_limit = int(options.get("varchar2_limit", 4000))
    raw_limit = int(options.get("raw_limit", 2000))
    sample_size = int(options.get("object_sample_size", 5000))
    json_threshold = float(options.get("json_text_threshold", 0.85))
    prefer_json = bool(options.get("prefer_json_datatype", True))

    if pd.api.types.is_bool_dtype(s.dtype):
        return "NUMBER(1,0)", semantic, warnings
    if pd.api.types.is_integer_dtype(s.dtype):
        p, w = _int_precision(s)
        warnings.extend(w)
        return f"NUMBER({p},0)", semantic, warnings
    if pd.api.types.is_float_dtype(s.dtype):
        if s.dtype == np.float32:
            return "BINARY_FLOAT", semantic, warnings
        return "BINARY_DOUBLE", semantic, warnings
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "TIMESTAMP", semantic, warnings
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        warnings.append("Timedelta mapped to NUMBER (seconds)")
        return "NUMBER", semantic, warnings
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        ins = inspect_object_series(s, sample_size, json_threshold)
        semantic = ins.semantic_type
        warnings.extend(ins.warnings)
        if semantic == SEMANTIC_BINARY_OBJECT:
            return _raw_or_blob(ins.max_bin_len, raw_limit), semantic, warnings
        if semantic == SEMANTIC_JSON_OBJECT:
            if prefer_json:
                return "JSON", semantic, warnings
            return "CLOB", semantic, warnings
        if semantic == SEMANTIC_STRING_OBJECT:
            return _varchar_or_clob(ins.max_str_len, varchar2_limit), semantic, warnings
        warnings.append("Ambiguous object values; CLOB fallback")
        return "CLOB", semantic, warnings
    warnings.append(f"Unmapped dtype '{s.dtype}'; CLOB fallback")
    return "CLOB", semantic, warnings


def _create_table_sql(table_name: str, schema_name: str, cols: list[ColumnMeta], include_not_null: bool) -> str:
    t = _san(table_name)
    sch = _san(schema_name) if schema_name else ""
    fq = f"{sch}.{t}" if sch else t
    lines: list[str] = []
    for c in cols:
        nn = " NOT NULL" if include_not_null and not c.nullable else ""
        lines.append(f"    {c.sanitized_name} {c.sql_type}{nn}")
    body = ",\n".join(lines)
    return f"CREATE TABLE {fq} (\n{body}\n)"


def _sa_type(sql_t: str) -> Any:
    t = sql_t.lower()
    if t.startswith("number"):
        p = None
        s = None
        if "(" in t and ")" in t:
            body = t[t.find("(") + 1:t.find(")")]
            parts = [p.strip() for p in body.split(",")]
            if parts and parts[0].isdigit():
                p = int(parts[0])
            if len(parts) > 1 and parts[1].lstrip("-").isdigit():
                s = int(parts[1])
        return satypes.Numeric(precision=p, scale=s)
    if t == "binary_float":
        return satypes.Float(precision=24)
    if t == "binary_double":
        return satypes.Float(precision=53)
    if t == "timestamp":
        return satypes.DateTime()
    if t.startswith("varchar2("):
        l = None
        if "(" in t and ")" in t:
            body = t[t.find("(") + 1:t.find(")")]
            tok = body.split()[0]
            if tok.isdigit():
                l = int(tok)
        return satypes.String(length=l)
    if t.startswith("raw("):
        l = None
        if "(" in t and ")" in t:
            body = t[t.find("(") + 1:t.find(")")]
            if body.isdigit():
                l = int(body)
        return satypes.LargeBinary(length=l)
    if t == "blob":
        return satypes.LargeBinary()
    if t == "clob":
        return satypes.CLOB()
    if t == "json":
        return satypes.JSON()
    return satypes.Text()


def generate_ddl(df: pd.DataFrame, table_name: str, options: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    validate_dataframe(df)
    validate_table_name(table_name)

    schema_name = str(options.get("schema", "") or "")
    include_not_null = bool(options.get("include_not_null", False))

    original_cols = [str(c) for c in df.columns.tolist()]
    sanitized_base = [_san(c) for c in original_cols]
    sanitized_cols = dedupe_identifiers(sanitized_base, ORACLE_MAX_IDENT)

    warnings: list[str] = []
    for o, s in zip(original_cols, sanitized_cols):
        if o != s:
            warnings.append(f"Column renamed '{o}' -> '{s}' for Oracle compatibility")

    columns: list[ColumnMeta] = []
    rename_map: dict[str, str] = {}
    json_cols: set[str] = set()
    binary_cols: set[str] = set()
    timedelta_cols: set[str] = set()

    for o, s in zip(original_cols, sanitized_cols):
        sql_t, semantic, w = _map_type(df, o, options)
        nullable = _nullable(df[o])
        columns.append(ColumnMeta(o, s, str(df[o].dtype), sql_t, semantic, nullable, w))
        rename_map[o] = s
        if sql_t == "JSON":
            json_cols.add(s)
        if semantic == SEMANTIC_BINARY_OBJECT or sql_t.startswith("RAW") or sql_t == "BLOB":
            binary_cols.add(s)
        if pd.api.types.is_timedelta64_dtype(df[o].dtype):
            timedelta_cols.add(s)

    processed_df, norm_warnings = normalize_df(df, rename_map, json_cols, binary_cols, timedelta_cols)
    warnings.extend(norm_warnings)

    create_sql = _create_table_sql(table_name, schema_name, columns, include_not_null)

    decisions = {
        "schema": schema_name or None,
        "include_not_null": include_not_null,
        "varchar2_limit": int(options.get("varchar2_limit", 4000)),
        "raw_limit": int(options.get("raw_limit", 2000)),
        "object_sample_size": int(options.get("object_sample_size", 5000)),
        "json_text_threshold": float(options.get("json_text_threshold", 0.85)),
        "prefer_json_datatype": bool(options.get("prefer_json_datatype", True)),
    }

    if any(c.sql_type == "JSON" for c in columns) and decisions["prefer_json_datatype"]:
        warnings.append("Oracle JSON data type requires Oracle 21c+; set prefer_json_datatype=False for older versions")

    base_schema = build_base_schema("oracle", _san(table_name), _san(schema_name) if schema_name else "", columns, warnings, decisions, create_sql)
    
    constraint_sql: list[str] = []
    pk_columns = options.get("pk")
    if pk_columns:
        if isinstance(pk_columns, str):
            pk_columns = [pk_columns]
        
        san_pk = [_san(c) for c in pk_columns]
        san_set = {c.sanitized_name for c in columns}
        
        pk_sql, pk_meta = build_pk_constraint(
            "oracle", table_name, schema_name, pk_columns, _san, san_set, options.get("pk_constraint_name")
        )
        constraint_sql.append(pk_sql)
        base_schema["constraints"]["primary_key"] = pk_meta

    sqlalchemy_schema = {c.sanitized_name: _sa_type(c.sql_type) for c in columns}
    return processed_df, create_sql, constraint_sql, base_schema, sqlalchemy_schema