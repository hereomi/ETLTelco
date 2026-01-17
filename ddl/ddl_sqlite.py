from __future__ import annotations

from typing import Any, Optional, List

import pandas as pd
from sqlalchemy import types as satypes

from .ddl_common import ColumnMeta, SEMANTIC_BINARY_OBJECT, SEMANTIC_JSON_OBJECT, SEMANTIC_MIXED_OBJECT, SEMANTIC_STRING_OBJECT, SEMANTIC_UNKNOWN_OBJECT
from .ddl_common import build_base_schema, dedupe_identifiers, inspect_object_series, normalize_df, sanitize_identifier, validate_dataframe, validate_table_name


SQLITE_MAX_IDENT = 128

SQLITE_RESERVED = {
    "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ALWAYS", "ANALYZE", "AND",
    "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN", "BY",
    "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN", "COMMIT", "CONFLICT",
    "CONSTRAINT", "CREATE", "CROSS", "CURRENT", "CURRENT_DATE", "CURRENT_TIME",
    "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT", "DEFERRABLE", "DEFERRED", "DELETE",
    "DESC", "DETACH", "DISTINCT", "DO", "DROP", "EACH", "ELSE", "END", "ESCAPE",
    "EXCEPT", "EXCLUDE", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL", "FILTER",
    "FIRST", "FOLLOWING", "FOR", "FOREIGN", "FROM", "FULL", "GENERATED", "GLOB",
    "GROUP", "GROUPS", "HAVING", "IF", "IGNORE", "IMMEDIATE", "IN", "INDEX",
    "INDEXED", "INITIALLY", "INNER", "INSERT", "INSTEAD", "INTERSECT", "INTO",
    "IS", "ISNULL", "JOIN", "KEY", "LAST", "LEFT", "LIKE", "LIMIT", "MATCH",
    "MATERIALIZED", "NATURAL", "NO", "NOT", "NOTHING", "NOTNULL", "NULL", "NULLS",
    "OF", "OFFSET", "ON", "OR", "ORDER", "OTHERS", "OUTER", "OVER", "PARTITION",
    "PLAN", "PRAGMA", "PRECEDING", "PRIMARY", "QUERY", "RAISE", "RANGE", "RECURSIVE",
    "REFERENCES", "REGEXP", "REINDEX", "RELEASE", "RENAME", "REPLACE", "RESTRICT",
    "RETURNING", "RIGHT", "ROLLBACK", "ROW", "ROWS", "SAVEPOINT", "SELECT", "SET",
    "TABLE", "TEMP", "TEMPORARY", "THEN", "TIES", "TO", "TRANSACTION", "TRIGGER",
    "UNBOUNDED", "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES", "VIEW",
    "VIRTUAL", "WHEN", "WHERE", "WINDOW", "WITH", "WITHOUT",
}


def _san(name: str) -> str:
    return sanitize_identifier(name, SQLITE_MAX_IDENT, SQLITE_RESERVED).lower()


def _nullable(s: pd.Series) -> bool:
    return bool(s.isna().any())


def _map_type(df: pd.DataFrame, col: str, options: dict[str, Any]) -> tuple[str, str, list[str]]:
    s = df[col]
    warnings: list[str] = []
    semantic = SEMANTIC_UNKNOWN_OBJECT

    sample_size = int(options.get("object_sample_size", 5000))
    json_threshold = float(options.get("json_text_threshold", 0.85))

    if pd.api.types.is_bool_dtype(s.dtype):
        return "integer", semantic, warnings
    if pd.api.types.is_integer_dtype(s.dtype):
        return "integer", semantic, warnings
    if pd.api.types.is_float_dtype(s.dtype):
        return "real", semantic, warnings
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "text", semantic, ["Datetime mapped to TEXT (ISO) unless adapter supports datetime"]
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        warnings.append("Timedelta mapped to REAL (seconds)")
        return "real", semantic, warnings
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        ins = inspect_object_series(s, sample_size, json_threshold)
        semantic = ins.semantic_type
        warnings.extend(ins.warnings)
        if semantic == SEMANTIC_BINARY_OBJECT:
            return "blob", semantic, warnings
        if semantic == SEMANTIC_JSON_OBJECT:
            return "text", semantic, ["JSON stored as TEXT"]
        if semantic == SEMANTIC_STRING_OBJECT:
            return "text", semantic, warnings
        warnings.append("Ambiguous object values; TEXT fallback")
        return "text", semantic, warnings
    warnings.append(f"Unmapped dtype '{s.dtype}'; TEXT fallback")
    return "text", semantic, warnings


def _create_table_sql(table_name: str, schema_name: str, cols: list[ColumnMeta], include_not_null: bool, pk_cols: Optional[List[str]] = None) -> str:
    t = _san(table_name)
    lines: list[str] = []
    for c in cols:
        nn = " NOT NULL" if include_not_null and not c.nullable else ""
        lines.append(f"    {c.sanitized_name} {c.sql_type}{nn}")
    
    if pk_cols:
        san_pk = [_san(c) for c in pk_cols]
        lines.append(f"    PRIMARY KEY ({', '.join(san_pk)})")
        
    body = ",\n".join(lines)
    return f"CREATE TABLE {t} (\n{body}\n)"


def _sa_type(sql_t: str) -> Any:
    t = sql_t.lower()
    if t == "integer":
        return satypes.Integer()
    if t == "real":
        return satypes.Float()
    if t == "text":
        return satypes.Text()
    if t == "blob":
        return satypes.LargeBinary()
    return satypes.Text()


def generate_ddl(df: pd.DataFrame, table_name: str, options: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    validate_dataframe(df)
    validate_table_name(table_name)

    schema_name = str(options.get("schema", "") or "")
    include_not_null = bool(options.get("include_not_null", False))
    pk_columns = options.get("pk")
    if isinstance(pk_columns, str):
        pk_columns = [pk_columns]

    original_cols = [str(c) for c in df.columns.tolist()]
    sanitized_base = [_san(c) for c in original_cols]
    sanitized_cols = dedupe_identifiers(sanitized_base, SQLITE_MAX_IDENT)

    warnings: list[str] = []
    for o, s in zip(original_cols, sanitized_cols):
        if o != s:
            warnings.append(f"Column renamed '{o}' -> '{s}' for SQLite compatibility")

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
        if semantic == SEMANTIC_JSON_OBJECT:
            json_cols.add(s)
        if sql_t == "blob":
            binary_cols.add(s)
        if pd.api.types.is_timedelta64_dtype(df[o].dtype):
            timedelta_cols.add(s)

    processed_df, norm_warnings = normalize_df(df, rename_map, json_cols, binary_cols, timedelta_cols)
    warnings.extend(norm_warnings)

    create_sql = _create_table_sql(table_name, schema_name, columns, include_not_null, pk_columns)

    decisions = {
        "schema": schema_name or None,
        "include_not_null": include_not_null,
        "object_sample_size": int(options.get("object_sample_size", 5000)),
        "json_text_threshold": float(options.get("json_text_threshold", 0.85)),
        "pk": pk_columns
    }

    constraint_sql: list[str] = []
    sqlalchemy_schema = {c.sanitized_name: _sa_type(c.sql_type) for c in columns}
    base_schema = build_base_schema("sqlite", _san(table_name), "", columns, warnings, decisions, create_sql)
    return processed_df, create_sql, constraint_sql, base_schema, sqlalchemy_schema