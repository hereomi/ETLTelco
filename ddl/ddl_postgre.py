from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import types as satypes

from .ddl_common import ColumnMeta, SEMANTIC_BINARY_OBJECT, SEMANTIC_JSON_OBJECT, SEMANTIC_MIXED_OBJECT, SEMANTIC_STRING_OBJECT, SEMANTIC_UNKNOWN_OBJECT
from .ddl_common import build_base_schema, dedupe_identifiers, inspect_object_series, normalize_df, sanitize_identifier, validate_dataframe, validate_table_name


POSTGRE_MAX_IDENT = 63

POSTGRE_RESERVED = {
    "ALL", "ANALYSE", "ANALYZE", "AND", "ANY", "ARRAY", "AS", "ASC", "ASYMMETRIC",
    "AUTHORIZATION", "BINARY", "BOTH", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN",
    "CONSTRAINT", "CREATE", "CURRENT_CATALOG", "CURRENT_DATE", "CURRENT_ROLE", "CURRENT_TIME",
    "CURRENT_TIMESTAMP", "CURRENT_USER", "DEFAULT", "DEFERRABLE", "DESC", "DISTINCT", "DO",
    "ELSE", "END", "EXCEPT", "FALSE", "FETCH", "FOR", "FOREIGN", "FROM", "GRANT", "GROUP",
    "HAVING", "IN", "INITIALLY", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN", "LEADING",
    "LEFT", "LIKE", "LIMIT", "LOCALTIME", "LOCALTIMESTAMP", "NATURAL", "NOT", "NOTNULL",
    "NULL", "OFFSET", "ON", "ONLY", "OR", "ORDER", "PLACING", "PRIMARY", "REFERENCES",
    "RETURNING", "RIGHT", "SELECT", "SESSION_USER", "SIMILAR", "SOME", "SYMMETRIC",
    "TABLE", "THEN", "TO", "TRAILING", "TRUE", "UNION", "UNIQUE", "USER", "USING",
    "VARIADIC", "VERBOSE", "WHEN", "WHERE", "WINDOW", "WITH",
}


def _san(name: str) -> str:
    return sanitize_identifier(name, POSTGRE_MAX_IDENT, POSTGRE_RESERVED).lower()


def _nullable(s: pd.Series) -> bool:
    return bool(s.isna().any())


def _varchar_or_text(max_len: int, limit: int) -> str:
    if max_len <= 0:
        return "varchar(1)"
    if max_len <= limit:
        return f"varchar({max_len})"
    return "text"


def _map_type(df: pd.DataFrame, col: str, options: dict[str, Any]) -> tuple[str, str, list[str]]:
    s = df[col]
    warnings: list[str] = []
    semantic = SEMANTIC_UNKNOWN_OBJECT

    varchar_limit = int(options.get("varchar_limit", 10485760))
    sample_size = int(options.get("object_sample_size", 5000))
    json_threshold = float(options.get("json_text_threshold", 0.85))

    if pd.api.types.is_bool_dtype(s.dtype):
        return "boolean", semantic, warnings
    if pd.api.types.is_integer_dtype(s.dtype):
        return "bigint", semantic, warnings
    if pd.api.types.is_float_dtype(s.dtype):
        return "double precision", semantic, warnings
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "timestamp", semantic, warnings
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        warnings.append("Timedelta mapped to double precision (seconds)")
        return "double precision", semantic, warnings
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        ins = inspect_object_series(s, sample_size, json_threshold)
        semantic = ins.semantic_type
        warnings.extend(ins.warnings)
        if semantic == SEMANTIC_BINARY_OBJECT:
            return "bytea", semantic, warnings
        if semantic == SEMANTIC_JSON_OBJECT:
            return "jsonb", semantic, warnings
        if semantic == SEMANTIC_STRING_OBJECT:
            return _varchar_or_text(ins.max_str_len, varchar_limit), semantic, warnings
        warnings.append("Ambiguous object values; text fallback")
        return "text", semantic, warnings
    warnings.append(f"Unmapped dtype '{s.dtype}'; text fallback")
    return "text", semantic, warnings


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
    if t == "boolean":
        return satypes.Boolean()
    if t == "bigint":
        return satypes.BigInteger()
    if t == "double precision":
        return satypes.Float()
    if t == "timestamp":
        return satypes.DateTime()
    if t.startswith("varchar("):
        try:
            l = int(t[8:-1])
        except Exception:
            l = None
        return satypes.String(length=l)
    if t == "text":
        return satypes.Text()
    if t == "json" or t == "jsonb":
        return satypes.JSON()
    if t == "bytea":
        return satypes.LargeBinary()
    return satypes.Text()


def generate_ddl(df: pd.DataFrame, table_name: str, options: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    validate_dataframe(df)
    validate_table_name(table_name)

    schema_name = str(options.get("schema", "") or "")
    include_not_null = bool(options.get("include_not_null", False))

    original_cols = [str(c) for c in df.columns.tolist()]
    sanitized_base = [_san(c) for c in original_cols]
    sanitized_cols = dedupe_identifiers(sanitized_base, POSTGRE_MAX_IDENT)

    warnings: list[str] = []
    for o, s in zip(original_cols, sanitized_cols):
        if o != s:
            warnings.append(f"Column renamed '{o}' -> '{s}' for PostgreSQL compatibility")

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
        if sql_t in {"jsonb", "json"}:
            json_cols.add(s)
        if sql_t == "bytea":
            binary_cols.add(s)
        if pd.api.types.is_timedelta64_dtype(df[o].dtype):
            timedelta_cols.add(s)

    processed_df, norm_warnings = normalize_df(df, rename_map, json_cols, binary_cols, timedelta_cols)
    warnings.extend(norm_warnings)

    create_sql = _create_table_sql(table_name, schema_name, columns, include_not_null)

    decisions = {
        "schema": schema_name or None,
        "include_not_null": include_not_null,
        "varchar_limit": int(options.get("varchar_limit", 10485760)),
        "object_sample_size": int(options.get("object_sample_size", 5000)),
        "json_text_threshold": float(options.get("json_text_threshold", 0.85)),
    }

    base_schema = build_base_schema("postgre", _san(table_name), _san(schema_name) if schema_name else "", columns, warnings, decisions, create_sql)
    
    constraint_sql: list[str] = []
    pk_columns = options.get("pk")
    if pk_columns:
        if isinstance(pk_columns, str):
            pk_columns = [pk_columns]
        
        san_pk = [_san(c) for c in pk_columns]
        san_set = {c.sanitized_name for c in columns}
        
        from .ddl_func import build_pk_constraint
        pk_sql, pk_meta = build_pk_constraint(
            "postgresql", table_name, schema_name, pk_columns, _san, san_set, options.get("pk_constraint_name")
        )
        constraint_sql.append(pk_sql)
        base_schema["constraints"]["primary_key"] = pk_meta

    sqlalchemy_schema = {c.sanitized_name: _sa_type(c.sql_type) for c in columns}
    return processed_df, create_sql, constraint_sql, base_schema, sqlalchemy_schema