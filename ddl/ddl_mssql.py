from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import types as satypes

from .ddl_common import ColumnMeta, SEMANTIC_BINARY_OBJECT, SEMANTIC_JSON_OBJECT, SEMANTIC_MIXED_OBJECT, SEMANTIC_STRING_OBJECT, SEMANTIC_UNKNOWN_OBJECT
from .ddl_common import build_base_schema, dedupe_identifiers, inspect_object_series, normalize_df, sanitize_identifier, validate_dataframe, validate_table_name


MSSQL_MAX_IDENT = 128

MSSQL_RESERVED = {
    "ADD", "ALL", "ALTER", "AND", "ANY", "AS", "ASC", "AUTHORIZATION", "BACKUP",
    "BEGIN", "BETWEEN", "BREAK", "BROWSE", "BULK", "BY", "CASCADE", "CASE", "CHECK",
    "CHECKPOINT", "CLOSE", "CLUSTERED", "COALESCE", "COLLATE", "COLUMN", "COMMIT",
    "COMPUTE", "CONSTRAINT", "CONTAINS", "CONTAINSTABLE", "CONTINUE", "CONVERT",
    "CREATE", "CROSS", "CURRENT", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
    "CURRENT_USER", "CURSOR", "DATABASE", "DBCC", "DEALLOCATE", "DECLARE", "DEFAULT",
    "DELETE", "DENY", "DESC", "DISK", "DISTINCT", "DISTRIBUTED", "DOUBLE", "DROP",
    "DUMP", "ELSE", "END", "ERRLVL", "ESCAPE", "EXCEPT", "EXEC", "EXECUTE", "EXISTS",
    "EXIT", "EXTERNAL", "FETCH", "FILE", "FILLFACTOR", "FOR", "FOREIGN", "FREETEXT",
    "FREETEXTTABLE", "FROM", "FULL", "FUNCTION", "GOTO", "GRANT", "GROUP", "HAVING",
    "HOLDLOCK", "IDENTITY", "IDENTITY_INSERT", "IDENTITYCOL", "IF", "IN", "INDEX",
    "INNER", "INSERT", "INTERSECT", "INTO", "IS", "JOIN", "KEY", "KILL", "LEFT",
    "LIKE", "LINENO", "LOAD", "MERGE", "NATIONAL", "NOCHECK", "NONCLUSTERED", "NOT",
    "NULL", "NULLIF", "OF", "OFF", "OFFSETS", "ON", "OPEN", "OPENDATASOURCE",
    "OPENQUERY", "OPENROWSET", "OPENXML", "OPTION", "OR", "ORDER", "OUTER", "OVER",
    "PERCENT", "PIVOT", "PLAN", "PRECISION", "PRIMARY", "PRINT", "PROC", "PROCEDURE",
    "PUBLIC", "RAISERROR", "READ", "READTEXT", "RECONFIGURE", "REFERENCES", "REPLICATION",
    "RESTORE", "RESTRICT", "RETURN", "REVERT", "REVOKE", "RIGHT", "ROLLBACK", "ROWCOUNT",
    "ROWGUIDCOL", "RULE", "SAVE", "SCHEMA", "SECURITYAUDIT", "SELECT", "SEMANTICKEYPHRASETABLE",
    "SEMANTICSIMILARITYDETAILSTABLE", "SEMANTICSIMILARITYTABLE", "SESSION_USER", "SET",
    "SETUSER", "SHUTDOWN", "SOME", "STATISTICS", "SYSTEM_USER", "TABLE", "TABLESAMPLE",
    "TEXTSIZE", "THEN", "TO", "TOP", "TRAN", "TRANSACTION", "TRIGGER", "TRUNCATE",
    "TRY_CONVERT", "TSEQUAL", "UNION", "UNIQUE", "UNPIVOT", "UPDATE", "UPDATETEXT",
    "USE", "USER", "VALUES", "VARYING", "VIEW", "WAITFOR", "WHEN", "WHERE", "WHILE",
    "WITH", "WRITETEXT",
}


def _san(name: str) -> str:
    return sanitize_identifier(name, MSSQL_MAX_IDENT, MSSQL_RESERVED)


def _nullable(s: pd.Series) -> bool:
    return bool(s.isna().any())


def _nvarchar_or_ntext(max_len: int) -> str:
    if max_len <= 0:
        return "nvarchar(1)"
    if max_len <= 4000:
        return f"nvarchar({max_len})"
    return "nvarchar(max)"


def _map_type(df: pd.DataFrame, col: str, options: dict[str, Any]) -> tuple[str, str, list[str]]:
    s = df[col]
    warnings: list[str] = []
    semantic = SEMANTIC_UNKNOWN_OBJECT

    sample_size = int(options.get("object_sample_size", 5000))
    json_threshold = float(options.get("json_text_threshold", 0.85))

    if pd.api.types.is_bool_dtype(s.dtype):
        return "bit", semantic, warnings
    if pd.api.types.is_integer_dtype(s.dtype):
        return "bigint", semantic, warnings
    if pd.api.types.is_float_dtype(s.dtype):
        return "float", semantic, warnings
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "datetime2", semantic, warnings
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        warnings.append("Timedelta mapped to float (seconds)")
        return "float", semantic, warnings
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        ins = inspect_object_series(s, sample_size, json_threshold)
        semantic = ins.semantic_type
        warnings.extend(ins.warnings)
        if semantic == SEMANTIC_BINARY_OBJECT:
            return "varbinary(max)", semantic, warnings
        if semantic == SEMANTIC_JSON_OBJECT:
            return "nvarchar(max)", semantic, ["JSON stored as NVARCHAR(MAX)"]
        if semantic == SEMANTIC_STRING_OBJECT:
            return _nvarchar_or_ntext(ins.max_str_len), semantic, warnings
        warnings.append("Ambiguous object values; NVARCHAR(MAX) fallback")
        return "nvarchar(max)", semantic, warnings
    warnings.append(f"Unmapped dtype '{s.dtype}'; NVARCHAR(MAX) fallback")
    return "nvarchar(max)", semantic, warnings


def _create_table_sql(table_name: str, schema_name: str, cols: list[ColumnMeta], include_not_null: bool) -> str:
    t = _san(table_name)
    sch = _san(schema_name) if schema_name else ""
    fq = f"{sch}.{t}" if sch else t
    lines: list[str] = []
    for c in cols:
        nn = " NOT NULL" if include_not_null and not c.nullable else ""
        lines.append(f"    [{c.sanitized_name}] {c.sql_type}{nn}")
    body = ",\n".join(lines)
    return f"CREATE TABLE {fq} (\n{body}\n)"


def _sa_type(sql_t: str) -> Any:
    t = sql_t.lower()
    if t == "bit":
        return satypes.Boolean()
    if t == "bigint":
        return satypes.BigInteger()
    if t == "float":
        return satypes.Float()
    if t.startswith("nvarchar("):
        try:
            l = int(t[9:-1])
        except Exception:
            l = None
        return satypes.Unicode(length=l)
    if t in {"nvarchar(max)", "ntext"}:
        return satypes.UnicodeText()
    if t == "datetime2":
        return satypes.DateTime()
    if t.startswith("varbinary"):
        return satypes.LargeBinary()
    return satypes.UnicodeText()


def generate_ddl(df: pd.DataFrame, table_name: str, options: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    validate_dataframe(df)
    validate_table_name(table_name)

    schema_name = str(options.get("schema", "") or "")
    include_not_null = bool(options.get("include_not_null", False))

    original_cols = [str(c) for c in df.columns.tolist()]
    sanitized_base = [_san(c) for c in original_cols]
    sanitized_cols = dedupe_identifiers(sanitized_base, MSSQL_MAX_IDENT)

    warnings: list[str] = []
    for o, s in zip(original_cols, sanitized_cols):
        if o != s:
            warnings.append(f"Column renamed '{o}' -> '{s}' for SQL Server compatibility")

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
        if sql_t.startswith("varbinary"):
            binary_cols.add(s)
        if pd.api.types.is_timedelta64_dtype(df[o].dtype):
            timedelta_cols.add(s)

    processed_df, norm_warnings = normalize_df(df, rename_map, json_cols, binary_cols, timedelta_cols)
    warnings.extend(norm_warnings)

    create_sql = _create_table_sql(table_name, schema_name, columns, include_not_null)

    decisions = {
        "schema": schema_name or None,
        "include_not_null": include_not_null,
        "object_sample_size": int(options.get("object_sample_size", 5000)),
        "json_text_threshold": float(options.get("json_text_threshold", 0.85)),
    }

    constraint_sql: list[str] = []
    sqlalchemy_schema = {c.sanitized_name: _sa_type(c.sql_type) for c in columns}
    base_schema = build_base_schema("mssql", _san(table_name), _san(schema_name) if schema_name else "", columns, warnings, decisions, create_sql)
    return processed_df, create_sql, constraint_sql, base_schema, sqlalchemy_schema