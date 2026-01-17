from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import types as satypes

from .ddl_common import ColumnMeta, SEMANTIC_BINARY_OBJECT, SEMANTIC_JSON_OBJECT, SEMANTIC_MIXED_OBJECT, SEMANTIC_STRING_OBJECT, SEMANTIC_UNKNOWN_OBJECT
from .ddl_common import build_base_schema, dedupe_identifiers, inspect_object_series, normalize_df, sanitize_identifier, validate_dataframe, validate_table_name


MYSQL_MAX_IDENT = 64

MYSQL_RESERVED = {
    "ADD", "ALL", "ALTER", "ANALYZE", "AND", "AS", "ASC", "ASENSITIVE", "BEFORE",
    "BETWEEN", "BIGINT", "BINARY", "BLOB", "BOTH", "BY", "CALL", "CASCADE", "CASE",
    "CHANGE", "CHAR", "CHECK", "COLLATE", "COLUMN", "CONDITION", "CONSTRAINT", "CONTINUE",
    "CONVERT", "CREATE", "CROSS", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
    "CURRENT_USER", "CURSOR", "DATABASE", "DATABASES", "DAY_HOUR", "DAY_MICROSECOND",
    "DAY_MINUTE", "DAY_SECOND", "DEC", "DECIMAL", "DECLARE", "DEFAULT", "DELAYED",
    "DELETE", "DESC", "DESCRIBE", "DETERMINISTIC", "DISTINCT", "DISTINCTROW", "DIV",
    "DOUBLE", "DROP", "DUAL", "EACH", "ELSE", "ELSEIF", "ENCLOSED", "ESCAPED", "EXISTS",
    "EXIT", "EXPLAIN", "FALSE", "FETCH", "FLOAT", "FOR", "FORCE", "FOREIGN", "FROM",
    "FULLTEXT", "GENERATED", "GRANT", "GROUP", "HAVING", "HIGH_PRIORITY", "HOUR_MICROSECOND",
    "HOUR_MINUTE", "HOUR_SECOND", "IF", "IGNORE", "IN", "INDEX", "INFILE", "INNER",
    "INOUT", "INSENSITIVE", "INSERT", "INT", "INTEGER", "INTERVAL", "INTO", "IS",
    "ITERATE", "JOIN", "KEY", "KEYS", "KILL", "LEADING", "LEAVE", "LEFT", "LIKE",
    "LIMIT", "LINEAR", "LINES", "LOAD", "LOCALTIME", "LOCALTIMESTAMP", "LOCK", "LONG",
    "LONGBLOB", "LONGTEXT", "LOOP", "LOW_PRIORITY", "MASTER_BIND", "MASTER_SSL_VERIFY_SERVER_CERT",
    "MATCH", "MAXVALUE", "MEDIUMBLOB", "MEDIUMINT", "MEDIUMTEXT", "MIDDLEINT", "MINUTE_MICROSECOND",
    "MINUTE_SECOND", "MOD", "MODIFIES", "NATURAL", "NOT", "NO_WRITE_TO_BINLOG", "NULL",
    "NUMERIC", "ON", "OPTIMIZE", "OPTION", "OPTIONALLY", "OR", "ORDER", "OUT", "OUTER",
    "OUTFILE", "PARTITION", "PRECISION", "PRIMARY", "PROCEDURE", "PURGE", "RANGE",
    "READ", "READS", "READ_WRITE", "REAL", "REFERENCES", "REGEXP", "RELEASE", "RENAME",
    "REPEAT", "REPLACE", "REQUIRE", "RESTRICT", "RETURN", "REVOKE", "RIGHT", "RLIKE",
    "SCHEMA", "SCHEMAS", "SECOND_MICROSECOND", "SELECT", "SENSITIVE", "SEPARATOR",
    "SET", "SHOW", "SMALLINT", "SPATIAL", "SPECIFIC", "SQL", "SQLEXCEPTION", "SQLSTATE",
    "SQLWARNING", "SQL_BIG_RESULT", "SQL_CALC_FOUND_ROWS", "SQL_SMALL_RESULT", "SSL",
    "STARTING", "STORED", "STRAIGHT_JOIN", "TABLE", "TERMINATED", "THEN", "TINYBLOB",
    "TINYINT", "TINYTEXT", "TO", "TRAILING", "TRIGGER", "TRUE", "UNDO", "UNION", "UNIQUE",
    "UNLOCK", "UNSIGNED", "UPDATE", "USAGE", "USE", "USING", "UTC_DATE", "UTC_TIME",
    "UTC_TIMESTAMP", "VALUES", "VARBINARY", "VARCHAR", "VARYING", "VIRTUAL", "WHEN",
    "WHERE", "WHILE", "WITH", "WRITE", "XOR", "YEAR_MONTH", "ZEROFILL",
}


def _san(name: str) -> str:
    return sanitize_identifier(name, MYSQL_MAX_IDENT, MYSQL_RESERVED).lower()


def _nullable(s: pd.Series) -> bool:
    return bool(s.isna().any())


def _varchar_or_text(max_len: int) -> str:
    if max_len <= 0:
        return "varchar(1)"
    if max_len <= 255:
        return f"varchar({max_len})"
    return "longtext"


def _map_type(df: pd.DataFrame, col: str, options: dict[str, Any]) -> tuple[str, str, list[str]]:
    s = df[col]
    warnings: list[str] = []
    semantic = SEMANTIC_UNKNOWN_OBJECT

    sample_size = int(options.get("object_sample_size", 5000))
    json_threshold = float(options.get("json_text_threshold", 0.85))

    if pd.api.types.is_bool_dtype(s.dtype):
        return "boolean", semantic, warnings
    if pd.api.types.is_integer_dtype(s.dtype):
        # Determine the appropriate integer type based on the range of values
        non_null = s.dropna()
        if not non_null.empty:
            min_val, max_val = int(non_null.min()), int(non_null.max())
            if min_val >= -128 and max_val <= 127:
                return "tinyint", semantic, warnings
            elif min_val >= -32768 and max_val <= 32767:
                return "smallint", semantic, warnings
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return "int", semantic, warnings
        return "bigint", semantic, warnings
    if pd.api.types.is_float_dtype(s.dtype):
        return "double", semantic, warnings
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "datetime", semantic, warnings
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        warnings.append("Timedelta mapped to bigint (microseconds)")
        return "bigint", semantic, warnings
    if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
        ins = inspect_object_series(s, sample_size, json_threshold)
        semantic = ins.semantic_type
        warnings.extend(ins.warnings)
        if semantic == SEMANTIC_BINARY_OBJECT:
            if ins.max_bin_len <= 255:
                return f"varbinary({ins.max_bin_len})", semantic, warnings
            return "longblob", semantic, warnings
        if semantic == SEMANTIC_JSON_OBJECT:
            return "json", semantic, warnings
        if semantic == SEMANTIC_STRING_OBJECT:
            return _varchar_or_text(ins.max_str_len), semantic, warnings
        warnings.append("Ambiguous object values; longtext fallback")
        return "longtext", semantic, warnings
    warnings.append(f"Unmapped dtype '{s.dtype}'; longtext fallback")
    return "longtext", semantic, warnings


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
    if t in {"tinyint", "smallint"}:
        return satypes.SmallInteger()
    if t == "int":
        return satypes.Integer()
    if t == "bigint":
        return satypes.BigInteger()
    if t == "double":
        return satypes.Float()
    if t == "datetime":
        return satypes.DateTime()
    if t == "json":
        return satypes.JSON()
    if t.startswith("varchar("):
        try:
            l = int(t[8:-1])
        except Exception:
            l = None
        return satypes.String(length=l)
    if t in {"longtext", "text"}:
        return satypes.Text()
    if t.startswith("varbinary("):
        try:
            l = int(t[10:-1])
        except Exception:
            l = None
        return satypes.LargeBinary(length=l)
    if t in {"blob", "mediumblob", "longblob"}:
        return satypes.LargeBinary()
    return satypes.Text()


def generate_ddl(df: pd.DataFrame, table_name: str, options: dict[str, Any]) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    validate_dataframe(df)
    validate_table_name(table_name)

    schema_name = str(options.get("schema", "") or "")
    include_not_null = bool(options.get("include_not_null", False))

    original_cols = [str(c) for c in df.columns.tolist()]
    sanitized_base = [_san(c) for c in original_cols]
    sanitized_cols = dedupe_identifiers(sanitized_base, MYSQL_MAX_IDENT)

    warnings: list[str] = []
    for o, s in zip(original_cols, sanitized_cols):
        if o != s:
            warnings.append(f"Column renamed '{o}' -> '{s}' for MySQL compatibility")

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
        if sql_t == "json":
            json_cols.add(s)
        if sql_t.startswith("varbinary") or sql_t in {"blob", "mediumblob", "longblob"}:
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

    base_schema = build_base_schema("mysql", _san(table_name), _san(schema_name) if schema_name else "", columns, warnings, decisions, create_sql)
    constraint_sql: list[str] = []
    sqlalchemy_schema = {c.sanitized_name: _sa_type(c.sql_type) for c in columns}
    return processed_df, create_sql, constraint_sql, base_schema, sqlalchemy_schema