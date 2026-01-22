'''
ddl, cleaned = ddl_via_pandas_internals(df, "my_table", "postgresql")
use pandas: 2.1.4 or 2.1.5
This implementation:
- Uses `create_mock_engine()` (no DB connection).
- Attempts pandas `SQLDatabase` + `SQLTable` private `_sqlalchemy_type` and `_create_table_setup`.
- Compiles a dialect-specific `CREATE TABLE` statement.
- Renames reserved-word columns to `{orig}_col` after normalization.
- Applies your per-dialect type overrides (PG JSONB, MySQL LONGTEXT/VARCHAR(255), Oracle CLOB/BLOB).
- Infers `NOT NULL` when a column has no missing values.
- Silently falls back to a manual mapping if any internal/private method changes or fails.
'''

import json
import math
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.schema import CreateTable


def make_mock_engine(dialect_name: str) -> sa.Engine:
    dialects = {"postgresql", "mysql", "oracle"}
    if dialect_name not in dialects:
        raise ValueError(f"Unsupported dialect_name: {dialect_name}")
    url = f"{dialect_name}://"
    engine = sa.create_mock_engine(url, executor=lambda *args, **kwargs: None)
    return engine


def get_reserved_words(dialect: sa.engine.Dialect) -> set[str]:
    words = getattr(dialect, "reserved_words", None)
    if words is None:
        return set()
    return set(words)


def normalize_identifier(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", cleaned)
    if cleaned == "":
        cleaned = "col"
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def dedupe_names(names: list[str]) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for name in names:
        base = name
        suffix = seen.get(base, 0)
        candidate = base if suffix == 0 else f"{base}_{suffix}"
        while candidate in seen:
            suffix += 1
            candidate = f"{base}_{suffix}"
        seen[base] = suffix + 1
        seen[candidate] = 1
        out.append(candidate)
    return out


def rename_reserved_and_normalize(df: pd.DataFrame, dialect: sa.engine.Dialect) -> tuple[pd.DataFrame, dict[str, str]]:
    reserved = get_reserved_words(dialect)
    max_len = getattr(dialect, "max_identifier_length", None)
    limit = int(max_len) if max_len and int(max_len) > 0 else 0
    orig_cols = [str(c) for c in df.columns]
    norm_cols = [normalize_identifier(c) for c in orig_cols]
    if limit:
        norm_cols = [c[:limit] for c in norm_cols]
    norm_cols = dedupe_names(norm_cols)
    names: list[str] = []
    for norm in norm_cols:
        cand = norm
        if cand.lower() in reserved or cand in reserved:
            cand = f"{cand}_col"
        if limit:
            cand = cand[:limit]
        names.append(cand)
    if limit:
        names = dedupe_names([n[:limit] for n in names])
    else:
        names = dedupe_names(names)
    mapping = {orig: new for orig, new in zip(orig_cols, names)}
    df2 = df.rename(columns=mapping)
    return df2, mapping


def infer_not_null_columns(df: pd.DataFrame) -> set[str]:
    not_null: set[str] = set()
    for col in df.columns:
        s = df[col]
        if s.isna().any():
            continue
        not_null.add(str(col))
    return not_null


def sample_series_nonnull(s: pd.Series, sample_size: int) -> pd.Series:
    nonnull = s.dropna()
    if len(nonnull) <= sample_size:
        return nonnull
    sample = nonnull.sample(sample_size, random_state=0)
    return sample


def looks_like_json(x: Any) -> bool:
    if isinstance(x, (dict, list)):
        return True
    if not isinstance(x, str):
        return False
    stripped = x.strip()
    if stripped == "":
        return False
    if (stripped[0], stripped[-1]) not in {("{", "}"), ("[", "]")}:
        return False
    try:
        json.loads(stripped)
    except Exception:
        return False
    return True


def max_string_length(values: pd.Series) -> int:
    if values.empty:
        return 0
    lengths = values.map(lambda x: len(x) if isinstance(x, str) else len(str(x)))
    m = int(lengths.max())
    return m


def is_binary_series(s: pd.Series, sample_size: int) -> bool:
    sample = sample_series_nonnull(s, sample_size)
    if sample.empty:
        return False
    is_bin = sample.map(lambda x: isinstance(x, (bytes, bytearray, memoryview))).any()
    return bool(is_bin)


def is_json_series(s: pd.Series, sample_size: int) -> bool:
    sample = sample_series_nonnull(s, sample_size)
    if sample.empty:
        return False
    ratio = float(sample.map(looks_like_json).mean())
    return ratio > 0.6


def manual_type_for_non_object(s: pd.Series) -> sa.types.TypeEngine:
    dt = s.dtype
    if pd.api.types.is_datetime64_any_dtype(dt):
        if isinstance(dt, pd.DatetimeTZDtype):
            return sa.DateTime(timezone=True)
        return sa.DateTime(timezone=False)
    if pd.api.types.is_timedelta64_dtype(dt):
        return sa.Float()
    if pd.api.types.is_bool_dtype(dt):
        return sa.Boolean()
    if pd.api.types.is_integer_dtype(dt):
        maxv = s.dropna().max()
        if pd.isna(maxv):
            return sa.Integer()
        if int(maxv) > np.iinfo(np.int32).max:
            return sa.BigInteger()
        return sa.Integer()
    if pd.api.types.is_float_dtype(dt):
        return sa.Float()
    return sa.Text()


def apply_dialect_overrides(dialect: sa.engine.Dialect, df: pd.DataFrame, dtype_map: dict[str, sa.types.TypeEngine], sample_size: int) -> dict[str, sa.types.TypeEngine]:
    out = dict(dtype_map)
    for col in df.columns:
        s = df[col]
        is_obj = pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)
        if not is_obj:
            continue
        if is_binary_series(s, sample_size):
            out[str(col)] = oracle_blob_or_binary(dialect)
            continue
        if is_json_series(s, sample_size):
            out[str(col)] = pg_jsonb_or_json(dialect)
            continue
        out[str(col)] = text_type_by_dialect(dialect, s, sample_size)
    return out


def pg_jsonb_or_json(dialect: sa.engine.Dialect) -> sa.types.TypeEngine:
    if dialect.name == "postgresql":
        try:
            from sqlalchemy.dialects.postgresql import JSONB
        except Exception:
            return sa.JSON()
        return JSONB()
    return sa.JSON()


def oracle_blob_or_binary(dialect: sa.engine.Dialect) -> sa.types.TypeEngine:
    if dialect.name != "oracle":
        return sa.LargeBinary()
    try:
        from sqlalchemy.dialects.oracle import BLOB
    except Exception:
        return sa.LargeBinary()
    return BLOB()


def text_type_by_dialect(dialect: sa.engine.Dialect, s: pd.Series, sample_size: int) -> sa.types.TypeEngine:
    sample = sample_series_nonnull(s, sample_size)
    m = max_string_length(sample)
    if dialect.name == "mysql":
        return mysql_text_type(m)
    if dialect.name == "oracle":
        return oracle_text_type(m)
    return sa.Text()


def mysql_text_type(max_len: int) -> sa.types.TypeEngine:
    if max_len > 65535:
        try:
            from sqlalchemy.dialects.mysql import LONGTEXT
        except Exception:
            return sa.Text()
        return LONGTEXT()
    return sa.String(length=255)


def oracle_text_type(max_len: int) -> sa.types.TypeEngine:
    if max_len > 4000:
        try:
            from sqlalchemy.dialects.oracle import CLOB
        except Exception:
            return sa.Text()
        return CLOB()
    if max_len == 0:
        return sa.String(length=1)
    return sa.String(length=min(max_len, 4000))


def coerce_series_for_type(s: pd.Series, t: sa.types.TypeEngine) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(s.dtype) and isinstance(t, sa.Float):
        return s.dt.total_seconds()
    if t.__class__.__name__.upper() in {"JSON", "JSONB"}:
        return s.map(jsonify_value)
    if isinstance(t, sa.LargeBinary):
        return s.map(bytesify_value)
    return s


def jsonify_value(x: Any) -> Any:
    if pd.isna(x):
        return None
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return x


def bytesify_value(x: Any) -> Any:
    if pd.isna(x):
        return None
    if isinstance(x, memoryview):
        return x.tobytes()
    if isinstance(x, bytearray):
        return bytes(x)
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8")
    return bytes(x)


def coerce_dataframe(df: pd.DataFrame, dtype_map: dict[str, sa.types.TypeEngine]) -> pd.DataFrame:
    out = df.copy()
    for col, t in dtype_map.items():
        out[col] = coerce_series_for_type(out[col], t)
    return out


def compile_create_table(table: sa.Table, dialect: sa.engine.Dialect) -> str:
    ddl = str(CreateTable(table).compile(dialect=dialect)).rstrip()
    if not ddl.endswith(";"):
        ddl = ddl + ";"
    return ddl


def build_table_manual(table_name: str, df: pd.DataFrame, dialect: sa.engine.Dialect, dtype_map: dict[str, sa.types.TypeEngine], not_null_cols: set[str]) -> sa.Table:
    md = sa.MetaData()
    cols: list[sa.Column] = []
    for col in df.columns:
        nullable = str(col) not in not_null_cols
        cols.append(sa.Column(str(col), dtype_map[str(col)], nullable=nullable))
    table = sa.Table(table_name, md, *cols)
    return table


def infer_types_via_pandas_internals(df: pd.DataFrame, table_name: str, engine: sa.Engine) -> dict[str, sa.types.TypeEngine]:
    import pandas.io.sql as psql

    pdb = psql.SQLDatabase(engine)
    SQLTable = psql.SQLTable
    dtype_map: dict[str, sa.types.TypeEngine] = {}

    for col in df.columns:
        ser = df[col]
        inferred = infer_one_type_via_pandas(SQLTable, pdb, table_name, df[[col]], ser)
        dtype_map[str(col)] = inferred
    return dtype_map


def infer_one_type_via_pandas(SQLTable: Any, pdb: Any, table_name: str, frame: pd.DataFrame, ser: pd.Series) -> sa.types.TypeEngine:
    st = SQLTable(name=table_name, pandas_sql_engine=pdb, frame=frame, index=False, if_exists="fail", keys=None, dtype=None, schema=None)
    if hasattr(st, "_sqlalchemy_type"):
        try:
            t = st._sqlalchemy_type(ser)
            return t
        except TypeError:
            t = st._sqlalchemy_type(ser.dtype)
            return t
    raise AttributeError("SQLTable._sqlalchemy_type not available")


def build_table_via_pandas_internals(df: pd.DataFrame, table_name: str, engine: sa.Engine, dtype_map: dict[str, sa.types.TypeEngine]) -> sa.Table:
    import pandas.io.sql as psql

    pdb = psql.SQLDatabase(engine)
    st = psql.SQLTable(name=table_name, pandas_sql_engine=pdb, frame=df, index=False, if_exists="fail", keys=None, dtype=dtype_map, schema=None)
    if not hasattr(st, "_create_table_setup"):
        raise AttributeError("SQLTable._create_table_setup not available")
    table = st._create_table_setup()
    return table


def ddl_via_pandas_internals(df: pd.DataFrame, table_name: str, dialect_name: str) -> tuple[str, pd.DataFrame, str, Optional[str]]:
    engine = make_mock_engine(dialect_name)
    dialect = engine.dialect
    df2, _ = rename_reserved_and_normalize(df, dialect)
    not_null_cols = infer_not_null_columns(df2)

    try:
        dtype_map = infer_types_via_pandas_internals(df2, table_name, engine)
        dtype_map = apply_dialect_overrides(dialect, df2, dtype_map, sample_size=1000)
        cleaned = coerce_dataframe(df2, dtype_map)
        table = build_table_via_pandas_internals(df2, table_name, engine, dtype_map)
        enforce_not_null(table, not_null_cols)
        ddl = compile_create_table(table, dialect)
        return ddl, cleaned, "primary", None
    except Exception as exc:
        ddl, cleaned = ddl_via_manual_fallback(df2, table_name, dialect, not_null_cols)
        return ddl, cleaned, "fallback", repr(exc)


def enforce_not_null(table: sa.Table, not_null_cols: set[str]) -> None:
    for c in table.columns:
        if c.name in not_null_cols:
            c.nullable = False


def ddl_via_manual_fallback(df: pd.DataFrame, table_name: str, dialect: sa.engine.Dialect, not_null_cols: set[str]) -> tuple[str, pd.DataFrame]:
    dtype_map: dict[str, sa.types.TypeEngine] = {}
    for col in df.columns:
        s = df[col]
        is_obj = pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)
        if is_obj:
            dtype_map[str(col)] = sa.Text()
        else:
            dtype_map[str(col)] = manual_type_for_non_object(s)
    dtype_map = apply_dialect_overrides(dialect, df, dtype_map, sample_size=1000)
    cleaned = coerce_dataframe(df, dtype_map)
    table = build_table_manual(table_name, cleaned, dialect, dtype_map, not_null_cols)
    ddl = compile_create_table(table, dialect)
    return ddl, cleaned

