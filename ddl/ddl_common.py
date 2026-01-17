from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


SEMANTIC_STRING_OBJECT = "STRING_OBJECT"
SEMANTIC_BINARY_OBJECT = "BINARY_OBJECT"
SEMANTIC_JSON_OBJECT = "JSON_OBJECT"
SEMANTIC_MIXED_OBJECT = "MIXED_OBJECT"
SEMANTIC_UNKNOWN_OBJECT = "UNKNOWN_OBJECT"


@dataclass(frozen=True)
class ObjectInspection:
    semantic_type: str
    non_null_count: int
    sample_count: int
    max_str_len: int
    max_bin_len: int
    json_text_ratio: float
    warnings: list[str]


@dataclass(frozen=True)
class ColumnMeta:
    original_name: str
    sanitized_name: str
    pandas_dtype: str
    sql_type: str
    semantic_object_type: str
    nullable: bool
    warnings: list[str]


def validate_dataframe(df: Any) -> None:
    if isinstance(df, pd.DataFrame):
        return
    raise TypeError("df must be a pandas DataFrame")


def validate_table_name(table_name: Any) -> None:
    if isinstance(table_name, str) and table_name.strip():
        return
    raise ValueError("table_name must be a non-empty string")


def _looks_like_json_text(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            json.loads(t)
            return True
        except Exception:
            return False
    return False


def inspect_object_series(s: pd.Series, sample_size: int, json_text_threshold: float) -> ObjectInspection:
    warnings: list[str] = []
    if s is None:
        return ObjectInspection(SEMANTIC_UNKNOWN_OBJECT, 0, 0, 0, 0, 0.0, ["Missing series"])
    non_null = s.dropna()
    non_null_count = int(non_null.shape[0])
    if non_null_count == 0:
        return ObjectInspection(SEMANTIC_STRING_OBJECT, 0, 0, 0, 0, 0.0, ["Empty object column defaulted to STRING_OBJECT"])
    sample = non_null.sample(sample_size, random_state=0) if non_null_count > sample_size else non_null
    sample_count = int(sample.shape[0])

    c_string = 0
    c_binary = 0
    c_json_obj = 0
    c_other = 0

    max_str_len = 0
    max_bin_len = 0
    json_text_hits = 0

    for v in sample.tolist():
        if isinstance(v, str):
            c_string += 1
            lv = len(v)
            if lv > max_str_len:
                max_str_len = lv
            if _looks_like_json_text(v):
                json_text_hits += 1
            continue
        if isinstance(v, (bytes, bytearray, memoryview)):
            c_binary += 1
            bv = bytes(v)
            lv = len(bv)
            if lv > max_bin_len:
                max_bin_len = lv
            continue
        if isinstance(v, (dict, list)):
            c_json_obj += 1
            continue
        c_other += 1

    json_text_ratio = (json_text_hits / c_string) if c_string else 0.0
    kinds = int(c_string > 0) + int(c_binary > 0) + int(c_json_obj > 0) + int(c_other > 0)

    if c_other > 0:
        return ObjectInspection(SEMANTIC_MIXED_OBJECT, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, ["Non-string/binary/json values detected; mixed fallback"])
    if kinds > 1:
        if c_string > 0 and c_binary == 0 and c_json_obj == 0:
            semantic = SEMANTIC_JSON_OBJECT if json_text_ratio >= json_text_threshold else SEMANTIC_STRING_OBJECT
            return ObjectInspection(semantic, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, [])
        return ObjectInspection(SEMANTIC_MIXED_OBJECT, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, ["Multiple object kinds detected; mixed fallback"])
    if c_json_obj > 0:
        return ObjectInspection(SEMANTIC_JSON_OBJECT, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, [])
    if c_binary > 0:
        return ObjectInspection(SEMANTIC_BINARY_OBJECT, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, [])
    if c_string > 0:
        semantic = SEMANTIC_JSON_OBJECT if json_text_ratio >= json_text_threshold else SEMANTIC_STRING_OBJECT
        return ObjectInspection(semantic, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, [])
    return ObjectInspection(SEMANTIC_UNKNOWN_OBJECT, non_null_count, sample_count, max_str_len, max_bin_len, json_text_ratio, ["Unclassifiable object values"])


def sanitize_identifier(name: str, max_len: int, reserved_words: set[str]) -> str:
    raw = str(name).strip()
    if not raw:
        raw = "COL"
    txt = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    txt = re.sub(r"_+", "_", txt)
    txt = txt.strip("_")
    if not txt:
        txt = "COL"
    if re.match(r"^\d", txt):
        txt = f"C_{txt}"
    txt = txt.upper()
    if txt in reserved_words:
        txt = f"{txt}_COL"
    if len(txt) > max_len:
        txt = txt[:max_len]
    return txt


def dedupe_identifiers(names: list[str], max_len: int) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    used: set[str] = set()
    for n in names:
        if n not in used:
            out.append(n)
            used.add(n)
            seen[n] = 0
            continue
        base = n
        idx = seen.get(base, 0) + 1
        while True:
            suffix = f"_{idx}"
            trimmed = base[:max_len - len(suffix)]
            cand = f"{trimmed}{suffix}"
            if cand not in used:
                out.append(cand)
                used.add(cand)
                seen[base] = idx
                break
            idx += 1
    return out


def build_base_schema(dialect: str, table_name: str, schema_name: str, columns: list[ColumnMeta], warnings: list[str], decisions: dict[str, Any], create_table_sql: str) -> dict[str, Any]:
    return {
        "dialect": dialect,
        "table": {
            "name": table_name,
            "schema": schema_name,
            "fully_qualified_name": f"{schema_name}.{table_name}" if schema_name else table_name,
        },
        "columns": [
            {
                "original_name": c.original_name,
                "sanitized_name": c.sanitized_name,
                "pandas_dtype": c.pandas_dtype,
                "semantic_object_type": c.semantic_object_type,
                "sql_type": c.sql_type,
                "nullable": c.nullable,
                "warnings": list(c.warnings),
            }
            for c in columns
        ],
        "constraints": {
            "primary_key": None,
            "unique": [],
            "foreign_keys": [],
        },
        "auto_increment": None,
        "warnings": list(warnings),
        "decisions": dict(decisions),
        "sql": {
            "create_table": create_table_sql,
            "constraints": [],
            "auto_increment": [],
            "full_ddl": create_table_sql,
        },
    }


def resolve_columns(cols: list[str], orig_to_san: dict[str, str], sanitized_set: set[str], sanitizer: Callable[[str], str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        k = str(c)
        if k in sanitized_set:
            out.append(k)
            continue
        if k in orig_to_san:
            out.append(orig_to_san[k])
            continue
        s = sanitizer(k)
        if s in sanitized_set:
            out.append(s)
            continue
        raise ValueError(f"Column '{c}' not found after sanitization")
    return out


def normalize_df(df: pd.DataFrame, rename_map: dict[str, str], json_cols: set[str], binary_cols: set[str], timedelta_cols: set[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    warnings: list[str] = []
    out.rename(columns=rename_map, inplace=True)

    for c in json_cols:
        s = out[c]
        def to_json(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, (dict, list)):
                try:
                    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    return str(v)
            if isinstance(v, str):
                return v
            return str(v)
        out[c] = s.map(to_json).astype("string")

    for c in binary_cols:
        s = out[c]
        def to_bytes(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, bytes):
                return v
            if isinstance(v, bytearray):
                return bytes(v)
            if isinstance(v, memoryview):
                return v.tobytes()
            if isinstance(v, str):
                return v.encode("utf-8")
            return str(v).encode("utf-8")
        out[c] = s.map(to_bytes)

    for c in timedelta_cols:
        s = out[c]
        out[c] = s.dt.total_seconds().astype("float64")
        warnings.append(f"Timedelta column '{c}' normalized to seconds float64")
    return out, warnings