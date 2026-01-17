from __future__ import annotations

from typing import Any, Callable


def _validate_ident(name: str, sanitizer: Callable[[str], str]) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Identifier must be a non-empty string")
    return sanitizer(name)


def _validate_cols_exist(cols: list[str], sanitized_set: set[str]) -> None:
    for c in cols:
        if c in sanitized_set:
            continue
        raise ValueError(f"Column '{c}' not present in table columns")


def build_pk_constraint(dialect: str, table_name: str, schema_name: str, pk_columns: list[str], sanitizer: Callable[[str], str], sanitized_set: set[str], constraint_name: str) -> tuple[str, dict[str, Any]]:
    if not pk_columns:
        raise ValueError("pk_columns cannot be empty")
    t = _validate_ident(table_name, sanitizer)
    s = sanitizer(schema_name) if schema_name else ""
    cols = [sanitizer(c) for c in pk_columns]
    _validate_cols_exist(cols, sanitized_set)
    cname = sanitizer(constraint_name) if constraint_name else sanitizer(f"PK_{t}")
    fq = f"{s}.{t}" if s else t
    sql = f"ALTER TABLE {fq} ADD CONSTRAINT {cname} PRIMARY KEY ({', '.join(cols)})"
    meta = {"name": cname, "columns": cols}
    return sql, meta


def build_unique_constraint(dialect: str, table_name: str, schema_name: str, unique_columns: list[str], sanitizer: Callable[[str], str], sanitized_set: set[str], constraint_name: str) -> tuple[str, dict[str, Any]]:
    if not unique_columns:
        raise ValueError("unique_columns cannot be empty")
    t = _validate_ident(table_name, sanitizer)
    s = sanitizer(schema_name) if schema_name else ""
    cols = [sanitizer(c) for c in unique_columns]
    _validate_cols_exist(cols, sanitized_set)
    cname = sanitizer(constraint_name) if constraint_name else sanitizer(f"UQ_{t}")
    fq = f"{s}.{t}" if s else t
    sql = f"ALTER TABLE {fq} ADD CONSTRAINT {cname} UNIQUE ({', '.join(cols)})"
    meta = {"name": cname, "columns": cols}
    return sql, meta


def build_fk_constraint(dialect: str, table_name: str, schema_name: str, fk_columns: list[str], ref_table: str, ref_schema: str, ref_columns: list[str], sanitizer: Callable[[str], str], sanitized_set: set[str], constraint_name: str, on_delete: str) -> tuple[str, dict[str, Any]]:
    if not fk_columns:
        raise ValueError("fk_columns cannot be empty")
    if not ref_columns:
        raise ValueError("ref_columns cannot be empty")
    if len(fk_columns) != len(ref_columns):
        raise ValueError("fk_columns and ref_columns must have same length")

    t = _validate_ident(table_name, sanitizer)
    s = sanitizer(schema_name) if schema_name else ""
    cols = [sanitizer(c) for c in fk_columns]
    _validate_cols_exist(cols, sanitized_set)

    rt = _validate_ident(ref_table, sanitizer)
    rs = sanitizer(ref_schema) if ref_schema else ""

    rcols = [sanitizer(c) for c in ref_columns]
    cname = sanitizer(constraint_name) if constraint_name else sanitizer(f"FK_{t}")

    fq = f"{s}.{t}" if s else t
    rfq = f"{rs}.{rt}" if rs else rt

    od = ""
    if on_delete:
        od_u = on_delete.strip().upper()
        if od_u in {"CASCADE", "SET NULL", "RESTRICT", "NO ACTION"}:
            od = f" ON DELETE {od_u}"

    sql = f"ALTER TABLE {fq} ADD CONSTRAINT {cname} FOREIGN KEY ({', '.join(cols)}) REFERENCES {rfq} ({', '.join(rcols)}){od}"
    meta = {"name": cname, "columns": cols, "ref_table": rt, "ref_schema": rs or None, "ref_columns": rcols, "on_delete": on_delete.upper() if on_delete else None}
    return sql, meta


def build_sequence(dialect: str, sequence_name: str, schema_name: str, sanitizer: Callable[[str], str], start_with: int, increment_by: int, cache: int) -> tuple[str, dict[str, Any]]:
    seq = _validate_ident(sequence_name, sanitizer)
    sch = sanitizer(schema_name) if schema_name else ""
    fq = f"{sch}.{seq}" if sch else seq

    if dialect == "oracle":
        c = int(cache) if cache and int(cache) > 0 else 0
        cache_sql = f"CACHE {c}" if c > 0 else "NOCACHE"
        sql = f"CREATE SEQUENCE {fq} START WITH {int(start_with)} INCREMENT BY {int(increment_by)} {cache_sql} NOCYCLE"
        meta = {"name": seq, "schema": sch or None, "start_with": int(start_with), "increment_by": int(increment_by), "cache": int(cache)}
        return sql, meta

    if dialect in {"postgre", "postgresql"}:
        sql = f"CREATE SEQUENCE {fq} START WITH {int(start_with)} INCREMENT BY {int(increment_by)}"
        meta = {"name": seq, "schema": sch or None, "start_with": int(start_with), "increment_by": int(increment_by)}
        return sql, meta

    raise ValueError(f"Dialect '{dialect}' does not support sequences via build_sequence")


def build_trigger(dialect: str, trigger_name: str, table_name: str, schema_name: str, column_name: str, sequence_name: str, sanitizer: Callable[[str], str]) -> tuple[str, dict[str, Any]]:
    trg = _validate_ident(trigger_name, sanitizer)
    t = _validate_ident(table_name, sanitizer)
    sch = sanitizer(schema_name) if schema_name else ""
    col = sanitizer(column_name)
    seq = sanitizer(sequence_name)

    fq_trg = f"{sch}.{trg}" if sch else trg
    fq_tbl = f"{sch}.{t}" if sch else t
    fq_seq = f"{sch}.{seq}" if sch else seq

    if dialect == "oracle":
        sql = (
            f"CREATE OR REPLACE TRIGGER {fq_trg}\n"
            f"BEFORE INSERT ON {fq_tbl}\n"
            f"FOR EACH ROW\n"
            f"WHEN (NEW.{col} IS NULL)\n"
            f"BEGIN\n"
            f"  SELECT {fq_seq}.NEXTVAL INTO :NEW.{col} FROM DUAL;\n"
            f"END;"
        )
        meta = {"name": trg, "schema": sch or None, "table": t, "column": col, "sequence": seq}
        return sql, meta

    raise ValueError(f"Dialect '{dialect}' does not support triggers via build_trigger")