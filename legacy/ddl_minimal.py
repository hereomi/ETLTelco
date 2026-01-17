"""
Minimal DDL Generation Module
Simplified version of ddl_create with essential functionality only
"""
import re
import json
import pandas as pd
from keyword import iskeyword
from typing import Optional, Union, List, Tuple, Dict, Any
from logger import log_call, log_json, log_string

# Simplified constants for PostgreSQL only
RESERVED_WORDS = {
    'postgresql': {
        'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'GROUP', 'ORDER',
        'BY', 'CREATE', 'TABLE', 'PRIMARY', 'KEY', 'FOREIGN', 'CONSTRAINT',
        'REFERENCES', 'BIGINT', 'INTEGER', 'SMALLINT', 'VARCHAR', 'TIMESTAMP',
        'BOOLEAN', 'NUMERIC', 'DECIMAL', 'REAL'
    }
}

DEFAULT_VARCHAR_SIZE = 255

# Simplified type mapping for PostgreSQL only
DTYPE_MAP = {
    'postgresql': {
        'object': 'TEXT',
        'string': 'TEXT',
        'category': 'TEXT',
        'Int64': 'BIGINT',
        'int64': 'BIGINT',
        'Float64': 'DOUBLE PRECISION',
        'float64': 'DOUBLE PRECISION',
        'float32': 'REAL',
        'boolean': 'BOOLEAN',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'datetime64[us]': 'TIMESTAMP',
        'timedelta64[ns]': 'INTERVAL',
        'timedelta64[us]': 'INTERVAL',
        'date': 'DATE',
        'time': 'TIME',
        'datetime': 'TIMESTAMP',
        'timedelta': 'INTERVAL'
    }
}


def sanitize_cols(obj, to_lower: bool = True, fallback: str = 'col_'):
    """Simplified column sanitization for PostgreSQL"""
    sql_kw = {kw.upper() for kw in RESERVED_WORDS.get('postgresql', set())}
    sql_kw.update({
        "SELECT", "FROM", "WHERE", "GROUP", "ORDER", "LIMIT",
        "JOIN", "TABLE", "COLUMN", "INSERT", "UPDATE", "DELETE",
        "CREATE", "DROP", "ALTER", "SCHEMA", "INDEX", "VIEW"
    })
    blank_count = iter(range(1, 1_000_000))

    def _c(s: Any):
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        s = re.sub(r'[^\w]', '_', s)  # Removed space handling for simplicity
        if to_lower:
            s = s.lower()
        s = re.sub(r'__+', '_', s).strip('_')
        if re.match(r'^\d', s):
            s = '_' + s
        if not s:
            s = f'{fallback}{next(blank_count)}'
        if s.upper() in sql_kw or iskeyword(s):
            s += '_'
        return s

    if isinstance(obj, (list, tuple, pd.Index)) and not isinstance(obj, str):
        seen: Dict[str, int] = {}
        clean = []
        for n in obj:
            c = _c(n)
            i = seen.get(c, 0)
            seen[c] = i + 1
            clean.append(c if i == 0 else f'{c}_{i}')
        return clean

    if isinstance(obj, dict):
        k = sanitize_cols(list(obj.keys()), to_lower, fallback)
        return dict(zip(k, obj.values()))

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.columns = sanitize_cols(df.columns.tolist(), to_lower, fallback)
        return df

    if isinstance(obj, str):
        return _c(obj)
    return obj


def escape_identifier(name: str) -> str:
    """Escape PostgreSQL identifiers"""
    reserved = RESERVED_WORDS.get('postgresql', set())
    name_upper = name.upper()

    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) and name_upper not in reserved:
        return name
    return '"{}"'.format(name.replace('"', '""'))


def _normalize_dtype(dtype) -> str:
    """Normalize pandas dtypes to canonical form"""
    dtype = str(dtype).lower()
    if dtype in ('int64', 'int32', 'int16', 'int8'):
        return 'Int64'
    if dtype in ('float64', 'float32'):
        return 'Float64'
    if dtype == 'bool':
        return 'boolean'
    return dtype


def get_sql_type(col_dtype, varchar_sizes: Optional[Dict[str, int]] = None, col_name: Optional[str] = None) -> str:
    """
    Maps pandas dtype to PostgreSQL type.
    Respects per-column varchar_sizes for generic string/unknown types.
    """
    dtype_map = DTYPE_MAP.get('postgresql', DTYPE_MAP['postgresql'])
    norm_dtype = _normalize_dtype(col_dtype)
    sql_type = dtype_map.get(norm_dtype)

    if sql_type is None:
        cd = str(col_dtype).lower()
        if any(x in cd for x in ('datetime', 'date', 'time')):
            sql_type = dtype_map.get('datetime', 'VARCHAR(255)')
        elif 'timedelta' in cd:
            sql_type = dtype_map.get('timedelta', 'VARCHAR(255)')
        else:
            size = DEFAULT_VARCHAR_SIZE
            if varchar_sizes and col_name is not None:
                size = varchar_sizes.get(col_name, DEFAULT_VARCHAR_SIZE)
            sql_type = f'VARCHAR({size})'
    
    try:
        log_json(f"type_mapping_{col_name}", {"pandas_dtype": str(col_dtype), "sql_type": sql_type, "server": 'postgresql'})
    except Exception:
        # Fixed: Avoid silent failures - log the error
        print(f"WARNING: Failed to log type mapping for {col_name}")
    
    return sql_type


def normalize_cols(cols) -> List[str]:
    """Normalize column input to list"""
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols) if hasattr(cols, '__iter__') else [cols]


def build_pk_constraint(table: str, pk) -> str:
    """Build primary key constraint SQL"""
    pk_list = ", ".join(escape_identifier(x) for x in normalize_cols(pk))
    name = escape_identifier(f"{table}_PK")
    return f"CONSTRAINT {name} PRIMARY KEY ({pk_list})"


def build_fk_constraint(table: str, col: str, ref_tab: str, ref_col: str, idx: int) -> str:
    """Build foreign key constraint SQL"""
    col_esc = escape_identifier(col)
    ref_tab_esc = escape_identifier(ref_tab)
    ref_col_esc = escape_identifier(ref_col)
    name = escape_identifier(f"{table}_fk{idx}")
    return f"CONSTRAINT {name} FOREIGN KEY ({col_esc}) REFERENCES {ref_tab_esc}({ref_col_esc})"


def build_unique_constraint(table: str, cols, idx: int) -> str:
    """Build unique constraint SQL"""
    col_list = ", ".join(escape_identifier(x) for x in normalize_cols(cols))
    name = escape_identifier(f"{table}_uq{idx}")
    return f"CONSTRAINT {name} UNIQUE ({col_list})"


@log_call
def df_ddl(
    input_df: pd.DataFrame,
    table: str,
    schema_name: Optional[str] = None,
    pk: Optional[Union[str, List[str]]] = None,
    fk: Optional[List[Tuple[str, str, str]]] = None,
    unique: Optional[List[Union[str, List[str]]]] = None,
    varchar_sizes: Optional[Dict[str, int]] = None,
    cast: bool = True,
    cast_kwargs: Optional[Dict[str, Any]] = None,
    sanitize: bool = False,
    rename_column: bool = False,
    return_dtype_meta: bool = False
) -> Union[Tuple[pd.DataFrame, str, Dict[str, Any]], Tuple[pd.DataFrame, str, Dict[str, Any], Dict[str, Any]]]:
    """
    Generate CREATE TABLE DDL and schema metadata from a DataFrame.
    Minimal implementation supporting PostgreSQL only.
    """
    if input_df.empty:
        raise ValueError("DataFrame is empty")
    if not table or not isinstance(table, str):
        raise ValueError("Table name must be non-empty string")

    # Copy the dataframe to avoid modifying the original
    df = input_df.copy()

    # Validation and Sanitization
    reserved = RESERVED_WORDS.get('postgresql', set())
    problem_cols = []
    
    for col in df.columns:
        col_name = str(col)
        # Check for reserved words or invalid characters
        is_reserved = col_name.upper() in reserved
        is_invalid_chars = not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', col_name)
        
        if is_reserved or is_invalid_chars:
            problem_cols.append(col_name)

    if problem_cols and not (sanitize or rename_column):
        msg = f"Column names {problem_cols} contain reserved words or invalid characters! Please remove or rename them manually, or set rename_column=True for auto-renaming."
        print(msg)
        raise ValueError(msg)

    mapping = None
    if sanitize or rename_column:
        orig_cols = df.columns.tolist()
        df = sanitize_cols(df)
        new_cols = df.columns.tolist()
        mapping = dict(zip(orig_cols, new_cols))

        if pk:
            if isinstance(pk, str):
                pk = mapping.get(pk, pk)
            else:
                pk = [mapping.get(c, c) for c in pk]
        if fk:
            fk = [(mapping.get(c, c), rt, rc) for c, rt, rc in fk]
        if unique:
            new_unique = []
            for u in unique:
                if isinstance(u, str):
                    new_unique.append(mapping.get(u, u))
                else:
                    new_unique.append([mapping.get(c, c) for c in u])
            unique = new_unique

    # Type casting if enabled
    if cast:
        from casting import cast_df
        cast_kwargs = cast_kwargs or {}
        cast_kwargs = {**cast_kwargs, "return_dtype_meta": True}
        df, cast_meta = cast_df(df, **cast_kwargs)
    else:
        cast_meta = None

    # Build column definitions
    cols: List[str] = []
    for col in df.columns:
        col_dtype = str(df[col].dtype)
        sql_type = get_sql_type(col_dtype, varchar_sizes, col_name=col)
        col_esc = escape_identifier(col)
        nullable = 'NOT NULL' if not df[col].isna().any() else 'NULL'
        cols.append(f"{col_esc} {sql_type} {nullable}")

    # Add constraints
    if pk:
        cols.append(build_pk_constraint(table, pk))

    if fk:
        for i, (c, rt, rc) in enumerate(fk, 1):
            cols.append(build_fk_constraint(table, c, rt, rc, i))

    if unique:
        for i, ug in enumerate(unique, 1):
            cols.append(build_unique_constraint(table, ug, i))

    # Build table name with schema if provided
    tbl_name = escape_identifier(table)
    if schema_name:
        tbl_name = f"{escape_identifier(schema_name)}.{tbl_name}"
    
    ddl = f"CREATE TABLE {tbl_name} (\n    " + ",\n    ".join(cols) + "\n)"

    # Build metadata
    columns_meta = []
    for col in df.columns:
        col_dtype = str(df[col].dtype)
        sql_type = get_sql_type(col_dtype, varchar_sizes, col_name=col)
        nullable = df[col].isna().any()
        non_null = df[col].dropna()
        sample = str(non_null.iloc[0]) if not non_null.empty else None

        col_entry = {
            'name': col,
            'pandas_dtype': col_dtype,
            'sql_dtype': sql_type,
            'nullable': nullable,
            'sample_value': sample
        }
        columns_meta.append(col_entry)

    pk_cols = normalize_cols(pk)
    fk_list = [{'column': c, 'references_table': rt, 'references_column': rc}
               for c, rt, rc in (fk or [])]
    unique_list = [normalize_cols(u) for u in (unique or [])]

    meta = {
        'server': 'postgresql',
        'schema': schema_name,
        'table': table,
        'columns': columns_meta,
        'primary_key': pk_cols,
        'foreign_keys': fk_list,
        'unique_constraints': unique_list,
        'column_mapping': mapping,
        'row_count': len(df),
        'column_count': len(df.columns)
    }

    # Log results
    try:
        ddl_log = ddl if len(ddl) < 5000 else ddl[:5000] + "\n... (truncated)"
        log_string(f"ddl_postgresql_{table}", ddl_log)
        log_json(f"schema_meta_postgresql_{table}", {"table": table, "server": 'postgresql', "columns": len(meta.get('columns', [])), "pk": meta.get('primary_key')})
    except Exception:
        # Fixed: Avoid silent failures - log the error
        print(f"WARNING: Failed to log DDL for table {table}")

    if return_dtype_meta and cast_meta:
        dtype_meta = {}
        for col in df.columns:
            col_dtype = df[col].dtype
            output_type = get_sql_type(col_dtype, varchar_sizes, col_name=col)
            dtype_meta[col] = {"input_dtype": str(col_dtype), "output_dtype": output_type}
        return df, ddl, meta, dtype_meta

    return df, ddl, meta