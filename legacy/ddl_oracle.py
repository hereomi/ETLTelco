"""
Oracle-specific DDL Generation Module
Simplified version focused on Oracle database requirements
"""
import re
import json
import pandas as pd
from keyword import iskeyword
from typing import Optional, Union, List, Tuple, Dict, Any
from logger import log_call, log_json, log_string

# Oracle-specific constants
ORACLE_RESERVED_WORDS = {
    'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'GROUP', 'ORDER',
    'BY', 'CREATE', 'TABLE', 'INDEX', 'VIEW', 'PRIMARY', 'KEY', 'FOREIGN',
    'CONSTRAINT', 'NUMBER', 'DATE', 'UNION', 'ALL', 'DISTINCT', 'JOIN',
    'INNER', 'OUTER', 'LEFT', 'RIGHT', 'ON', 'USING', 'HAVING', 'LIMIT',
    'OFFSET', 'ALTER', 'DROP', 'TRUNCATE', 'USER', 'LEVEL', 'ACCESS', 'MODE'
}

ORACLE_MAX_IDENTIFIER = 30  # Oracle identifier length limit
DEFAULT_VARCHAR_SIZE = 255

# Oracle-specific type mapping
ORACLE_DTYPE_MAP = {
    'object': 'VARCHAR2(255)',
    'string': 'VARCHAR2(255)',
    'category': 'VARCHAR2(255)',
    'Int64': 'NUMBER',
    'int64': 'NUMBER',
    'Float64': 'BINARY_DOUBLE',
    'float64': 'BINARY_DOUBLE',
    'float32': 'BINARY_FLOAT',
    'boolean': 'NUMBER(1,0)',
    'bool': 'NUMBER(1,0)',
    'datetime64[ns]': 'TIMESTAMP',
    'datetime64[us]': 'TIMESTAMP',
    'timedelta64[ns]': 'INTERVAL DAY TO SECOND',
    'timedelta64[us]': 'INTERVAL DAY TO SECOND',
    'date': 'DATE',
    'time': 'DATE',  # Oracle doesn't have a dedicated TIME type
    'datetime': 'TIMESTAMP',
    'timedelta': 'INTERVAL DAY TO SECOND'
}


def sanitize_oracle_cols(obj, to_lower: bool = True, fallback: str = 'col_'):
    """Oracle-specific column sanitization respecting 30 char limit"""
    sql_kw = {kw.upper() for kw in ORACLE_RESERVED_WORDS}
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
        s = re.sub(r'[^\w]', '_', s)
        if to_lower:
            s = s.lower()
        s = re.sub(r'__+', '_', s).strip('_')
        # Ensure Oracle identifier length compliance
        if len(s) > ORACLE_MAX_IDENTIFIER:
            s = s[:ORACLE_MAX_IDENTIFIER-4] + "_" + str(hash(s) % 10000).zfill(4)  # Add hash to differentiate
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
        k = sanitize_oracle_cols(list(obj.keys()), to_lower, fallback)
        return dict(zip(k, obj.values()))

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.columns = sanitize_oracle_cols(df.columns.tolist(), to_lower, fallback)
        return df

    if isinstance(obj, str):
        return _c(obj)
    return obj


def _truncate_identifier(name: str, max_len: int = ORACLE_MAX_IDENTIFIER) -> str:
    """Truncate identifiers to Oracle's 30-character limit"""
    if len(name) > max_len:
        # Add a hash to preserve uniqueness when truncating
        truncated = name[:max_len-5]  # Leave space for hash and underscore
        name_hash = str(abs(hash(name)) % 10000).zfill(4)
        return f"{truncated}_{name_hash}"
    return name


def escape_oracle_identifier(name: str) -> str:
    """Escape Oracle identifiers"""
    name_upper = name.upper()

    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$#]*', name) and name_upper not in ORACLE_RESERVED_WORDS:
        return name.upper()  # Oracle identifiers are typically uppercase
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


def _oracle_object_sql_type(semantic_type: Optional[str], varchar_size: int) -> Optional[str]:
    """Maps semantic object types to Oracle-specific types"""
    if semantic_type == "STRING_OBJECT":
        return f"VARCHAR2({varchar_size})"
    if semantic_type in ("STRUCTURED_OBJECT", "TRUE_OBJECT"):
        return "BLOB"
    return None


def get_oracle_sql_type(
    col_dtype,
    varchar_sizes: Optional[Dict[str, int]] = None,
    col_name: Optional[str] = None,
    semantic_type: Optional[str] = None
) -> str:
    """
    Maps pandas dtype to Oracle SQL type.
    Respects per-column varchar_sizes for generic string/unknown types.
    """
    norm_dtype = _normalize_dtype(col_dtype)
    
    if norm_dtype == 'object' and semantic_type:
        obj_type = _oracle_object_sql_type(semantic_type, varchar_sizes.get(col_name, DEFAULT_VARCHAR_SIZE) if varchar_sizes else DEFAULT_VARCHAR_SIZE)
        if obj_type:
            try:
                log_json(f"type_mapping_{col_name}", {
                    "pandas_dtype": str(col_dtype), 
                    "sql_type": obj_type, 
                    "server": "oracle", 
                    "semantic_type": semantic_type
                })
            except Exception:
                print(f"WARNING: Failed to log type mapping for {col_name}")
            return obj_type
            
    sql_type = ORACLE_DTYPE_MAP.get(norm_dtype)

    if sql_type is None:
        cd = str(col_dtype).lower()
        if any(x in cd for x in ('datetime', 'date', 'time')):
            sql_type = ORACLE_DTYPE_MAP.get('datetime', 'VARCHAR2(255)')
        elif 'timedelta' in cd:
            sql_type = ORACLE_DTYPE_MAP.get('timedelta', 'VARCHAR2(255)')
        else:
            size = DEFAULT_VARCHAR_SIZE
            if varchar_sizes and col_name is not None:
                size = varchar_sizes.get(col_name, DEFAULT_VARCHAR_SIZE)
            sql_type = f'VARCHAR2({size})'
    
    try:
        log_json(f"type_mapping_{col_name}", {
            "pandas_dtype": str(col_dtype), 
            "sql_type": sql_type, 
            "server": "oracle", 
            "semantic_type": semantic_type
        })
    except Exception:
        print(f"WARNING: Failed to log type mapping for {col_name}")
    
    return sql_type


def normalize_cols(cols) -> List[str]:
    """Normalize column input to list"""
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols) if hasattr(cols, '__iter__') else [cols]


def build_oracle_pk_constraint(table: str, pk) -> str:
    """Build Oracle primary key constraint SQL"""
    pk_list = ", ".join(escape_oracle_identifier(x) for x in normalize_cols(pk))
    name = escape_oracle_identifier(_truncate_identifier(f"{table}_PK"))
    return f"CONSTRAINT {name} PRIMARY KEY ({pk_list})"


def build_oracle_fk_constraint(table: str, col: str, ref_tab: str, ref_col: str, idx: int) -> str:
    """Build Oracle foreign key constraint SQL"""
    col_esc = escape_oracle_identifier(col)
    ref_tab_esc = escape_oracle_identifier(ref_tab)
    ref_col_esc = escape_oracle_identifier(ref_col)
    name = escape_oracle_identifier(_truncate_identifier(f"{table}_fk{idx}"))
    return f"CONSTRAINT {name} FOREIGN KEY ({col_esc}) REFERENCES {ref_tab_esc}({ref_col_esc})"


def build_oracle_unique_constraint(table: str, cols, idx: int) -> str:
    """Build Oracle unique constraint SQL"""
    col_list = ", ".join(escape_oracle_identifier(x) for x in normalize_cols(cols))
    name = escape_oracle_identifier(_truncate_identifier(f"{table}_uq{idx}"))
    return f"CONSTRAINT {name} UNIQUE ({col_list})"


def _build_oracle_sequence_trigger(table: str, col: str, initial_value: int = 1) -> Tuple[str, str]:
    """Build Oracle sequence and trigger for auto-increment"""
    seq_name = escape_oracle_identifier(_truncate_identifier(f"{table}_{col}_seq"))
    trig_name = escape_oracle_identifier(_truncate_identifier(f"{table}_{col}_trg"))
    
    seq_sql = f"CREATE SEQUENCE {seq_name} START WITH {initial_value} INCREMENT BY 1 NOCACHE;"
    trig_sql = f"""
CREATE OR REPLACE TRIGGER {trig_name}
BEFORE INSERT ON {escape_oracle_identifier(table)}
FOR EACH ROW
BEGIN
  IF :NEW.{escape_oracle_identifier(col)} IS NULL THEN
    :NEW.{escape_oracle_identifier(col)} := {seq_name}.NEXTVAL;
  END IF;
END;"""
    
    return seq_sql, trig_sql


@log_call
def oracle_df_ddl(
    input_df: pd.DataFrame,
    table: str,
    schema_name: Optional[str] = None,
    pk: Optional[Union[str, List[str]]] = None,
    fk: Optional[List[Tuple[str, str, str]]] = None,
    unique: Optional[List[Union[str, List[str]]]] = None,
    autoincrement: Optional[Tuple[str, int]] = None,
    varchar_sizes: Optional[Dict[str, int]] = None,
    cast: bool = True,
    cast_kwargs: Optional[Dict[str, Any]] = None,
    sanitize: bool = False,
    rename_column: bool = False,
    return_dtype_meta: bool = False
) -> Union[Tuple[pd.DataFrame, str, Dict[str, Any]], Tuple[pd.DataFrame, str, Dict[str, Any], Dict[str, Any]]]:
    """
    Generate Oracle CREATE TABLE DDL and schema metadata from a DataFrame.
    """
    if input_df.empty:
        raise ValueError("DataFrame is empty")
    if not table or not isinstance(table, str):
        raise ValueError("Table name must be non-empty string")

    # Copy the dataframe to avoid modifying the original
    df = input_df.copy()

    # Validation and Sanitization
    problem_cols = []
    
    for col in df.columns:
        col_name = str(col)
        # Check for reserved words or invalid characters
        is_reserved = col_name.upper() in ORACLE_RESERVED_WORDS
        is_invalid_chars = not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$#]*', col_name)
        
        if is_reserved or is_invalid_chars:
            problem_cols.append(col_name)

    if problem_cols and not (sanitize or rename_column):
        msg = f"Column names {problem_cols} contain Oracle reserved words or invalid characters! Please remove or rename them manually, or set rename_column=True for auto-renaming."
        print(msg)
        raise ValueError(msg)

    mapping = None
    if sanitize or rename_column:
        orig_cols = df.columns.tolist()
        df = sanitize_oracle_cols(df)
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
        if autoincrement:
            col, val = autoincrement
            autoincrement = (mapping.get(col, col), val)

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
    auto_col = autoincrement[0] if autoincrement else None
    pk_cols = normalize_cols(pk)

    for col in df.columns:
        col_dtype = str(df[col].dtype)
        semantic_type = None
        if cast_meta and col in cast_meta:
            semantic_type = cast_meta[col].get("object_semantic_type")
        
        sql_type = get_oracle_sql_type(col_dtype, varchar_sizes, col_name=col, semantic_type=semantic_type)
        col_esc = escape_oracle_identifier(col)
        is_auto = (col == auto_col and autoincrement is not None)

        if is_auto:
            # For Oracle auto-increment, we'll create a NUMBER column and handle sequence separately
            cols.append(f"{col_esc} NUMBER GENERATED ALWAYS AS IDENTITY (START WITH {autoincrement[1]} INCREMENT BY 1) NOT NULL")
            continue

        nullable = 'NOT NULL' if not df[col].isna().any() else 'NULL'
        cols.append(f"{col_esc} {sql_type} {nullable}")

    # Add constraints
    add_pk = pk is not None
    if add_pk:
        is_single_auto_pk = (len(pk_cols) == 1 and pk_cols[0] == auto_col)
        if is_single_auto_pk:
            add_pk = False  # Skip PK if it's the auto-increment column
    if add_pk:
        cols.append(build_oracle_pk_constraint(table, pk))

    if fk:
        for i, (c, rt, rc) in enumerate(fk, 1):
            cols.append(build_oracle_fk_constraint(table, c, rt, rc, i))

    if unique:
        for i, ug in enumerate(unique, 1):
            cols.append(build_oracle_unique_constraint(table, ug, i))

    # Build table name with schema if provided
    tbl_name = escape_oracle_identifier(table)
    if schema_name:
        tbl_name = f"{escape_oracle_identifier(schema_name)}.{tbl_name}"
    
    ddl = f"CREATE TABLE {tbl_name} (\n    " + ",\n    ".join(cols) + "\n)"

    # Build metadata
    columns_meta = []
    for col in df.columns:
        col_dtype = str(df[col].dtype)
        semantic_type = None
        if cast_meta and col in cast_meta:
            semantic_type = cast_meta[col].get("object_semantic_type")
        
        sql_type = get_oracle_sql_type(col_dtype, varchar_sizes, col_name=col, semantic_type=semantic_type)
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
        if semantic_type:
            col_entry['semantic_type'] = semantic_type
        columns_meta.append(col_entry)

    pk_cols = normalize_cols(pk)
    fk_list = [{'column': c, 'references_table': rt, 'references_column': rc}
               for c, rt, rc in (fk or [])]
    unique_list = [normalize_cols(u) for u in (unique or [])]

    meta = {
        'server': 'oracle',
        'schema': schema_name,
        'table': table,
        'columns': columns_meta,
        'primary_key': pk_cols,
        'foreign_keys': fk_list,
        'unique_constraints': unique_list,
        'autoincrement': (
            {'column': autoincrement[0], 'initial_value': autoincrement[1]}
            if autoincrement else None
        ),
        'column_mapping': mapping,
        'row_count': len(df),
        'column_count': len(df.columns)
    }

    # Log results
    try:
        ddl_log = ddl if len(ddl) < 5000 else ddl[:5000] + "\n... (truncated)"
        log_string(f"ddl_oracle_{table}", ddl_log)
        log_json(f"schema_meta_oracle_{table}", {
            "table": table, 
            "server": 'oracle', 
            "columns": len(meta.get('columns', [])), 
            "pk": meta.get('primary_key')
        })
    except Exception:
        print(f"WARNING: Failed to log DDL for Oracle table {table}")

    if return_dtype_meta and cast_meta:
        dtype_meta = {}
        for col in df.columns:
            col_dtype = df[col].dtype
            semantic_type = None
            if cast_meta and col in cast_meta:
                semantic_type = cast_meta[col].get("object_semantic_type")
            
            output_type = get_oracle_sql_type(col_dtype, varchar_sizes, col_name=col, semantic_type=semantic_type)
            dtype_meta[col] = {"input_dtype": str(col_dtype), "output_dtype": output_type}
            if semantic_type:
                dtype_meta[col]["semantic_type"] = semantic_type
        return df, ddl, meta, dtype_meta

    return df, ddl, meta