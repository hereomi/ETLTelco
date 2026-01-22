import re
from keyword import iskeyword
from typing import Optional, Union, List, Tuple, Dict, Sequence, Any
import pandas as pd

RESERVED_WORDS = {
    'oracle': {
        'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'GROUP', 'ORDER',
        'BY', 'CREATE', 'TABLE', 'INDEX', 'VIEW', 'PRIMARY', 'KEY', 'FOREIGN',
        'CONSTRAINT', 'NUMBER', 'DATE', 'UNION', 'ALL', 'DISTINCT', 'JOIN',
        'INNER', 'OUTER', 'LEFT', 'RIGHT', 'ON', 'USING', 'HAVING', 'LIMIT',
        'OFFSET', 'ALTER', 'DROP', 'TRUNCATE', 'USER', 'LEVEL', 'ACCESS', 'MODE'
    },
    'postgresql': {
        'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'GROUP', 'ORDER',
        'BY', 'CREATE', 'TABLE', 'PRIMARY', 'KEY', 'FOREIGN', 'CONSTRAINT',
        'REFERENCES', 'BIGINT', 'INTEGER', 'SMALLINT', 'VARCHAR', 'TIMESTAMP',
        'BOOLEAN', 'NUMERIC', 'DECIMAL', 'REAL'
    },
    'mysql': {
        'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'TABLE',
        'CREATE', 'DROP', 'ALTER', 'PRIMARY', 'KEY', 'FOREIGN', 'CONSTRAINT',
        'INT', 'VARCHAR', 'DATETIME', 'BIGINT', 'SMALLINT', 'AUTO_INCREMENT',
        'UNIQUE', 'INDEX'
    },
    'mssql': {
        'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'TABLE',
        'CREATE', 'DROP', 'ALTER', 'PRIMARY', 'KEY', 'FOREIGN', 'CONSTRAINT',
        'IDENTITY', 'INT', 'VARCHAR', 'DATETIME', 'BIT', 'FLOAT', 'NUMERIC',
        'DECIMAL', 'BIGINT', 'SMALLINT'
    },
    'sqlite': {
        'SELECT', 'INSERT', 'DELETE', 'UPDATE', 'WHERE', 'FROM', 'TABLE',
        'CREATE', 'DROP', 'ALTER', 'CONSTRAINT', 'PRIMARY', 'KEY', 'FOREIGN',
        'UNIQUE', 'CHECK', 'DEFAULT'
    }
}


def escape_identifier(name: str, dialect: str = 'oracle') -> str:
    """Escapes SQL identifier (table/column name) based on the database server."""
    d_key = dialect.lower()
    if d_key == 'postgres':
        d_key = 'postgresql'
    reserved = RESERVED_WORDS.get(d_key, RESERVED_WORDS['oracle'])
    name_upper = name.upper()

    if d_key in ('oracle', 'postgresql', 'sqlite'):
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$#]*', name) and name_upper not in reserved:
            return name.upper() if d_key == 'oracle' else name
        return '"{}"'.format(name.replace('"', '""'))

    if d_key == 'mysql':
        if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', name) and name_upper not in reserved:
            return name
        return '`{}`'.format(name.replace('`', '``'))

    if d_key == 'mssql':
        return f'[{name}]'

    return name

def sanitize_cols(
    obj,
    allow_space: bool = False,
    to_lower: bool = True,
    fallback: str = 'col_',
    dialect: str = 'postgresql'
):
    d_key = dialect.lower()
    if d_key == 'postgres':
        d_key = 'postgresql'

    sql_kw = {kw.upper() for kw in RESERVED_WORDS.get(d_key, set())}
    sql_kw.update({
        "SELECT", "FROM", "WHERE", "GROUP", "ORDER", "LIMIT",
        "JOIN", "TABLE", "COLUMN", "INSERT", "UPDATE", "DELETE",
        "CREATE", "DROP", "ALTER", "SCHEMA", "INDEX", "VIEW",
        "TRIGGER", "PROCEDURE", "FUNCTION", "DATABASE", "USER",
        "ROLE", "GRANT", "REVOKE"
    })
    blank_count = iter(range(1, 1_000_000))

    def _c(s: Any):
        if not isinstance(s, str):
            s = str(s)
        s = s.strip()
        s = re.sub(r'[^\w\s]' if allow_space else r'[^\w]', '_', s)
        if allow_space:
            s = re.sub(r'\s+', ' ', s)
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

    if isinstance(obj, (list, tuple, pd.Index, Sequence)) and not isinstance(obj, str):
        seen: Dict[str, int] = {}
        clean = []
        for n in obj:
            c = _c(n)
            i = seen.get(c, 0)
            seen[c] = i + 1
            clean.append(c if i == 0 else f'{c}_{i}')
        return clean

    if isinstance(obj, dict):
        k = sanitize_cols(list(obj.keys()), allow_space, to_lower, fallback, d_key)
        return dict(zip(k, obj.values()))

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.columns = sanitize_cols(df.columns.tolist(), allow_space, to_lower, fallback, d_key)
        return df

    if isinstance(obj, str):
        return _c(obj)
    # If we reach here, it's an unsupported type for column/dict/dataframe processing
    return obj
