from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Union, List, Dict, Any
from contextlib import contextmanager
from itertools import islice

import sqlalchemy as sa
from sqlalchemy.engine import Engine, Connection

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

DataItem = Mapping[str, Any]
DataLike = Union[Sequence[DataItem], DataItem, "pd.DataFrame"]

def _write_sql_to_file(func_name: str, table_name: str, statement: Any, engine: Engine):
    """Writes the generated SQL statement to a file for tracing."""
    filename = f"{func_name}_{table_name}.txt"
    try:
        # Try to compile with literal binds to show values
        if hasattr(statement, 'compile'):
            sql_str = str(statement.compile(dialect=engine.dialect, compile_kwargs={"literal_binds": True}))
        else:
            sql_str = str(statement)
            
        with open(filename, "w", encoding="utf-8") as f:
            f.write(sql_str)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Could not write SQL trace to {filename}: {e}")


def _normalize_data(data: DataLike) -> List[Dict[str, Any]]:
    """Normalize input data into list[dict], with Ultra-Safe null handling."""
    if pd is not None and isinstance(data, pd.DataFrame):
        # Stage 1: Vectorized cleaning for speed
        data_clean = data.astype(object).where(pd.notnull(data), None)
        records = data_clean.to_dict(orient="records")
    elif isinstance(data, Mapping):
        records = [dict(data)]
    elif isinstance(data, Sequence):
        records = [dict(row) for row in data]
    else:
        raise TypeError("Unsupported data type; expected dict, list[dict], or DataFrame")
        
    # Stage 2: Recursive sweep for pure Python types and Timestamp conversion
    # (Assuming simple flat dicts for now as per original code)
    for r in records:
        for k, v in r.items():
            # Handle Pandas-specific types
            if hasattr(v, 'to_pydatetime'):
                r[k] = v.to_pydatetime()
            elif pd is not None and v is pd.NaT:
                r[k] = None
            # Aggressive null check for all varieties of nan
            elif v is None:
                r[k] = None
            elif pd is not None and pd.isna(v):
                r[k] = None
            elif np is not None and isinstance(v, (float, np.floating)) and np.isnan(v):
                r[k] = None
                
    return records


def _chunk_iter(rows: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield chunks of rows with given size."""
    it = iter(rows)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


@contextmanager
def _ensure_connection(eng_or_conn: Union[Engine, Connection]) -> Iterable[Connection]:
    """Yield a Connection; do not manage transaction if caller passed Connection."""
    if isinstance(eng_or_conn, Connection):
        yield eng_or_conn
        return
    engine = eng_or_conn
    with engine.begin() as conn:
        yield conn


def _get_table(conn: Connection, table: Union[str, sa.Table]) -> sa.Table:
    """Resolve table name or return Table as-is, reflecting if needed."""
    if isinstance(table, sa.Table):
        return table
    meta = sa.MetaData()
    return sa.Table(table, meta, autoload_with=conn)
