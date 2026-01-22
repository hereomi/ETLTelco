from typing import Union
import pandas as pd
from utils.data_profiler import get_pk, profile_dataframe
from utils.schema_manager import SchemaManager


def _process_pk(ctx, df: pd.DataFrame, table: str, pk_arg: Union[str, list]) -> list:
    """Processes and validates the primary key argument, prioritizing DB-side PKs."""
    if pk_arg == 'verify':
        print(f"Verifying existing PK for table '{table}'...")
        return _derive_pk(ctx, df, table)

    if pk_arg == 'derive':
        print(f"Deriving PK for '{table}' (checking DB first)...")
        return _derive_pk(ctx, df, table)

    if not pk_arg: return []

    if isinstance(pk_arg, str):
        pk_arg = [pk_arg]

    if not isinstance(pk_arg, list):
        raise ValueError("pk argument must be 'derive', 'verify', or a list of column names.")

    df_cols_map = {c.lower(): c for c in df.columns}
    final_pk = []
    missing = []
    for c in pk_arg:
        if c.lower() in df_cols_map:
            final_pk.append(df_cols_map[c.lower()])
        else:
            missing.append(c)

    if missing:
        print(f"PK validation FAILED: Columns {missing} not found in DataFrame.")
        raise ValueError(f"Columns {missing} not found in DataFrame for PK. Avail: {list(df.columns)}")

    pk_arg = final_pk

    if len(pk_arg) == 1:
        col = pk_arg[0]
        if df[col].duplicated().any():
            print(f"PK validation FAILED: Column '{col}' contains duplicates.")
            raise ValueError(f"Manual PK validation failure: Column '{col}' contains duplicates.")
    else:
        combined = df[pk_arg].astype(str).agg('_'.join, axis=1)
        if combined.duplicated().any():
            print(f"PK validation FAILED: Composite {pk_arg} contains duplicates.")
            raise ValueError(f"Manual PK validation failure: Composite columns {pk_arg} (underscore-joined) contain duplicates.")

    print(f"PK validation SUCCESS for {table}: {pk_arg}")
    return pk_arg


def _derive_pk(ctx, df: pd.DataFrame, table: str) -> list:
    """Attempts to find a PK from the database first, then falls back to profiling."""
    sm = SchemaManager(ctx.engine)
    if sm.has_table(table):
        db_pk = sm.get_primary_keys(table)
        if db_pk:
            print(f"Detected existing database PK for '{table}': {db_pk}")
            df_cols_map = {c.lower(): c for c in df.columns}
            final_pk = []
            missing = []
            for c in db_pk:
                if c.lower() in df_cols_map:
                    final_pk.append(df_cols_map[c.lower()])
                else:
                    missing.append(c)

            if not missing:
                print(f"Matching DB PK with data columns: {final_pk}")
                return final_pk
            else:
                print(f"Warning: Database PK columns {missing} missing in source data. Falling back to profiling.")

    print(f"Profiling DataFrame to find optimal PK for '{table}'...")
    pk_info = get_pk(df, profile_dataframe(df))
    derived = pk_info[2]['components']
    if derived:
        print(f"Derived PK from data: {derived}")
        return derived
    return []
