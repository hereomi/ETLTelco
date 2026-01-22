from typing import Optional, Dict, Any
import pandas as pd
from crud import auto_insert as insert, auto_upsert as upsert, auto_update as crud_update
from utils.schema_manager import SchemaManager


def _load_and_validate(
    ctx,
    sm: SchemaManager,
    table: str,
    df: pd.DataFrame,
    op_type: str = 'insert',
    pk: Optional[list] = None,
    chunksize: Optional[int] = None,
    add_missing_cols: Optional[bool] = None,
    failure_threshold: Optional[int] = None,
    semantic_meta: Optional[Dict[str, Any]] = None,
    trace_sql: bool = False
):
    """Orchestrates the actual data load and performs final row count validation."""
    csize = chunksize if chunksize is not None else ctx.default_chunksize
    amc = add_missing_cols if add_missing_cols is not None else ctx.default_add_missing_cols
    ft = failure_threshold if failure_threshold is not None else ctx.default_failure_threshold

    print(f"Performing '{op_type}' on '{table}' (chunksize={csize})...")
    try:
        if op_type == 'upsert':
            res = upsert(ctx.engine, df, table, pk, chunk=csize, add_missing_cols=amc, failure_threshold=ft, semantic_meta=semantic_meta, trace_sql=trace_sql)
            rows_done = res.get('success', 0) if isinstance(res, dict) else res
        else:
            rows_done = insert(ctx.engine, df, table, chunk_size=csize, add_missing_cols=amc, failure_threshold=ft, semantic_meta=semantic_meta, trace_sql=trace_sql)

        print(f"Load complete. {rows_done} rows processed for '{table}'.")

        print(f"Validating row counts for '{table}'...")
        db_count = sm.get_row_count(table)
        df_count = len(df)

        if op_type == 'insert' and db_count != df_count:
            print(f"Validation FAILURE: DB count ({db_count}) != DF count ({df_count}).")
            return False

        print(f"Validation finished. Final DB count: {db_count}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during data loading: {e}")
        return False
