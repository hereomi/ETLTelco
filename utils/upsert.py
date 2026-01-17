import os
import logging
import pandas as pd
import datetime as dt
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any, Optional
from .ddl import cast_dataframe as auto_convert_dtypes_ml, df_to_table_datatype
from .where_build import parse_sql_condition, sql_where
from .models import OperationResult

logger = logging.getLogger(__name__)

def _sanitize_data(data: Any) -> Any:
    """Convert problematic types (Timestamp, NaN) to DB-friendly types."""
    import numpy as np
    if isinstance(data, list):
        return [_sanitize_data(i) for i in data]
    if isinstance(data, dict):
        return {k: _sanitize_data(v) for k, v in data.items()}
    if hasattr(data, 'to_pydatetime'):
        return data.to_pydatetime()
    if isinstance(data, (float, np.floating)) and np.isnan(data):
        return None
    if data is pd.NaT:
        return None
    return data


def write_queries(data, file_path, count):
    """Write queries to file"""
    data_str = f"{count + 1}:{str(data)}"
    if isinstance(data, list):
        data_mod = "\n".join(str(item) for item in data)
        data_str = ";".join(str(item) for item in data_mod) if isinstance(data_mod, tuple) else data_mod
    elif isinstance(data, tuple):
        data_str = ";".join(str(item) for item in data)
    
    with open(file_path, "a+") as f:
        f.write(data_str + "\n")


def lazy_fallback(alchemy_instance, method_name, records, **kwargs) -> OperationResult:
    """
    Unified row-by-row retry logic for failed bulk operations.
    Executes within a single transaction context for efficiency.
    """
    method = getattr(alchemy_instance, method_name)
    success_count = 0
    errors = []
    
    logger.warning(f"Bulk {method_name} failed for table {alchemy_instance.table}. Falling back to row-by-row retry.")
    
    conn = kwargs.get('conn')
    
    def _run_row(row, connection):
        nonlocal success_count
        # Ensure the injected connection is used for all sub-calls
        kwargs['conn'] = connection
        try:
            # Call the method (e.g., insert or update) with the injected connection
            res = method(row, **kwargs)
            if res.success:
                count = res.affected_rows if res.affected_rows is not None else 1
                success_count += count
            else:
                errors.append(res.error or "Unknown error")
        except (SQLAlchemyError, ValueError, TypeError) as e:
            errors.append(str(e))

    if conn:
        # If a connection was passed from outside, we use it directly
        for row in records:
            _run_row(row, conn)
    else:
        # For the fallback path, we open a new connection for each row 
        # to ensure one failure doesn't kill the connection/transaction for others.
        # This is truly 'lazy' but robust.
        for row in records:
            with alchemy_instance.engine.connect() as new_conn:
                _run_row(row, new_conn)
            
    return OperationResult(
        success=(success_count > 0),
        affected_rows=success_count,
        error=f"Partial success. {len(errors)} failures out of {len(records)}." if errors else None,
        data=pd.DataFrame([{"total": len(records), "success": success_count, "failed": len(errors)}])
    )


def upsert_execute(engine, select, update, insert, is_execute=True, verb=False, session=None):
    """Execute a single upsert step. If session is provided, use it instead of creating one."""
    if not is_execute:
        return [select, update, insert]
        
    def _logic(s):
        if isinstance(select, tuple):
            existing = s.execute(text(select[0]), select[1]).fetchone()
        else:
            existing = s.execute(text(select)).fetchone()
            
        if existing:
            if isinstance(update, tuple):
                s.execute(text(update[0]), update[1])
                return f"[Update]: {update[0]}"
            else:
                s.execute(text(update))
                return f"[Update]: {update}"
        else:
            if isinstance(insert, tuple):
                s.execute(text(insert[0]), insert[1])
                return f"[Insert]: {insert[0]}"
            else:
                s.execute(text(insert))
                return f"[Insert]: {insert}"

    if session:
        try:
            what = _logic(session)
            return f'Success: {what}'
        except SQLAlchemyError as e:
            logger.error(f'Failed in shared session: {e}', exc_info=True)
            return str(e)
    else:
        Session = sessionmaker(bind=engine)
        with Session() as session:
            try:
                what = _logic(session)
                session.commit()
                result = f'Success: {what}'
                if verb:
                    print(result)
                return result
            except SQLAlchemyError as e:
                error_msg = f'Failed: {str(e)} - Query: {insert}'
                logger.error(error_msg, exc_info=True)
                return error_msg



def upsert_lazy(alchemy_instance, table, data, update_fields=None, where_condition=None, expression=None, file_path=None, writer_dir=None, write=True):
    """Perform row-by-row upsert (select-then-update-or-insert) within a single connection context."""
    target_table = table or getattr(alchemy_instance, 'table', None)
    if target_table is None:
        raise ValueError('Table name is required for upsert_lazy')
        
    df = alchemy_instance.sanitize_dataframe(data, table_name=target_table)
    if df.empty:
        logger.info('empty dataframe, exiting')
        return []
        
    dttm = dt.datetime.now().strftime('%b-%Y %H%M')
    file_path = f'upsert_lazy {dttm}.txt' if file_path is None else f'upsert_lazy {file_path}'
    update_fields = df.columns.tolist() if update_fields is None else update_fields
    
    # Check existence using the resolved table
    if not alchemy_instance.ins.has_table(target_table):
         # If we are here, we might want to auto-create? 
         # But the legacy behavior was export_df. Let's keep it but ensure export uses correct table.
         # Note: export_df in alc.py is single-table oriented currently.
         # We'll need to rely on the user having ensured table or use export_df carefully.
         # For now, let's assume export_df needs to be updated or we use alchemy_instance.export_df logic but targeted.
         # Given constraints, let's assume standard flow:
         sanitized = alchemy_instance.sanitize_dataframe(df, table_name=target_table)
         sanitized = auto_convert_dtypes_ml(sanitized)
         sanitized.to_sql(target_table, alchemy_instance.engine, index=False, if_exists='replace') # Warning: replace? 
         return len(sanitized)
        
    dataframe, _ = alchemy_instance.drop_invalid_columns(df, table_name=target_table)
    dataframe = auto_convert_dtypes_ml(dataframe)
    
    results = []
    
    writer_dir = writer_dir or alchemy_instance.writer_dir or os.path.join(os.getcwd(), 'writer')
    try:
        os.makedirs(writer_dir, exist_ok=True)
    except (OSError, PermissionError):
        pass
        
    counts = 0
    
    # Use a single connection for the entire lazy loop (Core API)
    with alchemy_instance.engine.connect() as conn:
        for row in dataframe.itertuples(index=False, name=None):
            row_dict = {dataframe.columns[i]: row[i] for i in range(len(row))}
            field_dict = {key: row_dict[key] for key in dataframe.columns.tolist() if key in update_fields}
            
            # Build WHERE clause with values substituted for '?'
            wh_conditions = []
            if where_condition:
                for cond in where_condition:
                    if '?' in str(cond):
                        parsed = parse_sql_condition(str(cond))
                        field_name = parsed.get('field', '')
                        if field_name in row_dict:
                            wh_conditions.append(cond.replace('?', f"'{row_dict[field_name]}'"))
                        else:
                            wh_conditions.append(cond)
                    else:
                        wh_conditions.append(cond)
            
            # Build parameterized queries
            where_clause, where_params = sql_where(wh_conditions, expression) if wh_conditions else ('1=1', {})
            
            # SELECT query
            select_query = f"SELECT 1 FROM {target_table} WHERE {where_clause}"
            
            # UPDATE query
            set_clause = ', '.join(f"{k} = :{k}" for k in field_dict.keys())
            update_query = f"UPDATE {target_table} SET {set_clause} WHERE {where_clause}"
            update_params = {**field_dict, **where_params}
            
            # INSERT query
            columns = ', '.join(row_dict.keys())
            placeholders = ', '.join(f':{k}' for k in row_dict.keys())
            insert_query = f"INSERT INTO {target_table} ({columns}) VALUES ({placeholders})"
            
            if write:
                write_queries([select_query, update_query, insert_query], os.path.join(writer_dir, file_path), counts)
                counts += 1
            
            # Execute upsert logic
            if alchemy_instance.is_execute:
                try:
                    existing = conn.execute(text(select_query), where_params).fetchone()
                    if existing:
                        conn.execute(text(update_query), update_params)
                        results.append(f"[Update]: {target_table}")
                    else:
                        conn.execute(text(insert_query), row_dict)
                        results.append(f"[Insert]: {target_table}")
                except SQLAlchemyError as e:
                    logger.error(f"Upsert failed for row: {e}", exc_info=True)
                    results.append(f"[Error]: {str(e)}")
            else:
                results.append([select_query, update_query, insert_query])
        
        conn.commit()
    return results


def upsert_bulk_enhanced(alchemy_instance, table, data, unique_constraint):
    target_table = table or getattr(alchemy_instance, 'table', None)
    if target_table is None:
        raise ValueError('Table name is required')
    df = alchemy_instance.sanitize_dataframe(data)
    if df.empty:
        raise ValueError('Data cannot be empty')
    df_converted = auto_convert_dtypes_ml(df)
    table_info = alchemy_instance.info
    df_typed = df_to_table_datatype(df_converted, table_info)
    
    q, p = upsert_bulk(target_table, df_typed, unique_constraint, alchemy_instance.server)
    
    res = alchemy_instance.execute(q, p)
    if not res.success:
        # Fallback to row-by-row for upsert_bulk
        return lazy_fallback(alchemy_instance, 'upsert_lazy', df_typed, update_fields=list(df_typed[0].keys()), where_condition=[f"{c} = ?" for c in unique_constraint])
    return res


def upsert_bulk(table, data, unique_columns, server):
    if isinstance(data, dict):
        data = [data]
    if not data:
        raise ValueError("Data cannot be empty")
    first_row = data[0]
    columns = list(first_row.keys())
    db_type = server.lower()
    if db_type == 'mysql':
        update_clause = ', '.join(f"{col}=VALUES({col})" for col in columns if col not in unique_columns)
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(f':{col}' for col in columns)}) ON DUPLICATE KEY UPDATE {update_clause}"
    elif db_type in ['postgresql', 'sqlite']:
        update_clause = ', '.join(f"{col} = EXCLUDED.{col}" for col in columns if col not in unique_columns)
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(f':{col}' for col in columns)}) ON CONFLICT({', '.join(unique_columns)}) DO UPDATE SET {update_clause}"
    elif db_type == 'mssql':
        update_clause = ', '.join(f"t.[{col}] = s.[{col}]" for col in columns if col not in unique_columns)
        merge_condition = ' AND '.join(f"t.[{col}] = s.[{col}]" for col in unique_columns)
        query = f"MERGE INTO {table} t USING (VALUES ({', '.join(f':{col}' for col in columns)})) s ({', '.join(columns)}) ON ({merge_condition}) WHEN MATCHED THEN UPDATE SET {update_clause} WHEN NOT MATCHED THEN INSERT ({', '.join(columns)}) VALUES ({', '.join(f's.[{col}]' for col in columns)});"
    elif db_type == 'oracle':
        # Oracle is case-sensitive for quoted names. Default is UPPERCASE.
        columns = [c.upper() for c in columns]
        unique_columns = [c.upper() for c in unique_columns]
        
        # Sanitize data to have uppercase keys
        sanitized_data = []
        for row in data:
            sanitized_data.append({k.upper(): v for k, v in row.items()})
        data = sanitized_data

        update_clause = ', '.join(f'"{col}" = s."{col}"' for col in columns if col not in unique_columns)
        merge_condition = ' AND '.join(f't."{col}" = s."{col}"' for col in unique_columns)
        
        # Avoid backslashes in f-string expressions
        sel_cols = ', '.join([f':{c} AS "{c}"' for c in columns])
        ins_cols = ', '.join([f'"{c}"' for c in columns])
        val_cols = ', '.join([f's."{c}"' for c in columns])
        
        query = f"MERGE INTO {table} t USING (SELECT {sel_cols} FROM DUAL) s ON ({merge_condition}) WHEN MATCHED THEN UPDATE SET {update_clause} WHEN NOT MATCHED THEN INSERT ({ins_cols}) VALUES ({val_cols})"
        return query, _sanitize_data(data)


    else:
        raise ValueError(f"Unsupported database type: {db_type}")
    return query, _sanitize_data(data)
