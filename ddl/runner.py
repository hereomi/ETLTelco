from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from .create_ddl import df_ddl
# Ideally we would reuse logging, but for now we'll stick to print/basic logging to avoid circular deps or complexity
# If logging is needed, we can import it.

def run_ddl_and_load(
    conn_dict: Mapping[str, str],
    df: pd.DataFrame,
    table: str,
    schema_name: Optional[str] = None,
    pk: Optional[Union[str, List[str]]] = None,
    fk: Optional[List[Tuple[str, str, str]]] = None,
    unique: Optional[List[Union[str, List[str]]]] = None,
    autoincrement: Optional[Tuple[str, int]] = None,
    varchar_sizes: Optional[Dict[str, int]] = None,
    # options for df_ddl
    include_not_null: bool = False,
    object_sample_size: int = 5000,
    json_text_threshold: float = 0.85,
    use_ddl_io: bool = False, # New option to use the user's ddl_io utility
) -> Dict[str, Any]:
    """
    Generate DDL, execute it, and load data for multiple database connections.
    Ported from ddl_create.df_ddl_create.
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    if not table:
        raise ValueError("Table name is required")

    results: Dict[str, Any] = {"df_shape": df.shape, "servers": {}}
    os.makedirs("schema", exist_ok=True)

    for server_key, conn_str in conn_dict.items():
        # Identify dialect from connection string/key or engine
        # We'll create the engine first to be sure
        try:
            engine = create_engine(conn_str)
        except Exception as e:
            results["servers"][server_key] = {"status": "ERROR", "message": f"Engine creation failed: {e}"}
            continue
            
        dialect_name = engine.dialect.name.lower()
        print(f"\n--- Processing Database: {dialect_name.upper()} (Key: {server_key}) ---")

        # Initialize variables to avoid unbound errors
        processed_df, create_sql, constraint_sql, meta_schema = None, None, [], {}
        
        if use_ddl_io:
            try:
                from .ddl_io import ddl_via_pandas_internals
                print(f"  ➡️ Using ddl_io utility for {dialect_name}...")
                create_sql, processed_df, ddl_path, ddl_error = ddl_via_pandas_internals(df, table, dialect_name)
                constraint_sql = []
                meta_schema = {"method": f"ddl_io.{ddl_path}"}
                if ddl_error:
                    meta_schema["primary_error"] = ddl_error
            except Exception as e:
                msg = f"ddl_io Generation failed: {e}. Falling back to standard."
                print(f"⚠️ {msg}")
                use_ddl_io = False

        if not use_ddl_io:
            try:
                processed_df, create_sql, constraint_sql, meta_schema, sa_schema = df_ddl(
                    engine,
                    df,
                    table,
                    pk=pk,
                    fk=fk,
                    schema=schema_name,
                    include_not_null=include_not_null,
                    object_sample_size=object_sample_size,
                    json_text_threshold=json_text_threshold
                )
            except Exception as e:
                msg = f"DDL Generation failed: {e}"
                print(f"❌ FAIL ({msg})")
                results["servers"][server_key] = {"status": "DDL_GEN_FAILED", "message": msg}
                continue

        server_status: Dict[str, Any] = {"ddl_success": False, "data_load_success": False}
        
        # Execute DDL
        try:
            if create_sql is None:
                raise ValueError("No DDL generated for table")
                
            print(f"  ➡️ Attempting Table Creation (DDL) for {table}...", end=' ')
            with engine.begin() as conn:
                # Split multi-statement DDL for drivers that don't support it (e.g. cx_Oracle)
                # and wrap in text() for SQLAlchemy 2.0 compatibility
                for stmt in create_sql.split(';'):
                    if stmt.strip():
                        conn.execute(text(stmt))
                for c_stmt in constraint_sql:
                    for sub_stmt in c_stmt.split(';'):
                        if sub_stmt.strip():
                            conn.execute(text(sub_stmt))
            
            server_status["ddl_success"] = True
            print("✅ SUCCESS")
            
            # Load Data
            if processed_df is None:
                raise ValueError("No processed data available to load")

            print("  ➡️ Attempting Data Load...", end=' ')
            processed_df.to_sql(
                table,
                engine,
                schema=schema_name if dialect_name != 'sqlite' else None,
                if_exists='append',
                index=False,
                chunksize=1000, # Added chunksize to prevent parameter limit errors
                method='multi'
            )
            server_status["data_load_success"] = True
            print("✅ SUCCESS")

            # Save Meta
            json_filename = f"schema/{dialect_name}_{table}_schema.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(meta_schema, f, indent=2, default=str)
                
            results["servers"][server_key] = {
                **server_status,
                "status": "COMPLETED",
                "ddl_file": json_filename,
                "schema_metadata": meta_schema
            }

        except SQLAlchemyError as err:
            msg = f"SQLAlchemy Error: {err.__class__.__name__}. {str(err).splitlines()[0]}"
            print(f"❌ FAIL ({msg})")
            status = "DDL_FAILED" if not server_status["ddl_success"] else "DATA_LOAD_FAILED"
            results["servers"][server_key] = {
                **server_status,
                "status": status,
                "message": msg
            }
        except Exception as err:
            msg = f"Unexpected Error: {err}"
            print(f"❌ FAIL ({msg})")
            results["servers"][server_key] = {
                **server_status,
                "status": "UNEXPECTED_ERROR",
                "message": msg
            }

    return results
