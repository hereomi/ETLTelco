import pandas as pd
from typing import Optional, Union, List, Tuple, Dict, Any
import json
from pathlib import Path
from sqlalchemy import create_engine
from utils.casting import cast_df
from ddl.create_ddl import df_ddl, DDLConfig
from utils.sanitization import sanitize_cols
from utils.data_profiler import get_pk, profile_dataframe
from utils.schema_manager import SchemaManager
from utils.crud_v2 import auto_insert as insert, auto_upsert as upsert, auto_update as crud_update
from utils.data_cleaner import quick_clean

class ETL:
    """
    Class-based ETL engine. Handles database state and provides methods 
    for full replacement (create_load), data addition (append), 
    and record updates (upsert).
    """
    def __init__(
        self, 
        db_uri: str, 
        schema_dir: Optional[Union[Path, str]] = None, 
        logging_enabled: bool = True,
        pk: Union[str, list] = 'verify',
        chunksize: int = 10000,
        add_missing_cols: bool = False,
        failure_threshold: int = 1,
        rename_column: bool = False,
        schema_aware_casting: bool = True,
        **cast_kwargs
    ):
        self.engine = create_engine(db_uri)
        self.dialect = self.engine.dialect.name
        # Standardize dialect for common mappings
        if self.dialect == 'postgresql': 
            self.dialect = 'postgresql'
        elif self.dialect == 'oracle':
            self.dialect = 'oracle'
            
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.logging_enabled = logging_enabled
        
        # Instance-level defaults
        self.default_pk = pk
        self.default_chunksize = chunksize
        self.default_add_missing_cols = add_missing_cols
        self.default_failure_threshold = failure_threshold
        self.default_rename_column = rename_column
        self.schema_aware_casting = schema_aware_casting
        
        # Internal components
        self.sm = SchemaManager(self.engine)
        
        # Casting configuration: Unified threshold logic
        self.cast_kwargs = cast_kwargs
        if 'infer_threshold' not in self.cast_kwargs:
            self.cast_kwargs['infer_threshold'] = self.default_failure_threshold
        
        # Initialize global logging state
        from utils import logger as etl_logger
        etl_logger.ENABLE_LOGGING = self.logging_enabled

    # --- PRIVATE HELPERS ---
    
    def _filter_cast_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filters kwargs to only include those valid for CastConfig."""
        valid_keys = {
            'use_transform', 'use_ml', 'infer_threshold', 'nan_threshold', 
            'max_sample_size', 'parallel', 'validate_conversions', 
            'max_null_increase', 'chunk_size', 'dtype'
        }
        return {k: v for k, v in kwargs.items() if k in valid_keys}

    def _save_ddl_schema(self, table: str, ddl: str, meta: Dict[str, Any]):
        """Saves generated DDL and metadata to the schema directory if configured."""
        if not self.schema_dir:
            return
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        ddl_file = self.schema_dir / f"{table}.txt"
        json_file = self.schema_dir / f"{table}.json"
        with ddl_file.open("w", encoding="utf-8") as f:
            f.write(ddl)
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"Saved DDL and Metadata to {self.schema_dir}")

    def _to_dataframe(self, source: Union[Path, str, pd.DataFrame, list]) -> pd.DataFrame:
        """Converts various source types (CSV, Excel, DF, List) to a pandas DataFrame."""
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, list):
            return pd.DataFrame(source)
        
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
        
        suffix = path.suffix.lower()
        if suffix == '.csv':
            return pd.read_csv(path)
        elif suffix in ('.xlsx', '.xls'):
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _process_pk(self, df: pd.DataFrame, table: str, pk_arg: Union[str, list]) -> list:
        """Processes and validates the primary key argument, prioritizing DB-side PKs."""
        if pk_arg == 'verify':
            print(f"Verifying existing PK for table '{table}'...")
            return self._derive_pk(df, table)
            
        if pk_arg == 'derive':
            print(f"Deriving PK for '{table}' (checking DB first)...")
            return self._derive_pk(df, table)
        
        if not pk_arg: return []
        
        if isinstance(pk_arg, str):
            pk_arg = [pk_arg]
        
        if not isinstance(pk_arg, list):
            raise ValueError("pk argument must be 'derive', 'verify', or a list of column names.")
        
        # Validate columns exist (Case-Insensitive)
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
        
        # Use the matched cased columns
        pk_arg = final_pk

        # Validate uniqueness
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

    def _derive_pk(self, df: pd.DataFrame, table: str) -> list:
        """Attempts to find a PK from the database first, then falls back to profiling."""
        sm = SchemaManager(self.engine)
        if sm.has_table(table):
            db_pk = sm.get_primary_keys(table)
            if db_pk:
                print(f"Detected existing database PK for '{table}': {db_pk}")
                # Case-Insensitive matching against DF columns
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

        # Fallback to profiling
        print(f"Profiling DataFrame to find optimal PK for '{table}'...")
        pk_info = get_pk(df, profile_dataframe(df))
        derived = pk_info[2]['components']
        if derived:
            print(f"Derived PK from data: {derived}")
            return derived
        return []

    def _get_schema_guided_dtypes(self, table: str) -> Dict[str, str]:
        """Queries the database for existing column types to guide the casting engine."""
        sm = SchemaManager(self.engine)
        if not sm.has_table(table):
            return {}
            
        print(f"Retrieving schema hints for '{table}'...")
        try:
            cols = sm.get_columns(table)
            hints = {}
            for c in cols:
                name = c['name']
                st = c['type']
                
                # Map SQLAlchemy types to cast_df hints
                from sqlalchemy.sql import sqltypes as sat
                if isinstance(st, (sat.Integer, sat.BIGINT, sat.SmallInteger)):
                    hints[name] = "integer"
                elif isinstance(st, (sat.Float, sat.Numeric, sat.REAL)):
                    hints[name] = "float"
                elif isinstance(st, (sat.Boolean)):
                    hints[name] = "boolean"
                elif isinstance(st, (sat.DateTime, sat.Date, sat.TIMESTAMP)):
                    hints[name] = "timestamp"
                elif isinstance(st, (sat.String, sat.Text, sat.Unicode)):
                    hints[name] = "string"
            return hints
        except Exception as e:
            print(f"Warning: Could not retrieve schema hints for '{table}': {e}")
            return {}

    def _load_and_sanitize_data(self, source: Union[Path, str, pd.DataFrame, list], table: str, pk_arg: Union[str, list], failure_threshold: Optional[int] = None, **kwargs):
        """Full data preparation: extraction, type casting, and PK validation/derivation."""
        df_raw = self._to_dataframe(source)
        # Apply robust cleaning to both headers and data
        df_raw = quick_clean(df_raw)
        
        ft = failure_threshold if failure_threshold is not None else self.default_failure_threshold
        current_cast_kwargs = self.cast_kwargs.copy()
        current_cast_kwargs.update(self._filter_cast_kwargs(kwargs)) # Overlay from batch process (e.g. dtype={})
        current_cast_kwargs['infer_threshold'] = ft
        
        # Schema-Guided Casting: Use existing DB types if available
        if self.schema_aware_casting:
            hints = self._get_schema_guided_dtypes(table)
            if hints:
                current_cast_kwargs['dtype'] = hints

        df_cast, semantic_meta = cast_df(df_raw, return_dtype_meta=True, **current_cast_kwargs)
        pk_cols = self._process_pk(df_cast, table, pk_arg)
        # Extract only the semantic types for SchemaAligner
        clean_meta = self._extract_semantic_meta(semantic_meta)
        return df_raw, df_cast, pk_cols, clean_meta

    def _extract_semantic_meta(self, meta: Dict[str, Any]) -> Dict[str, str]:
        """Extracts simple semantic type strings from the complex cast_df metadata."""
        return {col: info['object_semantic_type'] for col, info in meta.items() if 'object_semantic_type' in info}

    def _prepare_ddl_and_sync(
        self, 
        df_raw: pd.DataFrame, 
        df_cast: pd.DataFrame, 
        table: str, 
        pk_cols: list, 
        rename_column: bool, 
        source: Union[Path, str, pd.DataFrame, list],
        fk=None, unique=None, autoincrement=None, varchar_sizes=None
    ):
        """Generates DDL and syncs source file to disk if column renaming is active."""
        print(f"Generating DDL for {self.dialect}...")
        
        # New DDL library usage
        # Construct DDLConfig or pass kwargs. For now, we'll map existing args.
        # df_ddl signature: (engine, df, table_name, config=None, pk=None, fk=None, **options)
        
        # Create config object for better type safety, or pass valid options
        # We need to filter options that DDLConfig/df_ddl accepts vs what we have.
        # Current valid options from DDLConfig: schema, pk, fk, include_not_null, etc.
        
        ddl_options = {
            'pk': pk_cols,
            'fk': fk,
            # 'unique': unique, # ddl libs might handle unique differently or via constraints param? 
            # Looking at create_ddl.py, 'unique' isn't explicitly in DDLConfig but might be passed via **options if supported by dialect functions?
            # Creating DDLConfig instance is safer if we match fields.
            'varchar_limit': 10485760 # Default
        }
        
        # Since unique and autoincrement were supported in legacy, we pass them in options if the new lib supports them in kwargs
        if unique:
            ddl_options['unique'] = unique
        if autoincrement:
            ddl_options['autoincrement'] = autoincrement
        
        # The new df_ddl returns: processed_df, create_sql, constraint_sql, schema_dict, sqlalchemy_schema
        df_final, create_sql, constraint_sql, meta, sa_schema = df_ddl(
            self.engine,
            df_cast,
            table,
            **ddl_options
        )

        # Legacy expectation: ddl should be the full SQL string
        full_ddl = f"{create_sql}\n\n{constraint_sql}" if constraint_sql else create_sql

        # Sync CSV if names changed (df_final might have sanitized cols)
        if rename_column and isinstance(source, (Path, str)):
            path = Path(source)
            if path.suffix.lower() == '.csv' and not list(df_raw.columns) == list(df_final.columns):
                print(f"Column names changed during sanitization. Updating CSV: {path}")
                df_final.to_csv(path, index=False)
        
        return df_final, full_ddl, meta

    def _setup_db_table(self, table: str, ddl: str, drop_if_exists: bool):
        """Ensures the target table exists, dropping and recreating it if requested."""
        sm = SchemaManager(self.engine)
        if drop_if_exists and sm.has_table(table):
            print(f"Table '{table}' exists and drop_if_exists=True. Dropping...")
            try:
                sm.drop_table(table)
                sm.refresh()
            except Exception as e:
                print(f"Warning: Failed to drop table '{table}': {e}")

        if not sm.has_table(table):
            print(f"Table '{table}' not found in {self.dialect}. Creating...")
            try:
                sm.execute_ddl(ddl, f"Table '{table}' created successfully.")
                sm.refresh()
            except Exception as e:
                print(f"Error creating table: {e}")
                return sm, False
        else:
            print(f"Table '{table}' already exists.")
        return sm, True

    def _load_and_validate(
        self, 
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
        csize = chunksize if chunksize is not None else self.default_chunksize
        amc = add_missing_cols if add_missing_cols is not None else self.default_add_missing_cols
        ft = failure_threshold if failure_threshold is not None else self.default_failure_threshold

        print(f"Performing '{op_type}' on '{table}' (chunksize={csize})...")
        try:
            if op_type == 'upsert':
                # crud_v2 upsert signature: auto_upsert(engine, data, table, constrain, ...)
                res = upsert(self.engine, df, table, pk, chunk=csize, add_missing_cols=amc, failure_threshold=ft, semantic_meta=semantic_meta, trace_sql=trace_sql)
                rows_done = res.get('success', 0) if isinstance(res, dict) else res
            else:
                # crud_v2 insert signature: auto_insert(engine, data, table, ...)
                rows_done = insert(self.engine, df, table, chunk_size=csize, add_missing_cols=amc, failure_threshold=ft, semantic_meta=semantic_meta, trace_sql=trace_sql)
            
            print(f"Load complete. {rows_done} rows processed for '{table}'.")
            
            # Validation
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

    # --- PUBLIC API ---

    def create_load(
        self,
        source: Union[Path, str, pd.DataFrame, list],
        table: str,
        pk: Optional[Union[str, list]] = None,
        drop_if_exists: bool = True,
        rename_column: Optional[bool] = None,
        chunksize: Optional[int] = None,
        add_missing_cols: Optional[bool] = None,
        failure_threshold: Optional[int] = None,
        fk=None, unique=None, autoincrement=None, varchar_sizes=None,
        trace_sql: bool = False,
        **kwargs
    ):
        """Full Lifecycle: Extractions -> Sanitize -> DDL -> Drop/Create -> Insert."""
        res_rc = rename_column if rename_column is not None else self.default_rename_column

        df_raw, df_cast, pk_cols, semantic_meta = self._load_and_sanitize_data(source, table, pk, failure_threshold, **kwargs)
        df_final, ddl, meta = self._prepare_ddl_and_sync(df_raw, df_cast, table, pk_cols, res_rc, source, fk, unique, autoincrement, varchar_sizes)
        
        self._save_ddl_schema(table, ddl, meta)
        sm, setup_ok = self._setup_db_table(table, ddl, drop_if_exists)
        if not setup_ok: return False
        
        return self._load_and_validate(sm, table, df_final, 'insert', None, chunksize, add_missing_cols, failure_threshold, semantic_meta, trace_sql=trace_sql)

    def append(
        self,
        source: Union[Path, str, pd.DataFrame, list],
        table: str,
        rename_column: Optional[bool] = None,
        chunksize: Optional[int] = None,
        add_missing_cols: Optional[bool] = None,
        failure_threshold: Optional[int] = None,
        trace_sql: bool = False,
        **kwargs
    ):
        """Add data to an existing table. Skips DDL generation and table management."""
        res_rc = rename_column if rename_column is not None else self.default_rename_column

        df_raw = self._to_dataframe(source)
        
        # Consistent Casting
        ft = failure_threshold if failure_threshold is not None else self.default_failure_threshold
        current_cast_kwargs = self.cast_kwargs.copy()
        current_cast_kwargs.update(self._filter_cast_kwargs(kwargs)) # Overlay filtered dtype/etc from batch config
        current_cast_kwargs['infer_threshold'] = ft
        
        # Schema-Guided Casting
        if self.schema_aware_casting:
            hints = self._get_schema_guided_dtypes(table)
            if hints:
                current_cast_kwargs['dtype'] = hints

        df_cast, semantic_meta = cast_df(df_raw, return_dtype_meta=True, **current_cast_kwargs)

        clean_meta = self._extract_semantic_meta(semantic_meta)
        
        df_final = sanitize_cols(df_cast, dialect=self.dialect) if res_rc else df_cast
        
        sm = SchemaManager(self.engine)
        if not sm.has_table(table):
            print(f"Error: Table '{table}' does not exist. Use create_load() first.")
            return False
            
        return self._load_and_validate(sm, table, df_final, 'insert', None, chunksize, add_missing_cols, failure_threshold, clean_meta, trace_sql=trace_sql)

    def upsert(
        self,
        source: Union[Path, str, pd.DataFrame, list],
        table: str,
        pk: Optional[Union[str, list]] = None,
        rename_column: Optional[bool] = None,
        chunksize: Optional[int] = None,
        add_missing_cols: Optional[bool] = None,
        failure_threshold: Optional[int] = None,
        trace_sql: bool = False,
        **kwargs
    ):
        """Update existing records or insert new ones based on the primary key."""
        res_pk = pk if pk is not None else self.default_pk
        res_rc = rename_column if rename_column is not None else self.default_rename_column

        df_raw, df_cast, pk_cols, semantic_meta = self._load_and_sanitize_data(source, table, res_pk, failure_threshold, **kwargs)
        df_final = sanitize_cols(df_cast, dialect=self.dialect) if res_rc else df_cast
        
        sm = SchemaManager(self.engine)
        if not sm.has_table(table):
            print(f"Error: Table '{table}' does not exist for upsert. Use create_load() first.")
            return False
            
        return self._load_and_validate(sm, table, df_final, 'upsert', pk_cols, chunksize, add_missing_cols, failure_threshold, semantic_meta, trace_sql=trace_sql)

    def update(
        self,
        source: Union[Path, str, pd.DataFrame, list],
        table: str,
        where: list,
        expression: Optional[str] = None,
        rename_column: Optional[bool] = None,
        add_missing_cols: Optional[bool] = None,
        failure_threshold: Optional[int] = None,
        trace_sql: bool = False
    ):
        """Update records based on a specific where condition."""
        res_rc = rename_column if rename_column is not None else self.default_rename_column
        amc = add_missing_cols if add_missing_cols is not None else self.default_add_missing_cols
        ft = failure_threshold if failure_threshold is not None else self.default_failure_threshold

        df_raw = self._to_dataframe(source)
        # Apply robust cleaning
        df_raw = quick_clean(df_raw)
        
        # Consistent Casting
        current_cast_kwargs = self.cast_kwargs.copy()
        current_cast_kwargs['infer_threshold'] = ft
        
        # Schema-Guided Casting
        if self.schema_aware_casting:
            hints = self._get_schema_guided_dtypes(table)
            if hints:
                current_cast_kwargs['dtype'] = hints

        df_cast, semantic_meta = cast_df(df_raw, return_dtype_meta=True, **current_cast_kwargs)
        clean_meta = self._extract_semantic_meta(semantic_meta)
        
        df_final = sanitize_cols(df_cast, dialect=self.dialect) if res_rc else df_cast
        
        return crud_update(self.engine, table, df_final, where, expression, add_missing_cols=amc, failure_threshold=ft, semantic_meta=clean_meta, trace_sql=trace_sql)

    def auto_etl(
        self,
        source: Union[Path, str, pd.DataFrame, list],
        table: str,
        pk: Optional[Union[str, list]] = None,
        trace_sql: bool = False,
        **kwargs
    ):
        """
        Smart dispatcher: Choose between create_load, upsert, or append based on state.
        """
        res_pk = pk if pk is not None else self.default_pk

        sm = SchemaManager(self.engine)
        exists = sm.has_table(table)
        
        if not exists:
            print(f"AutoETL: Table '{table}' not found. Redirecting to create_load.")
            return self.create_load(source, table, pk=res_pk, trace_sql=trace_sql, **kwargs)
        
        # If exists, we need to decide between upsert and append
        try:
            # Pass kwargs (like failure_threshold, dtype) to ensure consistent detection
            df_raw, df_cast, pk_cols, semantic_meta = self._load_and_sanitize_data(source, table, res_pk, kwargs.get('failure_threshold'), **kwargs)
            if pk_cols:
                print(f"AutoETL: Table '{table}' exists. Performing upsert using keys: {pk_cols}")
                return self.upsert(source, table, pk=res_pk, trace_sql=trace_sql, **kwargs)
        except Exception as e:
            print(f"AutoETL: PK detection or preparation failed: {e}")
            pass
            
        print(f"AutoETL: Table '{table}' exists but no valid PK for upsert. Redirecting to append.")
        return self.append(source, table, trace_sql=trace_sql, **kwargs)

# --- BACKWARD COMPATIBILITY ---

def run_etl(
    source: Union[Path, str, pd.DataFrame, list],
    table: str,
    db_uri: str,
    pk: Optional[Union[str, list]] = None,
    schema_dir: Optional[Path] = None,
    logging_enabled: bool = True,
    **kwargs
):
    """Unified entry point using the smart auto_etl dispatcher."""
    etl = ETL(db_uri, schema_dir=schema_dir, logging_enabled=logging_enabled, **kwargs)
    return etl.auto_etl(source, table, pk=pk, **kwargs)
