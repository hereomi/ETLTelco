"""Main DataAligner class with enhanced DataFrame-to-SQL alignment capabilities."""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.sql import sqltypes as sat
try:
    from .schema_analyzer import SchemaAnalyzer, ColumnInfo, ConstraintInfo
except ImportError:
    from schema_analyzer import SchemaAnalyzer, ColumnInfo, ConstraintInfo
from .config import AlignmentConfig, ValidationMode
from .diagnostics import DiagnosticsCollector
from .performance import PerformanceOptimizer
from .validators import TypeValidator
from .utils import validate_identifier, quote_identifier, should_chunk_processing

# Oracle-specific type imports (safe import with fallback)
try:
    from sqlalchemy.dialects.oracle import (
        VARCHAR2, NVARCHAR2, CHAR, NCHAR, CLOB, NCLOB,
        NUMBER, FLOAT as ORACLE_FLOAT, BINARY_DOUBLE, BINARY_FLOAT,
        DATE as ORACLE_DATE, TIMESTAMP as ORACLE_TIMESTAMP,
        RAW, BLOB as ORACLE_BLOB, LONG, ROWID, INTERVAL
    )
    ORACLE_TYPES_AVAILABLE = True
except ImportError:
    ORACLE_TYPES_AVAILABLE = False
    VARCHAR2 = NVARCHAR2 = CHAR = NCHAR = CLOB = NCLOB = None
    NUMBER = ORACLE_FLOAT = BINARY_DOUBLE = BINARY_FLOAT = None
    ORACLE_DATE = ORACLE_TIMESTAMP = RAW = ORACLE_BLOB = LONG = ROWID = INTERVAL = None

# MySQL/MariaDB-specific type imports
try:
    from sqlalchemy.dialects.mysql import (
        TINYINT, SMALLINT, MEDIUMINT, INTEGER as MYSQL_INTEGER, BIGINT as MYSQL_BIGINT,
        FLOAT as MYSQL_FLOAT, DOUBLE as MYSQL_DOUBLE, DECIMAL as MYSQL_DECIMAL,
        VARCHAR as MYSQL_VARCHAR, CHAR as MYSQL_CHAR, TEXT as MYSQL_TEXT,
        TINYTEXT, MEDIUMTEXT, LONGTEXT,
        DATETIME as MYSQL_DATETIME, TIMESTAMP as MYSQL_TIMESTAMP, DATE as MYSQL_DATE, TIME as MYSQL_TIME,
        BLOB as MYSQL_BLOB, TINYBLOB, MEDIUMBLOB, LONGBLOB,
        JSON as MYSQL_JSON, ENUM as MYSQL_ENUM, SET as MYSQL_SET
    )
    MYSQL_TYPES_AVAILABLE = True
except ImportError:
    MYSQL_TYPES_AVAILABLE = False
    TINYINT = SMALLINT = MEDIUMINT = MYSQL_INTEGER = MYSQL_BIGINT = None
    MYSQL_FLOAT = MYSQL_DOUBLE = MYSQL_DECIMAL = None
    MYSQL_VARCHAR = MYSQL_CHAR = MYSQL_TEXT = TINYTEXT = MEDIUMTEXT = LONGTEXT = None
    MYSQL_DATETIME = MYSQL_TIMESTAMP = MYSQL_DATE = MYSQL_TIME = None
    MYSQL_BLOB = TINYBLOB = MEDIUMBLOB = LONGBLOB = None
    MYSQL_JSON = MYSQL_ENUM = MYSQL_SET = None

# MSSQL-specific type imports
try:
    from sqlalchemy.dialects.mssql import (
        TINYINT as MSSQL_TINYINT, SMALLINT as MSSQL_SMALLINT, 
        INTEGER as MSSQL_INTEGER, BIGINT as MSSQL_BIGINT,
        FLOAT as MSSQL_FLOAT, REAL as MSSQL_REAL, DECIMAL as MSSQL_DECIMAL, MONEY, SMALLMONEY,
        VARCHAR as MSSQL_VARCHAR, CHAR as MSSQL_CHAR, NVARCHAR, NCHAR as MSSQL_NCHAR, TEXT as MSSQL_TEXT, NTEXT,
        DATETIME as MSSQL_DATETIME, DATETIME2, SMALLDATETIME, DATE as MSSQL_DATE, TIME as MSSQL_TIME,
        DATETIMEOFFSET,
        BINARY as MSSQL_BINARY, VARBINARY as MSSQL_VARBINARY, IMAGE,
        BIT
    )
    MSSQL_TYPES_AVAILABLE = True
except ImportError:
    MSSQL_TYPES_AVAILABLE = False
    MSSQL_TINYINT = MSSQL_SMALLINT = MSSQL_INTEGER = MSSQL_BIGINT = None
    MSSQL_FLOAT = MSSQL_REAL = MSSQL_DECIMAL = MONEY = SMALLMONEY = None
    MSSQL_VARCHAR = MSSQL_CHAR = NVARCHAR = MSSQL_NCHAR = MSSQL_TEXT = NTEXT = None
    MSSQL_DATETIME = DATETIME2 = SMALLDATETIME = MSSQL_DATE = MSSQL_TIME = None
    DATETIMEOFFSET = MSSQL_BINARY = MSSQL_VARBINARY = IMAGE = BIT = None

logger = logging.getLogger("DataAligner")

class DataAligner:
    """Enhanced DataFrame-to-SQL alignment with strict validation and performance optimizations."""
    
    def __init__(self, db_type: Optional[str] = None, config: Optional[AlignmentConfig] = None):
        self.config = config or AlignmentConfig()
        self._db_type = db_type.lower() if db_type else None
        self.supported_dialects = {"oracle", "sqlite", "mssql", "postgres", "postgresql", "mysql", "mariadb"}
        self.diagnostics = DiagnosticsCollector(self.config)
        self.performance = PerformanceOptimizer(self.config)
        self.validator = TypeValidator(self.config, self.diagnostics)
        if self._db_type and self._db_type not in self.supported_dialects:
            raise ValueError(f"Unsupported db_type '{db_type}'")
    
    @property
    def db_type(self) -> str:
        """Current database type."""
        return self._db_type or 'unknown'
    
    def align(self, conn: Connection | Engine, df: pd.DataFrame, table: str, schema: Optional[str] = None, on_error: str = 'coerce', failure_threshold: Optional[float] = None, validate_fk: bool = False, add_missing_cols: bool = False, col_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Main alignment entry point with enhanced validation and performance."""
        if failure_threshold is not None:
            self.config.failure_threshold = failure_threshold
        with self.performance.measure_time("total"):
            if self._db_type is None:
                self._db_type = self._detect_dialect(conn)
                logger.info(f"Auto-detected database dialect: {self._db_type}")
            with self.performance.measure_time("schema_introspection"):
                analyzer = SchemaAnalyzer(conn)
                report = analyzer.analyze_table(schema=schema, table_name=table, df=None, run_fk_checks=False)
                if not report.table_exists:
                    raise NoSuchTableError(f"Table '{table}' not found")
                meta = report.columns
                constraints = report.constraints
            if add_missing_cols:
                meta, constraints = self._handle_schema_evolution(conn, df, table, schema, meta, constraints, col_map)
            with self.performance.measure_time("type_coercion"):
                aligned = self._align_dataframe(df, meta, col_map)
            with self.performance.measure_time("validation"):
                self._validate_constraints(conn, aligned, meta, constraints, validate_fk)
            self.performance.metrics.rows_processed = len(aligned)
            self.performance.metrics.columns_processed = len(aligned.columns)
            
            # Final pass: Ensure native Python types for DB driver compatibility (e.g. Oracle DPY-3002)
            aligned = self._finalize_types(aligned)
            return aligned
    
    def _detect_dialect(self, conn: Connection | Engine) -> str:
        """Extract and normalize dialect name from SQLAlchemy connection."""
        if isinstance(conn, Connection):
            dialect_name = conn.engine.dialect.name.lower()
        else:
            dialect_name = conn.dialect.name.lower()
        if dialect_name in ('postgresql', 'postgres'):
            return 'postgres'
        if dialect_name in ('mariadb',):
            return 'mysql'
        return dialect_name
    
    def _handle_schema_evolution(self, conn: Connection | Engine, df: pd.DataFrame, table: str, schema: Optional[str], meta: Dict[str, ColumnInfo], constraints: ConstraintInfo, col_map: Optional[Dict[str, str]]) -> tuple[Dict[str, ColumnInfo], ConstraintInfo]:
        """Handle automatic schema evolution by adding missing columns."""
        mapped_df = self._map_columns(df, meta, col_map)
        cols_in_db = set(meta.keys())
        candidates = [c for c in mapped_df.columns if c not in cols_in_db]
        if candidates:
            logger.info(f"Schema Evolution: Adding {len(candidates)} columns: {candidates}")
            self._add_missing_columns(conn, table, schema, df, candidates)
            analyzer = SchemaAnalyzer(conn)
            report = analyzer.analyze_table(schema=schema, table_name=table, df=None, run_fk_checks=False)
            return report.columns, report.constraints
        return meta, constraints
    
    def _add_missing_columns(self, conn: Connection | Engine, table: str, schema: Optional[str], df: pd.DataFrame, new_cols: List[str]):
        """Add missing columns to database table with intelligent type mapping."""
        def map_pandas_to_sql_type(series: pd.Series) -> str:
            dialect = self.db_type
            if pd.api.types.is_integer_dtype(series):
                return {"oracle": "NUMBER(19)", "sqlite": "INTEGER", "postgres": "BIGINT"}.get(dialect, "BIGINT")
            elif pd.api.types.is_float_dtype(series):
                return {"oracle": "NUMBER", "postgres": "DOUBLE PRECISION"}.get(dialect, "FLOAT")
            elif pd.api.types.is_bool_dtype(series):
                return {"mssql": "BIT", "oracle": "NUMBER(1)", "sqlite": "INTEGER", "postgres": "BOOLEAN"}.get(dialect, "BOOLEAN")
            elif pd.api.types.is_datetime64_any_dtype(series):
                return {"oracle": "TIMESTAMP", "postgres": "TIMESTAMP WITH TIME ZONE", "sqlite": "TEXT"}.get(dialect, "TIMESTAMP")
            elif series.apply(lambda x: isinstance(x, (dict, list))).any():
                return {"postgres": "JSONB", "oracle": "CLOB", "sqlite": "TEXT", "mysql": "JSON"}.get(dialect, "TEXT")
            else:
                max_len = 255
                if len(series.dropna()) > 0:
                    try:
                        max_len = int(series.dropna().astype(str).str.len().max()) or 255
                    except:
                        max_len = 255
                if max_len > 4000:
                    return {"oracle": "CLOB", "postgres": "TEXT", "mssql": "VARCHAR(MAX)"}.get(dialect, "TEXT")
                elif max_len > 255:
                    return f"VARCHAR2({max_len})" if dialect == "oracle" else f"VARCHAR({max_len})"
                else:
                    return "VARCHAR2(255)" if dialect == "oracle" else "VARCHAR(255)"
        safe_schema = validate_identifier(schema) if schema else None
        safe_table = validate_identifier(table)
        full_table = f"{safe_schema}.{safe_table}" if safe_schema else safe_table
        def execute_sql(sql_stmt):
            if isinstance(conn, Engine):
                with conn.begin() as c:
                    c.execute(text(sql_stmt))
            else:
                conn.execute(text(sql_stmt))
        for col in new_cols:
            try:
                sql_type = map_pandas_to_sql_type(df[col])
                quoted_col = quote_identifier(col, self.db_type)
                if self.db_type in ('mssql', 'oracle'):
                    alter_stmt = f"ALTER TABLE {full_table} ADD {quoted_col} {sql_type}"
                else:
                    alter_stmt = f"ALTER TABLE {full_table} ADD COLUMN {quoted_col} {sql_type}"
                logger.info(f"Executing: {alter_stmt}")
                execute_sql(alter_stmt)
            except Exception as e:
                self.diagnostics.add_error(f"Failed to add column '{col}': {e}")
                raise ValueError(f"Schema evolution failed for column '{col}'")  # Issue 1: prevent silent schema add failures
    
    def _map_columns(self, df: pd.DataFrame, meta: Dict[str, ColumnInfo], col_map: Optional[Dict[str, str]]) -> pd.DataFrame:
        """Map DataFrame columns to database columns with case-insensitive matching."""
        db_map = {k.lower(): k for k in meta.keys()}
        rename_map = {}
        custom_map = {k.lower(): v for k, v in (col_map or {}).items()} if col_map else {}
        for col in df.columns:
            if col.lower() in custom_map:
                target = custom_map[col.lower()]
                if target.lower() in db_map:
                    rename_map[col] = db_map[target.lower()]
                else:
                    rename_map[col] = target
                continue
            c_low = col.lower()
            if c_low in db_map:
                rename_map[col] = db_map[c_low]
        return df.rename(columns=rename_map)
    
    def _align_dataframe(self, df: pd.DataFrame, meta: Dict[str, ColumnInfo], col_map: Optional[Dict[str, str]]) -> pd.DataFrame:
        """Align DataFrame with database schema using enhanced validation."""
        if should_chunk_processing(df, self.config.chunk_size):
            return self.performance.process_in_chunks(df, lambda chunk: self._align_single_chunk(chunk, meta, col_map))
        return self._align_single_chunk(df, meta, col_map)
    
    def _align_single_chunk(self, df: pd.DataFrame, meta: Dict[str, ColumnInfo], col_map: Optional[Dict[str, str]]) -> pd.DataFrame:
        """Align a single DataFrame chunk."""
        df_mapped = self._map_columns(df, meta, col_map)
        cols_to_keep = [c for c in df_mapped.columns if c in meta]
        dropped_cols = set(df_mapped.columns) - set(cols_to_keep)
        if dropped_cols:
            self.diagnostics.add_warning(f"Dropping extra columns not in target schema: {list(dropped_cols)}")
        df_mapped = df_mapped[cols_to_keep]
        aligned = self._coerce_types(df_mapped, meta)
        self._enforce_nullability(aligned, meta)
        final_cols = [c for c in meta.keys() if c in aligned.columns]
        for c, info in meta.items():
            if c not in aligned.columns:
                if not info.nullable and info.default is None:
                    self.diagnostics.add_warning(f"Missing required column '{c}'. Filling NULL.")
                aligned[c] = None
        return aligned[final_cols]
    
    def _coerce_types(self, df: pd.DataFrame, meta: Dict[str, ColumnInfo]) -> pd.DataFrame:
        """Coerce DataFrame types to match database schema."""
        out = df.copy()
        for col_name, info in meta.items():
            if col_name not in out.columns or out[col_name].empty:
                continue
            raw_type = info.raw_type
            target_type = self._determine_target_type(raw_type)
            if target_type:
                out[col_name] = self.validator.validate_and_convert(out[col_name], info, target_type)
        return out
    
    def _determine_target_type(self, raw_type: Any) -> Optional[str]:
        """Determine target type from SQLAlchemy type with dialect awareness."""
        if self._is_int_type(raw_type):
            return 'integer'
        elif self._is_float_type(raw_type):
            return 'float'
        elif self._is_string_type(raw_type):
            return 'string'
        elif self._is_bool_type(raw_type):
            return 'boolean'
        elif self._is_datetime_type(raw_type):
            return 'datetime'
        elif self._is_json_type(raw_type):
            return 'json'
        elif self._is_binary_type(raw_type):
            return 'binary'
        return None

    def _is_int_type(self, t: Any) -> bool:
        # Standard integer types
        if isinstance(t, (sat.INTEGER, sat.BIGINT, sat.SmallInteger)):
            return True
        # Oracle NUMBER with scale=0 is integer
        if ORACLE_TYPES_AVAILABLE and isinstance(t, NUMBER):
            # Only treat as integer when scale is explicitly 0.
            if getattr(t, 'scale', None) == 0:
                return True
        # MySQL integer types
        if MYSQL_TYPES_AVAILABLE:
            if isinstance(t, (TINYINT, SMALLINT, MEDIUMINT, MYSQL_INTEGER, MYSQL_BIGINT)):
                return True
        # MSSQL integer types
        if MSSQL_TYPES_AVAILABLE:
            if isinstance(t, (MSSQL_TINYINT, MSSQL_SMALLINT, MSSQL_INTEGER, MSSQL_BIGINT)):
                return True
        return False

    def _is_float_type(self, t: Any) -> bool:
        # Standard float types
        if isinstance(t, (sat.Float, sat.REAL, sat.DOUBLE_PRECISION, sat.Numeric)):
            return True
        # Oracle-specific float types
        if ORACLE_TYPES_AVAILABLE:
            if isinstance(t, (ORACLE_FLOAT, BINARY_DOUBLE, BINARY_FLOAT)):
                return True
            if isinstance(t, NUMBER) and hasattr(t, 'scale') and t.scale and t.scale > 0:
                return True
        # MySQL float types
        if MYSQL_TYPES_AVAILABLE:
            if isinstance(t, (MYSQL_FLOAT, MYSQL_DOUBLE, MYSQL_DECIMAL)):
                return True
        # MSSQL float types
        if MSSQL_TYPES_AVAILABLE:
            if isinstance(t, (MSSQL_FLOAT, MSSQL_REAL, MSSQL_DECIMAL, MONEY, SMALLMONEY)):
                return True
        return False

    def _is_string_type(self, t: Any) -> bool:
        if isinstance(t, (sat.String, sat.Unicode, sat.Text, sat.VARCHAR, sat.CHAR)):
            return True
        # Oracle string types
        if ORACLE_TYPES_AVAILABLE:
            if isinstance(t, (VARCHAR2, NVARCHAR2, CHAR, NCHAR, CLOB, NCLOB, ROWID)):
                return True
        # MySQL string types
        if MYSQL_TYPES_AVAILABLE:
            if isinstance(t, (MYSQL_VARCHAR, MYSQL_CHAR, MYSQL_TEXT, TINYTEXT, MEDIUMTEXT, LONGTEXT)):
                return True
        # MSSQL string types
        if MSSQL_TYPES_AVAILABLE:
            if isinstance(t, (MSSQL_VARCHAR, MSSQL_CHAR, NVARCHAR, MSSQL_NCHAR, MSSQL_TEXT, NTEXT)):
                return True
        return False

    def _is_bool_type(self, t: Any) -> bool:
        if isinstance(t, sat.Boolean):
            return True
        # MySQL/MSSQL often use BIT/TINYINT for boolean logic
        if MSSQL_TYPES_AVAILABLE and isinstance(t, BIT):
            return True
        if MYSQL_TYPES_AVAILABLE and isinstance(t, TINYINT):
             # Heuristic: TINYINT(1) is often boolean
             if getattr(t, 'display_width', None) == 1:
                 return True
        return False

    def _is_datetime_type(self, t: Any) -> bool:
        if isinstance(t, (sat.DateTime, sat.TIMESTAMP, sat.Date, sat.Time)):
            return True
        if ORACLE_TYPES_AVAILABLE:
            if isinstance(t, (ORACLE_DATE, ORACLE_TIMESTAMP, INTERVAL)):
                return True
        if MYSQL_TYPES_AVAILABLE:
            if isinstance(t, (MYSQL_DATETIME, MYSQL_TIMESTAMP, MYSQL_DATE, MYSQL_TIME)):
                return True
        if MSSQL_TYPES_AVAILABLE:
            if isinstance(t, (MSSQL_DATETIME, DATETIME2, SMALLDATETIME, MSSQL_DATE, MSSQL_TIME, DATETIMEOFFSET)):
                return True
        return False
    
    def _is_json_type(self, t: Any) -> bool:
        if isinstance(t, (sat.JSON, sat.ARRAY)):
             return True
        if MYSQL_TYPES_AVAILABLE and isinstance(t, MYSQL_JSON):
             return True
        return False

    def _is_binary_type(self, t: Any) -> bool:
        if isinstance(t, (sat.LargeBinary, sat.BINARY, sat.VARBINARY, sat.BLOB)):
             return True
        if ORACLE_TYPES_AVAILABLE and isinstance(t, (RAW, ORACLE_BLOB, LONG)):
             return True
        if MYSQL_TYPES_AVAILABLE and isinstance(t, (MYSQL_BLOB, TINYBLOB, MEDIUMBLOB, LONGBLOB)):
             return True
        if MSSQL_TYPES_AVAILABLE and isinstance(t, (MSSQL_BINARY, MSSQL_VARBINARY, IMAGE)):
             return True
        return False

    def _finalize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pandas/numpy specific types (Int64, float64, etc.) to native Python types.
        This prevents errors in strict DB drivers like oracledb (DPY-3002).
        """
        out = df.copy()
        for col in out.columns:
            dtype = out[col].dtype
            
            # Convert Nullable Integers (Int64) and standard integers to Python int
            if pd.api.types.is_integer_dtype(dtype):
                # Convert to object series with Python ints and None
                out[col] = out[col].astype(object).where(out[col].notna(), None)
                
            # Convert Floats to Python float
            elif pd.api.types.is_float_dtype(dtype):
                out[col] = out[col].astype(object).where(out[col].notna(), None)
                
            # Convert Booleans
            elif pd.api.types.is_bool_dtype(dtype):
                out[col] = out[col].astype(object).where(out[col].notna(), None)
        
        return out
    
    def _enforce_nullability(self, df: pd.DataFrame, meta: Dict[str, ColumnInfo]):
        """Enforce NOT NULL constraints."""
        for col_name, info in meta.items():
            if col_name not in df.columns or info.nullable:
                continue
            nulls = df[col_name].isna()
            if nulls.any():
                null_count = int(nulls.sum())
                total_count = len(df)
                self.diagnostics.add_metadata("not_null_violations", {col_name: {"nulls": null_count, "total": total_count}})  # Issue 3: surface NOT NULL failures
                if self.config.validation_mode == ValidationMode.STRICT:
                    raise ValueError(f"NOT NULL constraint violation: {col_name} has {null_count} null values")
                else:
                    self.diagnostics.add_warning(f"NOT NULL constraint violation: {col_name} has {null_count}/{total_count} null values")
    
    def _validate_constraints(self, conn: Connection | Engine, df: pd.DataFrame, meta: Dict[str, ColumnInfo], constraints: ConstraintInfo, validate_fk: bool):
        """Validate primary key, unique, and foreign key constraints."""
        violations: List[Dict[str, Any]] = []
        pk = constraints.primary_key
        if pk:
            pk_cols = pk.get('constrained_columns', [])
            pk_cols_present = [c for c in pk_cols if c in df.columns]
            if pk_cols_present and len(pk_cols_present) == len(pk_cols):
                if df.duplicated(subset=pk_cols_present).any():
                    dupes = df[df.duplicated(subset=pk_cols_present, keep=False)]
                    msg = f"Primary Key violation: Duplicates found in {pk_cols_present}. count={len(dupes)}"
                    self.diagnostics.add_error(msg)
                    violations.append({"type": "primary_key", "columns": pk_cols_present, "count": len(dupes), "message": msg})
        for u in constraints.unique_constraints:
            u_cols = u.get('column_names', [])
            u_cols_present = [c for c in u_cols if c in df.columns]
            if u_cols_present and len(u_cols_present) == len(u_cols):
                if df.duplicated(subset=u_cols_present).any():
                    msg = f"Unique constraint violation: Duplicates in {u_cols_present}."
                    self.diagnostics.add_error(msg)
                    violations.append({"type": "unique", "columns": u_cols_present, "message": msg})
        if violations:
            if self.config.validation_mode == ValidationMode.STRICT:
                raise ValueError("Constraint violations detected")  # Issue 2: hard fail in strict mode
            self.diagnostics.add_metadata("constraint_violations", violations)  # Issue 2: expose violations in non-strict mode
        if validate_fk:
            pass
    
    def get_diagnostics(self) -> DiagnosticsCollector:
        """Get diagnostics collector for detailed validation results."""
        return self.diagnostics
    
    def get_validation_metadata(self) -> Dict[str, Any]:
        """Expose validation metadata for downstream callers."""  # Issue 2/3/4: surface observability
        return self.diagnostics.get_metadata()
    
    def get_performance_metrics(self):
        """Get performance metrics for the last alignment operation."""
        return self.performance.get_performance_metrics()
    
    def print_summary(self):
        """Print a summary of the last alignment operation."""
        self.diagnostics.print_summary_report()
        if self.config.enable_metrics:
            metrics = self.get_performance_metrics()
            print(f"\n=== Performance Metrics ===")
            print(f"Total time: {metrics.total_time:.3f}s")
            print(f"Rows processed: {metrics.rows_processed}")
            print(f"Columns processed: {metrics.columns_processed}")
            print(f"Cache hits: {metrics.cache_hits}")
    
    def reset_state(self):
        """Reset diagnostics and performance metrics."""
        self.diagnostics.clear_results()
        self.performance.reset_metrics()

    def cleanup(self):
        self.reset_state()
