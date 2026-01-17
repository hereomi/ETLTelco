from __future__ import annotations
import logging
import re
from typing import Optional, Set, Any
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

def normalize_db_type(dialect_name: str) -> str:
    """Normalize database dialect name"""
    name = dialect_name.lower()
    if name == 'postgresql':
        return 'postgres'
    return name

logger = logging.getLogger(__name__)

class SchemaManager:
    """Consolidated database schema inspection and management utility."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.refresh()

    @property
    def db_type(self) -> str:
        return normalize_db_type(self.engine.dialect.name)

    def refresh(self) -> None:
        """Refresh the SQLAlchemy inspector."""
        self._inspector = inspect(self.engine)

    # --- Inspection & Exploration ---

    def list_tables(self, schema: str = None, pattern: str = None) -> list:
        """Get list of tables, optionally filtered by regex pattern."""
        tables = self._inspector.get_table_names(schema=schema)
        if pattern:
            tables = [x for x in tables if re.search(pattern, x, re.I)]
        return tables

    def get_columns(self, table: str, schema: str = None) -> list:
        """Get raw column metadata for a table."""
        return self._inspector.get_columns(table, schema=schema)

    def get_column_info_map(self, table: str, schema: str = None) -> dict:
        """Get column info as dictionary keyed by column name."""
        columns = self.get_columns(table, schema)
        return {col['name']: col for col in columns}

    def find_column(self, col_pattern: str, schema: str = None) -> dict:
        """Find tables containing columns matching regex pattern."""
        res, pat = {}, re.compile(col_pattern, re.I)
        for t in self.list_tables(schema):
            matches = [c['name'] for c in self.get_columns(t, schema=schema) if pat.search(c['name'])]
            if matches:
                res[t] = matches
        return res

    def get_primary_keys(self, table: str, schema: str = None) -> list:
        """Get primary key columns for table."""
        pk_constraint = self._inspector.get_pk_constraint(table, schema=schema)
        return pk_constraint.get('constrained_columns', []) if pk_constraint else []

    def get_unique_constraints(self, table: str, schema: str = None) -> list:
        """Get unique constraint column sets for table."""
        unique_constraints = self._inspector.get_unique_constraints(table, schema=schema)
        return [uc.get('column_names', []) for uc in unique_constraints]

    def get_identity_columns(self, table: str, schema: str = None) -> Set[str]:
        """Detect identity/autoincrement columns."""
        is_oracle = self.db_type == 'oracle'
        try:
            cols = self.get_columns(table, schema=schema)
        except Exception:
            return set()
        
        identity_cols = set()
        for c in cols:
            if c.get('autoincrement', False):
                identity_cols.add(c['name'])
            elif is_oracle and 'identity' in str(c.get('default', '') or '').lower():
                identity_cols.add(c['name'])
        return identity_cols

    def get_table_details(self, table: str, schema: str = None) -> dict:
        """Deep inspection of table structure including keys, indexes, and constraints."""
        insp = self._inspector
        return {
            'columns': self.get_column_info_map(table, schema),
            'pk': self.get_primary_keys(table, schema),
            'fk': insp.get_foreign_keys(table, schema),
            'indexes': insp.get_indexes(table, schema),
            'constraints': insp.get_unique_constraints(table, schema),
            'identity': list(self.get_identity_columns(table, schema))
        }

    # --- Validation & Diffing ---

    def has_table(self, table: str, schema: Optional[str] = None) -> bool:
        """Check if table exists in current schema."""
        with self.engine.connect() as conn:
            return self.engine.dialect.has_table(conn, table, schema=schema)

    def validate_upsert_constraints(self, table: str, key_cols: list, schema: str = None) -> None:
        """Validate that PK or UNIQUE constraint exists on key columns (for supported DBs)."""
        if self.db_type not in ('postgres', 'mysql', 'sqlite'):
            return
            
        pk_cols = set(self.get_primary_keys(table, schema))
        unique_cons = [set(uc) for uc in self.get_unique_constraints(table, schema)]
        
        key_set = set(key_cols)
        if not (key_set <= pk_cols or any(key_set <= uc for uc in unique_cons)):
            raise ValueError(
                f"Upsert requires PK or UNIQUE constraint on {key_cols} for '{table}'. "
                f"Found PK: {list(pk_cols)}, UNIQUE constraints: {unique_cons}"
            )

    def compare_to_structure(self, table: str, structure: dict, schema: str = None) -> dict:
        """Compare database table against expected {col_name: type_str} structure."""
        curr = {c['name']: str(c['type']) for c in self.get_columns(table, schema=schema)}
        return {
            'missing_in_db': [k for k in structure if k not in curr],
            'extra_in_db': [k for k in curr if k not in structure],
            'type_mismatch': {
                k: f"{curr[k]} != {structure[k]}" 
                for k in structure 
                if k in curr and str(structure[k]).lower() not in str(curr[k]).lower()
            }
        }

    # --- Schema Modification (DDL) ---

    def add_column(self, table: str, col_name: str, col_type: str, schema: str = None) -> None:
        """Add a single column to an existing table."""
        q = self.engine.dialect.identifier_preparer.quote
        pre = f"{q(schema)}." if schema else ""
        fmt = "({0} {1})" if self.db_type == 'oracle' else "{0} {1}"
        sql = f"ALTER TABLE {pre}{q(table)} ADD {fmt.format(q(col_name), col_type)}"
        self.execute_ddl(sql, f"Added column {col_name} to {table}")

    def add_columns(self, table: str, new_columns: dict, schema: str = None) -> None:
        """Add multiple columns to a table (batch)."""
        for col_name, col_type in new_columns.items():
            self.add_column(table, col_name, col_type, schema)

    def alter_column_type(self, table: str, col_name: str, new_type: str, schema: str = None) -> None:
        """Alter column type across different database dialects."""
        q = self.engine.dialect.identifier_preparer.quote
        pre = f"{q(schema)}." if schema else ""
        db, tq, cq = self.db_type, q(table), q(col_name)
        
        if db == 'postgres':
            sql = f"ALTER TABLE {pre}{tq} ALTER COLUMN {cq} TYPE {new_type}"
        elif db == 'mysql':
            sql = f"ALTER TABLE {pre}{tq} MODIFY COLUMN {cq} {new_type}"
        elif db == 'oracle':
            sql = f"ALTER TABLE {pre}{tq} MODIFY ({cq} {new_type})"
        elif db == 'mssql':
            sql = f"ALTER TABLE {pre}{tq} ALTER COLUMN {cq} {new_type}"
        elif db == 'sqlite':
            logger.warning("Skipping column type alter for SQLite (limited support)")
            return
        else:
            sql = f"ALTER TABLE {pre}{tq} MODIFY {cq} {new_type}"
            
        self.execute_ddl(sql, f"Altered column {col_name} in {table} to {new_type}")

    def drop_table(self, table: str, schema: str = None) -> None:
        """Drop a table if it exists."""
        q = self.engine.dialect.identifier_preparer.quote
        pre = f"{q(schema)}." if schema else ""
        sql = f"DROP TABLE {pre}{q(table)}"
        self.execute_ddl(sql, f"Dropped table {table}")

    def get_row_count(self, table: str, schema: str = None) -> int:
        """Get the row count of a table."""
        q = self.engine.dialect.identifier_preparer.quote
        pre = f"{q(schema)}." if schema else ""
        sql = f"SELECT COUNT(*) FROM {pre}{q(table)}"
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return result.scalar()

    def execute_ddl(self, sql: str, msg: str) -> None:
        """Execute DDL statement with logging and error handling. Retries with semicolon if failed."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql))
            logger.info(msg)
        except SQLAlchemyError as first_err:
            if sql.strip().endswith(';'):
                raise first_err
            
            # Retry strategy: append semicolon
            logger.warning(f"DDL failed without semicolon, retrying with semicolon... (Error: {first_err})")
            try:
                with self.engine.begin() as conn:
                    conn.execute(text(sql.strip() + ";"))
                logger.info(f"{msg} (Successful after semicolon retry)")
            except SQLAlchemyError as second_err:
                logger.error(f"DDL failed even with semicolon: {second_err}")
                raise second_err from first_err
