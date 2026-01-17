#!/usr/bin/env python
"""
ETL Schema Evolution Test Suite

A flexible test harness for validating ETL engine behavior with any CSV file.
Supports command-line arguments for customization.

Usage:
    python test_etl_evolution.py                           # Use defaults
    python test_etl_evolution.py csv/mydata.csv            # Test specific CSV
    python test_etl_evolution.py csv/mydata.csv my_table   # Specify table name
    python test_etl_evolution.py --pk=id,name              # Specify PK columns
    python test_etl_evolution.py --subset-cols=3           # Use first N columns for Phase 1
    python test_etl_evolution.py --subset-rows=10          # Use first N rows for Phase 1
    python test_etl_evolution.py --all-tests               # Run all test cases
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from sqlalchemy import text

from etl import ETL
from utils.schema_manager import SchemaManager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_config(config_path: str = "config.yml") -> dict:
    """Load database configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def get_db_uri(config: dict) -> str:
    """Extract DB URI from config or use default."""
    if config:
        active = config.get("active_server", "oracle")
        servers = config.get("server", {})
        if active in servers:
            return servers[active]
    # Fallback default
    return "oracle+oracledb://akomi:PASSWORD_HERE@localhost:1521/?service_name=XEPDB1"


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def print_result(test_name: str, success: bool):
    """Print test result with visual indicator."""
    status = "[PASS]" if success else "[FAIL]"
    color_code = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"  {color_code}{status}{reset} - {test_name}")


class ETLTestRunner:
    """Test runner for ETL schema evolution scenarios."""
    
    def __init__(
        self,
        csv_path: Path,
        table_name: str,
        db_uri: str,
        pk: str = "derive",
        subset_cols: int = 3,
        subset_rows: int = 5,
        schema_dir: str = "schema",
        trace_sql: bool = False
    ):
        self.csv_path = csv_path
        self.table_name = table_name
        self.db_uri = db_uri
        self.pk = pk if pk != "derive" else pk
        self.subset_cols = subset_cols
        self.subset_rows = subset_rows
        self.schema_dir = schema_dir
        self.trace_sql = trace_sql
        
        self.etl: Optional[ETL] = None
        self.df_full: Optional[pd.DataFrame] = None
        self.results: dict = {}
    
    def setup(self) -> bool:
        """Initialize ETL engine and load CSV with explicit cleanup."""
        print_header(f"Setup: Loading {self.csv_path}")
        
        if not self.csv_path.exists():
            print(f"ERROR: CSV file not found: {self.csv_path}")
            return False
        
        try:
            # First pass: Cleanup
            print(f"  [CLEANUP] Dropping '{self.table_name}' and resetting engine...")
            temp_etl = ETL(self.db_uri)
            try:
                # Use the publicly accessible sm attribute
                temp_etl.sm.drop_table(self.table_name)
                print(f"  [CLEANUP] Table '{self.table_name}' dropped.")
            except Exception:
                pass
            finally:
                temp_etl.engine.dispose()
                print(f"  [CLEANUP] Engine disposed.")

            # Second pass: Actual setup
            self.etl = ETL(self.db_uri, schema_dir=self.schema_dir, logging_enabled=True)
            self.df_full = pd.read_csv(self.csv_path)
            
            print(f"  File: {self.csv_path}")
            print(f"  Shape: {self.df_full.shape}")
            print(f"  Columns: {list(self.df_full.columns)}")
            print(f"  DB Dialect: {self.etl.dialect}")
            return True
        except Exception as e:
            print(f"ERROR during setup: {e}")
            return False
    
    def test_create_load_subset(self) -> bool:
        """
        Test 1: Create table with subset of columns and rows.
        Validates: Table creation, DDL generation, initial data load.
        """
        test_name = "create_load with Subset Schema"
        print_header(f"Test: {test_name}")
        
        subset_cols = list(self.df_full.columns[:self.subset_cols])
        df_subset = self.df_full[subset_cols].head(self.subset_rows)
        
        print(f"  Using columns: {subset_cols}")
        print(f"  Using rows: {len(df_subset)}")
        
        try:
            success = self.etl.create_load(
                df_subset, 
                self.table_name, 
                pk=self.pk, 
                drop_if_exists=True
            )
            self.results[test_name] = success
            print_result(test_name, success)
            return success
        except Exception as e:
            print(f"  ERROR: {e}")
            self.results[test_name] = False
            print_result(test_name, False)
            return False
    
    def test_upsert_no_schema_evolution(self) -> bool:
        """
        Test 2: Upsert full data without schema evolution.
        Validates: Extra columns are dropped, existing columns updated.
        """
        test_name = "auto_etl Upsert (add_missing_cols=False)"
        print_header(f"Test: {test_name}")
        
        try:
            success = self.etl.auto_etl(
                self.df_full, 
                self.table_name, 
                pk=self.pk, 
                add_missing_cols=False
            )
            self.results[test_name] = success
            print_result(test_name, success)
            return success
        except Exception as e:
            print(f"  ERROR: {e}")
            self.results[test_name] = False
            print_result(test_name, False)
            return False
    
    def test_upsert_with_schema_evolution(self) -> bool:
        """
        Test 3: Upsert full data with schema evolution.
        Validates: Missing columns added, data merged correctly.
        """
        test_name = "auto_etl Upsert (add_missing_cols=True)"
        print_header(f"Test: {test_name}")
        
        try:
            success = self.etl.auto_etl(
                self.df_full, 
                self.table_name, 
                pk=self.pk, 
                add_missing_cols=True
            )
            self.results[test_name] = success
            print_result(test_name, success)
            return success
        except Exception as e:
            print(f"  ERROR: {e}")
            self.results[test_name] = False
            print_result(test_name, False)
            return False
    
    def test_append_data(self) -> bool:
        """
        Test 4: Append new data to existing table.
        Validates: Data addition without PK conflicts, row count increases.
        """
        test_name = "append New Data"
        print_header(f"Test: {test_name}")
        
        try:
            # Get current row count
            sm = SchemaManager(self.etl.engine)
            with self.etl.engine.connect() as conn:
                before_count = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}")).scalar()
            print(f"  Rows before append: {before_count}")
            
            # Create new data with different PK values
            df_new = self.df_full.copy()
            pk_col = self.df_full.columns[0]
            
            # Offset IDs to avoid conflicts
            if pd.api.types.is_numeric_dtype(df_new[pk_col]):
                max_id = df_new[pk_col].max()
                df_new[pk_col] = df_new[pk_col] + max_id + 1000
            else:
                df_new[pk_col] = df_new[pk_col].astype(str) + "_new"
            
            df_new = df_new.head(3)  # Just append 3 rows
            print(f"  Appending {len(df_new)} new rows")
            
            # Perform append (ignore internal validation return)
            self.etl.append(df_new, self.table_name, add_missing_cols=True)
            
            # Verify row count increased
            with self.etl.engine.connect() as conn:
                after_count = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}")).scalar()
            print(f"  Rows after append: {after_count}")
            
            expected_count = before_count + len(df_new)
            success = after_count == expected_count
            
            if success:
                print(f"  Append verified: {before_count} + {len(df_new)} = {after_count}")
            else:
                print(f"  Append failed: Expected {expected_count}, got {after_count}")
            
            self.results[test_name] = success
            print_result(test_name, success)
            return success
        except Exception as e:
            print(f"  ERROR: {e}")
            self.results[test_name] = False
            print_result(test_name, False)
            return False
    
    def test_update_records(self) -> bool:
        """
        Test 5: Update specific records using ETL.update.
        Validates: Targeted record modification via WHERE clause.
        
        Dynamic logic:
        - Uses first row from DataFrame
        - Finds updateable column (priority: bool > str > int > float)
        - Changes value based on type
        - Uses PK in WHERE clause
        """
        test_name = "update Specific Records"
        print_header(f"Test: {test_name}")
        
        try:
            # Get PK column(s) - use first column as PK
            pk_col = self.df_full.columns[0]
            
            # Get first row from DataFrame
            first_row = self.df_full.iloc[0]
            pk_value = first_row[pk_col]
            
            print(f"  Target record: {pk_col} = {pk_value}")
            
            # Find column to update (exclude PK)
            non_pk_cols = [c for c in self.df_full.columns if c != pk_col]
            
            if not non_pk_cols:
                print("  No non-PK columns to update")
                self.results[test_name] = False
                print_result(test_name, False)
                return False
            
            # Priority: bool > string > int > float
            update_col = None
            old_value = None
            new_value = None
            
            for col in non_pk_cols:
                dtype = self.df_full[col].dtype
                val = first_row[col]
                
                # Skip NaN/None values
                if pd.isna(val):
                    continue
                
                if dtype == 'bool' or str(dtype) == 'boolean':
                    update_col = col
                    old_value = val
                    new_value = not val  # Toggle boolean
                    print(f"  Found bool column: {col}")
                    break
                elif dtype == 'object' or str(dtype).startswith('str'):
                    update_col = col
                    old_value = val
                    new_value = f"{val}_UPDATED"
                    print(f"  Found string column: {col}")
                    break
                elif pd.api.types.is_integer_dtype(dtype):
                    update_col = col
                    old_value = val
                    new_value = int(val) + 999999
                    print(f"  Found int column: {col}")
                    break
                elif pd.api.types.is_float_dtype(dtype):
                    update_col = col
                    old_value = val
                    new_value = float(val) + 0.999
                    print(f"  Found float column: {col}")
                    break
            
            if update_col is None:
                print("  No suitable column found for update")
                self.results[test_name] = False
                print_result(test_name, False)
                return False
            
            print(f"  Updating {update_col}: {old_value} -> {new_value}")
            
            # Create update data
            update_df = pd.DataFrame([{update_col: new_value}])
            
            # Perform update using PK in WHERE clause
            rows_updated = self.etl.update(
                update_df,
                self.table_name,
                where=[(pk_col, '=', pk_value)],
                trace_sql=self.trace_sql
            )
            
            print(f"  Rows updated: {rows_updated}")
            
            # Verify the update
            with self.etl.engine.connect() as conn:
                # Use parameterized query for safety
                # Convert numpy/pandas types to native Python
                if hasattr(pk_value, 'item'):
                    safe_pk = pk_value.item()
                else:
                    safe_pk = pk_value
                
                # Use identifier quoting from dialect
                qp = self.etl.engine.dialect.identifier_preparer.quote
                target_col = update_col
                target_pk_col = pk_col
                
                # Postgres and Oracle are case-sensitive if quoted; 
                # we usually quote upper in create_ddl for consistency.
                if self.etl.dialect in ('oracle', 'postgresql'):
                    target_col = update_col.upper()
                    target_pk_col = pk_col.upper()
                
                sql = f"SELECT {qp(target_col)} FROM {qp(self.table_name)} WHERE {qp(target_pk_col)} = :pk"
                result = conn.execute(text(sql), {"pk": safe_pk}).scalar()

            
            # Compare values (handle type differences)
            if isinstance(new_value, bool):
                success = bool(result) == new_value
            elif isinstance(new_value, str):
                success = str(result) == new_value
            elif isinstance(new_value, (int, float)):
                success = abs(float(result) - float(new_value)) < 0.01
            else:
                success = str(result) == str(new_value)
            
            if success:
                print(f"  Update verified: {update_col} = {result}")
            else:
                print(f"  Update failed: Expected '{new_value}', got '{result}'")
            
            self.results[test_name] = success
            print_result(test_name, success)
            return success
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.results[test_name] = False
            print_result(test_name, False)
            return False
    
    def test_full_replace(self) -> bool:
        """
        Test 6: Full table replacement.
        Validates: Drop and recreate with new data.
        """
        test_name = "create_load Full Replace"
        print_header(f"Test: {test_name}")
        
        try:
            success = self.etl.create_load(
                self.df_full, 
                self.table_name, 
                pk=self.pk, 
                drop_if_exists=True
            )
            self.results[test_name] = success
            print_result(test_name, success)
            return success
        except Exception as e:
            print(f"  ERROR: {e}")
            self.results[test_name] = False
            print_result(test_name, False)
            return False
    
    def verify_final_state(self) -> dict:
        """Verify final database state."""
        print_header("Final Verification")
        
        verification = {
            "row_count": 0,
            "columns": [],
            "schema_match": False
        }
        
        try:
            sm = SchemaManager(self.etl.engine)
            
            with self.etl.engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}")).scalar()
                verification["row_count"] = count
                print(f"  Final row count: {count}")
            
            db_cols = [c['name'].upper() for c in sm.get_columns(self.table_name)]
            verification["columns"] = db_cols
            print(f"  Final columns ({len(db_cols)}): {db_cols}")
            
            # Check schema alignment
            source_cols = [c.upper() for c in self.df_full.columns]
            missing = [c for c in source_cols if c not in db_cols]
            
            if not missing:
                print("  Schema Match: All source columns present in database")
                verification["schema_match"] = True
            else:
                print(f"  Schema Mismatch: Missing columns {missing}")
                verification["schema_match"] = False
                
        except Exception as e:
            print(f"  Verification Error: {e}")
        
        return verification
    
    def run_core_tests(self) -> bool:
        """Run the core schema evolution test sequence."""
        if not self.setup():
            return False
        
        # Core test sequence
        if not self.test_create_load_subset():
            return False
        
        self.test_upsert_no_schema_evolution()
        self.test_upsert_with_schema_evolution()
        
        self.verify_final_state()
        return all(self.results.values())
    
    def run_all_tests(self) -> bool:
        """Run all available tests."""
        if not self.setup():
            return False
        
        self.test_create_load_subset()
        self.test_upsert_no_schema_evolution()
        self.test_upsert_with_schema_evolution()
        self.test_append_data()
        self.test_update_records()
        self.test_full_replace()
        
        verification = self.verify_final_state()
        
        # Summary
        print_header("Test Summary")
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        print(f"  Tests Passed: {passed}/{total}")
        
        for name, result in self.results.items():
            print_result(name, result)
        
        return all(self.results.values())


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ETL Schema Evolution Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="csv/quota.csv",
        help="Path to CSV file to test (default: csv/quota.csv)"
    )
    
    parser.add_argument(
        "table_name",
        nargs="?",
        default=None,
        help="Target table name (default: derived from CSV filename)"
    )
    
    parser.add_argument(
        "--pk",
        default="derive",
        help="Primary key column(s), comma-separated (default: derive)"
    )
    
    parser.add_argument(
        "--subset-cols",
        type=int,
        default=3,
        help="Number of columns for initial subset (default: 3)"
    )
    
    parser.add_argument(
        "--subset-rows",
        type=int,
        default=5,
        help="Number of rows for initial subset (default: 5)"
    )
    
    parser.add_argument(
        "--db-uri",
        default=None,
        help="Database connection URI (default: from config.yml)"
    )
    
    parser.add_argument(
        "--schema-dir",
        default="schema",
        help="Directory to save DDL schemas (default: schema)"
    )
    
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run all tests including append and full replace"
    )
    parser.add_argument(
        "--trace-sql",
        action="store_true",
        help="Enable SQL tracing in CRUD operations"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve CSV path
    csv_path = Path(args.csv_path)
    
    # Derive table name from CSV if not specified
    table_name = args.table_name or f"test_{csv_path.stem}"
    
    # Get DB URI
    if args.db_uri:
        db_uri = args.db_uri
    else:
        config = load_config()
        db_uri = get_db_uri(config)
    
    # Parse PK
    pk = args.pk
    if pk and pk != "derive" and "," in pk:
        pk = [p.strip() for p in pk.split(",")]
    
    # Create and run test runner
    runner = ETLTestRunner(
        csv_path=csv_path,
        table_name=table_name,
        db_uri=db_uri,
        pk=pk,
        subset_cols=args.subset_cols,
        subset_rows=args.subset_rows,
        schema_dir=args.schema_dir,
        trace_sql=args.trace_sql
    )
    
    if args.all_tests:
        success = runner.run_all_tests()
    else:
        success = runner.run_core_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
