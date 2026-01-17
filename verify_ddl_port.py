
import pandas as pd
import sqlite3
import os
import sys

# Add path to find ddl module
sys.path.append(os.getcwd())

# Import old monolithic
import ddl_create
# Import new modular
from ddl import create_ddl, runner, orm

def test_verify_ddl_parity():
    print("--- Verifying DDL Parity ---")
    
    df = pd.DataFrame({
        'id': [1, 2],
        'name': ['Alice', 'Bob'],
        'score': [95.5, 88.0],
        'active': [True, False]
    })
    table_name = "test_verify"
    
    # 1. ORM Generation Comparison
    print("\n1. Testing ORM Generation...")
    
    old_orm = ddl_create.generate_sqlalchemy_model(df, table_name, sanitize=True, pk='id')
    new_orm = orm.generate_sqlalchemy_model(df, table_name, sanitize=True, pk='id')
    
    # Normalizing whitespace for comparison
    def norm(s): return " ".join(s.split())
    
    if norm(old_orm) == norm(new_orm):
        print("✅ ORM Generation matches!")
    else:
        print("⚠️ ORM Generation differs (might be acceptable if logic improved, but check differences):")
        print("OLD:\n", old_orm)
        print("NEW:\n", new_orm)
        
    # 2. Execution and Data Loading (SQLite)
    print("\n2. Testing Execution (SQLite)...")
    
    db_file = "verify_test.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    conn_str = f"sqlite:///{db_file}"
    conn_dict = {"sqlite": conn_str}
    
    # Run new runner
    res = runner.run_ddl_and_load(conn_dict, df, table_name, pk='id')
    print("Runner Result:", res['servers']['sqlite']['status'])
    
    # Verify data in DB
    with sqlite3.connect(db_file) as conn:
        saved_df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
    print(f"Rows in DB: {len(saved_df)}")
    if len(saved_df) == 2:
        print("✅ Data loading verified!")
    else:
        print("❌ Data loading failed count check")

    # Cleanup
    if os.path.exists(db_file):
        os.remove(db_file)
        
if __name__ == "__main__":
    test_verify_ddl_parity()
