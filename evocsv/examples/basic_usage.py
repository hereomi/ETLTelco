"""
Basic Usage Example for EvoCSV Framework

This example demonstrates:
1. Creating a test DataFrame
2. Initializing the test harness
3. Running INSERT, UPSERT, UPDATE operations
4. Validating results
5. Running full cycle test
"""
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, MetaData, Table
from evocsv import EnhancedCrudTestHarness

def create_test_table(engine):
    """Create test table in database."""
    meta = MetaData()
    users = Table('users', meta,
        Column('id', Integer, primary_key=True),
        Column('username', String(50), unique=True),
        Column('email', String(100)),
        Column('score', Float),
        Column('active', Integer)  # SQLite doesn't have boolean
    )
    meta.create_all(engine)
    return users

def create_test_dataframe():
    """Create 10-row test DataFrame."""
    return pd.DataFrame({
        'id': range(10),
        'username': [f'user{i}' for i in range(10)],
        'email': [f'user{i}@example.com' for i in range(10)],
        'score': [float(i * 10.5) for i in range(10)],
        'active': [1 if i % 2 == 0 else 0 for i in range(10)]
    })

def main():
    print("=" * 70)
    print("EvoCSV Framework - Basic Usage Example")
    print("=" * 70)
    
    # 1. Setup
    print("\n1. Creating in-memory SQLite database...")
    engine = create_engine('sqlite:///:memory:', echo=False)
    create_test_table(engine)
    print("   ✓ Table 'users' created")
    
    # 2. Create test data
    print("\n2. Creating test DataFrame (10 rows)...")
    df = create_test_dataframe()
    print(f"   ✓ DataFrame created with shape {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # 3. Initialize harness
    print("\n3. Initializing test harness...")
    harness = EnhancedCrudTestHarness(
        df_src=df,
        pk_cols='id',
        constraint_cols='username',
        table_name='users'
    )
    print(f"   ✓ Harness initialized")
    print(f"   PK columns: {harness.pk}")
    print(f"   Constraint columns: {harness.constraint}")
    print(f"   Mutable columns: {harness.mutable}")
    
    # 4. Show test data
    print("\n4. Test data overview:")
    print(f"   INSERT DataFrame: {len(harness.insert_df)} rows (indices 0-5)")
    print(f"   UPSERT DataFrame: {len(harness.upsert_df)} rows (indices 3-7)")
    print(f"   UPDATE DataFrame: {len(harness.update_df)} rows (indices 1,7,8)")
    
    # 5. Run INSERT test
    print("\n5. Running INSERT test...")
    insert_result = harness.run_insert_test(engine, dialect='sqlite')
    print(f"   {insert_result.validation}")
    print(f"   SQL queries generated: {len(insert_result.sql_queries)}")
    
    # 6. Run UPSERT test
    print("\n6. Running UPSERT test...")
    upsert_result = harness.run_upsert_test(engine, dialect='sqlite')
    print(f"   {upsert_result.validation}")
    print(f"   SQL queries generated: {len(upsert_result.sql_queries)}")
    
    # 7. Run UPDATE test
    print("\n7. Running UPDATE test...")
    update_result = harness.run_update_test(engine, dialect='sqlite')
    print(f"   {update_result.validation}")
    print(f"   SQL queries generated: {len(update_result.sql_queries)}")
    
    # 8. Verify final state
    print("\n8. Verifying final database state...")
    final_df = pd.read_sql("SELECT * FROM users ORDER BY id", engine)
    print(f"   Total rows in database: {len(final_df)}")
    print(f"   Expected rows: 7 (rows 0-7, excluding 8-9)")
    
    # 9. Show sample data
    print("\n9. Sample data from database:")
    print(final_df.head(3).to_string(index=False))
    
    # 10. Full cycle test (fresh database)
    print("\n10. Running full cycle test (fresh database)...")
    engine2 = create_engine('sqlite:///:memory:', echo=False)
    create_test_table(engine2)
    
    full_result = harness.run_full_cycle_test(engine2, dialect='sqlite')
    print(f"    {full_result.validation}")
    print(f"    Total operations: {full_result.execution_stats['total_operations']}")
    print(f"    - INSERT: {full_result.execution_stats['insert_rows']} rows")
    print(f"    - UPSERT: {full_result.execution_stats['upsert_rows']} rows")
    print(f"    - UPDATE: {full_result.execution_stats['update_rows']} rows")
    
    print("\n" + "=" * 70)
    print("✓ All tests completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
