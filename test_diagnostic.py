"""
Diagnostic script to test DataAligner and auto_update flow.
This helps identify where the issue is occurring.
"""
import pandas as pd
from sqlalchemy import create_engine, text
from schema_align.core import DataAligner
from utils.crud_v2 import auto_update

# Setup
engine = create_engine('sqlite:///opflow_test.db')

print("=" * 60)
print("DIAGNOSTIC TEST: DataAligner and auto_update")
print("=" * 60)

# Step 1: Check current database value
print("\n1. Current database value:")
with engine.connect() as conn:
    result = conn.execute(text("SELECT cost_id, cost_code FROM test_quota WHERE cost_id = 1001"))
    current = result.fetchone()
    if current:
        print(f"   cost_id={current[0]}, cost_code='{current[1]}'")
    else:
        print("   No record found with cost_id=1001")

# Step 2: Test DataAligner
print("\n2. Testing DataAligner:")
df_input = pd.DataFrame([{'cost_code': 'ALIGNER_TEST_VALUE'}])
print(f"   Input DataFrame columns: {list(df_input.columns)}")
print(f"   Input DataFrame values: {df_input.to_dict(orient='records')}")

aligner = DataAligner(db_type='sqlite')
df_aligned = aligner.align(engine, df_input, 'test_quota')
print(f"   Aligned DataFrame columns: {list(df_aligned.columns)}")
print(f"   Aligned DataFrame values: {df_aligned.to_dict(orient='records')}")

# Step 3: Test auto_update
print("\n3. Testing auto_update:")
df_update = pd.DataFrame([{'cost_code': 'AUTO_UPDATE_TEST'}])
print(f"   Update DataFrame: {df_update.to_dict(orient='records')}")
print(f"   WHERE clause: [('COST_ID', '=', 1001)]")

rows_updated = auto_update(
    engine,
    'test_quota',
    df_update,
    [('COST_ID', '=', 1001)],
    trace_sql=True
)
print(f"   Rows updated: {rows_updated}")

# Step 4: Verify the update
print("\n4. Verifying update:")
with engine.connect() as conn:
    result = conn.execute(text("SELECT cost_id, cost_code FROM test_quota WHERE cost_id = 1001"))
    updated = result.fetchone()
    if updated:
        print(f"   cost_id={updated[0]}, cost_code='{updated[1]}'")
        if updated[1] == 'AUTO_UPDATE_TEST':
            print("   ✅ UPDATE SUCCESSFUL!")
        else:
            print(f"   ❌ UPDATE FAILED: Expected 'AUTO_UPDATE_TEST', got '{updated[1]}'")
    else:
        print("   ❌ No record found")

# Step 5: Check generated SQL
print("\n5. Generated SQL (check auto_update_test_quota.txt):")
try:
    with open('auto_update_test_quota.txt', 'r') as f:
        sql = f.read()
        print(f"   {sql}")
except FileNotFoundError:
    print("   File not found")

# Step 6: Check logs
print("\n6. Logs (check where_build.log):")
try:
    with open('where_build.log', 'r') as f:
        logs = f.readlines()
        print("   Last 5 log entries:")
        for line in logs[-5:]:
            print(f"   {line.strip()}")
except FileNotFoundError:
    print("   File not found")

print("\n" + "=" * 60)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 60)
