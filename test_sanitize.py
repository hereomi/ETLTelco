"""Test to check if sanitize_cols changes column names."""
import pandas as pd
from utils.sanitization import sanitize_cols

df = pd.DataFrame([{'cost_code': 'TEST_VALUE', 'COST_ID': 1001}])
print("Before sanitize_cols:")
print(f"  Columns: {list(df.columns)}")

df_sanitized = sanitize_cols(df, dialect='sqlite')
print("\nAfter sanitize_cols (sqlite):")
print(f"  Columns: {list(df_sanitized.columns)}")

df_sanitized_oracle = sanitize_cols(df, dialect='oracle')
print("\nAfter sanitize_cols (oracle):")
print(f"  Columns: {list(df_sanitized_oracle.columns)}")
