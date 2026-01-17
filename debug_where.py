
import pandas as pd
from utils.where_build import build_update, _row_conditions, _parse_where_item

def test_debug():
    record = {"COST_ID": 1001, "COST_CODE": "OLD"}
    table = "test_quota"
    where = [("COST_ID", "=", 1001)]
    dialect = "sqlite"
    
    df = pd.DataFrame([record])
    row = df.iloc[0]
    df_cols = {str(c) for c in row.index}
    
    print(f"Row index: {list(row.index)}")
    print(f"DF cols: {df_cols}")
    print(f"Where field: {where[0][0]}")
    
    try:
        d = _parse_where_item(where[0], df_cols)
        print(f"Parsed item: {d}")
        
        conds = _row_conditions(row, where)
        print(f"Row conditions: {conds}")
        
        sql, params = build_update(record, table, where, dialect=dialect)
        print(f"SQL: {sql}")
        print(f"Params: {params}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
