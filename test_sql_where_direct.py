
import sys
from utils.where_build import sql_where

def test_sql_where():
    print("--- Testing sql_where with complex conditions ---")
    
    # 1. Complex expression and mixed conditions
    conds = [
        "cell_id = '00123'",
        ("vendor", "IN", ["GP", "BL"]),
        {"field": "tech", "operator": "LIKE", "value": "2G"},
        "dt BETWEEN '2023-01-01' and '2023-01-31'"
    ]
    expr = "1 AND (2 OR 3) AND 4"
    try:
        sql, binds = sql_where(conds, expr, dialect="oracle")
        print(f"SQL: {sql}")
        print(f"Binds: {binds}")
        
        # Expected: cell_id = :cell_id_1 AND (vendor IN (:vendor_2_0, :vendor_2_1) OR tech LIKE :tech_3 ESCAPE '\') AND dt BETWEEN :dt_4_0 AND :dt_4_1
        if "cell_id" in sql and "vendor" in sql and "tech" in sql and "dt" in sql:
            print("PASS: Basic fields found")
        else:
            print("FAIL: Fields missing or mangled")
            
        if "ESCAPE '\\'" in sql:
            print("PASS: Oracle escape found")
            
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- Testing prepared row conditions (sitecode = ?) ---")
    # 2. sitecode = ? style
    conds_prepared = [
        {"field": "sitecode", "operator": "=", "value": "001"},
        "eventdate > '2023-01-01'"
    ]
    try:
        sql, binds = sql_where(conds_prepared, expression="1 AND 2", dialect="sqlite")
        print(f"SQL: {sql}")
        print(f"Binds: {binds}")
    except Exception as e:
        print(f"ERROR: {e}")

    print("\n--- Testing numeric literals in expression (BUG FIX check) ---")
    # 3. Numeric literals that are NOT condition indices
    try:
        sql, binds = sql_where([("kpi", ">", 90)], "1 AND kpi < 100", dialect="sqlite")
        print(f"SQL: {sql}")
        print(f"Binds: {binds}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_sql_where()
