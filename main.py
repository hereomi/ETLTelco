import argparse
import json
import sys
from pathlib import Path
from etl import ETL

def parse_json(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON string provided: {s}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="EtlTele CLI - Robust Data Pipeline Engine")
    
    # --- Global Initialization Arguments ---
    parser.add_argument("--db-uri", required=True, help="SQLAlchemy connection URI (e.g., sqlite:///data.db)")
    parser.add_argument("--schema-dir", help="Directory for DDL/history")
    parser.add_argument("--chunksize", type=int, default=10000, help="Batch size for DB operations")
    parser.add_argument("--add-missing-cols", action="store_true", help="Enable schema evolution (auto ALTER TABLE)")
    parser.add_argument("--rename-column", action="store_true", help="Sanitize column headers automatically")
    parser.add_argument("--no-logging", action="store_true", help="Disable extensive logging")
    parser.add_argument("--no-schema-aware-casting", action="store_true", help="Disable DB-guided type casting")
    
    # --- Casting / Configuration Overrides ---
    parser.add_argument("--infer-threshold", type=float, default=0.9, help="Threshold for type inference (0.0 to 1.0)")
    parser.add_argument("--max-null-increase", type=float, default=0.1, help="Tolerance for data loss during casting")
    parser.add_argument("--dtype", help="Manual type mapping as JSON string (e.g., '{\"id\": \"string\"}')")
    
    subparsers = parser.add_subparsers(dest="command", help="ETL Commands")

    # --- create-load ---
    p_create = subparsers.add_parser("create-load", help="Recreate table and load data")
    p_create.add_argument("source", help="Source file path (CSV/Excel) or data")
    p_create.add_argument("table", help="Target table name")
    p_create.add_argument("--no-drop", action="store_false", dest="drop_if_exists", help="Do not drop table if exists")
    p_create.add_argument("--pk", help="Primary Key(s) if not auto-verified")

    # --- append ---
    p_append = subparsers.add_parser("append", help="Append data to existing table")
    p_append.add_argument("source", help="Source file or data")
    p_append.add_argument("table", help="Target table name")

    # --- upsert ---
    p_upsert = subparsers.add_parser("upsert", help="Update existing or insert new records")
    p_upsert.add_argument("source", help="Source file or data")
    p_upsert.add_argument("table", help="Target table name")
    p_upsert.add_argument("--constrain", required=True, help="Comma-separated unique keys (e.g., 'ID' or 'ID,CODE')")

    # --- update ---
    p_update = subparsers.add_parser("update", help="Update records using WHERE clauses")
    p_update.add_argument("source", help="Source file or data")
    p_update.add_argument("table", help="Target table name")
    p_update.add_argument("--where", help="JSON list of filter tuples (e.g., '[[\"ID\", \"=\", 10]]')")
    p_update.add_argument("--expression", help="Raw SQL string for WHERE clause")
    p_update.add_argument("--trace-sql", action="store_true", help="Save final SQL to file for review")

    # --- auto-etl ---
    p_auto = subparsers.add_parser("auto-etl", help="Intelligent Load (Create or Upsert)")
    p_auto.add_argument("source", help="Source file or data")
    p_auto.add_argument("table", help="Target table name")
    p_auto.add_argument("--pk", default="derive", help="PK strategy (verify, derive, or list)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize ETL
    etl = ETL(
        db_uri=args.db_uri,
        schema_dir=args.schema_dir,
        logging_enabled=not args.no_logging,
        chunksize=args.chunksize,
        add_missing_cols=args.add_missing_cols,
        rename_column=args.rename_column,
        schema_aware_casting=not args.no_schema_aware_casting,
        infer_threshold=args.infer_threshold,
        max_null_increase=args.max_null_increase,
        dtype=parse_json(args.dtype)
    )

    print(f"\nRunning {args.command} on table '{args.table}'...")

    try:
        if args.command == "create-load":
            etl.create_load(args.source, args.table, drop_if_exists=args.drop_if_exists, pk=args.pk)
        elif args.command == "append":
            etl.append(args.source, args.table)
        elif args.command == "upsert":
            constrain = args.constrain.split(",")
            etl.upsert(args.source, args.table, constrain=constrain)
        elif args.command == "update":
            where = parse_json(args.where)
            etl.update(args.source, args.table, where=where, expression=args.expression, trace_sql=args.trace_sql)
        elif args.command == "auto-etl":
            etl.auto_etl(args.source, args.table, pk=args.pk)
            
        print("Operation completed successfully.")
    except Exception as e:
        print(f"Error during operation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
