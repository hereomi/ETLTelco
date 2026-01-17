import os, yaml
from datetime import datetime
from pathlib import Path
from etl import ETL

def run_batch_etl(engine_or_uri, csv_dir, config_path="csv/config.yml"):
    """Compact batch processor: checks mtime, but DOES NOT write to config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for file in Path(csv_dir).glob("*.csv"):
        cfg = config.get(file.stem)
        if not cfg: continue

        # Time-based execution check (based on manual 'executed_at' in config)
        mtime = datetime.fromtimestamp(file.stat().st_mtime)
        last_exec = cfg.get('executed_at')
        if last_exec and mtime <= (datetime.fromisoformat(last_exec) if isinstance(last_exec, str) else last_exec):
            print(f"Skipping {file.name} (not modified since {last_exec})")
            continue

        print(f"Processing {file.name}...")
        try:
            etl = ETL(engine_or_uri, add_missing_cols=cfg.get('table_column_addition', False))
            op = cfg.get('operation', 'upsert').lower()
            pk = cfg.get('pk')
            table_name = cfg.get('table_name', file.stem)
            
            # Execute ETL
            if op == 'upsert':
                # auto_etl handles PK derivation if pk='derive'
                success = etl.auto_etl(file, table_name, pk=pk, dtype=cfg.get('dtypes', {}))
            else:
                success = etl.append(file, table_name, dtype=cfg.get('dtypes', {})) if etl.engine.dialect.has_table(etl.engine.connect(), table_name) else \
                          etl.create_load(file, table_name, pk=pk, drop_if_exists=False, dtype=cfg.get('dtypes', {}))

            if success:
                print(f"Success: {file.name}")
                # Print info for manual update as requested
                print(f"  [INFO] Processed at: {datetime.now().isoformat()}")
                if pk == 'derive':
                    # The ETL instance might have derived the PK during the process.
                    # We print a reminder for the user to update their config.
                    print(f"  [MANUAL UPDATE] Suggested PK for '{file.stem}': Check logs above for derived components.")
        except Exception as e:
            print(f"Error {file.name}: {e}")

    # Section to write back config removed per user request: "config file won't be modified by programitically"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_batch_etl(sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[1])
