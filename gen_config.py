import os
import yaml
import pandas as pd
from pathlib import Path
from utils.data_profiler import profile_dataframe, get_pk

# Default template as per user spec
TEMPLATE = {
    'pk': None,
    'table_name': None,
    'operation': 'insert',
    'dtypes': None,
    'table_column_addition': True,
    'outlier_percentage': 1.0,
    'executed_at': None
}

def derive_config_for_csv(csv_path: Path):
    """Profile CSV to suggest PK and operation."""
    try:
        # Load a sample for speed (5000 rows is usually enough for PK heuristic)
        df = pd.read_csv(csv_path, nrows=5000)
        if df.empty:
            return TEMPLATE.copy()

        dfinfo = profile_dataframe(df)
        _, _, pk_meta = get_pk(df, dfinfo)

        cfg = TEMPLATE.copy()
        if pk_meta.get('is_unique'):
            # Convert components to a clean list or string for YAML
            cfg['pk'] = pk_meta['components']
            cfg['operation'] = 'upsert'
            print(f"  [AUTO] Detected PK for {csv_path.name}: {cfg['pk']}")
        else:
            print(f"  [AUTO] No unique PK for {csv_path.name}. Defaulting to insert.")
        return cfg
    except Exception as e:
        print(f"  [ERROR] Could not profile {csv_path.name}: {e}")
        return TEMPLATE.copy()

def generate_configs(root_dir, force=False):
    """Walk through folders and manage folder-specific config.yml files."""
    for dirpath, _, filenames in os.walk(root_dir):
        csv_files = [f for f in filenames if f.endswith('.csv')]
        if not csv_files:
            continue

        folder_path = Path(dirpath)
        config_path = folder_path / "config.yml"
        
        # Force delete if requested
        if force and config_path.exists():
            print(f"Forcing reset of: {config_path}")
            config_path.unlink()

        # Load existing config or start fresh
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            print(f"Updating existing config in: {folder_path}")
        else:
            config = {}
            print(f"Creating new config in: {folder_path}")

        updated = False
        for csv_file in csv_files:
            stem = Path(csv_file).stem
            if stem in config:
                # User spec: "escape existing config"
                continue
            
            print(f"  Generating config for: {csv_file}")
            # Calculate table name from folder and filename
            folder_name = folder_path.name if folder_path.name else "root"
            derived_table_name = f"{folder_name}_{stem}"
            
            file_cfg = derive_config_for_csv(folder_path / csv_file)
            file_cfg['table_name'] = derived_table_name
            config[stem] = file_cfg
            updated = True

        if updated:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            print(f"  Saved {config_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate folder-specific config.yml files.")
    parser.add_argument("path", nargs="?", default=".", help="Root path to walk")
    parser.add_argument("--force", action="store_true", help="Delete existing config.yml before generating")
    
    args = parser.parse_args()
    print(f"Walking through: {os.path.abspath(args.path)}")
    generate_configs(args.path, force=args.force)
