from etl.gen_config import TEMPLATE, derive_config_for_csv, generate_configs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate folder-specific config.yml files.")
    parser.add_argument("path", nargs="?", default=".", help="Root path to walk")
    parser.add_argument("--force", action="store_true", help="Delete existing config.yml before generating")

    args = parser.parse_args()
    import os
    print(f"Walking through: {os.path.abspath(args.path)}")
    generate_configs(args.path, force=args.force)
