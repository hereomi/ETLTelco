from etl.dir_tosql import run_batch_etl

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_batch_etl(sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[1])
