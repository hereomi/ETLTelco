from pathlib import Path

_pkg_dir = Path(__file__).with_name("etl")
__path__ = [str(_pkg_dir)]

from etl.ops import ETL, run_etl

__all__ = ["ETL", "run_etl"]
