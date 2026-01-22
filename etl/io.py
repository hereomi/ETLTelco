from typing import Union
from pathlib import Path
import pandas as pd


def _to_dataframe(source: Union[Path, str, pd.DataFrame, list]) -> pd.DataFrame:
    """Converts various source types (CSV, Excel, DF, List) to a pandas DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, list):
        return pd.DataFrame(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    elif suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
