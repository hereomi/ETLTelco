from __future__ import annotations
from __future__ import annotations
from typing import Any, Callable, Optional, Union, List
from dataclasses import dataclass, asdict

import pandas as pd

import pandas as pd
import sqlalchemy

from . import ddl_mysql
from . import ddl_oracle
from . import ddl_postgre
from . import ddl_sqlite
from . import ddl_mssql

from sqlalchemy import create_engine
from .ddl_common import validate_dataframe, validate_table_name


_DIALECTS: dict[str, Callable[[pd.DataFrame, str, dict[str, Any]], tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]]] = {
    "oracle": ddl_oracle.generate_ddl,
    "postgresql": ddl_postgre.generate_ddl,
    "mysql": ddl_mysql.generate_ddl,
    "sqlite": ddl_sqlite.generate_ddl,
    "mssql": ddl_mssql.generate_ddl,
}


@dataclass
class DDLConfig:
    schema: Optional[str] = None
    pk: Optional[Union[str, List[str]]] = None
    fk: Optional[Any] = None
    include_not_null: bool = False
    object_sample_size: int = 5000
    json_text_threshold: float = 0.85
    varchar_limit: int = 10485760


def df_ddl(engine: sqlalchemy.engine.Engine, df: pd.DataFrame, table_name: str, config: Optional[DDLConfig] = None, pk=None, fk=None, **options) -> tuple[pd.DataFrame, str, list[str], dict[str, Any], dict[str, Any]]:
    if df is None:
        raise ValueError("df is required")
    if table_name is None or not str(table_name).strip():
        raise ValueError("table_name is required")
    if engine is None or not hasattr(engine, "dialect") or not getattr(engine.dialect, "name", ""):
        raise ValueError("SQLAlchemy engine with dialect is required")
    dialect = str(engine.dialect.name).lower()
    func = _DIALECTS.get(dialect)
    if func is None:
        raise ValueError(f"Unsupported dialect '{dialect}'")
    
    if config is None:
        # Backward compatibility: Construct config from kwargs and explicit args
        # Note: We prioritize kwargs > explicit args > defaults if we were merging, 
        # but here we just take what's given.
        
        # Helper to get from options or default
        def _get(key, default):
            return options.get(key, default)
            
        config = DDLConfig(
            schema=_get("schema", None),
            pk=pk if pk is not None else _get("pk", None),
            fk=fk if fk is not None else _get("fk", None),
            include_not_null=_get("include_not_null", False),
            object_sample_size=_get("object_sample_size", 5000),
            json_text_threshold=_get("json_text_threshold", 0.85),
            varchar_limit=_get("varchar_limit", 10485760)
        )
    
    # Convert config to dict options for the dialect function
    # Filter out None values to avoid overriding defaults if logic existed downstram (though usually strict dict is fine)
    # Actually, asdict returns everything.
    opts = asdict(config)
    # Merge any extra options just in case (though config should cover standard ones)
    # If the user passed extra kwargs that are not in config, we might want to preserve them for plugins?
    # For now, let's just use the config dict plus any leftover options that aren't in the config keys.
    for k, v in options.items():
        if k not in opts:
            opts[k] = v

    processed_df, create_sql, constraint_sql, schema, sqlalchemy_schema = func(df, table_name, opts)
    return processed_df, create_sql, constraint_sql, schema, sqlalchemy_schema
