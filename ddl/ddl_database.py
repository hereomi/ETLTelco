from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy import create_engine
from .ddl_common import validate_dataframe, validate_table_name


class DatabaseManager:
   
    def __init__(self, connection_strings: Dict[str, str]):
        self.connection_strings = connection_strings
        self.engines: Dict[str, Any] = {}
    
    def get_engine(self, dialect: str) -> Any:
        if dialect not in self.connection_strings:
            raise ValueError(f"Connection string for dialect '{dialect}' not found")
        if dialect not in self.engines:
            connection_string = self.connection_strings[dialect]
            self.engines[dialect] = create_engine(connection_string)
        return self.engines[dialect]
    
    def execute_ddl(self, sql: str, dialect: str) -> None:
        from sqlalchemy import text
        engine = self.get_engine(dialect)
        with engine.connect() as conn:
            # Wrap raw SQL in text() for SQLAlchemy 2.0 compatibility
            conn.execute(text(sql))
            conn.commit()
    
    def insert_data(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        dialect: str, 
        schema: Optional[str] = None, 
        if_exists: str = 'fail',
        **kwargs
    ) -> None:
        
        validate_dataframe(df)
        validate_table_name(table_name)
        engine = self.get_engine(dialect)
        default_kwargs = {
            'if_exists': if_exists,
            'index': False,
            'method': 'multi'
        }
        default_kwargs.update(kwargs)
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            **default_kwargs
        )


# Connection strings for different database systems
CONNECTION_STRINGS = {
    'mysql': "mysql+pymysql://root:PASSWORD_HERE@127.0.0.1:3306/akomi",
    'mssql': "mssql+pyodbc://sa:PASSWORD_HERE@127.0.0.1:1433/master?driver=ODBC+Driver+17+for+SQL+Server",
    'postgre': "postgresql+psycopg2://postgres:PASSWORD_HERE@127.0.0.1:5432/akomi",
    'oracle': "oracle+oracledb://akomi:PASSWORD_HERE@localhost:1521/?service_name=XEPDB1",
    'sqlite': "sqlite:///opflow.db"
}


def get_default_db_manager() -> DatabaseManager:
    return DatabaseManager(CONNECTION_STRINGS)