from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .ddl_database import get_default_db_manager, DatabaseManager
from .ddl_mysql import generate_ddl as mysql_generate_ddl
from .ddl_postgre import generate_ddl as postgre_generate_ddl
from .ddl_oracle import generate_ddl as oracle_generate_ddl
from .ddl_mssql import generate_ddl as mssql_generate_ddl
from .ddl_sqlite import generate_ddl as sqlite_generate_ddl
from .ddl_common import ColumnMeta
from .log_util import setup_logger


logger = setup_logger('ETLDemo', 'log.txt')


def create_demo_dataframe() -> pd.DataFrame:
    """
    Creates a demo DataFrame with various data types to showcase the ETL capabilities.
    
    Returns:
        A pandas DataFrame with diverse data types
    """
    logger.info("Creating demo DataFrame...")
    
    np.random.seed(42)
    n_rows = 100
    
    data = {
        'id': range(1, n_rows + 1),
        'name': [f'Person_{i}' for i in range(1, n_rows + 1)],
        'email': [f'user{i}@example.com' for i in range(1, n_rows + 1)],
        'age': np.random.randint(18, 80, size=n_rows),
        'salary': np.random.uniform(30000, 150000, size=n_rows),
        'is_active': np.random.choice([True, False], size=n_rows),
        'created_date': [datetime.now() - timedelta(days=int(x)) for x in np.random.randint(0, 365, size=n_rows)],
        'profile': [
            f'{{"bio": "Sample bio {i}", "preferences": {{"theme": "dark", "lang": "en"}}}}' 
            for i in range(1, n_rows + 1)
        ],
        'metadata': [bytes(f'metadata_{i}', 'utf-8') for i in range(1, n_rows + 1)],
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Demo DataFrame created with shape: {df.shape}")
    return df


def create_table_in_databases(df: pd.DataFrame, table_name: str, db_manager: DatabaseManager) -> None:
    """
    Creates a table in all supported databases based on the DataFrame schema using the unified runner.
    """
    from .runner import run_ddl_and_load
    
    logger.info(f"Creating table '{table_name}' in all databases using the unified runner...")
    
    # Use the connection strings from db_manager or the default ones
    conn_dict = db_manager.connection_strings
    
    # Define options for the DDL generator
    # We can pass these directly to run_ddl_and_load
    results = run_ddl_and_load(
        conn_dict=conn_dict,
        df=df,
        table=table_name,
        include_not_null=True,
        use_ddl_io=True  # Feature provided by the user
    )
    
    for server, status in results.get("servers", {}).items():
        if status.get("status") == "COMPLETED":
            logger.info(f"✅ {server}: Success")
        else:
            logger.error(f"❌ {server}: {status.get('status')} - {status.get('message', 'No error message')}")
    

def main():
    """
    Main function to demonstrate ETL process with demo data across multiple databases.
    """
    logger.info("Starting ETL Demo")
    
    try:
        logger.info("Creating demo DataFrame...")
        df = create_demo_dataframe()
        logger.info(f"Demo DataFrame created with shape: {df.shape}")
        
        logger.info("DataFrame head:")
        for index, row in df.head().iterrows():
            logger.info(f"Row {index}: {row.to_dict()}")
        
        logger.info("DataFrame info:")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        # Initialize database manager
        logger.info("Initializing database manager...")
        db_manager = get_default_db_manager()
        
        # Create table in all databases
        table_name = "demo_users"
        create_table_in_databases(df, table_name, db_manager)
        
        logger.info("ETL Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"ETL Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()