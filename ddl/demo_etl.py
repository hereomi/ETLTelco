from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ddl_database import get_default_db_manager, DatabaseManager
from ddl_mysql import generate_ddl as mysql_generate_ddl
from ddl_postgre import generate_ddl as postgre_generate_ddl
from ddl_oracle import generate_ddl as oracle_generate_ddl
from ddl_mssql import generate_ddl as mssql_generate_ddl
from ddl_sqlite import generate_ddl as sqlite_generate_ddl
from ddl_common import ColumnMeta
from log_util import setup_logger


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
        'created_date': [datetime.now() - timedelta(days=x) for x in np.random.randint(0, 365, size=n_rows)],
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
    Creates a table in all supported databases based on the DataFrame schema.
    
    Args:
        df: DataFrame to use for schema generation
        table_name: Name of the table to create
        db_manager: DatabaseManager instance with connections
    """
    logger.info(f"Creating table '{table_name}' in all databases...")
    
    # Define database-specific options
    options = {
        'mysql': {},
        'postgre': {},
        'oracle': {'varchar2_limit': 4000, 'raw_limit': 2000},
        'mssql': {},
        'sqlite': {}
    }
    
    # Create table in MySQL
    try:
        logger.info("Generating MySQL DDL...")
        mysql_processed_df, mysql_create_sql, mysql_schema = mysql_generate_ddl(df.copy(), table_name, options['mysql'])
        logger.info("MySQL DDL generated successfully")
        
        logger.info("Executing MySQL DDL...")
        db_manager.execute_ddl(mysql_create_sql, 'mysql')
        logger.info("MySQL table created successfully")
        
        logger.info("Inserting data to MySQL...")
        db_manager.insert_data(mysql_processed_df, table_name, 'mysql', if_exists='replace')
        logger.info("Data inserted to MySQL successfully")
    except Exception as e:
        logger.error(f"MySQL operation failed: {e}")
    
    # Create table in PostgreSQL
    try:
        logger.info("Generating PostgreSQL DDL...")
        postgre_processed_df, postgre_create_sql, postgre_schema = postgre_generate_ddl(df.copy(), table_name, options['postgre'])
        logger.info("PostgreSQL DDL generated successfully")
        
        logger.info("Executing PostgreSQL DDL...")
        db_manager.execute_ddl(postgre_create_sql, 'postgre')
        logger.info("PostgreSQL table created successfully")
        
        logger.info("Inserting data to PostgreSQL...")
        db_manager.insert_data(postgre_processed_df, table_name, 'postgre', if_exists='replace')
        logger.info("Data inserted to PostgreSQL successfully")
    except Exception as e:
        logger.error(f"PostgreSQL operation failed: {e}")
    
    # Create table in Oracle
    try:
        logger.info("Generating Oracle DDL...")
        oracle_processed_df, oracle_create_sql, oracle_schema = oracle_generate_ddl(df.copy(), table_name, options['oracle'])
        logger.info("Oracle DDL generated successfully")
        
        logger.info("Executing Oracle DDL...")
        db_manager.execute_ddl(oracle_create_sql, 'oracle')
        logger.info("Oracle table created successfully")
        
        logger.info("Inserting data to Oracle...")
        db_manager.insert_data(oracle_processed_df, table_name, 'oracle', if_exists='replace')
        logger.info("Data inserted to Oracle successfully")
    except Exception as e:
        logger.error(f"Oracle operation failed: {e}")
    
    # Create table in MSSQL
    try:
        logger.info("Generating MSSQL DDL...")
        mssql_processed_df, mssql_create_sql, mssql_schema = mssql_generate_ddl(df.copy(), table_name, options['mssql'])
        logger.info("MSSQL DDL generated successfully")
        
        logger.info("Executing MSSQL DDL...")
        db_manager.execute_ddl(mssql_create_sql, 'mssql')
        logger.info("MSSQL table created successfully")
        
        logger.info("Inserting data to MSSQL...")
        db_manager.insert_data(mssql_processed_df, table_name, 'mssql', if_exists='replace')
        logger.info("Data inserted to MSSQL successfully")
    except Exception as e:
        logger.error(f"MSSQL operation failed: {e}")
    
    # Create table in SQLite
    try:
        logger.info("Generating SQLite DDL...")
        sqlite_processed_df, sqlite_create_sql, sqlite_schema = sqlite_generate_ddl(df.copy(), table_name, options['sqlite'])
        logger.info("SQLite DDL generated successfully")
        
        logger.info("Executing SQLite DDL...")
        db_manager.execute_ddl(sqlite_create_sql, 'sqlite')
        logger.info("SQLite table created successfully")
        
        logger.info("Inserting data to SQLite...")
        db_manager.insert_data(sqlite_processed_df, table_name, 'sqlite', if_exists='replace')
        logger.info("Data inserted to SQLite successfully")
    except Exception as e:
        logger.error(f"SQLite operation failed: {e}")
    

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