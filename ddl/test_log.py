from log_util import setup_logger
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logger
logger = setup_logger('TestLogger', 'log.txt')

def test_logging():
    """Function to test logging functionality"""
    logger.info("Starting log test")
    
    # Create a simple dataframe
    df = pd.DataFrame({
        'id': range(1, 6),
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'date': [datetime.now() for _ in range(5)]
    })
    
    logger.info(f"Created DataFrame with {len(df)} rows")
    logger.info(f"DataFrame columns: {list(df.columns)}")
    
    # Simulate database operations
    databases = ['mysql', 'postgre', 'oracle', 'mssql', 'sqlite']
    
    for db in databases:
        logger.info(f"Processing data for {db}...")
        # Simulate some processing
        processed_rows = len(df) * 2
        logger.info(f"Processed {processed_rows} records for {db}")
    
    logger.info("Log test completed successfully")

if __name__ == "__main__":
    test_logging()