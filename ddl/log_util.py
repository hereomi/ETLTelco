import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = 'ETLLogger',
    log_file: str = 'log.txt',
    level: int = logging.INFO,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        level: Logging level
        format_string: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_log_filename(base_name: str = 'log') -> str:
    """
    Generate a timestamped log filename.
    
    Args:
        base_name: Base name for the log file
        
    Returns:
        Timestamped log filename
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}.txt"


def rotate_log_file_if_needed(log_file: str, max_size_mb: int = 10) -> str:
    """
    Rotate the log file if it exceeds the specified size.
    
    Args:
        log_file: Path to the log file
        max_size_mb: Maximum size in MB before rotation
        
    Returns:
        Path to the current log file (possibly rotated)
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if os.path.exists(log_file):
        if os.path.getsize(log_file) > max_size_bytes:
            # Create backup filename with timestamp
            log_path = Path(log_file)
            backup_name = f"{log_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{log_path.suffix}"
            backup_path = log_path.parent / backup_name
            
            # Rename the current log file to backup
            os.rename(log_file, backup_path)
            return str(backup_path)
    
    return log_file