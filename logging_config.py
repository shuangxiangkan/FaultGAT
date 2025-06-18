import logging
import os
from typing import Optional

def setup_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Setup logging configuration
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log format
        log_file: Log file path, if specified, output to file simultaneously
        console_output: Whether to output to console
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(format_string)
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get logger with specified name
    
    Args:
        name: Logger name, usually use __name__
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

# Default configuration
def init_default_logging():
    """Initialize default logging configuration"""
    setup_logging(
        level="INFO",
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        console_output=True
    ) 