import logging
import os
from datetime import datetime

# Define a function to create the logs directory and file
def setup_logging():
    # Define log folder and filename
    log_dir = 'logs'
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = f'{log_dir}/log_{current_time}.log'

    # Create log folder if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up logger
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Create console handler to also output logs to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# Set up logging
logger = setup_logging()

# Application code
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')