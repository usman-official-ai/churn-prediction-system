import logging 
import sys 
from pathlib import Path 
from logging.handlers import RotatingFileHandler 
from config.settings import settings 
 
def setup_logger(name: str = "churn_prediction"): 
    logger = logging.getLogger(name) 
    logger.setLevel(settings.LOG_LEVEL) 
    console_formatter = logging.Formatter( 
        "%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s", 
        datefmt="%%Y-%%m-%%d %%H:%%M:%%S" 
    ) 
    console_handler = logging.StreamHandler(sys.stdout) 
    console_handler.setLevel(logging.INFO) 
    console_handler.setFormatter(console_formatter) 
    logger.addHandler(console_handler) 
    if settings.LOGS_DIR: 
        settings.LOGS_DIR.mkdir(parents=True, exist_ok=True) 
        log_file = settings.LOGS_DIR / "app.log" 
        file_handler = RotatingFileHandler( 
            log_file, 
            maxBytes=10_485_760, 
            backupCount=5 
        ) 
        file_formatter = logging.Formatter( 
            "%%(asctime)s - %%(name)s - %%(levelname)s - %%(pathname)s:%%(lineno)d - %%(message)s", 
            datefmt="%%Y-%%m-%%d %%H:%%M:%%S" 
        ) 
        file_handler.setLevel(logging.DEBUG) 
        file_handler.setFormatter(file_formatter) 
        logger.addHandler(file_handler) 
    return logger 
 
logger = setup_logger() 
