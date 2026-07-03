import os 
from pathlib import Path 
from dotenv import load_dotenv 
 
load_dotenv() 
 
class Settings: 
    BASE_DIR = Path(__file__).resolve().parent.parent 
    DATA_DIR = BASE_DIR / "data" 
    RAW_DATA_DIR = DATA_DIR / "raw" 
    PROCESSED_DATA_DIR = DATA_DIR / "processed" 
    MODELS_DIR = BASE_DIR / "models" 
    LOGS_DIR = BASE_DIR / "logs" 
 
    MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost") 
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2")) 
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42")) 
    N_TRIALS = int(os.getenv("N_TRIALS", "100")) 
 
    API_HOST = os.getenv("API_HOST", "0.0.0.0") 
    API_PORT = int(os.getenv("API_PORT", "8000")) 
    API_WORKERS = int(os.getenv("API_WORKERS", "4")) 
 
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000") 
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction") 
 
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") 
 
    @classmethod 
    def create_dirs(cls): 
        directories = [ 
            cls.RAW_DATA_DIR, 
            cls.PROCESSED_DATA_DIR, 
            cls.MODELS_DIR, 
            cls.LOGS_DIR, 
        ] 
        for directory in directories: 
            directory.mkdir(parents=True, exist_ok=True) 
 
settings = Settings() 
settings.create_dirs() 
