import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from utils.logger import logger
from utils.exceptions import DataLoadError
from config.settings import settings

class DataLoader:
    """Handles data loading and initial validation"""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or settings.RAW_DATA_DIR
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file and perform initial validation"""
        try:
            file_path = self.data_path / filename
            logger.info(f"Loading data from {file_path}")
            
            if not file_path.exists():
                raise DataLoadError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            self._validate_dataframe(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise DataLoadError(f"Failed to load data: {str(e)}")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Perform basic validation on loaded dataframe"""
        if df.empty:
            raise DataLoadError("Loaded dataframe is empty")
        
        if 'Churn' not in df.columns:
            raise DataLoadError("Required column 'Churn' not found")
        
        logger.info("Data validation passed")
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """Get comprehensive dataset information"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'numeric_stats': df.describe().to_dict(),
            'categorical_stats': {
                col: df[col].value_counts().to_dict() 
                for col in df.select_dtypes(include=['object']).columns
            }
        }