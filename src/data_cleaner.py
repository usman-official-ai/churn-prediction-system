import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.logger import logger
from utils.exceptions import DataProcessError
from config.settings import settings

class DataCleaner:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_features = []
        self.categorical_features = []
        self.X = None
        self.y = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main cleaning pipeline"""
        try:
            logger.info("Starting data cleaning process")
            df_clean = df.copy()
            
            # Remove customer ID if exists
            if 'customerID' in df_clean.columns:
                df_clean = df_clean.drop('customerID', axis=1)
            
            # Handle TotalCharges conversion
            if 'TotalCharges' in df_clean.columns:
                df_clean['TotalCharges'] = pd.to_numeric(
                    df_clean['TotalCharges'], 
                    errors='coerce'
                )
                # Fill missing with median
                df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(
                    df_clean['TotalCharges'].median()
                )
            
            # Convert SeniorCitizen to object
            if 'SeniorCitizen' in df_clean.columns:
                df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype('object')
            
            # Separate features and target
            self.X = df_clean.drop('Churn', axis=1)
            self.y = df_clean['Churn'].map({'Yes': 1, 'No': 0})
            
            # Identify feature types
            self.numerical_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = self.X.select_dtypes(include=['object']).columns.tolist()
            
            logger.info(f"Numerical features: {self.numerical_features}")
            logger.info(f"Categorical features: {self.categorical_features}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            raise DataProcessError(f"Data cleaning failed: {str(e)}")
    
    def encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        try:
            X_encoded = X.copy()
            
            for col in self.categorical_features:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col])
                else:
                    X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
            
            logger.info("Categorical encoding completed")
            return X_encoded
            
        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            raise DataProcessError(f"Encoding failed: {str(e)}")
    
    def scale_numerical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        try:
            X_scaled = X.copy()
            
            if fit:
                X_scaled[self.numerical_features] = self.scaler.fit_transform(
                    X_scaled[self.numerical_features]
                )
            else:
                X_scaled[self.numerical_features] = self.scaler.transform(
                    X_scaled[self.numerical_features]
                )
            
            logger.info("Numerical scaling completed")
            return X_scaled
            
        except Exception as e:
            logger.error(f"Scaling failed: {str(e)}")
            raise DataProcessError(f"Scaling failed: {str(e)}")