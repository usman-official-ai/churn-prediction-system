import pandas as pd
import numpy as np
from utils.logger import logger
from utils.exceptions import DataProcessError

class FeatureEngineer:
    """Feature engineering for churn prediction"""
    
    def __init__(self):
        self.features = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        try:
            logger.info("Creating new features")
            df_fe = df.copy()
            
            # Tenure categories
            if 'tenure' in df_fe.columns:
                df_fe['tenure_category'] = pd.cut(
                    df_fe['tenure'],
                    bins=[0, 12, 24, 48, 100],
                    labels=['0-12 months', '13-24 months', '25-48 months', '48+ months']
                )
                logger.info("  Created tenure_category")
            
            # Monthly charges categories
            if 'MonthlyCharges' in df_fe.columns:
                df_fe['charge_category'] = pd.cut(
                    df_fe['MonthlyCharges'],
                    bins=[0, 30, 60, 150],
                    labels=['Low', 'Medium', 'High']
                )
                logger.info("  Created charge_category")
            
            # Average monthly charge
            if 'TotalCharges' in df_fe.columns and 'tenure' in df_fe.columns:
                df_fe['avg_monthly_charge'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 1)
                logger.info("  Created avg_monthly_charge")
            
            # Number of services
            service_cols = ['PhoneService', 'MultipleLines', 'InternetService',
                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            service_cols_existing = [col for col in service_cols if col in df_fe.columns]
            if service_cols_existing:
                df_fe['num_services'] = df_fe[service_cols_existing].apply(
                    lambda x: (x != 'No').sum(), axis=1
                )
                logger.info("  Created num_services")
            
            # Total monthly charges per service
            if 'MonthlyCharges' in df_fe.columns and 'num_services' in df_fe.columns:
                df_fe['charge_per_service'] = df_fe['MonthlyCharges'] / (df_fe['num_services'] + 1)
                logger.info("  Created charge_per_service")
            
            # Has internet service
            if 'InternetService' in df_fe.columns:
                df_fe['has_internet'] = (df_fe['InternetService'] != 'No').astype(int)
                logger.info("  Created has_internet")
            
            # Has security package
            security_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
            security_existing = [col for col in security_cols if col in df_fe.columns]
            if security_existing:
                df_fe['has_security'] = df_fe[security_existing].apply(
                    lambda x: (x == 'Yes').sum(), axis=1
                )
                logger.info("  Created has_security")
            
            # Interaction: Tenure * MonthlyCharges
            if 'tenure' in df_fe.columns and 'MonthlyCharges' in df_fe.columns:
                df_fe['tenure_charge_interaction'] = df_fe['tenure'] * df_fe['MonthlyCharges']
                logger.info("  Created tenure_charge_interaction")
            
            logger.info(f"Created {len(df_fe.columns) - len(df.columns)} new features")
            return df_fe
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise DataProcessError(f"Feature engineering failed: {str(e)}")
    
    def get_feature_names(self) -> list:
        """Get list of created feature names"""
        return self.features