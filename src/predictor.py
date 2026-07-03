import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
from utils.logger import logger
from utils.exceptions import ModelPredictionError
from config.settings import settings

class Predictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.feature_columns = None
        self.model_path = model_path or settings.MODELS_DIR / "churn_model.pkl"
        
    def load_model(self) -> bool:
        """Load trained model and preprocessors"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load preprocessors
            base_path = Path(self.model_path).parent
            self.label_encoders = joblib.load(base_path / "label_encoders.pkl")
            self.scaler = joblib.load(base_path / "scaler.pkl")
            self.feature_columns = joblib.load(base_path / "feature_columns.pkl")
            
            logger.info("✅ Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess input data for prediction"""
        try:
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Map input fields to expected column names
            column_mapping = {
                'gender': 'gender',
                'senior_citizen': 'SeniorCitizen',
                'partner': 'Partner',
                'dependents': 'Dependents',
                'phone_service': 'PhoneService',
                'multiple_lines': 'MultipleLines',
                'internet_service': 'InternetService',
                'online_security': 'OnlineSecurity',
                'online_backup': 'OnlineBackup',
                'device_protection': 'DeviceProtection',
                'tech_support': 'TechSupport',
                'streaming_tv': 'StreamingTV',
                'streaming_movies': 'StreamingMovies',
                'contract': 'Contract',
                'paperless_billing': 'PaperlessBilling',
                'payment_method': 'PaymentMethod',
                'tenure': 'tenure',
                'monthly_charges': 'MonthlyCharges',
                'total_charges': 'TotalCharges'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Encode categorical features
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except:
                        # If value not seen during training, use most common
                        df[col] = 0
                else:
                    df[col] = 0
            
            # Scale numerical features
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
            # Ensure all feature columns are present
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[self.feature_columns]
            
            return df
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise ModelPredictionError(f"Preprocessing failed: {str(e)}")
    
    def predict(self, data: Dict[str, Any]) -> Tuple[float, int]:
        """Make prediction for a single customer"""
        try:
            if self.model is None:
                if not self.load_model():
                    raise ModelPredictionError("Model not loaded")
            
            # Preprocess input
            processed_data = self.preprocess_input(data)
            
            # Make prediction
            proba = self.model.predict_proba(processed_data)[0, 1]
            pred = 1 if proba > 0.5 else 0
            
            logger.info(f"Prediction: {pred} with probability {proba:.4f}")
            return proba, pred
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise ModelPredictionError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, data_list: list) -> list:
        """Make predictions for multiple customers"""
        try:
            results = []
            for data in data_list:
                proba, pred = self.predict(data)
                results.append({
                    'churn_probability': proba,
                    'prediction': 'Will Churn' if pred == 1 else 'Will Not Churn'
                })
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise ModelPredictionError(f"Batch prediction failed: {str(e)}")
    
    def get_prediction_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed prediction explanation"""
        try:
            proba, pred = self.predict(data)
            
            # Get top factors
            factors = []
            if data.get('contract') == 'Month-to-month':
                factors.append("Month-to-month contract")
            if data.get('tenure', 0) < 12:
                factors.append("Short tenure")
            if data.get('monthly_charges', 0) > 100:
                factors.append("High monthly charges")
            if data.get('online_security') == 'No':
                factors.append("No online security")
            if data.get('tech_support') == 'No':
                factors.append("No tech support")
            
            # Get recommendations
            recommendations = []
            if proba > 0.7:
                recommendations = [
                    "Offer 25% discount on 2-year contract",
                    "Provide free premium security",
                    "Priority customer support",
                    "Loyalty rewards program"
                ]
            elif proba > 0.4:
                recommendations = [
                    "Offer 15% discount on 1-year contract",
                    "Free online security trial",
                    "Enhanced customer service"
                ]
            else:
                recommendations = [
                    "Continue excellent service",
                    "Loyalty rewards",
                    "Regular feedback surveys"
                ]
            
            return {
                'churn_probability': proba,
                'prediction': 'Will Churn' if pred == 1 else 'Will Not Churn',
                'risk_level': 'High' if proba > 0.7 else 'Medium' if proba > 0.4 else 'Low',
                'confidence': max(proba, 1 - proba),
                'factors': factors,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {str(e)}")
            raise ModelPredictionError(f"Prediction explanation failed: {str(e)}")