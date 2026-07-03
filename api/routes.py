from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from api.schemas import CustomerInput, PredictionResponse
from utils.logger import logger

router = APIRouter()

# Global variables
model = None
label_encoders = None
scaler = None
feature_columns = None

def load_artifacts():
    """Load trained model and preprocessors"""
    global model, label_encoders, scaler, feature_columns
    
    try:
        model = joblib.load('models/churn_model.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        logger.info("✅ Artifacts loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        return False

def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess input data for prediction"""
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
    
    df = df.rename(columns=column_mapping)
    
    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                df[col] = 0
    
    # Scale numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_columns]

def get_recommendations(proba: float, data: dict) -> list:
    """Get personalized recommendations"""
    recommendations = []
    
    if proba > 0.7:
        recommendations.append("Offer 25% discount on 2-year contract")
        recommendations.append("Provide free premium security package")
        recommendations.append("Priority customer support")
        recommendations.append("Loyalty rewards program")
        
        if data.get('contract') == 'Month-to-month':
            recommendations.append("Offer to switch to annual contract with benefits")
        
        if data.get('online_security') == 'No':
            recommendations.append("Free online security setup")
            
    elif proba > 0.4:
        recommendations.append("Offer 15% discount on 1-year contract")
        recommendations.append("Free online security trial")
        recommendations.append("Enhanced customer service")
        
        if data.get('tenure', 0) < 12:
            recommendations.append("Welcome bonus for new customers")
            
    else:
        recommendations.append("Continue excellent service")
        recommendations.append("Loyalty rewards")
        recommendations.append("Regular feedback surveys")
    
    return recommendations

def get_risk_level(proba: float) -> str:
    """Get risk level"""
    if proba > 0.7:
        return "High"
    elif proba > 0.4:
        return "Medium"
    else:
        return "Low"

@router.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerInput):
    """Predict churn probability for a customer"""
    try:
        logger.info("📊 Prediction request received")
        
        # Load artifacts if not loaded
        if model is None:
            if not load_artifacts():
                raise HTTPException(status_code=503, detail="Model not available")
        
        # Convert input to dict
        data = customer.dict()
        
        # Preprocess
        processed_data = preprocess_input(data)
        
        # Predict
        proba = model.predict_proba(processed_data)[0, 1]
        
        # Determine prediction
        prediction = "Will Churn" if proba > 0.5 else "Will Not Churn"
        
        # Calculate confidence
        confidence = max(proba, 1 - proba)
        
        # Get risk level
        risk_level = get_risk_level(proba)
        
        # Get recommendations
        recommendations = get_recommendations(proba, data)
        
        # Get top factors
        top_factors = []
        if data.get('contract') == 'Month-to-month':
            top_factors.append("Month-to-month contract (high risk)")
        if data.get('tenure', 0) < 12:
            top_factors.append(f"Short tenure ({data['tenure']} months)")
        if data.get('monthly_charges', 0) > 100:
            top_factors.append(f"High monthly charges (${data['monthly_charges']})")
        if data.get('online_security') == 'No':
            top_factors.append("No online security")
        if data.get('tech_support') == 'No':
            top_factors.append("No tech support")
        if data.get('internet_service') == 'Fiber optic' and data.get('online_security') == 'No':
            top_factors.append("Fiber optic without security")
            
        if not top_factors:
            top_factors.append("Customer profile shows low risk factors")
        
        response = PredictionResponse(
            churn_probability=round(proba, 4),
            prediction=prediction,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            recommendations=recommendations[:5],
            top_factors=top_factors[:5]
        )
        
        logger.info(f"✅ Prediction complete: {prediction} ({proba:.4f})")
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """Get model information"""
    if model is None:
        load_artifacts()
    
    return {
        "model_type": "XGBoost" if model else "Not loaded",
        "features": feature_columns,
        "status": "loaded" if model is not None else "not loaded",
        "performance": {
            "accuracy": 0.8048,
            "f1_score": 0.5914
        }
    }