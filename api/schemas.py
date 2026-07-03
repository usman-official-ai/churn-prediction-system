from pydantic import BaseModel, Field 
from typing import List 
from enum import Enum 
 
class ContractType(str, Enum): 
    MONTH_TO_MONTH = "Month-to-month" 
    ONE_YEAR = "One year" 
    TWO_YEAR = "Two year" 
 
class PaymentMethod(str, Enum): 
    BANK_TRANSFER = "Bank transfer (automatic)" 
    CREDIT_CARD = "Credit card (automatic)" 
    ELECTRONIC_CHECK = "Electronic check" 
    MAILED_CHECK = "Mailed check" 
 
class CustomerInput(BaseModel): 
    gender: str 
    senior_citizen: str 
    partner: str 
    dependents: str 
    phone_service: str 
    multiple_lines: str 
    internet_service: str 
    online_security: str 
    online_backup: str 
    device_protection: str 
    tech_support: str 
    streaming_tv: str 
    streaming_movies: str 
    contract: ContractType 
    paperless_billing: str 
    payment_method: PaymentMethod 
    tenure: int 
    monthly_charges: float 
    total_charges: float 
 
class PredictionResponse(BaseModel): 
    churn_probability: float 
    prediction: str 
    risk_level: str 
    confidence: float 
    recommendations: List[str] 
    top_factors: List[str] 
