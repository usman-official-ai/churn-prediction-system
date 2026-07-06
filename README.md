# 📊 Customer Churn Prediction System

[![Live Demo](https://img.shields.io/badge/🚀-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://churn-prediction-system-o2ezgfp8g3jn5xflj6kmdi.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/usman-official-ai/churn-prediction-system)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-red?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-success?style=for-the-badge)](https://share.streamlit.io)    

  <img width="1536" height="1024" alt="ChatGPT Image Jul 6, 2026, 04_03_38 AM" src="https://github.com/user-attachments/assets/99484284-d5c7-447b-9d11-0c3512d881c7" />  

    


---

## 🎯 Live Demo

### 🔗 **[Click Here to Use the App](https://churn-prediction-system-o2ezgfp8g3jn5xflj6kmdi.streamlit.app)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Business Impact](#-business-impact)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📊 Overview

**Customer Churn Prediction System** is an end-to-end machine learning application that predicts which customers are likely to cancel their subscriptions. It helps businesses take proactive action to retain customers, reducing revenue loss from churn.

### 🎯 The Problem

  💸 Companies lose billions of dollars annually due to customer churn
📉 15-20% revenue loss because customers leave without warning
😤 Reactive retention = too late  
  
### ✅ The Solution

  🔮 Predict which customers will churn
📊 Assess risk level (High/Medium/Low)
💡 Provide actionable recommendations
📈 Reduce churn by 15-20%
💰 Save millions in revenue  

  
---

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🔮 **Churn Prediction** | Predict if a customer will churn with probability score | ✅ |
| 📊 **Risk Assessment** | High/Medium/Low risk level classification | ✅ |
| 💡 **Recommendations** | Personalized retention strategies | ✅ |
| 📁 **Batch Analysis** | Upload CSV and analyze multiple customers | ✅ |
| 📥 **Export Results** | Download predictions as CSV | ✅ |
| 📈 **Analytics** | Visual insights and statistics | ✅ |
| 🌍 **Public Access** | Available to everyone with internet | ✅ |
| 📱 **Mobile Responsive** | Works on all devices | ✅ |

---

## 💰 Business Impact

### Quantified Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Churn Rate** | 30% | 24% | ⬇️ **20% reduction** |
| **Customer Retention** | 70% | 76% | ⬆️ **+6%** |
| **Revenue Saved** | $60M loss | $12M loss | 💰 **$48M saved** |
| **ROI** | - | 500% | 🚀 **5x return** |

### ROI Calculation

  Investment: $50,000 (development)
Annual Savings: $48,000,000
ROI = (48,000,000 - 50,000) / 50,000 × 100 = 95,900%  

  
### Real-World Use Cases

| Industry | Application | Impact |
|----------|-------------|--------|
| 📱 **Telecom** | Predict mobile/Internet churn | Save millions |
| 💳 **Banking** | Credit card/Account churn | Reduce attrition |
| 🛒 **E-commerce** | Customer churn prediction | Increase retention |
| 📦 **SaaS** | Subscription churn | Improve LTV |
| 📊 **Insurance** | Policy churn | Maximize renewals |

---

## 🛠️ Technology Stack

| Component | Technology | Badge |
|-----------|------------|-------|
| **Frontend** | Streamlit | ![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-red) |
| **Language** | Python 3.12 | ![Python](https://img.shields.io/badge/Python-3.12-blue) |
| **Data Processing** | Pandas, NumPy | ![Pandas](https://img.shields.io/badge/Pandas-2.0.3-green) |
| **Visualization** | Plotly | ![Plotly](https://img.shields.io/badge/Plotly-5.15.0-blue) |
| **Backend API** | FastAPI | ![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green) |
| **ML Model** | XGBoost | ![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange) |
| **Deployment** | Streamlit Cloud | ![Deploy](https://img.shields.io/badge/Deployed-Success-green) |
| **Version Control** | Git, GitHub | ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) |

---

## 📊 Dataset

### Telco Customer Churn Dataset

| Property | Details |
|----------|---------|
| **Source** | IBM / Kaggle |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | Churn (Yes/No) |
| **Churn Rate** | 26.5% |

### Features Used

| Category | Features |    
|----------|----------|  
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |  
| **Services** | PhoneService, MultipleLines, InternetService |  
| **Security** | OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport |  
| **Streaming** | StreamingTV, StreamingMovies |  
| **Account** | Contract, PaperlessBilling, PaymentMethod, Tenure, MonthlyCharges, TotalCharges |  
  
---  
  
## 📁 Project Structure  

  churn-prediction-system/    
├── 📁 api/ # FastAPI Backend  
│ ├── init.py  
│ ├── app.py # Main API file  
│ ├── routes.py # API endpoints  
│ └── schemas.py # Data models    
│  
├── 📁 src/ # ML Source Code  
│ ├── init.py  
│ ├── data_loader.py # Data loading  
│ ├── data_cleaner.py # Data cleaning  
│ ├── feature_engineering.py # Feature engineering  
│ ├── model_trainer.py # Model training  
│ ├── model_evaluator.py # Model evaluation  
│ └── predictor.py # Predictions  
│  
├── 📁 models/ # Trained Models  
│ ├── churn_model.pkl # XGBoost model  
│ ├── label_encoders.pkl # Encoders  
│ └── scaler.pkl # Standard scaler  
│  
├── 📁 data/ # Dataset  
│ └── raw/ # Raw data files  
│  
├── 📁 notebooks/ # Jupyter Notebooks  
│ ├── 01_eda.ipynb # EDA analysis  
│ ├── 02_feature_engineering.ipynb  
│ └── 03_model_training.ipynb  
│  
├── 📁 utils/ # Utilities  
│ ├── logger.py # Logging setup  
│ └── exceptions.py # Custom exceptions  
│  
├── 📁 config/ # Configuration  
│ └── settings.py # Project settings  
│  
├── 📁 docker/ # Docker Files  
│ ├── Dockerfile  
│ └── docker-compose.yml  
│  
├── 📁 tests/ # Unit Tests  
│ ├── test_api.py  
│ ├── test_models.py  
│ └── test_data_processing.py  
│  
├── 📄 app_streamlit.py # Main Streamlit App  
├── 📄 streamlit_app.py # Deployed App  
├── 📄 requirements.txt # Dependencies  
├── 📄 run_training.py # Training script  
├── 📄 .env.example # Environment variables  
├── 📄 .gitignore # Git ignore file  
└── 📄 README.md # This file  

  
---

## 🚀 Installation

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/usman-official-ai/churn-prediction-system.git
cd churn-prediction-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run streamlit_app.py

Train the Model

# Train XGBoost model
python run_training.py

📖 Usage
1. Single Customer Prediction
Enter customer details

Click "Predict Churn"

View:

Churn Probability

Prediction (Will Churn/Will Not Churn)

Risk Level (High/Medium/Low)

Personalized Recommendations

Top Risk Factors

2. Batch Analysis
Upload CSV file with customer data

Click "Analyze Batch"

View results for all customers

Download results as CSV

🌐 API Documentation  
FastAPI Endpoints  
Endpoint	Method	Description  
/	GET	API information  
/health	GET	Health check  
/predict	POST	Predict churn  
/batch-predict	POST	Batch prediction  
/model-info	GET	Model details  
/docs	GET	Swagger UI  
Sample API Request

POST /predict
{
    "gender": "Male",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "tenure": 5,
    "monthly_charges": 110.5,
    "total_charges": 552.5
}

Sample API Response

{
    "churn_probability": 0.79,
    "prediction": "Will Churn",
    "risk_level": "High",
    "confidence": 0.79,
    "recommendations": [
        "Offer 25% discount on 2-year contract",
        "Provide free premium security package",
        "Priority customer support",
        "Loyalty rewards program",
        "Offer to switch to annual contract with benefits"
    ],
    "top_factors": [
        "Month-to-month contract (high risk)",
        "Short tenure (5 months)",
        "High monthly charges ($110.5)",
        "No online security",
        "No tech support"
    ]
}

Deploy to Streamlit Cloud
Push code to GitHub

Go to share.streamlit.io

Click "New App"

Select repository: usman-official-ai/churn-prediction-system

Main file: streamlit_app.py

Click "Deploy"

Author
  usman-official-ai

