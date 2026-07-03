import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, f1_score, classification_report 
import joblib 
import warnings 
warnings.filterwarnings('ignore') 
import os 
 
print("="*60) 
print("CUSTOMER CHURN MODEL TRAINING") 
print("="*60) 
 
print("\n1. Loading data...") 
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv') 
print(f"   Loaded {len(df)} rows") 
 
print("\n2. Cleaning data...") 
df = df.drop('customerID', axis=1) 
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') 
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True) 
df['SeniorCitizen'] = df['SeniorCitizen'].astype('object') 
print("   Data cleaned!") 
 
print("\n3. Preparing features...") 
X = df.drop('Churn', axis=1) 
y = df['Churn'].map({'Yes': 1, 'No': 0}) 
categorical_cols = X.select_dtypes(include=['object']).columns.tolist() 
label_encoders = {} 
for col in categorical_cols: 
    le = LabelEncoder() 
    X[col] = le.fit_transform(X[col]) 
    label_encoders[col] = le 
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges'] 
scaler = StandardScaler() 
X[numerical_cols] = scaler.fit_transform(X[numerical_cols]) 
print("   Features prepared!") 
 
print("\n4. Splitting data...") 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
print(f"   Train: {len(X_train)} samples") 
print(f"   Test: {len(X_test)} samples") 
 
print("\n5. Training models...") 
models = { 
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), 
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss') 
} 
results = {} 
for name, model in models.items(): 
    print(f"   Training {name}...") 
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test) 
    accuracy = accuracy_score(y_test, y_pred) 
    f1 = f1_score(y_test, y_pred) 
    results[name] = {'model': model, 'accuracy': accuracy, 'f1_score': f1} 
    print(f"      Accuracy: {accuracy:.4f}, F1: {f1:.4f}") 
 
print("\n6. Selecting best model...") 
best_name = max(results, key=lambda x: results[x]['f1_score']) 
best_model = results[best_name]['model'] 
print(f"   Best model: {best_name}") 
print(f"   Accuracy: {results[best_name]['accuracy']:.4f}") 
print(f"   F1 Score: {results[best_name]['f1_score']:.4f}") 
 
print("\n7. Saving models...") 
os.makedirs('models', exist_ok=True) 
joblib.dump(best_model, 'models/churn_model.pkl') 
joblib.dump(label_encoders, 'models/label_encoders.pkl') 
joblib.dump(scaler, 'models/scaler.pkl') 
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl') 
print("   Models saved successfully!") 
 
print("\n" + "="*60) 
print("TRAINING COMPLETE!") 
print("="*60) 
