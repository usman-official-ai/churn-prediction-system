import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("IMPROVED CUSTOMER CHURN MODEL TRAINING")
print("=" * 60)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"   Loaded {len(df)} rows and {len(df.columns)} columns")

# 2. Clean data (Fixed pandas warning)
print("\n2. Cleaning data...")
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Fix: Use assignment instead of inplace
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

if 'SeniorCitizen' in df.columns:
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')

print(f"   Shape after cleaning: {df.shape}")

# 3. Prepare features
print("\n3. Preparing features...")
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

# 4. Encode categorical features
print("\n4. Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"   Encoded: {col}")

# 5. Scale numerical features
print("\n5. Scaling numerical features...")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print(f"   Scaled: {numerical_cols}")

# 6. Split data
print("\n6. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# 7. Train models with hyperparameter tuning
print("\n7. Training models with hyperparameter tuning...")

# Random Forest with Grid Search
print("\n   🔄 Training Random Forest...")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"      Best params: {rf_grid.best_params_}")
print(f"      Accuracy: {rf_accuracy:.4f}, F1-Score: {rf_f1:.4f}")

# XGBoost with Grid Search
print("\n   🔄 Training XGBoost...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='f1', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
xgb_pred = xgb_best.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
print(f"      Best params: {xgb_grid.best_params_}")
print(f"      Accuracy: {xgb_accuracy:.4f}, F1-Score: {xgb_f1:.4f}")

# 8. Select best model
print("\n8. Selecting best model...")
results = {
    'Random Forest': {'model': rf_best, 'accuracy': rf_accuracy, 'f1_score': rf_f1},
    'XGBoost': {'model': xgb_best, 'accuracy': xgb_accuracy, 'f1_score': xgb_f1}
}

best_name = max(results, key=lambda x: results[x]['f1_score'])
best_model = results[best_name]['model']
best_score = results[best_name]['f1_score']

print(f"   🏆 Best model: {best_name} with F1-Score: {best_score:.4f}")

# 9. Save artifacts
print("\n9. Saving artifacts...")
Path('models').mkdir(exist_ok=True)

joblib.dump(best_model, 'models/churn_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

print("   ✅ models/churn_model.pkl")
print("   ✅ models/label_encoders.pkl")
print("   ✅ models/scaler.pkl")
print("   ✅ models/feature_columns.pkl")

# 10. Detailed evaluation
print("\n10. Detailed evaluation...")
y_pred = best_model.predict(X_test)
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

print("\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   [[{cm[0][0]:4d} {cm[0][1]:4d}]")
print(f"    [{cm[1][0]:4d} {cm[1][1]:4d}]]")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\n📊 Final Results:")
for name, result in results.items():
    print(f"   {name}: Accuracy={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")

print(f"\n🏆 Best: {best_name} with F1={best_score:.4f}")
print("\n🚀 Model is ready! Restart your API:")
print("   uvicorn api.app:app --reload --host 0.0.0.0 --port 8000")