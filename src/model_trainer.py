import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
import mlflow
from utils.logger import logger
from utils.exceptions import ModelTrainingError
from config.settings import settings

class ModelTrainer:
    """Train and evaluate multiple models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of models to train"""
        return {
            'logistic_regression': LogisticRegression(
                random_state=settings.RANDOM_STATE,
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=settings.RANDOM_STATE,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=settings.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        }
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train all models and return results"""
        try:
            logger.info("Starting model training")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=settings.TEST_SIZE,
                random_state=settings.RANDOM_STATE,
                stratify=y
            )
            
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            
            # Get models
            self.models = self.get_models()
            
            # Train each model
            for name, model in self.models.items():
                logger.info(f"Training {name}...")
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Cross validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=settings.RANDOM_STATE),
                    scoring='f1'
                )
                
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
                
                # Log with MLflow
                with mlflow.start_run(run_name=name):
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics({
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'roc_auc': roc_auc,
                        'cv_mean': cv_scores.mean()
                    })
                    mlflow.sklearn.log_model(model, name)
            
            # Select best model
            self.select_best_model()
            
            # Save best model
            self.save_best_model()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise ModelTrainingError(f"Model training failed: {str(e)}")
    
    def select_best_model(self) -> None:
        """Select best model based on F1 score"""
        best_score = 0
        for name, result in self.results.items():
            if result['f1_score'] > best_score:
                best_score = result['f1_score']
                self.best_model_name = name
                self.best_model = result['model']
        
        logger.info(f"Best model: {self.best_model_name} with F1: {best_score:.4f}")
    
    def save_best_model(self) -> None:
        """Save the best model"""
        if self.best_model is None:
            raise ModelTrainingError("No best model to save")
        
        model_path = settings.MODELS_DIR / f"{self.best_model_name}.pkl"
        joblib.dump(self.best_model, model_path)
        logger.info(f"Saved best model to {model_path}")