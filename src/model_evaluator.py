import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, 
    recall_score, confusion_matrix, roc_curve, auc,
    classification_report, precision_recall_curve
)
from utils.logger import logger
from utils.exceptions import ModelPredictionError

class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self):
        self.results = {}
        self.metrics = {}
    
    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance"""
        try:
            logger.info("Evaluating model")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"Evaluation results:")
            logger.info(f"  Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {self.metrics['f1_score']:.4f}")
            logger.info(f"  AUC-ROC: {self.metrics['roc_auc']:.4f}")
            logger.info(f"  Precision: {self.metrics['precision']:.4f}")
            logger.info(f"  Recall: {self.metrics['recall']:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            raise ModelPredictionError(f"Model evaluation failed: {str(e)}")
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series, 
                              save_path: str = None):
        """Plot confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Confusion matrix saved to {save_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {str(e)}")
    
    def plot_roc_curve(self, y_true: pd.Series, y_proba: pd.Series,
                       save_path: str = None):
        """Plot ROC curve"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"ROC curve saved to {save_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot ROC curve: {str(e)}")
    
    def plot_precision_recall_curve(self, y_true: pd.Series, y_proba: pd.Series,
                                    save_path: str = None):
        """Plot Precision-Recall curve"""
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"PR curve saved to {save_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot PR curve: {str(e)}")
    
    def plot_feature_importance(self, model, feature_names: list,
                                top_n: int = 20, save_path: str = None):
        """Plot feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create DataFrame
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
                
                plt.figure(figsize=(10, 8))
                plt.barh(feature_importance['feature'], feature_importance['importance'])
                plt.xlabel('Importance')
                plt.title(f'Top {top_n} Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path)
                    logger.info(f"Feature importance saved to {save_path}")
                plt.show()
                
                return feature_importance
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return None
                
        except Exception as e:
            logger.error(f"Failed to plot feature importance: {str(e)}")
            return None
    
    def get_classification_report(self, y_true: pd.Series, y_pred: pd.Series) -> str:
        """Get classification report as string"""
        try:
            report = classification_report(y_true, y_pred, 
                                         target_names=['No Churn', 'Churn'])
            logger.info(f"\nClassification Report:\n{report}")
            return report
        except Exception as e:
            logger.error(f"Failed to get classification report: {str(e)}")
            return ""