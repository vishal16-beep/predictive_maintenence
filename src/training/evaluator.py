"""Model evaluation and metrics module."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation for predictive maintenance models."""

    def __init__(self):
        self.evaluation_results: Dict[str, Any] = {}
        self.comparison_results: Dict[str, Any] = {}

    def evaluate_rul_model(self, model, X_test: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate RUL prediction model."""
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAPE
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
        
        # Symmetric MAPE
        smape = np.mean(2 * np.abs(predictions - y_test) / 
                       (np.abs(predictions) + np.abs(y_test) + 1e-8)) * 100
        
        # Calculate accuracy within thresholds
        errors = np.abs(predictions - y_test)
        within_1_day = np.mean(errors <= 1) * 100
        within_3_days = np.mean(errors <= 3) * 100
        within_7_days = np.mean(errors <= 7) * 100
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "smape": float(smape),
            "within_1_day": float(within_1_day),
            "within_3_days": float(within_3_days),
            "within_7_days": float(within_7_days),
            "n_samples": len(y_test)
        }
        
        self.evaluation_results["rul"] = metrics
        logger.info(f"RUL Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
        
        return metrics

    def evaluate_anomaly_model(self, model, X_test: np.ndarray,
                               y_test: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate anomaly detection model."""
        results = model.detect_anomalies(X_test)
        
        predictions = results["is_anomaly"]
        scores = results["anomaly_scores"]
        
        metrics = {
            "n_samples": len(X_test),
            "n_anomalies_detected": int(np.sum(predictions)),
            "anomaly_rate": float(np.mean(predictions)),
            "mean_anomaly_score": float(np.mean(scores)),
            "max_anomaly_score": float(np.max(scores))
        }
        
        # If ground truth is available, calculate classification metrics
        if y_test is not None:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score, confusion_matrix
            )
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            
            try:
                roc_auc = roc_auc_score(y_test, scores)
            except Exception:
                roc_auc = 0.0
            
            cm = confusion_matrix(y_test, predictions)
            
            metrics.update({
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "confusion_matrix": cm.tolist()
            })
        
        self.evaluation_results["anomaly"] = metrics
        logger.info(f"Anomaly Evaluation: Detected {metrics['n_anomalies_detected']} anomalies")
        
        return metrics

    def evaluate_classification_model(self, model, X_test: np.ndarray,
                                     y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate failure classification model."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report, confusion_matrix,
            roc_auc_score
        )
        
        predictions, probabilities = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
        
        # Classification report
        report = classification_report(y_test, predictions, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # ROC AUC
        try:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, probabilities[:, 1])
            else:
                roc_auc = roc_auc_score(
                    pd.get_dummies(y_test).values,
                    probabilities,
                    multi_class="ovr",
                    average="weighted"
                )
        except Exception:
            roc_auc = 0.0
        
        # Per-class metrics
        classes = np.unique(y_test)
        per_class_metrics = {}
        for i, cls in enumerate(classes):
            per_class_metrics[cls] = {
                "precision": float(precision_score(y_test, predictions, average=None, labels=[cls])[0]),
                "recall": float(recall_score(y_test, predictions, average=None, labels=[cls])[0]),
                "f1": float(f1_score(y_test, predictions, average=None, labels=[cls])[0]),
                "support": int(np.sum(y_test == cls))
            }
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "n_classes": len(classes)
        }
        
        self.evaluation_results["classification"] = metrics
        logger.info(f"Classification Evaluation: Acc={accuracy:.4f}, F1={f1:.4f}")
        
        return metrics

    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray,
                       y_test: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Compare multiple models on the same test set."""
        comparisons = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, "predict"):
                    # Check model type
                    if hasattr(model, "detect_anomalies"):
                        # Anomaly detection model
                        metrics = self.evaluate_anomaly_model(model, X_test, y_test)
                    elif hasattr(model, "model") and hasattr(model.model, "predict"):
                        # Deep learning model (RUL)
                        if y_test is not None:
                            metrics = self.evaluate_rul_model(model, X_test, y_test)
                        else:
                            metrics = {"model_type": "deep_learning"}
                    else:
                        # Sklearn-style classifier
                        if y_test is not None:
                            metrics = self.evaluate_classification_model(model, X_test, y_test)
                        else:
                            metrics = {"model_type": "classifier"}
                    
                    comparisons.append({
                        "model_name": model_name,
                        "metrics": metrics
                    })
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                comparisons.append({
                    "model_name": model_name,
                    "error": str(e)
                })
        
        self.comparison_results = {
            "models": comparisons,
            "timestamp": datetime.now().isoformat()
        }
        
        return pd.DataFrame(comparisons)

    def calculate_business_metrics(self, predictions: np.ndarray,
                                  actuals: np.ndarray,
                                  cost_per_false_positive: float = 100,
                                  cost_per_false_negative: float = 1000,
                                  cost_per_true_positive: float = -50) -> Dict[str, float]:
        """Calculate business-relevant metrics."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(actuals, predictions)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Multi-class: calculate for each class
            tp = np.diag(cm)
            fp = cm.sum(axis=1) - tp
            fn = cm.sum(axis=0) - tp
            tn = cm.sum() - (tp + fp + fn)
        
        # Calculate costs
        total_cost = (
            fp.sum() * cost_per_false_positive +
            fn.sum() * cost_per_false_negative +
            tp.sum() * cost_per_true_positive
        )
        
        # Calculate savings from true positives
        # (correctly identifying issues before failure)
        prevented_failures = tp.sum()
        estimated_savings = prevented_failures * 5000  # Assume $5000 per prevented failure
        
        metrics = {
            "total_cost": float(total_cost),
            "prevented_failures": int(prevented_failures),
            "estimated_savings": float(estimated_savings),
            "net_benefit": float(estimated_savings - total_cost),
            "cost_per_prediction": float(total_cost / len(predictions)),
            "roi": float((estimated_savings - total_cost) / (total_cost + 1e-8) * 100)
        }
        
        return metrics

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "models_evaluated": list(self.evaluation_results.keys()),
            "metrics": self.evaluation_results,
            "summary": {}
        }
        
        # Add summary for each model
        for model_name, metrics in self.evaluation_results.items():
            if model_name == "rul":
                report["summary"][model_name] = {
                    "rmse": metrics.get("rmse", 0),
                    "mae": metrics.get("mae", 0),
                    "r2": metrics.get("r2", 0),
                    "within_7_days_accuracy": metrics.get("within_7_days", 0)
                }
            elif model_name == "anomaly":
                report["summary"][model_name] = {
                    "anomaly_rate": metrics.get("anomaly_rate", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0)
                }
            elif model_name == "classification":
                report["summary"][model_name] = {
                    "accuracy": metrics.get("accuracy", 0),
                    "f1_score": metrics.get("f1_score", 0),
                    "roc_auc": metrics.get("roc_auc", 0)
                }
        
        return report

    def save_report(self, filepath: str):
        """Save evaluation report to file."""
        report = self.generate_evaluation_report()
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {filepath}")

    def load_report(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation report from file."""
        with open(filepath, "r") as f:
            report = json.load(f)
        
        self.evaluation_results = report.get("metrics", {})
        return report


class CrossValidator:
    """Cross-validation utilities for model evaluation."""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.cv_results: Dict[str, Any] = {}

    def cross_validate_rul(self, model_factory, X: np.ndarray, y: np.ndarray,
                           **model_kwargs) -> Dict[str, Any]:
        """Perform cross-validation for RUL model."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_factory(**model_kwargs)
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            fold_scores.append(metrics)
            
            logger.info(f"Fold {fold + 1}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        
        # Aggregate results
        aggregated = {
            "mean_rmse": np.mean([s["rmse"] for s in fold_scores]),
            "std_rmse": np.std([s["rmse"] for s in fold_scores]),
            "mean_mae": np.mean([s["mae"] for s in fold_scores]),
            "std_mae": np.std([s["mae"] for s in fold_scores]),
            "mean_r2": np.mean([s["r2"] for s in fold_scores]),
            "std_r2": np.std([s["r2"] for s in fold_scores]),
            "fold_scores": fold_scores
        }
        
        self.cv_results["rul"] = aggregated
        return aggregated

    def cross_validate_classification(self, model_factory, X: np.ndarray, y: np.ndarray,
                                     **model_kwargs) -> Dict[str, Any]:
        """Perform cross-validation for classification model."""
        from sklearn.model_selection import StratifiedKFold
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = model_factory(**model_kwargs)
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            fold_scores.append(metrics)
            
            logger.info(f"Fold {fold + 1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        # Aggregate results
        aggregated = {
            "mean_accuracy": np.mean([s["accuracy"] for s in fold_scores]),
            "std_accuracy": np.std([s["accuracy"] for s in fold_scores]),
            "mean_f1": np.mean([s["f1_score"] for s in fold_scores]),
            "std_f1": np.std([s["f1_score"] for s in fold_scores]),
            "fold_scores": fold_scores
        }
        
        self.cv_results["classification"] = aggregated
        return aggregated
