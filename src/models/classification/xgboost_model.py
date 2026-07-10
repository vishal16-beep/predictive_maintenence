"""Failure classification model using ensemble methods."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import pickle

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import (
        RandomForestClassifier, 
        GradientBoostingClassifier,
        VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Classification model will not work.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class FailureClassifier:
    """Ensemble classifier for failure type prediction."""

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.models: Dict[str, Any] = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.training_results: Dict[str, Dict] = {}

    def build_models(self) -> Dict[str, Any]:
        """Build multiple classification models."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for classification")
        
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class="multinomial"
            )
        }
        
        if XGBOOST_AVAILABLE:
            self.models["xgboost"] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="multi:softprob",
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss"
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models["lightgbm"] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        logger.info(f"Built {len(self.models)} classification models")
        return self.models

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all models and select the best one."""
        if not self.models:
            self.build_models()
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.class_names = self.label_encoder.classes_.tolist()
        
        if y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
        
        best_score = 0
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train_encoded)
                
                # Evaluate on training set
                train_pred = model.predict(X_train)
                train_score = accuracy_score(y_train_encoded, train_pred)
                
                # Evaluate on validation set
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_score = accuracy_score(y_val_encoded, val_pred)
                    val_f1 = f1_score(y_val_encoded, val_pred, average="weighted")
                else:
                    val_score = train_score
                    val_f1 = 0
                
                results[name] = {
                    "train_accuracy": float(train_score),
                    "val_accuracy": float(val_score),
                    "val_f1": float(val_f1)
                }
                
                logger.info(f"{name}: Train Acc={train_score:.4f}, Val Acc={val_score:.4f}")
                
                # Update best model
                if val_score > best_score:
                    best_score = val_score
                    self.best_model = model
                    self.best_model_name = name
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        self.training_results = results
        
        logger.info(f"Best model: {self.best_model_name} with accuracy {best_score:.4f}")
        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict failure type and probabilities."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        # Get predictions
        predictions_encoded = self.best_model.predict(X)
        
        # Get probabilities
        probabilities = self.best_model.predict_proba(X)
        
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions, probabilities

    def predict_with_confidence(self, X: np.ndarray, 
                                confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Predict with confidence scores and flags low-confidence predictions."""
        predictions, probabilities = self.predict(X)
        
        max_probabilities = np.max(probabilities, axis=1)
        is_confident = max_probabilities >= confidence_threshold
        
        results = []
        for i in range(len(X)):
            results.append({
                "prediction": predictions[i],
                "confidence": float(max_probabilities[i]),
                "is_confident": bool(is_confident[i]),
                "probabilities": {
                    self.class_names[j]: float(probabilities[i, j])
                    for j in range(len(self.class_names))
                }
            })
        
        return {
            "predictions": results,
            "n_confident": int(np.sum(is_confident)),
            "n_low_confidence": int(np.sum(~is_confident)),
            "avg_confidence": float(np.mean(max_probabilities))
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        # Encode labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Get predictions
        predictions_encoded = self.best_model.predict(X_test)
        probabilities = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, predictions_encoded)
        precision = precision_score(y_test_encoded, predictions_encoded, average="weighted")
        recall = recall_score(y_test_encoded, predictions_encoded, average="weighted")
        f1 = f1_score(y_test_encoded, predictions_encoded, average="weighted")
        
        # Classification report
        report = classification_report(
            y_test_encoded, predictions_encoded,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, predictions_encoded)
        
        # ROC AUC (if binary or probability-based)
        try:
            if self.n_classes == 2:
                roc_auc = roc_auc_score(y_test_encoded, probabilities[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_test_encoded, probabilities, 
                    multi_class="ovr", average="weighted"
                )
        except Exception:
            roc_auc = 0.0
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
        logger.info(f"Classification Evaluation: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={roc_auc:.4f}")
        return metrics

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from the best model."""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_).mean(axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            "feature": feature_names[:len(importances)],
            "importance": importances
        })
        
        return importance_df.sort_values("importance", ascending=False)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation for all models."""
        if not self.models:
            self.build_models()
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        results = {}
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy")
                results[name] = {
                    "mean_accuracy": float(scores.mean()),
                    "std_accuracy": float(scores.std()),
                    "scores": scores.tolist()
                }
                logger.info(f"{name}: CV Accuracy = {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                logger.error(f"Error in cross-validation for {name}: {e}")
        
        return results

    def create_ensemble(self, voting: str = "soft") -> Any:
        """Create ensemble model from top performers."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ensemble")
        
        # Select top 3 models based on training results
        sorted_models = sorted(
            self.training_results.items(),
            key=lambda x: x[1].get("val_accuracy", 0),
            reverse=True
        )[:3]
        
        estimators = [
            (name, self.models[name])
            for name, _ in sorted_models
            if name in self.models
        ]
        
        if len(estimators) < 2:
            logger.warning("Not enough models for ensemble")
            return None
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting
        )
        
        self.models["ensemble"] = ensemble
        logger.info(f"Created ensemble from: {[name for name, _ in estimators]}")
        
        return ensemble

    def save_model(self, filepath: str):
        """Save the best model and encoders."""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "label_encoder": self.label_encoder,
            "class_names": self.class_names,
            "n_classes": self.n_classes,
            "training_results": self.training_results
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model and encoders."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.best_model = data["model"]
        self.best_model_name = data["model_name"]
        self.label_encoder = data["label_encoder"]
        self.class_names = data["class_names"]
        self.n_classes = data["n_classes"]
        self.training_results = data["training_results"]
        
        logger.info(f"Model loaded from {filepath}")


class MultiOutputClassifier:
    """Multi-output classifier for predicting multiple failure attributes."""

    def __init__(self):
        self.classifiers: Dict[str, FailureClassifier] = {}
        self.target_names: List[str] = []

    def train(self, X_train: np.ndarray, targets: Dict[str, np.ndarray],
              X_val: Optional[np.ndarray] = None,
              targets_val: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """Train separate classifiers for each target."""
        results = {}
        
        for target_name, y_train in targets.items():
            logger.info(f"Training classifier for: {target_name}")
            
            classifier = FailureClassifier(n_classes=len(np.unique(y_train)))
            classifier.build_models()
            
            y_val = targets_val.get(target_name) if targets_val else None
            classifier.train(X_train, y_train, X_val, y_val)
            
            self.classifiers[target_name] = classifier
            self.target_names.append(target_name)
            
            results[target_name] = classifier.training_results
        
        return results

    def predict(self, X: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Predict all targets."""
        predictions = {}
        
        for target_name, classifier in self.classifiers.items():
            preds, probs = classifier.predict(X)
            predictions[target_name] = (preds, probs)
        
        return predictions

    def evaluate(self, X_test: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Evaluate all classifiers."""
        results = {}
        
        for target_name, y_test in targets.items():
            if target_name in self.classifiers:
                results[target_name] = self.classifiers[target_name].evaluate(X_test, y_test)
        
        return results

    def save_models(self, directory: str):
        """Save all classifiers."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for target_name, classifier in self.classifiers.items():
            filepath = os.path.join(directory, f"{target_name}_classifier.pkl")
            classifier.save_model(filepath)

    def load_models(self, directory: str):
        """Load all classifiers."""
        import os
        
        for filename in os.listdir(directory):
            if filename.endswith("_classifier.pkl"):
                target_name = filename.replace("_classifier.pkl", "")
                filepath = os.path.join(directory, filename)
                
                classifier = FailureClassifier()
                classifier.load_model(filepath)
                
                self.classifiers[target_name] = classifier
                self.target_names.append(target_name)
