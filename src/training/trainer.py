"""Model training pipeline."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates the training of all predictive maintenance models."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.training_history: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}

    def train_rul_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None,
                        model_type: str = "lstm",
                        **kwargs) -> Dict[str, Any]:
        """Train RUL prediction model."""
        from ..models.rul.lstm_model import RULPredictor
        
        logger.info(f"Training RUL model ({model_type})...")
        
        predictor = RULPredictor(
            sequence_length=X_train.shape[1],
            n_features=X_train.shape[2]
        )
        
        predictor.build_model(model_type)
        history = predictor.train(X_train, y_train, X_val, y_val, **kwargs)
        
        self.models["rul"] = predictor
        self.training_history["rul"] = {
            "model_type": model_type,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
        
        return history

    def train_anomaly_model(self, X_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           model_type: str = "autoencoder",
                           **kwargs) -> Dict[str, Any]:
        """Train anomaly detection model."""
        if model_type == "autoencoder":
            from ..models.anomaly.autoencoder import AnomalyDetector
            
            detector = AnomalyDetector(
                n_features=X_train.shape[1] if len(X_train.shape) > 1 else X_train.shape[-1]
            )
            detector.build_model()
            history = detector.train(X_train, X_val, **kwargs)
            
            self.models["anomaly"] = detector
            
        elif model_type == "isolation_forest":
            from ..models.anomaly.autoencoder import IsolationForestDetector
            
            detector = IsolationForestDetector()
            
            # Reshape if needed
            if len(X_train.shape) == 3:
                X_train_2d = X_train.reshape(-1, X_train.shape[-1])
            else:
                X_train_2d = X_train
            
            detector.train(X_train_2d)
            history = {"trained": True}
            
            self.models["anomaly"] = detector
        
        else:
            raise ValueError(f"Unknown anomaly model type: {model_type}")
        
        self.training_history["anomaly"] = {
            "model_type": model_type,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
        
        return history

    def train_classification_model(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: Optional[np.ndarray] = None,
                                   y_val: Optional[np.ndarray] = None,
                                   **kwargs) -> Dict[str, Any]:
        """Train failure classification model."""
        from ..models.classification.xgboost_model import FailureClassifier
        
        logger.info("Training classification model...")
        
        n_classes = len(np.unique(y_train))
        classifier = FailureClassifier(n_classes=n_classes)
        
        classifier.build_models()
        results = classifier.train(X_train, y_train, X_val, y_val, **kwargs)
        
        self.models["classification"] = classifier
        self.training_history["classification"] = {
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        return results

    def train_all_models(self, sensor_data: pd.DataFrame,
                        maintenance_data: pd.DataFrame,
                        feature_engineer=None,
                        scaler=None) -> Dict[str, Any]:
        """Train all models with the provided data."""
        from ..preprocessing.feature_engine import FeatureEngineer
        from ..preprocessing.scaler import DataScaler, TimeSeriesScaler
        
        results = {}
        
        # Engineer features if not provided
        if feature_engineer is None:
            feature_engineer = FeatureEngineer()
        
        if scaler is None:
            scaler = DataScaler()
        
        # Prepare features
        logger.info("Engineering features...")
        featured_data = feature_engineer.create_sensor_features(sensor_data)
        
        # Create sequences for RUL model
        logger.info("Creating sequences for RUL model...")
        ts_scaler = TimeSeriesScaler()
        
        feature_cols = [col for col in featured_data.columns 
                       if col not in ["vehicle_id", "timestamp", "failure_type"]]
        
        # Add RUL labels
        featured_data = feature_engineer.create_rul_labels(featured_data)
        
        X_seq, y_seq = ts_scaler.create_sequences(
            featured_data, 
            feature_cols=feature_cols[:10],  # Use top 10 features
            target_col="rul_days",
            sequence_length=10
        )
        
        if len(X_seq) > 0:
            # Scale sequences
            X_seq_scaled = ts_scaler.fit_transform_sequence(X_seq)
            
            # Split data
            split_idx = int(len(X_seq_scaled) * 0.8)
            X_train_rul = X_seq_scaled[:split_idx]
            y_train_rul = y_seq[:split_idx]
            X_val_rul = X_seq_scaled[split_idx:]
            y_val_rul = y_seq[split_idx:]
            
            # Train RUL model
            results["rul"] = self.train_rul_model(
                X_train_rul, y_train_rul,
                X_val_rul, y_val_rul
            )
        
        # Prepare 2D features for other models
        featured_data_2d = scaler.fit_transform_numerical(featured_data, feature_cols)
        
        # Train anomaly detection
        logger.info("Training anomaly detection model...")
        split_idx = int(len(featured_data_2d) * 0.8)
        X_train_anomaly = featured_data_2d[feature_cols].values[:split_idx]
        X_val_anomaly = featured_data_2d[feature_cols].values[split_idx:]
        
        results["anomaly"] = self.train_anomaly_model(
            X_train_anomaly, X_val_anomaly
        )
        
        # Train classification model
        if "failure_type" in maintenance_data.columns:
            logger.info("Training classification model...")
            # Merge sensor features with maintenance data
            merged_data = self._merge_data(featured_data_2d, maintenance_data)
            
            if len(merged_data) > 0:
                X_class = merged_data[feature_cols].values
                y_class = merged_data["failure_type"].values
                
                split_idx = int(len(X_class) * 0.8)
                X_train_class = X_class[:split_idx]
                y_train_class = y_class[:split_idx]
                X_val_class = X_class[split_idx:]
                y_val_class = y_class[split_idx:]
                
                results["classification"] = self.train_classification_model(
                    X_train_class, y_train_class,
                    X_val_class, y_val_class
                )
        
        return results

    def _merge_data(self, sensor_features: pd.DataFrame, 
                   maintenance_data: pd.DataFrame) -> pd.DataFrame:
        """Merge sensor features with maintenance data."""
        # Find common vehicle IDs
        common_vehicles = set(sensor_features["vehicle_id"].unique()) & \
                         set(maintenance_data["vehicle_id"].unique())
        
        if not common_vehicles:
            logger.warning("No common vehicles between sensor and maintenance data")
            return pd.DataFrame()
        
        # Aggregate sensor features per vehicle
        sensor_agg = sensor_features[sensor_features["vehicle_id"].isin(common_vehicles)].groupby("vehicle_id").mean().reset_index()
        
        # Merge
        merged = pd.merge(sensor_agg, maintenance_data, on="vehicle_id", how="inner")
        
        return merged

    def evaluate_models(self, X_test: np.ndarray, y_test_rul: Optional[np.ndarray] = None,
                       y_test_class: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate all trained models."""
        results = {}
        
        # Evaluate RUL model
        if "rul" in self.models and y_test_rul is not None:
            results["rul"] = self.models["rul"].evaluate(X_test, y_test_rul)
        
        # Evaluate classification model
        if "classification" in self.models and y_test_class is not None:
            results["classification"] = self.models["classification"].evaluate(X_test, y_test_class)
        
        return results

    def save_models(self, model_dir: str):
        """Save all trained models."""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name == "rul":
                filepath = model_path / "rul_model.keras"
                model.save_model(str(filepath))
            elif model_name == "anomaly":
                filepath = model_path / "anomaly_model.keras"
                model.save_model(str(filepath))
            elif model_name == "classification":
                filepath = model_path / "classification_model.pkl"
                model.save_model(str(filepath))
        
        # Save training history
        history_path = model_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info(f"All models saved to {model_dir}")

    def load_models(self, model_dir: str):
        """Load trained models."""
        model_path = Path(model_dir)
        
        # Load RUL model
        rul_path = model_path / "rul_model.keras"
        if rul_path.exists():
            from ..models.rul.lstm_model import RULPredictor
            predictor = RULPredictor()
            predictor.load_model(str(rul_path))
            self.models["rul"] = predictor
        
        # Load anomaly model
        anomaly_path = model_path / "anomaly_model.keras"
        if anomaly_path.exists():
            from ..models.anomaly.autoencoder import AnomalyDetector
            detector = AnomalyDetector()
            detector.load_model(str(anomaly_path))
            self.models["anomaly"] = detector
        
        # Load classification model
        class_path = model_path / "classification_model.pkl"
        if class_path.exists():
            from ..models.classification.xgboost_model import FailureClassifier
            classifier = FailureClassifier()
            classifier.load_model(str(class_path))
            self.models["classification"] = classifier
        
        # Load training history
        history_path = model_path / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.training_history = json.load(f)
        
        logger.info(f"Models loaded from {model_dir}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training runs."""
        summary = {}
        
        for model_name, history in self.training_history.items():
            summary[model_name] = {
                "model_type": history.get("model_type", "unknown"),
                "timestamp": history.get("timestamp", "unknown"),
                "metrics": history.get("results", {})
            }
        
        return summary
