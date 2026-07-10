"""Model loader for inference."""
import os
import pickle
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load and manage trained models for inference."""

    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    def load_all_models(self):
        """Load all available models from the model directory."""
        if not self.model_dir.exists():
            logger.warning(f"Model directory not found: {self.model_dir}")
            return

        # Load RUL model
        rul_path = self.model_dir / "rul_model.keras"
        if rul_path.exists():
            self.load_rul_model(str(rul_path))

        # Load anomaly model
        anomaly_path = self.model_dir / "anomaly_model.keras"
        if anomaly_path.exists():
            self.load_anomaly_model(str(anomaly_path))

        # Load classification model
        class_path = self.model_dir / "classification_model.pkl"
        if class_path.exists():
            self.load_classification_model(str(class_path))

        # Load scalers
        scaler_path = self.model_dir / "scalers.pkl"
        if scaler_path.exists():
            self.load_scalers(str(scaler_path))

        logger.info(f"Loaded models: {list(self.models.keys())}")

    def load_rul_model(self, filepath: str) -> Any:
        """Load RUL prediction model."""
        try:
            from ..models.rul.lstm_model import RULPredictor
            
            predictor = RULPredictor()
            predictor.load_model(filepath)
            self.models["rul"] = predictor
            
            logger.info(f"Loaded RUL model from {filepath}")
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading RUL model: {e}")
            return None

    def load_anomaly_model(self, filepath: str) -> Any:
        """Load anomaly detection model."""
        try:
            from ..models.anomaly.autoencoder import AnomalyDetector
            
            detector = AnomalyDetector()
            detector.load_model(filepath)
            self.models["anomaly"] = detector
            
            logger.info(f"Loaded anomaly model from {filepath}")
            return detector
            
        except Exception as e:
            logger.error(f"Error loading anomaly model: {e}")
            return None

    def load_classification_model(self, filepath: str) -> Any:
        """Load failure classification model."""
        try:
            from ..models.classification.xgboost_model import FailureClassifier
            
            classifier = FailureClassifier()
            classifier.load_model(filepath)
            self.models["classification"] = classifier
            
            logger.info(f"Loaded classification model from {filepath}")
            return classifier
            
        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            return None

    def load_scalers(self, filepath: str):
        """Load preprocessing scalers."""
        try:
            from ..preprocessing.scaler import DataScaler, TimeSeriesScaler
            
            with open(filepath, "rb") as f:
                scaler_data = pickle.load(f)
            
            if "data_scaler" in scaler_data:
                self.scalers["data"] = scaler_data["data_scaler"]
            
            if "time_series_scaler" in scaler_data:
                self.scalers["time_series"] = scaler_data["time_series_scaler"]
            
            logger.info(f"Loaded scalers from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading scalers: {e}")

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a specific model by name."""
        return self.models.get(model_name)

    def get_scaler(self, scaler_name: str) -> Optional[Any]:
        """Get a specific scaler by name."""
        return self.scalers.get(scaler_name)

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models

    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of all models."""
        return {
            "rul": self.is_model_loaded("rul"),
            "anomaly": self.is_model_loaded("anomaly"),
            "classification": self.is_model_loaded("classification")
        }

    def reload_model(self, model_name: str):
        """Reload a specific model."""
        if model_name == "rul":
            filepath = self.model_dir / "rul_model.keras"
            if filepath.exists():
                self.load_rul_model(str(filepath))
        elif model_name == "anomaly":
            filepath = self.model_dir / "anomaly_model.keras"
            if filepath.exists():
                self.load_anomaly_model(str(filepath))
        elif model_name == "classification":
            filepath = self.model_dir / "classification_model.pkl"
            if filepath.exists():
                self.load_classification_model(str(filepath))

    def reload_all_models(self):
        """Reload all models."""
        self.models.clear()
        self.scalers.clear()
        self.load_all_models()
