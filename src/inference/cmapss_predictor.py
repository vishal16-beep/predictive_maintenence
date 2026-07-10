"""
C-MAPSS Prediction Service

Loads the trained XGBoost model and makes RUL predictions
using the correct feature engineering pipeline.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import xgboost as xgb


# Feature columns the model expects (in order)
FEATURE_COLUMNS = [
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21"
]

ROLLING_WINDOW = 10
RUL_CLIP = 125


class CMAPSSPredictor:
    """
    Predictor for C-MAPSS turbofan engine RUL.
    
    Handles:
    - Loading trained XGBoost model
    - Feature engineering from raw sensor readings
    - RUL prediction with urgency classification
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.metadata = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load the trained XGBoost model and metadata."""
        try:
            # Load XGBoost model
            model_path = self.model_dir / "xgboost_rul_model.json"
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                return False
            
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            
            # Load metadata
            metadata_path = self.model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            
            # Load feature columns
            features_path = self.model_dir / "feature_columns.json"
            if features_path.exists():
                with open(features_path, "r") as f:
                    self.feature_columns = json.load(f)
            else:
                self.feature_columns = FEATURE_COLUMNS
            
            self.is_loaded = True
            print(f"Model loaded successfully: {len(self.feature_columns)} features")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _compute_rolling_features(self, readings: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute rolling features from a sequence of readings.
        
        For each base feature, computes:
        - Rolling mean (window=10)
        - Rolling std (window=10)
        """
        if len(readings) < 1:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(readings)
        
        # Ensure all required columns exist
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
        
        # Use last ROLLING_WINDOW readings (or all if fewer)
        window = min(ROLLING_WINDOW, len(df))
        recent = df[FEATURE_COLUMNS].tail(window)
        
        features = {}
        
        # Base features (last reading)
        for col in FEATURE_COLUMNS:
            features[col] = float(df[col].iloc[-1])
        
        # Rolling features
        for col in FEATURE_COLUMNS:
            features[f"{col}_roll_mean"] = float(recent[col].mean())
            features[f"{col}_roll_std"] = float(recent[col].std()) if len(recent) > 1 else 0.0
        
        return features
    
    def _single_reading_features(self, reading: Dict[str, float]) -> Dict[str, float]:
        """
        Convert a single reading to features (no rolling).
        
        Rolling features will be 0 since we don't have history.
        """
        features = {}
        
        # Base features
        for col in FEATURE_COLUMNS:
            features[col] = float(reading.get(col, 0.0))
        
        # Rolling features (zero for single reading)
        for col in FEATURE_COLUMNS:
            features[f"{col}_roll_mean"] = features[col]
            features[f"{col}_roll_std"] = 0.0
        
        return features
    
    def _classify_urgency(self, rul: float) -> str:
        """Classify RUL into urgency levels."""
        if rul <= 10:
            return "critical"
        elif rul <= 20:
            return "urgent"
        elif rul <= 40:
            return "warning"
        elif rul <= 80:
            return "monitoring"
        else:
            return "healthy"
    
    def predict_from_features(self, features: Dict[str, float], unit_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Make prediction from pre-computed features.
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Build feature vector in correct order
        feature_values = []
        for col in self.feature_columns:
            feature_values.append(features.get(col, 0.0))
        
        X = np.array([feature_values])
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_columns)
        
        # Predict
        raw_prediction = self.model.predict(dmatrix)[0]
        
        # Clip to valid RUL range
        clipped_prediction = np.clip(raw_prediction, 0, RUL_CLIP)
        
        # Classify urgency
        urgency = self._classify_urgency(clipped_prediction)
        
        return {
            "unit_number": unit_number,
            "predicted_rul": float(raw_prediction),
            "rul_clipped": float(clipped_prediction),
            "urgency": urgency,
            "timestamp": datetime.now().isoformat(),
            "features_used": len(self.feature_columns)
        }
    
    def predict_from_readings(self, readings: List[Dict[str, float]], unit_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Make prediction from raw sensor readings.
        Computes rolling features internally.
        """
        # Compute features from readings
        features = self._compute_rolling_features(readings)
        
        return self.predict_from_features(features, unit_number)
    
    def predict_single(self, reading: Dict[str, float], unit_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Make prediction from a single reading (no rolling history).
        """
        features = self._single_reading_features(reading)
        
        return self.predict_from_features(features, unit_number)
    
    def predict_batch(self, engines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple engines.
        
        Each engine dict should have either:
        - "readings": list of sensor readings
        - "features": pre-computed feature dict
        - "single_reading": single reading dict
        """
        predictions = []
        
        for engine in engines:
            unit_number = engine.get("unit_number")
            
            if "readings" in engine:
                pred = self.predict_from_readings(engine["readings"], unit_number)
            elif "features" in engine:
                pred = self.predict_from_features(engine["features"], unit_number)
            elif "single_reading" in engine:
                pred = self.predict_single(engine["single_reading"], unit_number)
            else:
                # Default: try to use engine dict as single reading
                pred = self.predict_single(engine, unit_number)
            
            predictions.append(pred)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if self.metadata:
            return self.metadata
        return {
            "model_type": "xgboost_regression",
            "dataset": "C-MAPSS FD001",
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns
        }


# Global predictor instance
_predictor: Optional[CMAPSSPredictor] = None


def get_predictor(model_dir: str = "./models") -> CMAPSSPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    
    if _predictor is None or not _predictor.is_loaded:
        _predictor = CMAPSSPredictor(model_dir)
        _predictor.load_model()
    
    return _predictor


def predict_rul(readings: Optional[List[Dict]] = None,
                features: Optional[Dict[str, float]] = None,
                single_reading: Optional[Dict[str, float]] = None,
                unit_number: Optional[int] = None,
                model_dir: str = "./models") -> Dict[str, Any]:
    """
    Convenience function for RUL prediction.
    """
    predictor = get_predictor(model_dir)
    
    if readings is not None:
        return predictor.predict_from_readings(readings, unit_number)
    elif features is not None:
        return predictor.predict_from_features(features, unit_number)
    elif single_reading is not None:
        return predictor.predict_single(single_reading, unit_number)
    else:
        raise ValueError("Provide readings, features, or single_reading")
