"""Prediction service for real-time inference."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making predictions using trained models."""

    def __init__(self, model_loader=None):
        self.model_loader = model_loader
        self.prediction_cache: Dict[str, Any] = {}

    def predict_rul(self, sensor_data: pd.DataFrame, 
                    feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Predict Remaining Useful Life for vehicles."""
        if not self.model_loader or not self.model_loader.is_model_loaded("rul"):
            raise ValueError("RUL model not loaded")
        
        rul_model = self.model_loader.get_model("rul")
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [col for col in sensor_data.columns 
                          if col not in ["vehicle_id", "timestamp"]]
        
        # Get sequences per vehicle
        results = []
        
        for vehicle_id in sensor_data["vehicle_id"].unique():
            vehicle_data = sensor_data[sensor_data["vehicle_id"] == vehicle_id]
            vehicle_data = vehicle_data.sort_values("timestamp")
            
            if len(vehicle_data) < rul_model.sequence_length:
                continue
            
            # Create sequence
            sequence = vehicle_data[feature_cols].values[-rul_model.sequence_length:]
            sequence = sequence.reshape(1, rul_model.sequence_length, len(feature_cols))
            
            # Scale if scaler available
            if self.model_loader and self.model_loader.is_model_loaded("time_series"):
                ts_scaler = self.model_loader.get_scaler("time_series")
                sequence = ts_scaler.transform_sequence(sequence)
            
            # Predict
            rul_days = rul_model.predict(sequence)[0]
            
            # Get confidence interval
            try:
                mean_pred, lower_bound, upper_bound = rul_model.predict_with_confidence(sequence)
                confidence_interval = {
                    "lower": float(lower_bound[0]),
                    "upper": float(upper_bound[0])
                }
            except Exception:
                confidence_interval = None
            
            # Determine urgency
            urgency = self._get_urgency_level(rul_days)
            
            results.append({
                "vehicle_id": vehicle_id,
                "rul_days": float(rul_days),
                "confidence_interval": confidence_interval,
                "urgency": urgency,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "predictions": results,
            "n_vehicles": len(results),
            "timestamp": datetime.now().isoformat()
        }

    def detect_anomalies(self, sensor_data: pd.DataFrame,
                         feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect anomalies in sensor data."""
        if not self.model_loader or not self.model_loader.is_model_loaded("anomaly"):
            raise ValueError("Anomaly model not loaded")
        
        anomaly_model = self.model_loader.get_model("anomaly")
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [col for col in sensor_data.columns 
                          if col not in ["vehicle_id", "timestamp"]]
        
        X = sensor_data[feature_cols].values
        
        # Scale if scaler available
        if self.model_loader and self.model_loader.is_model_loaded("data"):
            data_scaler = self.model_loader.get_scaler("data")
            X = data_scaler.transform_numerical(pd.DataFrame(X, columns=feature_cols)).values
        
        # Detect anomalies
        results = anomaly_model.detect_anomalies(X)
        
        # Get explanations
        explanations = anomaly_model.get_anomaly_explanation(X, feature_cols)
        
        # Format results
        anomaly_results = []
        for i, vehicle_id in enumerate(sensor_data["vehicle_id"].unique()):
            if i < len(results["is_anomaly"]):
                anomaly_results.append({
                    "vehicle_id": vehicle_id,
                    "is_anomaly": bool(results["is_anomaly"][i]),
                    "anomaly_score": float(results["anomaly_scores"][i]),
                    "reconstruction_error": float(results["reconstruction_errors"][i])
                })
        
        return {
            "anomalies": anomaly_results,
            "summary": {
                "n_anomalies": int(np.sum(results["is_anomaly"])),
                "anomaly_rate": float(np.mean(results["is_anomaly"])),
                "mean_score": float(np.mean(results["anomaly_scores"]))
            },
            "explanations": explanations,
            "timestamp": datetime.now().isoformat()
        }

    def classify_failures(self, sensor_data: pd.DataFrame,
                         feature_cols: Optional[List[str]] = None,
                         confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Classify potential failure types."""
        if not self.model_loader or not self.model_loader.is_model_loaded("classification"):
            raise ValueError("Classification model not loaded")
        
        classifier = self.model_loader.get_model("classification")
        
        # Prepare features
        if feature_cols is None:
            feature_cols = [col for col in sensor_data.columns 
                          if col not in ["vehicle_id", "timestamp", "failure_type"]]
        
        X = sensor_data[feature_cols].values
        
        # Scale if scaler available
        if self.model_loader and self.model_loader.is_model_loaded("data"):
            data_scaler = self.model_loader.get_scaler("data")
            X = data_scaler.transform_numerical(pd.DataFrame(X, columns=feature_cols)).values
        
        # Get predictions with confidence
        predictions = classifier.predict_with_confidence(X, confidence_threshold)
        
        # Format results
        classification_results = []
        for i, vehicle_id in enumerate(sensor_data["vehicle_id"].unique()):
            if i < len(predictions["predictions"]):
                pred = predictions["predictions"][i]
                classification_results.append({
                    "vehicle_id": vehicle_id,
                    "predicted_failure": pred["prediction"],
                    "confidence": pred["confidence"],
                    "is_confident": pred["is_confident"],
                    "probabilities": pred["probabilities"]
                })
        
        return {
            "predictions": classification_results,
            "summary": {
                "n_predictions": len(classification_results),
                "n_confident": predictions["n_confident"],
                "n_low_confidence": predictions["n_low_confidence"],
                "avg_confidence": predictions["avg_confidence"]
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_comprehensive_prediction(self, sensor_data: pd.DataFrame,
                                    feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get comprehensive predictions from all models."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "vehicle_predictions": {}
        }
        
        # RUL predictions
        try:
            rul_results = self.predict_rul(sensor_data, feature_cols)
            for pred in rul_results["predictions"]:
                vehicle_id = pred["vehicle_id"]
                if vehicle_id not in results["vehicle_predictions"]:
                    results["vehicle_predictions"][vehicle_id] = {}
                results["vehicle_predictions"][vehicle_id]["rul"] = pred
        except Exception as e:
            logger.error(f"RUL prediction error: {e}")
            results["rul_error"] = str(e)
        
        # Anomaly detection
        try:
            anomaly_results = self.detect_anomalies(sensor_data, feature_cols)
            for pred in anomaly_results["anomalies"]:
                vehicle_id = pred["vehicle_id"]
                if vehicle_id not in results["vehicle_predictions"]:
                    results["vehicle_predictions"][vehicle_id] = {}
                results["vehicle_predictions"][vehicle_id]["anomaly"] = pred
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            results["anomaly_error"] = str(e)
        
        # Failure classification
        try:
            classification_results = self.classify_failures(sensor_data, feature_cols)
            for pred in classification_results["predictions"]:
                vehicle_id = pred["vehicle_id"]
                if vehicle_id not in results["vehicle_predictions"]:
                    results["vehicle_predictions"][vehicle_id] = {}
                results["vehicle_predictions"][vehicle_id]["classification"] = pred
        except Exception as e:
            logger.error(f"Classification error: {e}")
            results["classification_error"] = str(e)
        
        # Generate alerts
        results["alerts"] = self._generate_alerts(results["vehicle_predictions"])
        
        return results

    def _get_urgency_level(self, rul_days: float) -> str:
        """Determine urgency level based on RUL."""
        if rul_days <= 3:
            return "critical"
        elif rul_days <= 7:
            return "urgent"
        elif rul_days <= 14:
            return "warning"
        elif rul_days <= 30:
            return "monitoring"
        else:
            return "healthy"

    def _generate_alerts(self, vehicle_predictions: Dict) -> List[Dict[str, Any]]:
        """Generate alerts based on predictions."""
        alerts = []
        
        for vehicle_id, predictions in vehicle_predictions.items():
            # RUL alert
            if "rul" in predictions:
                rul_pred = predictions["rul"]
                if rul_pred["urgency"] in ["critical", "urgent"]:
                    alerts.append({
                        "vehicle_id": vehicle_id,
                        "type": "rul_warning",
                        "severity": rul_pred["urgency"],
                        "message": f"Vehicle {vehicle_id} has {rul_pred['rul_days']:.1f} days remaining useful life",
                        "rul_days": rul_pred["rul_days"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Anomaly alert
            if "anomaly" in predictions:
                anomaly_pred = predictions["anomaly"]
                if anomaly_pred["is_anomaly"]:
                    alerts.append({
                        "vehicle_id": vehicle_id,
                        "type": "anomaly_detected",
                        "severity": "high" if anomaly_pred["anomaly_score"] > 0.8 else "medium",
                        "message": f"Anomaly detected for vehicle {vehicle_id} (score: {anomaly_pred['anomaly_score']:.2f})",
                        "anomaly_score": anomaly_pred["anomaly_score"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Classification alert
            if "classification" in predictions:
                class_pred = predictions["classification"]
                if class_pred["is_confident"] and class_pred["confidence"] > 0.7:
                    alerts.append({
                        "vehicle_id": vehicle_id,
                        "type": "failure_prediction",
                        "severity": "high",
                        "message": f"Potential {class_pred['predicted_failure']} failure predicted for vehicle {vehicle_id}",
                        "predicted_failure": class_pred["predicted_failure"],
                        "confidence": class_pred["confidence"],
                        "timestamp": datetime.now().isoformat()
                    })
        
        return alerts

    def batch_predict(self, sensor_data: pd.DataFrame,
                     batch_size: int = 100) -> Dict[str, Any]:
        """Process predictions in batches."""
        results = {
            "predictions": [],
            "alerts": [],
            "timestamp": datetime.now().isoformat()
        }
        
        vehicle_ids = sensor_data["vehicle_id"].unique()
        
        for i in range(0, len(vehicle_ids), batch_size):
            batch_vehicles = vehicle_ids[i:i + batch_size]
            batch_data = sensor_data[sensor_data["vehicle_id"].isin(batch_vehicles)]
            
            batch_results = self.get_comprehensive_prediction(batch_data)
            results["predictions"].append(batch_results)
            results["alerts"].extend(batch_results.get("alerts", []))
        
        return results

    def cache_prediction(self, vehicle_id: str, prediction: Dict[str, Any],
                        ttl_seconds: int = 300):
        """Cache a prediction with TTL."""
        self.prediction_cache[vehicle_id] = {
            "prediction": prediction,
            "timestamp": datetime.now(),
            "ttl": ttl_seconds
        }

    def get_cached_prediction(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if still valid."""
        if vehicle_id in self.prediction_cache:
            cached = self.prediction_cache[vehicle_id]
            elapsed = (datetime.now() - cached["timestamp"]).total_seconds()
            
            if elapsed < cached["ttl"]:
                return cached["prediction"]
            else:
                del self.prediction_cache[vehicle_id]
        
        return None
