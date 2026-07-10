"""Pydantic schemas for API responses - C-MAPSS compatible."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RULPredictionResponse(BaseModel):
    """Response for RUL prediction."""
    unit_number: Optional[int] = None
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    rul_clipped: float = Field(..., description="RUL clipped to [0, 125] range")
    urgency: str = Field(..., description="Urgency level (critical/urgent/warning/monitoring/healthy)")
    timestamp: datetime
    features_used: int = Field(..., description="Number of features used for prediction")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[RULPredictionResponse]
    n_engines: int
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    dataset: str
    n_features: int
    feature_columns: List[str]
    dropped_sensors: List[str]
    remaining_sensors: List[str]
    op_settings: List[str]
    rolling_window: int
    rul_clip: int
    hyperparameters: Dict[str, Any]
    test_metrics: Dict[str, float]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    n_features: Optional[int] = None
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime
