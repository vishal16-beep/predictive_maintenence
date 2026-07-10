"""Pydantic schemas for API requests - C-MAPSS compatible."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class CMAPSSReading(BaseModel):
    """
    Single C-MAPSS sensor reading.
    
    Raw features (before rolling window engineering):
    - op_setting_1, op_setting_2, op_setting_3: Operational settings
    - sensor_2 through sensor_21: Sensor readings (some dropped as uninformative)
    
    Note: Rolling features are computed server-side from a sequence of readings.
    """
    # Operational settings
    op_setting_1: float = Field(..., description="Operational setting 1")
    op_setting_2: float = Field(..., description="Operational setting 2")
    op_setting_3: float = Field(..., description="Operational setting 3")
    
    # Sensors (keeping only informative ones after analysis)
    sensor_2: float = Field(..., description="Sensor 2 - Total temperature at fan inlet")
    sensor_3: float = Field(..., description="Sensor 3 - Total pressure at bypass duct")
    sensor_4: float = Field(..., description="Sensor 4 - Total pressure ratio")
    sensor_7: float = Field(..., description="Sensor 7 - Physical fan speed")
    sensor_8: float = Field(..., description="Sensor 8 - Physical core speed")
    sensor_9: float = Field(..., description="Sensor 9 - Engine pressure ratio (PR)")
    sensor_11: float = Field(..., description="Sensor 11 - Static pressure at burner")
    sensor_12: float = Field(..., description="Sensor 12 - Ratio of fuel flow to PS30")
    sensor_13: float = Field(..., description="Sensor 13 - Corrected fan speed")
    sensor_14: float = Field(..., description="Sensor 14 - Corrected core speed")
    sensor_15: float = Field(..., description="Sensor 15 - Bypass ratio")
    sensor_17: float = Field(..., description="Sensor 17 - Burner fuel-air ratio")
    sensor_20: float = Field(..., description="Sensor 20 - Bleed enthalpy")
    sensor_21: float = Field(..., description="Sensor 21 - Requested fan fan speed")


class RULPredictionRequest(BaseModel):
    """
    Request for RUL prediction.
    
    Option 1: Provide raw sensor readings (server computes rolling features)
    Option 2: Provide pre-computed 51 features directly
    """
    # Engine identifier
    unit_number: Optional[int] = Field(None, description="Engine unit number (for tracking)")
    
    # Option 1: Raw sensor readings (use this if you have a sequence)
    readings: Optional[List[CMAPSSReading]] = Field(
        None, 
        description="Sequence of recent sensor readings (min 10 for rolling features)"
    )
    
    # Option 2: Pre-computed features (51 features matching model schema)
    features: Optional[Dict[str, float]] = Field(
        None,
        description="Pre-computed 51 features (key=feature_name, value=feature_value)"
    )
    
    # Option 3: Simple single reading (will use only raw features, no rolling)
    single_reading: Optional[CMAPSSReading] = Field(
        None,
        description="Single sensor reading (uses raw features only, no rolling)"
    )


class RULPredictionResponse(BaseModel):
    """Response for RUL prediction."""
    unit_number: Optional[int] = None
    predicted_rul: float = Field(..., description="Predicted Remaining Useful Life in cycles")
    rul_clipped: float = Field(..., description="RUL clipped to [0, 125] range")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    urgency: str = Field(..., description="Urgency level (critical/urgent/warning/monitoring/healthy)")
    timestamp: datetime
    features_used: int = Field(..., description="Number of features used for prediction")


class BatchPredictionRequest(BaseModel):
    """Batch prediction for multiple engines."""
    engines: List[Dict[str, Any]] = Field(
        ..., 
        description="List of engine data (each with readings or features)"
    )


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
