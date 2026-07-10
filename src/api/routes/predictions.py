"""Prediction API routes."""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..schemas.request import (
    SensorReadingRequest, BatchSensorRequest,
    RULPredictionRequest, AnomalyDetectionRequest,
    FailureClassificationRequest
)
from ..schemas.response import (
    RULPredictionResponse, RULBatchResponse,
    AnomalyDetectionResponse, FailureClassificationResponse
)
from ..main import get_prediction_service

router = APIRouter()


def convert_readings_to_dataframe(readings: List[SensorReadingRequest]) -> pd.DataFrame:
    """Convert sensor readings to DataFrame."""
    data = [reading.model_dump() for reading in readings]
    df = pd.DataFrame(data)
    return df


@router.post("/rul", response_model=RULBatchResponse)
async def predict_rul(
    request: RULPredictionRequest,
    service = Depends(get_prediction_service)
):
    """Predict Remaining Useful Life for a vehicle."""
    try:
        # Convert to DataFrame
        df = convert_readings_to_dataframe(request.sensor_data)
        
        # Get predictions
        results = service.predict_rul(df)
        
        # Format response
        predictions = [
            RULPredictionResponse(**pred)
            for pred in results["predictions"]
        ]
        
        return RULBatchResponse(
            predictions=predictions,
            n_vehicles=results["n_vehicles"],
            timestamp=results["timestamp"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/anomaly", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    service = Depends(get_prediction_service)
):
    """Detect anomalies in sensor data."""
    try:
        # Convert to DataFrame
        df = convert_readings_to_dataframe(request.sensor_data)
        
        # Get predictions
        results = service.detect_anomalies(df)
        
        # Format response
        anomalies = [
            AnomalyResult(**anomaly)
            for anomaly in results["anomalies"]
        ]
        
        return AnomalyDetectionResponse(
            anomalies=anomalies,
            summary=results["summary"],
            explanations=results.get("explanations"),
            timestamp=results["timestamp"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.post("/failure", response_model=FailureClassificationResponse)
async def classify_failures(
    request: FailureClassificationRequest,
    service = Depends(get_prediction_service)
):
    """Classify potential failure types."""
    try:
        # Convert to DataFrame
        df = convert_readings_to_dataframe(request.sensor_data)
        
        # Get predictions
        results = service.classify_failures(
            df, 
            confidence_threshold=request.confidence_threshold
        )
        
        # Format response
        predictions = [
            FailurePrediction(**pred)
            for pred in results["predictions"]
        ]
        
        return FailureClassificationResponse(
            predictions=predictions,
            summary=results["summary"],
            timestamp=results["timestamp"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failure classification failed: {str(e)}"
        )


@router.post("/batch")
async def batch_predict(
    request: BatchSensorRequest,
    service = Depends(get_prediction_service)
):
    """Process batch sensor readings for all predictions."""
    try:
        # Convert to DataFrame
        df = convert_readings_to_dataframe(request.readings)
        
        # Get comprehensive predictions
        results = service.get_comprehensive_prediction(df)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.post("/comprehensive")
async def comprehensive_prediction(
    readings: List[SensorReadingRequest],
    service = Depends(get_prediction_service)
):
    """Get comprehensive predictions (RUL, anomaly, classification) for sensor data."""
    try:
        # Convert to DataFrame
        df = convert_readings_to_dataframe(readings)
        
        # Get all predictions
        results = service.get_comprehensive_prediction(df)
        
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive prediction failed: {str(e)}"
        )
