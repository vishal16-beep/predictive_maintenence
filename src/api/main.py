"""FastAPI application for Predictive Maintenance System - C-MAPSS Edition."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from config.settings import settings
from src.inference.cmapss_predictor import CMAPSSPredictor, get_predictor

logger = logging.getLogger(__name__)

# Global predictor
predictor: Optional[CMAPSSPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor
    
    # Startup
    logger.info("Starting Predictive Maintenance API (C-MAPSS FD001)...")
    
    # Load model
    predictor = get_predictor(settings.MODEL_DIR)
    
    if predictor.is_loaded:
        logger.info(f"Model loaded: {predictor.metadata.get('model_type', 'unknown')}")
        logger.info(f"Features: {len(predictor.feature_columns)}")
    else:
        logger.warning("Model not loaded - predictions will fail")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


app = FastAPI(
    title="Predictive Maintenance API - C-MAPSS FD001",
    version="2.0.0",
    description="RUL prediction for turbofan engines using NASA C-MAPSS FD001 dataset",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health & Info
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Predictive Maintenance API",
        "version": "2.0.0",
        "dataset": "NASA C-MAPSS FD001",
        "status": "running" if (predictor and predictor.is_loaded) else "model_not_loaded",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = predictor is not None and predictor.is_loaded
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": predictor.metadata.get("model_type") if predictor and predictor.metadata else None,
        "n_features": len(predictor.feature_columns) if predictor and predictor.feature_columns else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Get model information and metadata."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return predictor.get_model_info()


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/predict/rul")
async def predict_rul(request: Dict[str, Any]):
    """
    Predict Remaining Useful Life (RUL).
    
    Request body options:
    1. {"readings": [...]} - Sequence of sensor readings (computes rolling features)
    2. {"features": {...}} - Pre-computed 51 features
    3. {"single_reading": {...}} - Single sensor reading (no rolling)
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        unit_number = request.get("unit_number")
        
        if "readings" in request:
            result = predictor.predict_from_readings(request["readings"], unit_number)
        elif "features" in request:
            result = predictor.predict_from_features(request["features"], unit_number)
        elif "single_reading" in request:
            result = predictor.predict_single(request["single_reading"], unit_number)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide 'readings', 'features', or 'single_reading'"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/batch")
async def predict_batch(request: Dict[str, Any]):
    """
    Batch prediction for multiple engines.
    
    Request body: {"engines": [{"unit_number": 1, "readings": [...]}, ...]}
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        engines = request.get("engines", [])
        
        if not engines:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No engines provided"
            )
        
        predictions = predictor.predict_batch(engines)
        
        return {
            "predictions": predictions,
            "n_engines": len(predictions),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/quick")
async def predict_quick(reading: Dict[str, float]):
    """
    Quick prediction from a single sensor reading.
    
    Accepts raw sensor values and returns RUL prediction.
    """
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        result = predictor.predict_single(reading)
        return result
        
    except Exception as e:
        logger.error(f"Quick prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Info
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/features")
async def get_features():
    """Get list of expected features."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "feature_columns": predictor.feature_columns,
        "n_features": len(predictor.feature_columns),
        "base_features": predictor.metadata.get("remaining_sensors", []) + predictor.metadata.get("op_settings", []),
        "rolling_window": predictor.metadata.get("rolling_window", 10)
    }


@app.get("/api/v1/sensors")
async def get_sensors():
    """Get sensor information."""
    if not predictor or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "dropped_sensors": predictor.metadata.get("dropped_sensors", []),
        "remaining_sensors": predictor.metadata.get("remaining_sensors", []),
        "op_settings": predictor.metadata.get("op_settings", [])
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
