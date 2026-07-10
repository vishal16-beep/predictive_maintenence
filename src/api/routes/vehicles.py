"""Vehicle management API routes."""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from ..schemas.request import VehicleHealthRequest, MaintenanceScheduleRequest
from ..schemas.response import VehicleHealthResponse, MaintenanceScheduleResponse
from ..main import get_prediction_service

router = APIRouter()

# In-memory store for demonstration (would use database in production)
vehicle_store: Dict[str, Dict[str, Any]] = {}
sensor_data_store: Dict[str, List[Dict]] = {}


@router.get("/", response_model=List[Dict[str, Any]])
async def list_vehicles():
    """List all vehicles in the system."""
    vehicles = []
    for vehicle_id, info in vehicle_store.items():
        vehicles.append({
            "vehicle_id": vehicle_id,
            "last_updated": info.get("last_updated"),
            "status": info.get("status", "unknown")
        })
    return vehicles


@router.get("/{vehicle_id}", response_model=Dict[str, Any])
async def get_vehicle(vehicle_id: str):
    """Get vehicle details."""
    if vehicle_id not in vehicle_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found"
        )
    return vehicle_store[vehicle_id]


@router.post("/", response_model=Dict[str, Any])
async def register_vehicle(vehicle_data: Dict[str, Any]):
    """Register a new vehicle."""
    vehicle_id = vehicle_data.get("vehicle_id")
    if not vehicle_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="vehicle_id is required"
        )
    
    vehicle_store[vehicle_id] = {
        **vehicle_data,
        "registered_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "status": "active"
    }
    
    return vehicle_store[vehicle_id]


@router.put("/{vehicle_id}", response_model=Dict[str, Any])
async def update_vehicle(vehicle_id: str, vehicle_data: Dict[str, Any]):
    """Update vehicle information."""
    if vehicle_id not in vehicle_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found"
        )
    
    vehicle_store[vehicle_id].update(vehicle_data)
    vehicle_store[vehicle_id]["last_updated"] = datetime.now().isoformat()
    
    return vehicle_store[vehicle_id]


@router.delete("/{vehicle_id}")
async def delete_vehicle(vehicle_id: str):
    """Delete a vehicle from the system."""
    if vehicle_id not in vehicle_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vehicle {vehicle_id} not found"
        )
    
    del vehicle_store[vehicle_id]
    return {"message": f"Vehicle {vehicle_id} deleted"}


@router.post("/{vehicle_id}/health", response_model=VehicleHealthResponse)
async def get_vehicle_health(
    vehicle_id: str,
    service = Depends(get_prediction_service)
):
    """Get comprehensive health status for a vehicle."""
    # Check if we have sensor data for this vehicle
    if vehicle_id not in sensor_data_store or not sensor_data_store[vehicle_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sensor data available for vehicle {vehicle_id}"
        )
    
    try:
        # Convert stored sensor data to DataFrame
        readings = sensor_data_store[vehicle_id]
        df = pd.DataFrame(readings)
        
        # Get comprehensive predictions
        results = service.get_comprehensive_prediction(df)
        
        # Extract vehicle-specific results
        vehicle_results = results.get("vehicle_predictions", {}).get(vehicle_id, {})
        
        # Calculate overall health score
        health_score = calculate_health_score(vehicle_results)
        
        # Generate recommendations
        recommendations = generate_recommendations(vehicle_results)
        
        return VehicleHealthResponse(
            vehicle_id=vehicle_id,
            rul_prediction=vehicle_results.get("rul"),
            anomaly_detection=vehicle_results.get("anomaly"),
            failure_classification=vehicle_results.get("classification"),
            maintenance_history=None,  # Would fetch from database
            overall_health_score=health_score,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.post("/{vehicle_id}/sensor-data")
async def add_sensor_data(vehicle_id: str, readings: List[Dict[str, Any]]):
    """Add sensor data for a vehicle."""
    if vehicle_id not in sensor_data_store:
        sensor_data_store[vehicle_id] = []
    
    for reading in readings:
        reading["vehicle_id"] = vehicle_id
        sensor_data_store[vehicle_id].append(reading)
    
    # Keep only last 1000 readings per vehicle
    if len(sensor_data_store[vehicle_id]) > 1000:
        sensor_data_store[vehicle_id] = sensor_data_store[vehicle_id][-1000:]
    
    return {
        "message": f"Added {len(readings)} readings for vehicle {vehicle_id}",
        "total_readings": len(sensor_data_store[vehicle_id])
    }


@router.get("/{vehicle_id}/sensor-data")
async def get_sensor_data(
    vehicle_id: str,
    limit: int = 100,
    offset: int = 0
):
    """Get sensor data for a vehicle."""
    if vehicle_id not in sensor_data_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sensor data available for vehicle {vehicle_id}"
        )
    
    data = sensor_data_store[vehicle_id]
    total = len(data)
    
    # Apply pagination
    paginated_data = data[offset:offset + limit]
    
    return {
        "vehicle_id": vehicle_id,
        "readings": paginated_data,
        "total": total,
        "offset": offset,
        "limit": limit
    }


@router.post("/{vehicle_id}/schedule-maintenance", response_model=MaintenanceScheduleResponse)
async def schedule_maintenance(
    vehicle_id: str,
    request: MaintenanceScheduleRequest,
    service = Depends(get_prediction_service)
):
    """Schedule maintenance for a vehicle."""
    # Get vehicle health to determine maintenance needs
    if vehicle_id not in sensor_data_store or not sensor_data_store[vehicle_id]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sensor data available for vehicle {vehicle_id}"
        )
    
    try:
        # Get predictions
        readings = sensor_data_store[vehicle_id]
        df = pd.DataFrame(readings)
        results = service.get_comprehensive_prediction(df)
        
        vehicle_results = results.get("vehicle_predictions", {}).get(vehicle_id, {})
        
        # Determine maintenance schedule
        schedule = determine_maintenance_schedule(
            vehicle_id, vehicle_results, request
        )
        
        return MaintenanceScheduleResponse(**schedule)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Maintenance scheduling failed: {str(e)}"
        )


def calculate_health_score(vehicle_results: Dict[str, Any]) -> float:
    """Calculate overall health score (0-100)."""
    score = 100.0
    
    # RUL impact
    if "rul" in vehicle_results:
        rul_days = vehicle_results["rul"].get("rul_days", 30)
        if rul_days <= 7:
            score -= 40
        elif rul_days <= 14:
            score -= 25
        elif rul_days <= 30:
            score -= 10
    
    # Anomaly impact
    if "anomaly" in vehicle_results:
        if vehicle_results["anomaly"].get("is_anomaly", False):
            score -= 20
            score -= vehicle_results["anomaly"].get("anomaly_score", 0) * 10
    
    # Classification impact
    if "classification" in vehicle_results:
        if vehicle_results["classification"].get("is_confident", False):
            score -= 15
    
    return max(0.0, min(100.0, score))


def generate_recommendations(vehicle_results: Dict[str, Any]) -> List[str]:
    """Generate maintenance recommendations."""
    recommendations = []
    
    # RUL recommendations
    if "rul" in vehicle_results:
        rul_days = vehicle_results["rul"].get("rul_days", 30)
        urgency = vehicle_results["rul"].get("urgency", "healthy")
        
        if urgency == "critical":
            recommendations.append("URGENT: Schedule immediate maintenance inspection")
        elif urgency == "urgent":
            recommendations.append("Schedule maintenance within the next 7 days")
        elif urgency == "warning":
            recommendations.append("Plan maintenance within the next 2 weeks")
    
    # Anomaly recommendations
    if "anomaly" in vehicle_results:
        if vehicle_results["anomaly"].get("is_anomaly", False):
            score = vehicle_results["anomaly"].get("anomaly_score", 0)
            if score > 0.8:
                recommendations.append("High anomaly detected: Perform detailed diagnostic check")
            else:
                recommendations.append("Anomaly detected: Monitor vehicle closely")
    
    # Classification recommendations
    if "classification" in vehicle_results:
        failure_type = vehicle_results["classification"].get("predicted_failure")
        if failure_type:
            recommendations.append(f"Potential {failure_type} issue: Inspect relevant components")
    
    if not recommendations:
        recommendations.append("Vehicle is operating normally. Continue regular maintenance schedule.")
    
    return recommendations


def determine_maintenance_schedule(
    vehicle_id: str,
    vehicle_results: Dict[str, Any],
    request: MaintenanceScheduleRequest
) -> Dict[str, Any]:
    """Determine maintenance schedule based on predictions."""
    # Default schedule
    schedule = {
        "vehicle_id": vehicle_id,
        "recommended_date": datetime.now(),
        "maintenance_type": request.maintenance_type,
        "priority": "medium",
        "estimated_duration_hours": 4.0,
        "estimated_cost": 500.0,
        "parts_likely_needed": [],
        "reason": "Routine maintenance"
    }
    
    # Adjust based on RUL
    if "rul" in vehicle_results:
        rul_days = vehicle_results["rul"].get("rul_days", 30)
        urgency = vehicle_results["rul"].get("urgency", "healthy")
        
        if urgency in ["critical", "urgent"]:
            from datetime import timedelta
            schedule["recommended_date"] = datetime.now() + timedelta(days=min(3, rul_days))
            schedule["priority"] = "critical" if urgency == "critical" else "high"
            schedule["reason"] = f"Critical: {rul_days:.1f} days remaining useful life"
        elif urgency == "warning":
            from datetime import timedelta
            schedule["recommended_date"] = datetime.now() + timedelta(days=min(7, rul_days))
            schedule["priority"] = "medium"
            schedule["reason"] = f"Warning: {rul_days:.1f} days remaining useful life"
    
    # Adjust based on failure classification
    if "classification" in vehicle_results:
        failure_type = vehicle_results["classification"].get("predicted_failure")
        confidence = vehicle_results["classification"].get("confidence", 0)
        
        if failure_type and confidence > 0.7:
            schedule["parts_likely_needed"].append(failure_type)
            schedule["estimated_cost"] += 200.0
            schedule["reason"] += f" - Potential {failure_type} issue"
    
    # Use preferred date if provided
    if request.preferred_date:
        schedule["recommended_date"] = request.preferred_date
    
    return schedule
