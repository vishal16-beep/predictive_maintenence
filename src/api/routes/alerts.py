"""Alert management API routes."""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from ..schemas.request import AlertFilterRequest
from ..schemas.response import Alert, AlertListResponse

router = APIRouter()

# In-memory alert store (would use database in production)
alerts_store: List[Dict[str, Any]] = []


@router.get("/", response_model=AlertListResponse)
async def get_alerts(
    vehicle_id: Optional[str] = None,
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 100
):
    """Get all alerts with optional filtering."""
    filtered_alerts = alerts_store
    
    # Apply filters
    if vehicle_id:
        filtered_alerts = [a for a in filtered_alerts if a.get("vehicle_id") == vehicle_id]
    
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a.get("severity") == severity]
    
    if alert_type:
        filtered_alerts = [a for a in filtered_alerts if a.get("type") == alert_type]
    
    # Sort by created_at descending
    filtered_alerts.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Apply limit
    filtered_alerts = filtered_alerts[:limit]
    
    # Count unread
    unread_count = sum(1 for a in filtered_alerts if not a.get("acknowledged", False))
    
    return AlertListResponse(
        alerts=[Alert(**alert) for alert in filtered_alerts],
        total_count=len(filtered_alerts),
        unread_count=unread_count,
        timestamp=datetime.now()
    )


@router.get("/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str):
    """Get a specific alert by ID."""
    alert = next((a for a in alerts_store if a.get("alert_id") == alert_id), None)
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    
    return Alert(**alert)


@router.post("/", response_model=Alert)
async def create_alert(alert_data: Dict[str, Any]):
    """Create a new alert."""
    alert_id = str(uuid.uuid4())
    
    alert = {
        "alert_id": alert_id,
        "vehicle_id": alert_data.get("vehicle_id"),
        "type": alert_data.get("type"),
        "severity": alert_data.get("severity", "medium"),
        "message": alert_data.get("message"),
        "details": alert_data.get("details", {}),
        "acknowledged": False,
        "created_at": datetime.now().isoformat()
    }
    
    alerts_store.append(alert)
    
    return Alert(**alert)


@router.put("/{alert_id}/acknowledge", response_model=Alert)
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    alert = next((a for a in alerts_store if a.get("alert_id") == alert_id), None)
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    
    alert["acknowledged"] = True
    alert["acknowledged_at"] = datetime.now().isoformat()
    
    return Alert(**alert)


@router.delete("/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert."""
    global alerts_store
    
    alert = next((a for a in alerts_store if a.get("alert_id") == alert_id), None)
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    
    alerts_store = [a for a in alerts_store if a.get("alert_id") != alert_id]
    
    return {"message": f"Alert {alert_id} deleted"}


@router.get("/stats/summary")
async def get_alert_stats():
    """Get alert statistics summary."""
    total = len(alerts_store)
    unread = sum(1 for a in alerts_store if not a.get("acknowledged", False))
    
    # Count by severity
    by_severity = {}
    for alert in alerts_store:
        severity = alert.get("severity", "unknown")
        by_severity[severity] = by_severity.get(severity, 0) + 1
    
    # Count by type
    by_type = {}
    for alert in alerts_store:
        alert_type = alert.get("type", "unknown")
        by_type[alert_type] = by_type.get(alert_type, 0) + 1
    
    # Count by vehicle
    by_vehicle = {}
    for alert in alerts_store:
        vehicle_id = alert.get("vehicle_id", "unknown")
        by_vehicle[vehicle_id] = by_vehicle.get(vehicle_id, 0) + 1
    
    return {
        "total_alerts": total,
        "unread_alerts": unread,
        "by_severity": by_severity,
        "by_type": by_type,
        "by_vehicle": by_vehicle,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/bulk-acknowledge")
async def bulk_acknowledge_alerts(alert_ids: List[str]):
    """Acknowledge multiple alerts at once."""
    acknowledged = []
    
    for alert_id in alert_ids:
        alert = next((a for a in alerts_store if a.get("alert_id") == alert_id), None)
        if alert:
            alert["acknowledged"] = True
            alert["acknowledged_at"] = datetime.now().isoformat()
            acknowledged.append(alert_id)
    
    return {
        "acknowledged_count": len(acknowledged),
        "acknowledged_alerts": acknowledged
    }


@router.delete("/bulk-delete")
async def bulk_delete_alerts(alert_ids: List[str]):
    """Delete multiple alerts at once."""
    global alerts_store
    
    deleted = []
    for alert_id in alert_ids:
        alert = next((a for a in alerts_store if a.get("alert_id") == alert_id), None)
        if alert:
            deleted.append(alert_id)
    
    alerts_store = [a for a in alerts_store if a.get("alert_id") not in alert_ids]
    
    return {
        "deleted_count": len(deleted),
        "deleted_alerts": deleted
    }
