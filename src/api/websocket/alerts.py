"""WebSocket connection manager for real-time alerts."""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections for real-time alerts."""

    def __init__(self):
        # Map of vehicle_id -> set of WebSocket connections
        self.vehicle_connections: Dict[str, Set[WebSocket]] = {}
        # Map of connection -> vehicle_ids subscribed to
        self.connection_subscriptions: Dict[WebSocket, Set[str]] = {}
        # General connections (receive all alerts)
        self.general_connections: Set[WebSocket] = set()
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept new WebSocket connection."""
        await websocket.accept()
        
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.now().isoformat(),
            "messages_sent": 0
        }
        
        logger.info(f"WebSocket connected: {client_id}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        # Remove from vehicle subscriptions
        if websocket in self.connection_subscriptions:
            for vehicle_id in self.connection_subscriptions[websocket]:
                if vehicle_id in self.vehicle_connections:
                    self.vehicle_connections[vehicle_id].discard(websocket)
                    if not self.vehicle_connections[vehicle_id]:
                        del self.vehicle_connections[vehicle_id]
            del self.connection_subscriptions[websocket]
        
        # Remove from general connections
        self.general_connections.discard(websocket)
        
        # Remove metadata
        self.connection_metadata.pop(websocket, None)
        
        logger.info("WebSocket disconnected")

    async def subscribe_to_vehicle(self, websocket: WebSocket, vehicle_id: str):
        """Subscribe to alerts for a specific vehicle."""
        if vehicle_id not in self.vehicle_connections:
            self.vehicle_connections[vehicle_id] = set()
        
        self.vehicle_connections[vehicle_id].add(websocket)
        
        if websocket not in self.connection_subscriptions:
            self.connection_subscriptions[websocket] = set()
        
        self.connection_subscriptions[websocket].add(vehicle_id)
        
        # Send confirmation
        await websocket.send_json({
            "type": "subscription_confirmed",
            "vehicle_id": vehicle_id,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"WebSocket subscribed to vehicle {vehicle_id}")

    async def unsubscribe_from_vehicle(self, websocket: WebSocket, vehicle_id: str):
        """Unsubscribe from vehicle alerts."""
        if vehicle_id in self.vehicle_connections:
            self.vehicle_connections[vehicle_id].discard(websocket)
            if not self.vehicle_connections[vehicle_id]:
                del self.vehicle_connections[vehicle_id]
        
        if websocket in self.connection_subscriptions:
            self.connection_subscriptions[websocket].discard(vehicle_id)
        
        # Send confirmation
        await websocket.send_json({
            "type": "unsubscription_confirmed",
            "vehicle_id": vehicle_id,
            "timestamp": datetime.now().isoformat()
        })

    async def subscribe_to_all(self, websocket: WebSocket):
        """Subscribe to all alerts."""
        self.general_connections.add(websocket)
        
        await websocket.send_json({
            "type": "subscription_confirmed",
            "scope": "all",
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("WebSocket subscribed to all alerts")

    async def broadcast_to_vehicle(self, vehicle_id: str, alert: Dict[str, Any]):
        """Broadcast alert to all subscribers of a specific vehicle."""
        if vehicle_id not in self.vehicle_connections:
            return
        
        disconnected = []
        
        for websocket in self.vehicle_connections[vehicle_id]:
            try:
                await websocket.send_json({
                    "type": "alert",
                    "vehicle_id": vehicle_id,
                    "data": alert,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_to_all(self, alert: Dict[str, Any]):
        """Broadcast alert to all general subscribers."""
        disconnected = []
        
        for websocket in self.general_connections:
            try:
                await websocket.send_json({
                    "type": "alert",
                    "scope": "all",
                    "data": alert,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Update metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected
        for ws in disconnected:
            self.general_connections.discard(ws)
            self.connection_metadata.pop(ws, None)

    async def broadcast_alert(self, alert: Dict[str, Any], vehicle_id: Optional[str] = None):
        """Broadcast alert to appropriate subscribers."""
        # Send to vehicle-specific subscribers
        if vehicle_id:
            await self.broadcast_to_vehicle(vehicle_id, alert)
        
        # Send to general subscribers
        await self.broadcast_to_all(alert)

    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a personal message to a specific connection."""
        try:
            await websocket.send_json({
                "type": "personal_message",
                "data": message,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.connection_metadata)

    def get_vehicle_subscriber_count(self, vehicle_id: str) -> int:
        """Get number of subscribers for a vehicle."""
        return len(self.vehicle_connections.get(vehicle_id, set()))

    def get_connection_info(self, websocket: WebSocket) -> Optional[Dict[str, Any]]:
        """Get information about a connection."""
        return self.connection_metadata.get(websocket)

    def get_all_connections_info(self) -> List[Dict[str, Any]]:
        """Get information about all connections."""
        info = []
        for websocket, metadata in self.connection_metadata.items():
            subscriptions = self.connection_subscriptions.get(websocket, set())
            info.append({
                **metadata,
                "subscriptions": list(subscriptions),
                "is_general_subscriber": websocket in self.general_connections
            })
        return info


# Global connection manager
connection_manager = ConnectionManager()


class AlertBroadcaster:
    """Broadcast alerts through WebSocket connections."""

    def __init__(self, manager: ConnectionManager):
        self.manager = manager

    async def broadcast_rul_alert(self, vehicle_id: str, rul_days: float, urgency: str):
        """Broadcast RUL-based alert."""
        alert = {
            "alert_type": "rul_warning",
            "severity": urgency,
            "message": f"Vehicle {vehicle_id} has {rul_days:.1f} days remaining useful life",
            "details": {
                "rul_days": rul_days,
                "urgency": urgency
            }
        }
        
        await self.manager.broadcast_alert(alert, vehicle_id)

    async def broadcast_anomaly_alert(self, vehicle_id: str, anomaly_score: float):
        """Broadcast anomaly detection alert."""
        severity = "critical" if anomaly_score > 0.9 else "high" if anomaly_score > 0.7 else "medium"
        
        alert = {
            "alert_type": "anomaly_detected",
            "severity": severity,
            "message": f"Anomaly detected for vehicle {vehicle_id} (score: {anomaly_score:.2f})",
            "details": {
                "anomaly_score": anomaly_score
            }
        }
        
        await self.manager.broadcast_alert(alert, vehicle_id)

    async def broadcast_failure_alert(self, vehicle_id: str, failure_type: str, confidence: float):
        """Broadcast failure prediction alert."""
        alert = {
            "alert_type": "failure_prediction",
            "severity": "high",
            "message": f"Potential {failure_type} failure predicted for vehicle {vehicle_id}",
            "details": {
                "failure_type": failure_type,
                "confidence": confidence
            }
        }
        
        await self.manager.broadcast_alert(alert, vehicle_id)

    async def broadcast_maintenance_reminder(self, vehicle_id: str, maintenance_date: str):
        """Broadcast maintenance reminder."""
        alert = {
            "alert_type": "maintenance_reminder",
            "severity": "medium",
            "message": f"Maintenance scheduled for vehicle {vehicle_id} on {maintenance_date}",
            "details": {
                "maintenance_date": maintenance_date
            }
        }
        
        await self.manager.broadcast_alert(alert, vehicle_id)


# Global broadcaster
alert_broadcaster = AlertBroadcaster(connection_manager)
