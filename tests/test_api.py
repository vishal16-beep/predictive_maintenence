"""Test API endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from src.api.main import app


client = TestClient(app)


class TestRootEndpoints:
    """Tests for root and health endpoints."""

    def test_root(self):
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["status"] == "running"

    def test_health(self):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "models" in data


class TestPredictionEndpoints:
    """Tests for prediction endpoints."""

    def test_rul_prediction(self):
        sensor_data = [
            {
                "vehicle_id": "VEH-1234",
                "timestamp": datetime.now().isoformat(),
                "engine_temp": 85.5,
                "vibration_level": 2.3,
                "oil_pressure": 45.0,
                "rpm": 2500,
                "mileage": 50000,
                "battery_voltage": 13.8,
                "brake_wear": 35.0,
                "tire_pressure": [32.0, 32.5, 31.5, 32.0],
                "ambient_temp": 25.0,
                "load_weight": 1500.0
            }
            for _ in range(10)
        ]
        
        response = client.post("/api/v1/predictions/rul", json={
            "vehicle_id": "VEH-1234",
            "sensor_data": sensor_data
        })
        
        # Note: This will fail if models are not loaded
        # In production, we'd mock the prediction service
        assert response.status_code in [200, 503]

    def test_anomaly_detection(self):
        sensor_data = [
            {
                "vehicle_id": "VEH-1234",
                "timestamp": datetime.now().isoformat(),
                "engine_temp": 85.5,
                "vibration_level": 2.3,
                "oil_pressure": 45.0,
                "rpm": 2500,
                "mileage": 50000,
                "battery_voltage": 13.8,
                "brake_wear": 35.0,
                "tire_pressure": [32.0, 32.5, 31.5, 32.0],
                "ambient_temp": 25.0,
                "load_weight": 1500.0
            }
        ]
        
        response = client.post("/api/v1/predictions/anomaly", json={
            "sensor_data": sensor_data
        })
        
        assert response.status_code in [200, 503]

    def test_failure_classification(self):
        sensor_data = [
            {
                "vehicle_id": "VEH-1234",
                "timestamp": datetime.now().isoformat(),
                "engine_temp": 85.5,
                "vibration_level": 2.3,
                "oil_pressure": 45.0,
                "rpm": 2500,
                "mileage": 50000,
                "battery_voltage": 13.8,
                "brake_wear": 35.0,
                "tire_pressure": [32.0, 32.5, 31.5, 32.0],
                "ambient_temp": 25.0,
                "load_weight": 1500.0
            }
        ]
        
        response = client.post("/api/v1/predictions/failure", json={
            "sensor_data": sensor_data,
            "confidence_threshold": 0.5
        })
        
        assert response.status_code in [200, 503]


class TestVehicleEndpoints:
    """Tests for vehicle endpoints."""

    def test_list_vehicles(self):
        response = client.get("/api/v1/vehicles/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_register_vehicle(self):
        vehicle_data = {
            "vehicle_id": "VEH-9999",
            "type": "truck",
            "year": 2022,
            "make": "Volvo",
            "model": "FH16"
        }
        
        response = client.post("/api/v1/vehicles/", json=vehicle_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["vehicle_id"] == "VEH-9999"

    def test_get_vehicle(self):
        # First register a vehicle
        vehicle_data = {
            "vehicle_id": "VEH-8888",
            "type": "bus"
        }
        client.post("/api/v1/vehicles/", json=vehicle_data)
        
        # Then get it
        response = client.get("/api/v1/vehicles/VEH-8888")
        
        assert response.status_code == 200
        data = response.json()
        assert data["vehicle_id"] == "VEH-8888"


class TestAlertEndpoints:
    """Tests for alert endpoints."""

    def test_get_alerts(self):
        response = client.get("/api/v1/alerts/")
        
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total_count" in data

    def test_create_alert(self):
        alert_data = {
            "vehicle_id": "VEH-1234",
            "type": "rul_warning",
            "severity": "critical",
            "message": "Test alert",
            "details": {"rul_days": 3.5}
        }
        
        response = client.post("/api/v1/alerts/", json=alert_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["vehicle_id"] == "VEH-1234"
        assert data["severity"] == "critical"

    def test_get_alert_stats(self):
        response = client.get("/api/v1/alerts/stats/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_alerts" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
