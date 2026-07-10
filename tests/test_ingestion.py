"""Test data ingestion modules."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.ingestion.sensor_stream import SensorStreamProcessor, SensorReading, SensorDataBuffer
from src.ingestion.log_processor import LogProcessor, OperationalLog
from src.ingestion.history_loader import HistoryLoader, MaintenanceRecord


class TestSensorStreamProcessor:
    """Tests for SensorStreamProcessor."""

    def test_parse_sensor_reading(self):
        processor = SensorStreamProcessor()
        
        data = {
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
        
        reading = processor._parse_sensor_reading(data)
        
        assert reading.vehicle_id == "VEH-1234"
        assert reading.engine_temp == 85.5
        assert reading.vibration_level == 2.3
        assert len(reading.tire_pressure) == 4

    def test_validate_reading_valid(self):
        processor = SensorStreamProcessor()
        
        reading = SensorReading(
            vehicle_id="VEH-1234",
            timestamp=datetime.now(),
            engine_temp=85.5,
            vibration_level=2.3,
            oil_pressure=45.0,
            rpm=2500,
            mileage=50000,
            battery_voltage=13.8,
            brake_wear=35.0,
            tire_pressure=[32.0, 32.5, 31.5, 32.0],
            ambient_temp=25.0,
            load_weight=1500.0
        )
        
        assert processor.validate_reading(reading) is True

    def test_validate_reading_invalid(self):
        processor = SensorStreamProcessor()
        
        reading = SensorReading(
            vehicle_id="VEH-1234",
            timestamp=datetime.now(),
            engine_temp=250.0,  # Invalid: > 200
            vibration_level=2.3,
            oil_pressure=45.0,
            rpm=2500,
            mileage=50000,
            battery_voltage=13.8,
            brake_wear=35.0,
            tire_pressure=[32.0, 32.5, 31.5, 32.0],
            ambient_temp=25.0,
            load_weight=1500.0
        )
        
        assert processor.validate_reading(reading) is False


class TestSensorDataBuffer:
    """Tests for SensorDataBuffer."""

    def test_add_reading(self):
        buffer = SensorDataBuffer()
        
        reading = SensorReading(
            vehicle_id="VEH-1234",
            timestamp=datetime.now(),
            engine_temp=85.5,
            vibration_level=2.3,
            oil_pressure=45.0,
            rpm=2500,
            mileage=50000,
            battery_voltage=13.8,
            brake_wear=35.0,
            tire_pressure=[32.0, 32.5, 31.5, 32.0],
            ambient_temp=25.0,
            load_weight=1500.0
        )
        
        buffer.add_reading(reading)
        
        assert "VEH-1234" in buffer.get_all_vehicles()

    def test_get_recent_readings(self):
        buffer = SensorDataBuffer()
        
        # Add multiple readings
        for i in range(10):
            reading = SensorReading(
                vehicle_id="VEH-1234",
                timestamp=datetime.now() - timedelta(minutes=i),
                engine_temp=85.5 + i,
                vibration_level=2.3,
                oil_pressure=45.0,
                rpm=2500,
                mileage=50000,
                battery_voltage=13.8,
                brake_wear=35.0,
                tire_pressure=[32.0, 32.5, 31.5, 32.0],
                ambient_temp=25.0,
                load_weight=1500.0
            )
            buffer.add_reading(reading)
        
        readings = buffer.get_recent_readings("VEH-1234", count=5)
        
        assert len(readings) == 5


class TestLogProcessor:
    """Tests for LogProcessor."""

    def test_parse_log_entry(self):
        processor = LogProcessor()
        
        log_entry = "[2024-01-15 10:30:00] VEH-1234 start: {\"duration_seconds\": 3600, \"speed\": 60}"
        
        log = processor.parse_log_entry(log_entry)
        
        assert log is not None
        assert log.vehicle_id == "VEH-1234"
        assert log.event_type == "start"
        assert log.duration_seconds == 3600

    def test_process_json_logs(self):
        processor = LogProcessor()
        
        json_logs = [
            {
                "vehicle_id": "VEH-1234",
                "timestamp": datetime.now().isoformat(),
                "event_type": "start",
                "duration_seconds": 3600,
                "speed": 60
            }
        ]
        
        logs = processor.process_json_logs(json_logs)
        
        assert len(logs) == 1
        assert logs[0].vehicle_id == "VEH-1234"


class TestHistoryLoader:
    """Tests for HistoryLoader."""

    def test_load_from_csv(self):
        loader = HistoryLoader()
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("record_id,vehicle_id,date,maintenance_type,failure_type,severity,parts_replaced,repair_cost,downtime_hours\n")
            f.write("REC-001,VEH-1234,2024-01-15,corrective,engine,high,\"oil filter,air filter\",1500.00,24\n")
            temp_path = f.name
        
        try:
            records = loader.load_from_csv(temp_path)
            
            assert len(records) == 1
            assert records[0].vehicle_id == "VEH-1234"
            assert records[0].failure_type == "engine"
        finally:
            os.unlink(temp_path)

    def test_process_records(self):
        loader = HistoryLoader()
        
        records = [
            MaintenanceRecord(
                record_id="REC-001",
                vehicle_id="VEH-1234",
                date=datetime(2024, 1, 15),
                maintenance_type="corrective",
                failure_type="engine",
                severity="high",
                parts_replaced=["oil filter"],
                repair_cost=1500.0,
                downtime_hours=24,
                technician_id="TECH-001",
                workshop_id="WS-001",
                root_cause="Wear and tear",
                action_taken="Replaced oil filter",
                notes="Routine maintenance"
            )
        ]
        
        loader.process_records(records)
        
        assert "VEH-1234" in loader.vehicle_histories
        history = loader.vehicle_histories["VEH-1234"]
        assert history.total_maintenance_count == 1
        assert history.total_repair_cost == 1500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
