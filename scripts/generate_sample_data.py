"""Generate sample data for testing and demonstration."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate realistic sample data for predictive maintenance."""

    def __init__(self, n_vehicles: int = 20, n_days: int = 90):
        self.n_vehicles = n_vehicles
        self.n_days = n_days
        self.vehicle_ids = [f"VEH-{1000 + i}" for i in range(n_vehicles)]
        
        # Vehicle characteristics
        self.vehicle_types = np.random.choice(
            ["truck", "bus", "van", "car"],
            n_vehicles
        )
        
        # Failure patterns per vehicle type
        self.failure_patterns = {
            "truck": {"engine": 0.3, "brake": 0.25, "transmission": 0.2, "electrical": 0.15, "tire": 0.1},
            "bus": {"engine": 0.25, "brake": 0.3, "transmission": 0.15, "electrical": 0.2, "tire": 0.1},
            "van": {"engine": 0.2, "brake": 0.2, "transmission": 0.25, "electrical": 0.25, "tire": 0.1},
            "car": {"engine": 0.15, "brake": 0.15, "transmission": 0.2, "electrical": 0.35, "tire": 0.15}
        }

    def generate_sensor_data(self) -> pd.DataFrame:
        """Generate sensor readings data."""
        logger.info(f"Generating sensor data for {self.n_vehicles} vehicles over {self.n_days} days...")
        
        records = []
        
        for vehicle_id, vehicle_type in zip(self.vehicle_ids, self.vehicle_types):
            # Generate daily readings
            for day in range(self.n_days):
                date = datetime.now() - timedelta(days=self.n_days - day)
                
                # Multiple readings per day
                for hour in range(0, 24, 2):  # Every 2 hours
                    timestamp = date.replace(hour=hour)
                    
                    # Base values with some randomness
                    base_values = self._get_base_values(vehicle_type)
                    
                    # Add degradation over time
                    degradation = day / self.n_days
                    
                    # Add random anomalies (5% chance)
                    is_anomaly = np.random.random() < 0.05
                    
                    reading = {
                        "vehicle_id": vehicle_id,
                        "timestamp": timestamp.isoformat(),
                        "engine_temp": self._add_noise(base_values["engine_temp"] + degradation * 10, is_anomaly, 20),
                        "vibration_level": self._add_noise(base_values["vibration_level"] + degradation * 2, is_anomaly, 5),
                        "oil_pressure": self._add_noise(base_values["oil_pressure"] - degradation * 5, is_anomaly, 10),
                        "rpm": int(self._add_noise(base_values["rpm"], is_anomaly, 500)),
                        "mileage": 50000 + day * 100 + np.random.randint(0, 50),
                        "battery_voltage": self._add_noise(base_values["battery_voltage"], is_anomaly, 2),
                        "brake_wear": min(100, base_values["brake_wear"] + degradation * 20),
                        "tire_pressure": [
                            self._add_noise(base_values["tire_pressure"], is_anomaly, 3)
                            for _ in range(4)
                        ],
                        "ambient_temp": 25 + 10 * np.sin(day * 2 * np.pi / 365) + np.random.normal(0, 5),
                        "load_weight": np.random.uniform(500, 3000) if vehicle_type in ["truck", "bus"] else np.random.uniform(100, 500)
                    }
                    
                    records.append(reading)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} sensor readings")
        return df

    def generate_operational_logs(self) -> pd.DataFrame:
        """Generate operational logs data."""
        logger.info("Generating operational logs...")
        
        records = []
        event_types = ["start", "stop", "idle", "acceleration", "braking", "turn"]
        
        for vehicle_id in self.vehicle_ids:
            for day in range(self.n_days):
                date = datetime.now() - timedelta(days=self.n_days - day)
                
                # Generate events per day
                n_events = np.random.randint(10, 30)
                
                for _ in range(n_events):
                    hour = np.random.randint(6, 22)  # Operating hours
                    minute = np.random.randint(0, 60)
                    
                    timestamp = date.replace(hour=hour, minute=minute)
                    event_type = np.random.choice(event_types, p=[0.2, 0.2, 0.15, 0.2, 0.15, 0.1])
                    
                    record = {
                        "vehicle_id": vehicle_id,
                        "timestamp": timestamp.isoformat(),
                        "event_type": event_type,
                        "duration_seconds": np.random.randint(10, 3600),
                        "location": {
                            "lat": 40.7128 + np.random.normal(0, 0.1),
                            "lon": -74.0060 + np.random.normal(0, 0.1)
                        },
                        "speed": np.random.uniform(0, 120) if event_type != "idle" else 0,
                        "fuel_consumption": np.random.uniform(0.5, 5.0) if event_type != "idle" else 0.1,
                        "driver_id": f"DRV-{np.random.randint(1, 50):03d}",
                        "route_id": f"RTE-{np.random.randint(1, 20):03d}",
                        "weather_condition": np.random.choice(["clear", "cloudy", "rain", "snow"], p=[0.5, 0.3, 0.15, 0.05]),
                        "road_condition": np.random.choice(["good", "fair", "poor"], p=[0.6, 0.3, 0.1])
                    }
                    
                    records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} operational log entries")
        return df

    def generate_maintenance_history(self) -> pd.DataFrame:
        """Generate maintenance history data."""
        logger.info("Generating maintenance history...")
        
        records = []
        failure_types = ["engine", "brake", "transmission", "electrical", "tire", "battery"]
        severities = ["low", "medium", "high", "critical"]
        parts_by_failure = {
            "engine": ["oil filter", "air filter", "spark plugs", "piston rings"],
            "brake": ["brake pads", "brake rotors", "brake fluid", "calipers"],
            "transmission": ["transmission fluid", "clutch", "gears", "synchros"],
            "electrical": ["alternator", "starter motor", "battery", "wiring harness"],
            "tire": ["tires", "wheel bearings", "valve stems", "TPMS sensors"],
            "battery": ["battery", "alternator", "cables", "terminal connectors"]
        }
        
        for vehicle_id, vehicle_type in zip(self.vehicle_ids, self.vehicle_types):
            # Generate maintenance records
            n_records = np.random.randint(5, 20)
            
            for _ in range(n_records):
                # Random date in the past year
                days_ago = np.random.randint(0, 365)
                date = datetime.now() - timedelta(days=days_ago)
                
                # Choose failure type based on vehicle type
                failure_probs = self.failure_patterns[vehicle_type]
                failure_type = np.random.choice(
                    list(failure_probs.keys()),
                    p=list(failure_probs.values())
                )
                
                severity = np.random.choice(severities, p=[0.3, 0.4, 0.2, 0.1])
                
                # Number of parts replaced based on severity
                n_parts = {"low": 1, "medium": 2, "high": 3, "critical": 4}[severity]
                parts = np.random.choice(
                    parts_by_failure[failure_type],
                    size=min(n_parts, len(parts_by_failure[failure_type])),
                    replace=False
                ).tolist()
                
                # Cost based on severity and parts
                base_cost = {"low": 100, "medium": 500, "high": 1500, "critical": 5000}[severity]
                cost = base_cost + len(parts) * np.random.uniform(50, 200)
                
                # Downtime based on severity
                downtime_hours = {"low": 2, "medium": 8, "high": 24, "critical": 72}[severity]
                downtime_hours += np.random.uniform(-2, 4)
                
                record = {
                    "record_id": f"REC-{vehicle_id}-{days_ago}",
                    "vehicle_id": vehicle_id,
                    "date": date.isoformat(),
                    "maintenance_type": np.random.choice(["preventive", "corrective", "predictive"]),
                    "failure_type": failure_type,
                    "severity": severity,
                    "parts_replaced": parts,
                    "repair_cost": round(cost, 2),
                    "downtime_hours": round(max(1, downtime_hours), 1),
                    "technician_id": f"TECH-{np.random.randint(1, 20):03d}",
                    "workshop_id": f"WS-{np.random.randint(1, 5):03d}",
                    "root_cause": f"Wear and tear on {failure_type} components",
                    "action_taken": f"Replaced {', '.join(parts)}",
                    "notes": f"Routine {severity} maintenance",
                    "next_maintenance_date": (date + timedelta(days=np.random.randint(30, 90))).isoformat()
                }
                
                records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} maintenance records")
        return df

    def _get_base_values(self, vehicle_type: str) -> dict:
        """Get base sensor values based on vehicle type."""
        base_values = {
            "truck": {
                "engine_temp": 90,
                "vibration_level": 3.0,
                "oil_pressure": 45,
                "rpm": 2000,
                "battery_voltage": 13.5,
                "brake_wear": 30,
                "tire_pressure": 35
            },
            "bus": {
                "engine_temp": 85,
                "vibration_level": 2.5,
                "oil_pressure": 50,
                "rpm": 1800,
                "battery_voltage": 14.0,
                "brake_wear": 40,
                "tire_pressure": 33
            },
            "van": {
                "engine_temp": 80,
                "vibration_level": 2.0,
                "oil_pressure": 40,
                "rpm": 2200,
                "battery_voltage": 13.8,
                "brake_wear": 25,
                "tire_pressure": 32
            },
            "car": {
                "engine_temp": 75,
                "vibration_level": 1.5,
                "oil_pressure": 35,
                "rpm": 2500,
                "battery_voltage": 14.2,
                "brake_wear": 20,
                "tire_pressure": 32
            }
        }
        return base_values[vehicle_type]

    def _add_noise(self, value: float, is_anomaly: bool = False, anomaly_range: float = 10) -> float:
        """Add noise to a value, with larger noise for anomalies."""
        noise = np.random.normal(0, 1)
        if is_anomaly:
            noise = np.random.uniform(-anomaly_range, anomaly_range)
        return round(value + noise, 2)

    def save_sample_data(self, output_dir: str = "./data/sample_data"):
        """Generate and save all sample data."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        sensor_data = self.generate_sensor_data()
        operational_logs = self.generate_operational_logs()
        maintenance_history = self.generate_maintenance_history()
        
        # Save to CSV
        sensor_data.to_csv(output_path / "sensor_data.csv", index=False)
        operational_logs.to_csv(output_path / "operational_logs.csv", index=False)
        maintenance_history.to_csv(output_path / "maintenance_history.csv", index=False)
        
        # Save to JSON
        sensor_data.to_json(output_path / "sensor_data.json", orient="records", indent=2)
        operational_logs.to_json(output_path / "operational_logs.json", orient="records", indent=2)
        maintenance_history.to_json(output_path / "maintenance_history.json", orient="records", indent=2)
        
        # Save summary
        summary = {
            "generation_date": datetime.now().isoformat(),
            "n_vehicles": self.n_vehicles,
            "n_days": self.n_days,
            "vehicle_ids": self.vehicle_ids,
            "vehicle_types": self.vehicle_types.tolist(),
            "data_statistics": {
                "sensor_readings": len(sensor_data),
                "operational_logs": len(operational_logs),
                "maintenance_records": len(maintenance_history)
            }
        }
        
        with open(output_path / "data_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Sample data saved to {output_dir}")
        
        return {
            "sensor_data": sensor_data,
            "operational_logs": operational_logs,
            "maintenance_history": maintenance_history,
            "summary": summary
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = SampleDataGenerator(n_vehicles=20, n_days=90)
    data = generator.save_sample_data()
    
    print(f"\nSample Data Generated:")
    print(f"  - Sensor readings: {len(data['sensor_data'])}")
    print(f"  - Operational logs: {len(data['operational_logs'])}")
    print(f"  - Maintenance records: {len(data['maintenance_history'])}")
