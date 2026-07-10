"""Maintenance history loader and processor."""
import csv
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class MaintenanceRecord:
    """Schema for maintenance records."""
    record_id: str
    vehicle_id: str
    date: datetime
    maintenance_type: str  # preventive, corrective, predictive
    failure_type: Optional[str]  # engine, brake, electrical, transmission, tire, battery
    severity: Optional[str]  # low, medium, high, critical
    parts_replaced: List[str]
    repair_cost: float
    downtime_hours: float
    technician_id: Optional[str]
    workshop_id: Optional[str]
    root_cause: Optional[str]
    action_taken: Optional[str]
    notes: Optional[str]
    next_maintenance_date: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VehicleHistory:
    """Complete maintenance history for a vehicle."""
    vehicle_id: str
    total_maintenance_count: int
    total_repair_cost: float
    total_downtime_hours: float
    last_maintenance_date: Optional[datetime]
    failure_history: List[Dict[str, Any]]
    maintenance_frequency: float  # average days between maintenance
    common_failures: List[Dict[str, Any]]


class HistoryLoader:
    """Load and process maintenance history data."""

    def __init__(self):
        self.records: Dict[str, List[MaintenanceRecord]] = {}
        self.vehicle_histories: Dict[str, VehicleHistory] = {}

    def load_from_csv(self, file_path: str) -> List[MaintenanceRecord]:
        """Load maintenance records from CSV file."""
        records = []
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"CSV file not found: {file_path}")
            return records
        
        try:
            df = pd.read_csv(path)
            
            for _, row in df.iterrows():
                try:
                    record = MaintenanceRecord(
                        record_id=str(row.get("record_id", "")),
                        vehicle_id=str(row["vehicle_id"]),
                        date=pd.to_datetime(row["date"]),
                        maintenance_type=str(row.get("maintenance_type", "corrective")),
                        failure_type=row.get("failure_type"),
                        severity=row.get("severity"),
                        parts_replaced=self._parse_list_field(row.get("parts_replaced", "")),
                        repair_cost=float(row.get("repair_cost", 0)),
                        downtime_hours=float(row.get("downtime_hours", 0)),
                        technician_id=row.get("technician_id"),
                        workshop_id=row.get("workshop_id"),
                        root_cause=row.get("root_cause"),
                        action_taken=row.get("action_taken"),
                        notes=row.get("notes"),
                        next_maintenance_date=pd.to_datetime(row["next_maintenance_date"]) if pd.notna(row.get("next_maintenance_date")) else None
                    )
                    records.append(record)
                    
                    vehicle_id = record.vehicle_id
                    if vehicle_id not in self.records:
                        self.records[vehicle_id] = []
                    self.records[vehicle_id].append(record)
                    
                except Exception as e:
                    logger.error(f"Error parsing row: {e}")
                    continue
            
            logger.info(f"Loaded {len(records)} maintenance records from {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return records

    def load_from_json(self, file_path: str) -> List[MaintenanceRecord]:
        """Load maintenance records from JSON file."""
        records = []
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"JSON file not found: {file_path}")
            return records
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            if isinstance(data, list):
                json_records = data
            else:
                json_records = data.get("records", [])
            
            for json_record in json_records:
                try:
                    record = MaintenanceRecord(
                        record_id=str(json_record.get("record_id", "")),
                        vehicle_id=json_record["vehicle_id"],
                        date=datetime.fromisoformat(json_record["date"]),
                        maintenance_type=json_record.get("maintenance_type", "corrective"),
                        failure_type=json_record.get("failure_type"),
                        severity=json_record.get("severity"),
                        parts_replaced=json_record.get("parts_replaced", []),
                        repair_cost=float(json_record.get("repair_cost", 0)),
                        downtime_hours=float(json_record.get("downtime_hours", 0)),
                        technician_id=json_record.get("technician_id"),
                        workshop_id=json_record.get("workshop_id"),
                        root_cause=json_record.get("root_cause"),
                        action_taken=json_record.get("action_taken"),
                        notes=json_record.get("notes"),
                        next_maintenance_date=datetime.fromisoformat(json_record["next_maintenance_date"]) if json_record.get("next_maintenance_date") else None
                    )
                    records.append(record)
                    
                    vehicle_id = record.vehicle_id
                    if vehicle_id not in self.records:
                        self.records[vehicle_id] = []
                    self.records[vehicle_id].append(record)
                    
                except Exception as e:
                    logger.error(f"Error parsing JSON record: {e}")
                    continue
            
            logger.info(f"Loaded {len(records)} maintenance records from {file_path}")
            return records
            
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            return records

    def load_from_database(self, connection_string: str, query: str) -> List[MaintenanceRecord]:
        """Load maintenance records from database."""
        records = []
        
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(connection_string)
            
            df = pd.read_sql(query, engine)
            
            for _, row in df.iterrows():
                try:
                    record = MaintenanceRecord(
                        record_id=str(row.get("record_id", "")),
                        vehicle_id=str(row["vehicle_id"]),
                        date=pd.to_datetime(row["date"]),
                        maintenance_type=str(row.get("maintenance_type", "corrective")),
                        failure_type=row.get("failure_type"),
                        severity=row.get("severity"),
                        parts_replaced=self._parse_list_field(row.get("parts_replaced", "")),
                        repair_cost=float(row.get("repair_cost", 0)),
                        downtime_hours=float(row.get("downtime_hours", 0)),
                        technician_id=row.get("technician_id"),
                        workshop_id=row.get("workshop_id"),
                        root_cause=row.get("root_cause"),
                        action_taken=row.get("action_taken"),
                        notes=row.get("notes"),
                        next_maintenance_date=pd.to_datetime(row["next_maintenance_date"]) if pd.notna(row.get("next_maintenance_date")) else None
                    )
                    records.append(record)
                    
                    vehicle_id = record.vehicle_id
                    if vehicle_id not in self.records:
                        self.records[vehicle_id] = []
                    self.records[vehicle_id].append(record)
                    
                except Exception as e:
                    logger.error(f"Error parsing database row: {e}")
                    continue
            
            logger.info(f"Loaded {len(records)} maintenance records from database")
            return records
            
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
            return records

    def process_records(self, records: List[MaintenanceRecord]):
        """Process loaded records and build vehicle histories."""
        # Group by vehicle
        for record in records:
            vehicle_id = record.vehicle_id
            if vehicle_id not in self.records:
                self.records[vehicle_id] = []
            self.records[vehicle_id].append(record)
        
        # Build vehicle histories
        for vehicle_id, vehicle_records in self.records.items():
            self._build_vehicle_history(vehicle_id, vehicle_records)

    def _build_vehicle_history(self, vehicle_id: str, records: List[MaintenanceRecord]):
        """Build complete history for a vehicle."""
        if not records:
            return
        
        # Sort by date
        records.sort(key=lambda x: x.date)
        
        # Calculate statistics
        total_cost = sum(r.repair_cost for r in records)
        total_downtime = sum(r.downtime_hours for r in records)
        
        # Calculate maintenance frequency
        if len(records) > 1:
            date_diffs = [
                (records[i+1].date - records[i].date).days
                for i in range(len(records) - 1)
            ]
            avg_frequency = sum(date_diffs) / len(date_diffs)
        else:
            avg_frequency = 0
        
        # Count failure types
        failure_counts: Dict[str, int] = {}
        for record in records:
            if record.failure_type:
                failure_counts[record.failure_type] = failure_counts.get(record.failure_type, 0) + 1
        
        common_failures = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        history = VehicleHistory(
            vehicle_id=vehicle_id,
            total_maintenance_count=len(records),
            total_repair_cost=total_cost,
            total_downtime_hours=total_downtime,
            last_maintenance_date=records[-1].date,
            failure_history=[
                {
                    "date": r.date.isoformat(),
                    "failure_type": r.failure_type,
                    "severity": r.severity,
                    "cost": r.repair_cost
                }
                for r in records if r.failure_type
            ],
            maintenance_frequency=avg_frequency,
            common_failures=[
                {"failure_type": ft, "count": c}
                for ft, c in common_failures
            ]
        )
        
        self.vehicle_histories[vehicle_id] = history

    def get_vehicle_history(self, vehicle_id: str) -> Optional[VehicleHistory]:
        """Get complete history for a vehicle."""
        return self.vehicle_histories.get(vehicle_id)

    def get_vehicle_records(self, vehicle_id: str, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[MaintenanceRecord]:
        """Get maintenance records for a vehicle within date range."""
        if vehicle_id not in self.records:
            return []
        
        records = self.records[vehicle_id]
        
        if start_date:
            records = [r for r in records if r.date >= start_date]
        if end_date:
            records = [r for r in records if r.date <= end_date]
        
        return records

    def predict_next_maintenance(self, vehicle_id: str) -> Optional[datetime]:
        """Predict next maintenance date based on historical frequency."""
        history = self.vehicle_histories.get(vehicle_id)
        
        if not history or not history.last_maintenance_date:
            return None
        
        if history.maintenance_frequency > 0:
            from datetime import timedelta
            return history.last_maintenance_date + timedelta(days=history.maintenance_frequency)
        
        return None

    def get_failure_probability(self, vehicle_id: str, failure_type: str) -> float:
        """Calculate probability of specific failure type for a vehicle."""
        history = self.vehicle_histories.get(vehicle_id)
        
        if not history or not history.failure_history:
            return 0.0
        
        failure_count = sum(
            1 for f in history.failure_history
            if f["failure_type"] == failure_type
        )
        
        return failure_count / len(history.failure_history)

    def _parse_list_field(self, value: Any) -> List[str]:
        """Parse a field that might contain a list."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try JSON parsing first
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass
            # Try comma-separated
            if value:
                return [item.strip() for item in value.split(",")]
        return []

    def to_dataframe(self, vehicle_id: Optional[str] = None) -> pd.DataFrame:
        """Convert records to pandas DataFrame."""
        if vehicle_id:
            records = self.records.get(vehicle_id, [])
        else:
            records = []
            for vehicle_records in self.records.values():
                records.extend(vehicle_records)
        
        if not records:
            return pd.DataFrame()
        
        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)
        
        # Convert datetime columns
        for col in ["date", "next_maintenance_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
