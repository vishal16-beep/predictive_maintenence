"""Operational log processor for vehicle usage data."""
import re
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperationalLog:
    """Schema for operational logs."""
    vehicle_id: str
    timestamp: datetime
    event_type: str  # start, stop, idle, acceleration, braking, turn
    duration_seconds: Optional[float] = None
    location: Optional[Dict[str, float]] = None  # lat, lon
    speed: Optional[float] = None
    fuel_consumption: Optional[float] = None
    driver_id: Optional[str] = None
    route_id: Optional[str] = None
    weather_condition: Optional[str] = None
    road_condition: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TripSummary:
    """Summary of a vehicle trip."""
    vehicle_id: str
    trip_id: str
    start_time: datetime
    end_time: datetime
    total_distance: float  # km
    total_duration: float  # seconds
    average_speed: float  # km/h
    max_speed: float
    fuel_used: float  # liters
    idle_time: float  # seconds
    stops_count: int
    harsh_events: int  # harsh braking, acceleration
    route_efficiency: float  # percentage


class LogProcessor:
    """Process operational logs from vehicles."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self.logs: Dict[str, List[OperationalLog]] = {}
        self.trip_summaries: Dict[str, TripSummary] = {}

    def parse_log_entry(self, raw_log: str) -> Optional[OperationalLog]:
        """Parse a raw log entry string into OperationalLog."""
        try:
            # Expected format: [TIMESTAMP] VEHICLE_ID EVENT_TYPE: {json_data}
            pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+): (\w+): (.+)"
            match = re.match(pattern, raw_log)
            
            if not match:
                logger.warning(f"Failed to parse log entry: {raw_log[:100]}")
                return None
            
            timestamp_str, vehicle_id, event_type, json_data = match.groups()
            data = json.loads(json_data)
            
            return OperationalLog(
                vehicle_id=vehicle_id,
                timestamp=datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S"),
                event_type=event_type,
                duration_seconds=data.get("duration_seconds"),
                location=data.get("location"),
                speed=data.get("speed"),
                fuel_consumption=data.get("fuel_consumption"),
                driver_id=data.get("driver_id"),
                route_id=data.get("route_id"),
                weather_condition=data.get("weather_condition"),
                road_condition=data.get("road_condition"),
                notes=data.get("notes")
            )
            
        except Exception as e:
            logger.error(f"Error parsing log entry: {e}")
            return None

    def load_logs_from_file(self, file_path: str) -> List[OperationalLog]:
        """Load operational logs from a file."""
        logs = []
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"Log file not found: {file_path}")
            return logs
        
        try:
            with open(path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    log_entry = self.parse_log_entry(line)
                    if log_entry:
                        logs.append(log_entry)
                        
                        # Store in memory
                        vehicle_id = log_entry.vehicle_id
                        if vehicle_id not in self.logs:
                            self.logs[vehicle_id] = []
                        self.logs[vehicle_id].append(log_entry)
            
            logger.info(f"Loaded {len(logs)} log entries from {file_path}")
            return logs
            
        except Exception as e:
            logger.error(f"Error loading logs from {file_path}: {e}")
            return logs

    def load_logs_from_directory(self, dir_path: str, pattern: str = "*.log") -> List[OperationalLog]:
        """Load all log files from a directory."""
        all_logs = []
        path = Path(dir_path)
        
        if not path.exists():
            logger.error(f"Directory not found: {dir_path}")
            return all_logs
        
        for log_file in path.glob(pattern):
            logs = self.load_logs_from_file(str(log_file))
            all_logs.extend(logs)
        
        logger.info(f"Loaded {len(all_logs)} total log entries from {dir_path}")
        return all_logs

    def process_json_logs(self, json_logs: List[Dict[str, Any]]) -> List[OperationalLog]:
        """Process JSON format logs."""
        logs = []
        
        for json_log in json_logs:
            try:
                log_entry = OperationalLog(
                    vehicle_id=json_log["vehicle_id"],
                    timestamp=datetime.fromisoformat(json_log["timestamp"]),
                    event_type=json_log["event_type"],
                    duration_seconds=json_log.get("duration_seconds"),
                    location=json_log.get("location"),
                    speed=json_log.get("speed"),
                    fuel_consumption=json_log.get("fuel_consumption"),
                    driver_id=json_log.get("driver_id"),
                    route_id=json_log.get("route_id"),
                    weather_condition=json_log.get("weather_condition"),
                    road_condition=json_log.get("road_condition"),
                    notes=json_log.get("notes")
                )
                logs.append(log_entry)
                
                vehicle_id = log_entry.vehicle_id
                if vehicle_id not in self.logs:
                    self.logs[vehicle_id] = []
                self.logs[vehicle_id].append(log_entry)
                
            except Exception as e:
                logger.error(f"Error processing JSON log: {e}")
                continue
        
        return logs

    def calculate_trip_summary(self, vehicle_id: str, start_time: datetime, end_time: datetime) -> Optional[TripSummary]:
        """Calculate trip summary for a vehicle between two timestamps."""
        if vehicle_id not in self.logs:
            return None
        
        vehicle_logs = [
            log for log in self.logs[vehicle_id]
            if start_time <= log.timestamp <= end_time
        ]
        
        if not vehicle_logs:
            return None
        
        # Sort by timestamp
        vehicle_logs.sort(key=lambda x: x.timestamp)
        
        # Calculate metrics
        speeds = [log.speed for log in vehicle_logs if log.speed is not None]
        total_distance = sum(
            (log.speed or 0) * (log.duration_seconds or 0) / 3600
            for log in vehicle_logs
        )
        total_duration = sum(log.duration_seconds or 0 for log in vehicle_logs)
        fuel_used = sum(log.fuel_consumption or 0 for log in vehicle_logs)
        
        # Count harsh events (high speed changes)
        harsh_events = 0
        for i in range(1, len(vehicle_logs)):
            if vehicle_logs[i].speed and vehicle_logs[i-1].speed:
                speed_change = abs(vehicle_logs[i].speed - vehicle_logs[i-1].speed)
                if speed_change > 30:  # Harsh acceleration/braking threshold
                    harsh_events += 1
        
        # Count stops (speed < 5 km/h)
        stops_count = sum(1 for speed in speeds if speed and speed < 5)
        
        # Calculate idle time
        idle_time = sum(
            log.duration_seconds or 0
            for log in vehicle_logs
            if log.event_type == "idle"
        )
        
        trip_summary = TripSummary(
            vehicle_id=vehicle_id,
            trip_id=f"TRIP-{vehicle_id}-{start_time.strftime('%Y%m%d%H%M%S')}",
            start_time=start_time,
            end_time=end_time,
            total_distance=total_distance,
            total_duration=total_duration,
            average_speed=total_distance / (total_duration / 3600) if total_duration > 0 else 0,
            max_speed=max(speeds) if speeds else 0,
            fuel_used=fuel_used,
            idle_time=idle_time,
            stops_count=stops_count,
            harsh_events=harsh_events,
            route_efficiency=0.0  # Calculate based on actual vs optimal route
        )
        
        self.trip_summaries[trip_summary.trip_id] = trip_summary
        return trip_summary

    def get_vehicle_logs(self, vehicle_id: str, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[OperationalLog]:
        """Get logs for a specific vehicle within time range."""
        if vehicle_id not in self.logs:
            return []
        
        logs = self.logs[vehicle_id]
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        return logs

    def aggregate_usage_statistics(self, vehicle_id: str, days: int = 30) -> Dict[str, Any]:
        """Aggregate usage statistics for a vehicle over specified days."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logs = self.get_vehicle_logs(vehicle_id, start_time, end_time)
        
        if not logs:
            return {}
        
        total_distance = sum(
            (log.speed or 0) * (log.duration_seconds or 0) / 3600
            for log in logs
        )
        total_fuel = sum(log.fuel_consumption or 0 for log in logs)
        total_duration = sum(log.duration_seconds or 0 for log in logs)
        
        event_counts = {}
        for log in logs:
            event_counts[log.event_type] = event_counts.get(log.event_type, 0) + 1
        
        return {
            "vehicle_id": vehicle_id,
            "period_days": days,
            "total_distance_km": total_distance,
            "total_fuel_liters": total_fuel,
            "total_operating_hours": total_duration / 3600,
            "average_daily_distance": total_distance / days,
            "event_counts": event_counts,
            "log_entries_count": len(logs)
        }
