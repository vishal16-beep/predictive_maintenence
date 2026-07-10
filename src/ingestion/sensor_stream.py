"""Sensor data stream ingestion from IoT devices."""
import json
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Schema for sensor data readings."""
    vehicle_id: str
    timestamp: datetime
    engine_temp: float
    vibration_level: float
    oil_pressure: float
    rpm: int
    mileage: float
    battery_voltage: float
    brake_wear: float
    tire_pressure: List[float]
    ambient_temp: float
    load_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SensorStreamProcessor:
    """Processes real-time sensor data streams from vehicles."""

    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092"):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.consumers: Dict[str, Any] = {}
        self.processors: List[Callable] = []

    def register_processor(self, processor: Callable[[SensorReading], None]):
        """Register a callback function to process sensor readings."""
        self.processors.append(processor)

    async def consume_kafka_stream(self, topic: str, group_id: str):
        """Consume sensor data from Kafka topic."""
        try:
            from kafka import KafkaConsumer

            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="latest"
            )

            logger.info(f"Started consuming from topic: {topic}")

            for message in consumer:
                try:
                    data = message.value
                    reading = self._parse_sensor_reading(data)
                    
                    for processor in self.processors:
                        processor(reading)
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        except ImportError:
            logger.warning("kafka-python not installed. Using mock consumer.")
            await self._mock_kafka_consumer(topic)

    async def _mock_kafka_consumer(self, topic: str):
        """Mock Kafka consumer for testing without Kafka."""
        import random
        
        while True:
            reading = SensorReading(
                vehicle_id=f"VEH-{random.randint(1000, 9999)}",
                timestamp=datetime.now(),
                engine_temp=random.uniform(70, 120),
                vibration_level=random.uniform(0.1, 10.0),
                oil_pressure=random.uniform(20, 60),
                rpm=random.randint(800, 4000),
                mileage=random.uniform(10000, 200000),
                battery_voltage=random.uniform(11.5, 14.5),
                brake_wear=random.uniform(0, 100),
                tire_pressure=[random.uniform(30, 35) for _ in range(4)],
                ambient_temp=random.uniform(-10, 45),
                load_weight=random.uniform(0, 5000)
            )
            
            for processor in self.processors:
                processor(reading)
            
            await asyncio.sleep(0.1)

    async def consume_mqtt_stream(self, broker: str, port: int, topic: str):
        """Consume sensor data from MQTT broker."""
        try:
            import paho.mqtt.client as mqtt

            def on_connect(client, userdata, flags, rc):
                logger.info(f"Connected to MQTT broker with result code {rc}")
                client.subscribe(topic)

            def on_message(client, userdata, msg):
                try:
                    data = json.loads(msg.payload.decode())
                    reading = self._parse_sensor_reading(data)
                    
                    for processor in self.processors:
                        processor(reading)
                        
                except Exception as e:
                    logger.error(f"Error processing MQTT message: {e}")

            client = mqtt.Client()
            client.on_connect = on_connect
            client.on_message = on_message
            
            client.connect(broker, port, 60)
            client.loop_forever()

        except ImportError:
            logger.warning("paho-mqtt not installed. Using mock MQTT.")
            await self._mock_mqtt_consumer()

    async def _mock_mqtt_consumer(self):
        """Mock MQTT consumer for testing."""
        await self._mock_kafka_consumer("mock-mqtt-topic")

    def _parse_sensor_reading(self, data: Dict[str, Any]) -> SensorReading:
        """Parse raw data dict into SensorReading object."""
        return SensorReading(
            vehicle_id=data["vehicle_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            engine_temp=float(data["engine_temp"]),
            vibration_level=float(data["vibration_level"]),
            oil_pressure=float(data["oil_pressure"]),
            rpm=int(data["rpm"]),
            mileage=float(data["mileage"]),
            battery_voltage=float(data["battery_voltage"]),
            brake_wear=float(data["brake_wear"]),
            tire_pressure=[float(x) for x in data["tire_pressure"]],
            ambient_temp=float(data["ambient_temp"]),
            load_weight=float(data["load_weight"])
        )

    def validate_reading(self, reading: SensorReading) -> bool:
        """Validate sensor reading values are within acceptable ranges."""
        validations = [
            (reading.engine_temp >= 0 and reading.engine_temp <= 200, "Engine temp out of range"),
            (reading.vibration_level >= 0 and reading.vibration_level <= 50, "Vibration out of range"),
            (reading.oil_pressure >= 0 and reading.oil_pressure <= 100, "Oil pressure out of range"),
            (reading.rpm >= 0 and reading.rpm <= 10000, "RPM out of range"),
            (reading.battery_voltage >= 0 and reading.battery_voltage <= 20, "Battery voltage out of range"),
            (reading.brake_wear >= 0 and reading.brake_wear <= 100, "Brake wear out of range"),
        ]
        
        for is_valid, error_msg in validations:
            if not is_valid:
                logger.warning(f"Invalid reading for {reading.vehicle_id}: {error_msg}")
                return False
        
        return True


class SensorDataBuffer:
    """Buffer for storing recent sensor readings per vehicle."""

    def __init__(self, max_buffer_size: int = 1000):
        self.buffers: Dict[str, List[SensorReading]] = {}
        self.max_buffer_size = max_buffer_size

    def add_reading(self, reading: SensorReading):
        """Add a sensor reading to the buffer."""
        vehicle_id = reading.vehicle_id
        
        if vehicle_id not in self.buffers:
            self.buffers[vehicle_id] = []
        
        self.buffers[vehicle_id].append(reading)
        
        # Maintain buffer size
        if len(self.buffers[vehicle_id]) > self.max_buffer_size:
            self.buffers[vehicle_id].pop(0)

    def get_recent_readings(self, vehicle_id: str, count: int = 100) -> List[SensorReading]:
        """Get recent readings for a specific vehicle."""
        if vehicle_id not in self.buffers:
            return []
        return self.buffers[vehicle_id][-count:]

    def get_all_vehicles(self) -> List[str]:
        """Get list of all vehicles with buffered data."""
        return list(self.buffers.keys())

    def clear_buffer(self, vehicle_id: Optional[str] = None):
        """Clear buffer for specific vehicle or all vehicles."""
        if vehicle_id:
            self.buffers.pop(vehicle_id, None)
        else:
            self.buffers.clear()
