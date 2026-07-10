import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "PredictiveMaintenance"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/predictive_maintenance"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_SENSOR_TOPIC: str = "sensor-data"
    KAFKA_ALERT_TOPIC: str = "alerts"

    # MQTT Configuration
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_USERNAME: Optional[str] = None
    MQTT_PASSWORD: Optional[str] = None

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "predictive-maintenance"

    # Model Paths
    MODEL_DIR: str = "./models"
    RUL_MODEL_PATH: str = "./models/rul_model.keras"
    ANOMALY_MODEL_PATH: str = "./models/anomaly_model.keras"
    CLASSIFICATION_MODEL_PATH: str = "./models/classification_model.pkl"

    # Data Paths
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    FEATURES_DIR: str = "./data/features"

    # Alert Thresholds
    ANOMALY_THRESHOLD: float = 0.8
    RUL_WARNING_DAYS: int = 7
    CRITICAL_RUL_DAYS: int = 3

    # Feature Store
    FEATURE_STORE_PATH: str = "./data/features"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


def get_settings() -> Settings:
    return settings
