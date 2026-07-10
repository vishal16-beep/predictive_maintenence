"""Training script for predictive maintenance models."""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_engine import FeatureEngineer
from src.preprocessing.scaler import DataScaler
from src.models.classification.xgboost_model import FailureClassifier
from src.training.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(data_dir: str):
    """Load data from directory."""
    data_path = Path(data_dir)
    
    # Load sensor data
    sensor_file = data_path / "sensor_data.csv"
    if sensor_file.exists():
        sensor_data = pd.read_csv(sensor_file)
        sensor_data["timestamp"] = pd.to_datetime(sensor_data["timestamp"])
    else:
        logger.error(f"Sensor data not found: {sensor_file}")
        return None, None
    
    # Load maintenance history
    maintenance_file = data_path / "maintenance_history.csv"
    if maintenance_file.exists():
        maintenance_data = pd.read_csv(maintenance_file)
        maintenance_data["date"] = pd.to_datetime(maintenance_data["date"])
    else:
        logger.warning(f"Maintenance data not found: {maintenance_file}")
        maintenance_data = pd.DataFrame()
    
    return sensor_data, maintenance_data


def preprocess_data(sensor_data: pd.DataFrame):
    """Preprocess the data for classification."""
    logger.info("Preprocessing data...")
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_sensor_data(sensor_data)
    
    # Engineer features
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.create_sensor_features(cleaned_data)
    
    # Select only numeric columns for features
    numeric_cols = featured_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["vehicle_id", "timestamp", "failure_type", "rul_days"]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return featured_data, feature_cols


def train_classification_model(data: pd.DataFrame, feature_cols: list, output_dir: str):
    """Train classification model."""
    logger.info("Training classification model...")
    
    # Select only numeric features
    X = data[feature_cols].values
    
    # Generate synthetic failure labels for demo
    y = np.random.choice(
        ["engine", "brake", "electrical", "transmission", "tire"],
        len(data)
    )
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Train model
    classifier = FailureClassifier(n_classes=5)
    classifier.build_models()
    results = classifier.train(X_train, y_train, X_val, y_val)
    
    logger.info("Classification model trained successfully")
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification_model(classifier, X_val, y_val)
    
    logger.info(f"Classification Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Classification F1: {metrics['f1_score']:.4f}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    classifier.save_model(str(output_path / "classification_model.pkl"))
    
    # Save feature columns and scaler info
    with open(output_path / "model_metadata.pkl", "wb") as f:
        pickle.dump({
            "feature_cols": feature_cols,
            "class_names": classifier.class_names,
            "metrics": metrics
        }, f)
    
    logger.info(f"Model saved to {output_dir}")
    
    return classifier, metrics


def main():
    parser = argparse.ArgumentParser(description="Train predictive maintenance models")
    parser.add_argument("--data-dir", default="./data/sample_data", help="Data directory")
    parser.add_argument("--output-dir", default="./models", help="Output directory for models")
    
    args = parser.parse_args()
    
    # Load data
    sensor_data, maintenance_data = load_data(args.data_dir)
    
    if sensor_data is None:
        logger.error("Failed to load data")
        return
    
    # Preprocess data
    featured_data, feature_cols = preprocess_data(sensor_data)
    
    # Train classification model
    classifier, metrics = train_classification_model(featured_data, feature_cols, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel trained: Classification")
    print(f"\nFeatures used: {len(feature_cols)}")
    print(f"\nEvaluation Results:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
