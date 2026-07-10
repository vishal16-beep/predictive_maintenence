"""
XGBoost RUL Prediction Training for NASA C-MAPSS FD001

Features:
- XGBoost regression (reg:squarederror)
- Early stopping on validation set
- NASA asymmetric scoring function (penalizes late predictions)
- Saves model + feature list for API consumption
"""
import argparse
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cmapss_loader import prepare_data, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# NASA Asymmetric Scoring Function
# ─────────────────────────────────────────────────────────────────────────────
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.
    
    Penalizes late/underestimated RUL predictions MORE than early ones.
    
    For each prediction:
      s = exp(-d/13) - 1   if d < 0  (predicted too late / underestimated)
      s = exp(d/10) - 1    if d >= 0 (predicted too early / overestimated)
      
    where d = y_pred - y_true
    
    Lower total score is better.
    """
    d = y_pred - y_true
    
    # Late predictions (underestimated RUL) - penalized more heavily
    late_penalty = np.where(d < 0, np.exp(-d / 13) - 1, 0)
    
    # Early predictions (overestimated RUL) - penalized less
    early_penalty = np.where(d >= 0, np.exp(d / 10) - 1, 0)
    
    total_score = np.sum(late_penalty + early_penalty)
    
    return total_score


def asymmetric_penalty(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Per-sample asymmetric penalty.
    Returns array of penalties for each prediction.
    """
    d = y_pred - y_true
    
    penalty = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  # Late: stronger penalty
        np.exp(d / 10) - 1    # Early: weaker penalty
    )
    
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list,
    params: dict = None
) -> tuple:
    """
    Train XGBoost RUL regression model with early stopping.
    
    Returns: (model, train_metrics, val_metrics)
    """
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "verbosity": 0
        }
    
    early_stopping_rounds = params.pop("early_stopping_rounds", 30)
    
    # Prepare data
    X_train = train_df[feature_columns].values
    y_train = train_df["rul"].values
    
    X_val = val_df[feature_columns].values
    y_val = val_df["rul"].values
    
    logger.info(f"Training XGBoost:")
    logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"  Val:   {X_val.shape[0]} samples")
    logger.info(f"  Params: {params}")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    
    # Training with early stopping
    evals = [(dtrain, "train"), (dval, "eval")]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 500),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
    )
    
    logger.info(f"Best iteration: {model.best_iteration}")
    
    # Predictions
    y_pred_train = model.predict(dtrain)
    y_pred_val = model.predict(dval)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_nasa = nasa_score(y_train, y_pred_train)
    
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_nasa = nasa_score(y_val, y_pred_val)
    
    train_metrics = {"rmse": train_rmse, "mae": train_mae, "nasa_score": train_nasa}
    val_metrics = {"rmse": val_rmse, "mae": val_mae, "nasa_score": val_nasa}
    
    logger.info(f"\nTrain Metrics: RMSE={train_rmse:.4f}, MAE={train_mae:.4f}, NASA={train_nasa:.1f}")
    logger.info(f"Val Metrics:   RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, NASA={val_nasa:.1f}")
    
    return model, train_metrics, val_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(model, test_df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Evaluate on real test set (last cycle per engine).
    """
    X_test = test_df[feature_columns].values
    y_test = test_df["rul"].values
    
    dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
    y_pred = model.predict(dtest)
    
    # Clip predictions to [0, 125] to match training RUL scale
    y_pred_clipped = np.clip(y_pred, 0, 125)
    
    # Standard metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    mae = mean_absolute_error(y_test, y_pred_clipped)
    
    # NASA asymmetric score
    nasa = nasa_score(y_test, y_pred_clipped)
    
    # Per-engine breakdown
    engine_ids = test_df["unit_number"].values
    per_engine = []
    for i, eid in enumerate(engine_ids):
        per_engine.append({
            "engine_id": int(eid),
            "true_rul": float(y_test[i]),
            "predicted_rul": float(y_pred_clipped[i]),
            "error": float(y_pred_clipped[i] - y_test[i]),
            "penalty": float(asymmetric_penalty(np.array([y_test[i]]), np.array([y_pred_clipped[i]]))[0])
        })
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "nasa_score": float(nasa),
        "n_engines": len(engine_ids),
        "mean_error": float(np.mean(y_pred_clipped - y_test)),
        "std_error": float(np.std(y_pred_clipped - y_test)),
        "per_engine": per_engine
    }
    
    return metrics, y_pred_clipped


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────
def save_model_artifacts(
    model,
    feature_columns: list,
    metrics: dict,
    output_dir: str
):
    """
    Save trained model and metadata for API consumption.
    
    Saves:
    - xgboost_rul_model.json (XGBoost model)
    - model_metadata.json (feature columns, metrics, config)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save XGBoost model
    model_path = output_path / "xgboost_rul_model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "xgboost_regression",
        "objective": "reg:squarederror",
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "rul_clip": 125,
        "dataset": "C-MAPSS FD001",
        "training_metrics": metrics.get("train_metrics", {}),
        "validation_metrics": metrics.get("val_metrics", {}),
        "test_metrics": {
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "nasa_score": metrics.get("nasa_score")
        },
        "hyperparameters": {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "early_stopping_rounds": 30,
            "objective": "reg:squarederror"
        },
        "dropped_sensors": ["sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19"],
        "remaining_sensors": [
            "sensor_2", "sensor_3", "sensor_4",
            "sensor_7", "sensor_8", "sensor_9",
            "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
            "sensor_17", "sensor_20", "sensor_21"
        ],
        "op_settings": ["op_setting_1", "op_setting_2", "op_setting_3"],
        "rolling_window": 10,
        "description": "XGBoost RUL prediction model trained on NASA C-MAPSS FD001 turbofan engine degradation dataset"
    }
    
    metadata_path = output_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Also save feature columns as separate file for quick loading
    features_path = output_path / "feature_columns.json"
    with open(features_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    logger.info(f"Feature columns saved to {features_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train XGBoost RUL model on C-MAPSS FD001")
    parser.add_argument("--data-dir", default="./data", help="Directory containing FD001 files")
    parser.add_argument("--output-dir", default="./models", help="Output directory for model")
    parser.add_argument("--dataset", default="FD001", help="C-MAPSS dataset subset")
    
    args = parser.parse_args()
    
    # 1. Load and prepare data
    data = prepare_data(args.data_dir, args.dataset)
    
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]
    feature_columns = data["feature_columns"]
    
    # 2. Train XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("Training XGBoost Model")
    logger.info("=" * 60)
    
    model, train_metrics, val_metrics = train_xgboost(
        train_df, val_df, feature_columns
    )
    
    # 3. Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on Test Set")
    logger.info("=" * 60)
    
    test_metrics, predictions = evaluate_on_test(model, test_df, feature_columns)
    
    logger.info(f"\nTest Set Results:")
    logger.info(f"  RMSE:        {test_metrics['rmse']:.4f}")
    logger.info(f"  MAE:         {test_metrics['mae']:.4f}")
    logger.info(f"  NASA Score:  {test_metrics['nasa_score']:.1f}")
    logger.info(f"  Mean Error:  {test_metrics['mean_error']:.2f}")
    logger.info(f"  Std Error:   {test_metrics['std_error']:.2f}")
    
    # 4. Save model and metadata
    logger.info("\n" + "=" * 60)
    logger.info("Saving Model Artifacts")
    logger.info("=" * 60)
    
    all_metrics = {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "nasa_score": test_metrics["nasa_score"]
    }
    
    save_model_artifacts(model, feature_columns, all_metrics, args.output_dir)
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nDataset: NASA C-MAPSS {args.dataset}")
    print(f"Features: {len(feature_columns)}")
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  MAE:  {val_metrics['mae']:.4f}")
    print(f"  NASA: {val_metrics['nasa_score']:.1f}")
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  NASA: {test_metrics['nasa_score']:.1f}")
    print(f"\nModel saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
