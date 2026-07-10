"""
NASA C-MAPSS FD001 Dataset Loader and Processor

Handles:
- Loading train/test/RUL files with correct column schema
- RUL labeling with clipping at 125
- Dropping uninformative sensors
- Rolling feature engineering (window=10)
- Train/validation split BY ENGINE ID (no leakage)
- Test set: last cycle per engine only
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Column schema for C-MAPSS
COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21"
]

# Sensors to drop (near-constant / uninformative)
SENSORS_TO_DROP = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19"
]

# Remaining sensors after dropping
REMAINING_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21"
]

# Operational settings (always keep)
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# Rolling window size
ROLLING_WINDOW = 10

# RUL clipping threshold
RUL_CLIP = 125


def load_cmapss_data(data_dir: str, dataset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and RUL files for a C-MAPSS dataset.
    
    Returns:
        train_df, test_df, rul_df
    """
    data_path = Path(data_dir)
    
    train_file = data_path / f"train_{dataset}.txt"
    test_file = data_path / f"test_{dataset}.txt"
    rul_file = data_path / f"RUL_{dataset}.txt"
    
    # Load files (space-separated, no header)
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul"])
    
    print(f"Loaded {dataset}:")
    print(f"  Train: {train_df.shape[0]} rows, {train_df['unit_number'].nunique()} engines")
    print(f"  Test:  {test_df.shape[0]} rows, {test_df['unit_number'].nunique()} engines")
    print(f"  RUL:   {rul_df.shape[0]} engines")
    
    return train_df, test_df, rul_df


def label_rul(train_df: pd.DataFrame, clip_at: int = RUL_CLIP) -> pd.DataFrame:
    """
    Label RUL for training data.
    
    RUL = (max cycle for that engine) - current cycle, clipped at `clip_at`.
    """
    df = train_df.copy()
    
    # Max cycle per engine
    max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
    
    # RUL = max_cycle - current_cycle
    df["rul"] = max_cycles - df["time_in_cycles"]
    
    # Clip RUL (standard C-MAPSS practice)
    df["rul"] = df["rul"].clip(upper=clip_at)
    
    print(f"RUL labeled: min={df['rul'].min()}, max={df['rul'].max()}, mean={df['rul'].mean():.1f}")
    
    return df


def drop_uninformative_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-constant/uninformative sensors."""
    cols_to_drop = [c for c in SENSORS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def get_feature_columns(include_rul: bool = False) -> List[str]:
    """Get the list of feature columns after dropping uninformative sensors."""
    features = OP_SETTINGS + REMAINING_SENSORS
    if include_rul:
        features = features + ["rul"]
    return features


def engineer_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Engineer rolling mean and rolling std per engine for each remaining sensor.
    """
    df = df.copy()
    
    # Sort by engine and cycle
    df = df.sort_values(["unit_number", "time_in_cycles"]).reset_index(drop=True)
    
    # Feature columns to engineer
    feature_cols = OP_SETTINGS + REMAINING_SENSORS
    
    # Generate rolling features per engine
    rolling_cols = []
    for col in feature_cols:
        mean_col = f"{col}_roll_mean"
        std_col = f"{col}_roll_std"
        
        df[mean_col] = (
            df.groupby("unit_number")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df[std_col] = (
            df.groupby("unit_number")[col]
            .rolling(window=window, min_periods=1)
            .std()
            .fillna(0)
            .reset_index(0, drop=True)
        )
        
        rolling_cols.extend([mean_col, std_col])
    
    # Replace inf with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # Fill NaN
    df = df.fillna(0)
    
    print(f"Engineered {len(rolling_cols)} rolling features")
    
    return df


def prepare_test_set_last_cycle(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    For the test set, only use the LAST cycle per engine.
    This is what RUL_FD001.txt provides ground truth for.
    """
    # Get last cycle per engine
    last_cycle = (
        test_df.groupby("unit_number")
        .last()
        .reset_index()
    )
    
    # Add ground truth RUL
    last_cycle["rul"] = rul_df["rul"].values
    
    print(f"Test set: {len(last_cycle)} engines (last cycle only)")
    
    return last_cycle


def split_train_val_by_engine(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train/validation BY ENGINE ID (not random rows).
    This avoids leakage between rows of the same engine's life.
    """
    np.random.seed(random_seed)
    
    engine_ids = df["unit_number"].unique()
    np.random.shuffle(engine_ids)
    
    n_val = int(len(engine_ids) * val_fraction)
    val_engines = engine_ids[:n_val]
    train_engines = engine_ids[n_val:]
    
    train_split = df[df["unit_number"].isin(train_engines)].reset_index(drop=True)
    val_split = df[df["unit_number"].isin(val_engines)].reset_index(drop=True)
    
    print(f"Train/Val split by engine ID:")
    print(f"  Train: {len(train_engines)} engines, {len(train_split)} rows")
    print(f"  Val:   {len(val_engines)} engines, {len(val_split)} rows")
    
    return train_split, val_split


def prepare_data(data_dir: str, dataset: str = "FD001") -> dict:
    """
    Complete data preparation pipeline.
    
    Returns dict with:
        - train_df, val_df: Training/validation with RUL labels
        - test_df: Test set (last cycle per engine with RUL)
        - feature_columns: Ordered list of feature columns the model expects
    """
    print("=" * 60)
    print(f"Loading C-MAPSS {dataset} dataset")
    print("=" * 60)
    
    # 1. Load raw data
    train_raw, test_raw, rul_df = load_cmapss_data(data_dir, dataset)
    
    # 2. Label RUL for training
    train_labeled = label_rul(train_raw)
    
    # 3. Drop uninformative sensors
    train_clean = drop_uninformative_sensors(train_labeled)
    test_clean = drop_uninformative_sensors(test_raw)
    
    # 4. Engineer rolling features
    train_featured = engineer_rolling_features(train_clean)
    test_featured = engineer_rolling_features(test_clean)
    
    # 5. Prepare test set (last cycle only)
    test_final = prepare_test_set_last_cycle(test_featured, rul_df)
    
    # 6. Split train/val by engine ID
    train_split, val_split = split_train_val_by_engine(train_featured)
    
    # 7. Get feature columns
    feature_columns = get_feature_columns(include_rul=False)
    
    # Add rolling feature columns
    rolling_cols = []
    for col in feature_columns:
        rolling_cols.extend([f"{col}_roll_mean", f"{col}_roll_std"])
    
    all_features = feature_columns + rolling_cols
    
    # Filter to only columns that exist
    all_features = [c for c in all_features if c in train_split.columns]
    
    print("\n" + "=" * 60)
    print(f"Feature columns ({len(all_features)} total):")
    print("=" * 60)
    for i, col in enumerate(all_features):
        print(f"  {i+1:2d}. {col}")
    
    return {
        "train_df": train_split,
        "val_df": val_split,
        "test_df": test_final,
        "feature_columns": all_features,
        "dataset": dataset
    }


if __name__ == "__main__":
    # Quick test
    data = prepare_data("./data")
    
    print(f"\nTrain shape: {data['train_df'].shape}")
    print(f"Val shape:   {data['val_df'].shape}")
    print(f"Test shape:  {data['test_df'].shape}")
    print(f"Features:    {len(data['feature_columns'])}")
