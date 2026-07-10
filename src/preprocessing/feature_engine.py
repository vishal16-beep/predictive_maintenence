"""Feature engineering module for predictive maintenance."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features from sensor, operational, and maintenance data."""

    def __init__(self, window_sizes: List[int] = None):
        self.window_sizes = window_sizes or [5, 10, 30, 60]
        self.feature_names: List[str] = []

    def create_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from sensor data."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Rolling statistics
        df = self._add_rolling_features(df)
        
        # Rate of change features
        df = self._add_rate_of_change_features(df)
        
        # Interaction features
        df = self._add_interaction_features(df)
        
        # Statistical features over windows
        df = self._add_statistical_features(df)
        
        self.feature_names = [col for col in df.columns if col not in 
                             ["vehicle_id", "timestamp", "failure_type", "maintenance_type"]]
        
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features from timestamp."""
        if "timestamp" not in df.columns:
            return df
        
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_night"] = df["hour"].isin(range(20, 6)).astype(int)
        
        # Time since last maintenance (if maintenance_date column exists)
        if "maintenance_date" in df.columns:
            df["days_since_maintenance"] = (
                df["timestamp"] - pd.to_datetime(df["maintenance_date"])
            ).dt.days
        
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window statistics."""
        sensor_cols = ["engine_temp", "vibration_level", "oil_pressure", 
                      "rpm", "battery_voltage", "brake_wear"]
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
            
            for window in self.window_sizes:
                # Rolling mean
                df[f"{col}_rolling_mean_{window}"] = (
                    df.groupby("vehicle_id")[col]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Rolling std
                df[f"{col}_rolling_std_{window}"] = (
                    df.groupby("vehicle_id")[col]
                    .rolling(window=window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
                
                # Rolling min
                df[f"{col}_rolling_min_{window}"] = (
                    df.groupby("vehicle_id")[col]
                    .rolling(window=window, min_periods=1)
                    .min()
                    .reset_index(0, drop=True)
                )
                
                # Rolling max
                df[f"{col}_rolling_max_{window}"] = (
                    df.groupby("vehicle_id")[col]
                    .rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(0, drop=True)
                )
        
        return df

    def _add_rate_of_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features."""
        sensor_cols = ["engine_temp", "vibration_level", "oil_pressure", 
                      "rpm", "battery_voltage"]
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
            
            # First derivative (rate of change)
            df[f"{col}_roc"] = (
                df.groupby("vehicle_id")[col]
                .diff()
                .fillna(0)
            )
            
            # Second derivative (acceleration)
            df[f"{col}_acceleration"] = (
                df.groupby("vehicle_id")[f"{col}_roc"]
                .diff()
                .fillna(0)
            )
        
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between sensors."""
        # Temperature-RPM interaction
        if "engine_temp" in df.columns and "rpm" in df.columns:
            df["temp_rpm_interaction"] = df["engine_temp"] * df["rpm"] / 1000
        
        # Vibration-Load interaction
        if "vibration_level" in df.columns and "load_weight" in df.columns:
            df["vibration_load_interaction"] = df["vibration_level"] * df["load_weight"] / 1000
        
        # Oil pressure-RPM interaction
        if "oil_pressure" in df.columns and "rpm" in df.columns:
            df["oil_rpm_interaction"] = df["oil_pressure"] * df["rpm"] / 1000
        
        # Battery voltage - engine temp interaction
        if "battery_voltage" in df.columns and "engine_temp" in df.columns:
            df["battery_temp_interaction"] = df["battery_voltage"] / (df["engine_temp"] + 1)
        
        # Tire pressure variance
        if "tire_pressure" in df.columns:
            df["tire_pressure_mean"] = df["tire_pressure"].apply(
                lambda x: np.mean(x) if isinstance(x, list) else x
            )
            df["tire_pressure_std"] = df["tire_pressure"].apply(
                lambda x: np.std(x) if isinstance(x, list) else 0
            )
            df["tire_pressure_diff"] = df["tire_pressure"].apply(
                lambda x: max(x) - min(x) if isinstance(x, list) else 0
            )
        
        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features over time windows."""
        sensor_cols = ["engine_temp", "vibration_level", "oil_pressure", "rpm"]
        
        for col in sensor_cols:
            if col not in df.columns:
                continue
            
            for window in self.window_sizes:
                # Skewness
                df[f"{col}_skew_{window}"] = (
                    df.groupby("vehicle_id")[col]
                    .rolling(window=window, min_periods=1)
                    .skew()
                    .reset_index(0, drop=True)
                )
                
                # Kurtosis
                df[f"{col}_kurt_{window}"] = (
                    df.groupby("vehicle_id")[col]
                    .rolling(window=window, min_periods=1)
                    .apply(lambda x: stats.kurtosis(x) if len(x) > 2 else 0)
                    .reset_index(0, drop=True)
                )
        
        return df

    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from operational logs."""
        df = df.copy()
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Event frequency features
        df = self._add_event_frequency_features(df)
        
        # Trip-based features
        df = self._add_trip_features(df)
        
        # Usage intensity features
        df = self._add_usage_intensity_features(df)
        
        return df

    def _add_event_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add event frequency features."""
        if "event_type" not in df.columns:
            return df
        
        # Count events per vehicle per day
        df["date"] = df["timestamp"].dt.date
        
        event_counts = (
            df.groupby(["vehicle_id", "date", "event_type"])
            .size()
            .unstack(fill_value=0)
        )
        
        # Add event counts as features
        for event_type in event_counts.columns:
            df[f"event_count_{event_type}"] = df.apply(
                lambda row: event_counts.loc[
                    (row["vehicle_id"], row["date"], event_type)
                ] if (row["vehicle_id"], row["date"], event_type) in event_counts.index else 0,
                axis=1
            )
        
        return df

    def _add_trip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trip-based features."""
        if "speed" not in df.columns:
            return df
        
        # Calculate trip statistics per vehicle per day
        if "date" not in df.columns:
            df["date"] = df["timestamp"].dt.date
        
        trip_stats = (
            df.groupby(["vehicle_id", "date"])
            .agg({
                "speed": ["mean", "max", "min", "std"],
                "duration_seconds": "sum"
            })
            .reset_index()
        )
        
        trip_stats.columns = [
            "vehicle_id", "date", 
            "avg_speed", "max_speed", "min_speed", "speed_std",
            "total_duration"
        ]
        
        # Merge back
        df = df.merge(trip_stats, on=["vehicle_id", "date"], how="left")
        
        return df

    def _add_usage_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add usage intensity features."""
        if "mileage" in df.columns:
            # Daily mileage
            df["daily_mileage"] = (
                df.groupby(["vehicle_id", df["timestamp"].dt.date])["mileage"]
                .transform(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
            )
        
        if "fuel_consumption" in df.columns and "speed" in df.columns:
            # Fuel efficiency
            df["fuel_efficiency"] = df["fuel_consumption"] / (df["speed"] + 1)
        
        return df

    def create_maintenance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from maintenance history."""
        df = df.copy()
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        
        # Days since last maintenance
        df = self._add_days_since_maintenance(df)
        
        # Maintenance frequency
        df = self._add_maintenance_frequency(df)
        
        # Cost features
        df = self._add_cost_features(df)
        
        # Failure history features
        df = self._add_failure_history_features(df)
        
        return df

    def _add_days_since_maintenance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add days since last maintenance feature."""
        if "date" not in df.columns or "vehicle_id" not in df.columns:
            return df
        
        df = df.sort_values(["vehicle_id", "date"])
        
        df["days_since_last_maintenance"] = (
            df.groupby("vehicle_id")["date"]
            .diff()
            .dt.days
            .fillna(0)
        )
        
        return df

    def _add_maintenance_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maintenance frequency features."""
        if "date" not in df.columns or "vehicle_id" not in df.columns:
            return df
        
        # Calculate rolling maintenance frequency
        df["maintenance_count"] = (
            df.groupby("vehicle_id")
            .cumcount() + 1
        )
        
        # Average days between maintenance
        df["avg_days_between_maintenance"] = (
            df.groupby("vehicle_id")["days_since_last_maintenance"]
            .expanding()
            .mean()
            .reset_index(0, drop=True)
        )
        
        return df

    def _add_cost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cost-related features."""
        if "repair_cost" not in df.columns or "vehicle_id" not in df.columns:
            return df
        
        # Cumulative cost
        df["cumulative_repair_cost"] = (
            df.groupby("vehicle_id")["repair_cost"]
            .cumsum()
        )
        
        # Average cost per maintenance
        df["avg_repair_cost"] = (
            df.groupby("vehicle_id")["repair_cost"]
            .expanding()
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Cost trend
        df["cost_trend"] = (
            df.groupby("vehicle_id")["repair_cost"]
            .pct_change()
            .fillna(0)
        )
        
        return df

    def _add_failure_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add failure history features."""
        if "failure_type" not in df.columns or "vehicle_id" not in df.columns:
            return df
        
        # Count of each failure type per vehicle
        failure_dummies = pd.get_dummies(df["failure_type"], prefix="failure")
        failure_counts = (
            failure_dummies.groupby(df["vehicle_id"])
            .cumsum()
        )
        
        # Add as features
        for col in failure_counts.columns:
            df[f"{col}_count"] = failure_counts[col]
        
        # Has had critical failure
        if "severity" in df.columns:
            df["had_critical_failure"] = (
                df.groupby("vehicle_id")["severity"]
                .apply(lambda x: (x == "critical").any())
                .reset_index(0, drop=True)
                .astype(int)
            )
        
        return df

    def create_rul_labels(self, df: pd.DataFrame, failure_window_days: int = 30) -> pd.DataFrame:
        """Create Remaining Useful Life (RUL) labels for supervised learning."""
        df = df.copy()
        
        if "failure_date" not in df.columns and "next_failure_date" not in df.columns:
            # If no failure date, estimate from maintenance patterns
            logger.warning("No failure date column found. Using estimated RUL.")
            df["rul_days"] = failure_window_days
            return df
        
        # Use next_failure_date if available, otherwise failure_date
        date_col = "next_failure_date" if "next_failure_date" in df.columns else "failure_date"
        
        df[date_col] = pd.to_datetime(df[date_col])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Calculate RUL in days
        df["rul_days"] = (df[date_col] - df["timestamp"]).dt.days
        
        # Clip negative values (past failure)
        df["rul_days"] = df["rul_days"].clip(lower=0)
        
        # Create RUL categories
        df["rul_category"] = pd.cut(
            df["rul_days"],
            bins=[0, 7, 14, 30, 60, float("inf")],
            labels=["critical", "urgent", "warning", "monitoring", "healthy"]
        )
        
        return df

    def create_anomaly_labels(self, df: pd.DataFrame, threshold_std: float = 3.0) -> pd.DataFrame:
        """Create anomaly labels based on statistical thresholds."""
        df = df.copy()
        
        sensor_cols = ["engine_temp", "vibration_level", "oil_pressure", "rpm"]
        
        # Calculate Z-scores for each sensor
        for col in sensor_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                df[f"{col}_zscore"] = np.abs((df[col] - mean) / (std + 1e-8))
        
        # Anomaly if any sensor has high Z-score
        zscore_cols = [f"{col}_zscore" for col in sensor_cols if f"{col}_zscore" in df.columns]
        
        if zscore_cols:
            df["is_anomaly"] = (df[zscore_cols].max(axis=1) > threshold_std).astype(int)
            df["anomaly_score"] = df[zscore_cols].max(axis=1) / threshold_std
        
        # Drop temporary z-score columns
        df = df.drop(columns=zscore_cols, errors="ignore")
        
        return df

    def get_feature_importance(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Calculate feature importance using correlation and mutual information."""
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in 
                       ["vehicle_id", "timestamp", target, "failure_type", "maintenance_type"]]
        
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target]
        
        # Drop rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Calculate correlations
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # Calculate mutual information
        if y.dtype in ["float64", "float32"]:
            mi_scores = mutual_info_regression(X, y)
        else:
            mi_scores = mutual_info_classif(X, y)
        
        mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Combine into DataFrame
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "correlation": correlations.values,
            "mutual_information": mi_series.values
        })
        
        importance_df["combined_score"] = (
            importance_df["correlation"] * 0.5 + 
            importance_df["mutual_information"] * 0.5
        )
        
        return importance_df.sort_values("combined_score", ascending=False)

    def select_features(self, df: pd.DataFrame, target: str, 
                       top_k: int = 20) -> List[str]:
        """Select top-k features based on importance."""
        importance_df = self.get_feature_importance(df, target)
        return importance_df.head(top_k)["feature"].tolist()
