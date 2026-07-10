"""Data scaling and normalization module."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import logging

logger = logging.getLogger(__name__)


class DataScaler:
    """Scale and normalize features for ML models."""

    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict] = {}

    def fit_transform_numerical(self, df: pd.DataFrame, 
                               columns: Optional[List[str]] = None,
                               method: str = "standard") -> pd.DataFrame:
        """Fit and transform numerical columns."""
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID and timestamp columns
        exclude_cols = ["vehicle_id", "timestamp", "record_id"]
        columns = [col for col in columns if col not in exclude_cols]
        
        if not columns:
            return df
        
        # Select scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df[columns] = scaler.fit_transform(df[columns])
        
        # Store scaler
        self.scalers[method] = scaler
        self.feature_stats[method] = {
            "columns": columns,
            "mean": dict(zip(columns, scaler.mean_)) if hasattr(scaler, 'mean_') else None,
            "std": dict(zip(columns, scaler.scale_)) if hasattr(scaler, 'scale_') else None
        }
        
        logger.info(f"Fitted {method} scaler on {len(columns)} columns")
        return df

    def transform_numerical(self, df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Transform numerical columns using fitted scaler."""
        df = df.copy()
        
        if method not in self.scalers:
            raise ValueError(f"Scaler for method '{method}' not fitted yet")
        
        scaler = self.scalers[method]
        columns = self.feature_stats[method]["columns"]
        
        # Only transform columns that exist
        existing_cols = [col for col in columns if col in df.columns]
        
        if existing_cols:
            df[existing_cols] = scaler.transform(df[existing_cols])
        
        return df

    def inverse_transform_numerical(self, df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Inverse transform numerical columns."""
        df = df.copy()
        
        if method not in self.scalers:
            raise ValueError(f"Scaler for method '{method}' not fitted yet")
        
        scaler = self.scalers[method]
        columns = self.feature_stats[method]["columns"]
        
        existing_cols = [col for col in columns if col in df.columns]
        
        if existing_cols:
            df[existing_cols] = scaler.inverse_transform(df[existing_cols])
        
        return df

    def fit_transform_categorical(self, df: pd.DataFrame,
                                  columns: Optional[List[str]] = None,
                                  method: str = "onehot",
                                  max_cardinality: int = 20) -> pd.DataFrame:
        """Fit and transform categorical columns."""
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Exclude ID columns
        exclude_cols = ["vehicle_id", "record_id"]
        columns = [col for col in columns if col not in exclude_cols]
        
        if not columns:
            return df
        
        for col in columns:
            if col not in df.columns:
                continue
            
            cardinality = df[col].nunique()
            
            if cardinality <= max_cardinality:
                if method == "onehot":
                    # One-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    self.encoders[col] = {"method": "onehot", "categories": df[col].unique().tolist()}
                    
                elif method == "label":
                    # Label encoding
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[col] = {"method": "label", "encoder": encoder}
            else:
                # High cardinality: use frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq"] = df[col].map(freq_map)
                df = df.drop(columns=[col])
                self.encoders[col] = {"method": "frequency", "freq_map": freq_map}
            
            logger.info(f"Encoded column '{col}' with method: {method}")
        
        return df

    def transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted encoders."""
        df = df.copy()
        
        for col, encoder_info in self.encoders.items():
            if col not in df.columns:
                continue
            
            method = encoder_info["method"]
            
            if method == "onehot":
                # One-hot encoding
                categories = encoder_info["categories"]
                for cat in categories:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df = df.drop(columns=[col])
                
            elif method == "label":
                # Label encoding
                encoder = encoder_info["encoder"]
                # Handle unseen categories
                df[col] = df[col].apply(
                    lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                )
                df[col] = encoder.transform(df[col].astype(str))
                
            elif method == "frequency":
                # Frequency encoding
                freq_map = encoder_info["freq_map"]
                df[f"{col}_freq"] = df[col].map(freq_map).fillna(0)
                df = df.drop(columns=[col])
        
        return df

    def create_scaler_pipeline(self, df: pd.DataFrame, 
                              numerical_method: str = "standard",
                              categorical_method: str = "onehot") -> pd.DataFrame:
        """Complete preprocessing pipeline for features."""
        df = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Exclude ID and target columns
        exclude_cols = ["vehicle_id", "timestamp", "record_id", 
                       "failure_type", "maintenance_type", "rul_days", "is_anomaly"]
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # Transform numerical
        if numerical_cols:
            df = self.fit_transform_numerical(df, numerical_cols, numerical_method)
        
        # Transform categorical
        if categorical_cols:
            df = self.fit_transform_categorical(df, categorical_cols, categorical_method)
        
        return df

    def save_scalers(self, filepath: str):
        """Save all scalers and encoders to file."""
        data = {
            "scalers": self.scalers,
            "encoders": self.encoders,
            "feature_stats": self.feature_stats
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved scalers to {filepath}")

    def load_scalers(self, filepath: str):
        """Load scalers and encoders from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.scalers = data["scalers"]
        self.encoders = data["encoders"]
        self.feature_stats = data["feature_stats"]
        
        logger.info(f"Loaded scalers from {filepath}")


class TimeSeriesScaler:
    """Specialized scaler for time series data."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.scalers: Dict[str, StandardScaler] = {}

    def fit_transform_sequence(self, sequences: np.ndarray, 
                              feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform time series sequences."""
        # sequences shape: (n_samples, n_timesteps, n_features)
        
        n_samples, n_timesteps, n_features = sequences.shape
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Reshape to 2D for fitting
        sequences_2d = sequences.reshape(-1, n_features)
        
        # Fit scaler
        scaler = StandardScaler()
        sequences_2d_scaled = scaler.fit_transform(sequences_2d)
        
        # Reshape back to 3D
        sequences_scaled = sequences_2d_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Store scaler
        self.scalers["global"] = scaler
        
        logger.info(f"Fitted time series scaler on {n_features} features")
        return sequences_scaled

    def transform_sequence(self, sequences: np.ndarray) -> np.ndarray:
        """Transform time series sequences using fitted scaler."""
        if "global" not in self.scalers:
            raise ValueError("Scaler not fitted yet")
        
        n_samples, n_timesteps, n_features = sequences.shape
        
        # Reshape to 2D
        sequences_2d = sequences.reshape(-1, n_features)
        
        # Transform
        sequences_2d_scaled = self.scalers["global"].transform(sequences_2d)
        
        # Reshape back to 3D
        sequences_scaled = sequences_2d_scaled.reshape(n_samples, n_timesteps, n_features)
        
        return sequences_scaled

    def create_sequences(self, df: pd.DataFrame, 
                        feature_cols: List[str],
                        target_col: Optional[str] = None,
                        sequence_length: int = 10,
                        stride: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create sequences from DataFrame for time series models."""
        sequences = []
        targets = []
        
        for vehicle_id in df["vehicle_id"].unique():
            vehicle_data = df[df["vehicle_id"] == vehicle_id].sort_values("timestamp")
            
            for i in range(0, len(vehicle_data) - sequence_length + 1, stride):
                seq = vehicle_data[feature_cols].iloc[i:i + sequence_length].values
                sequences.append(seq)
                
                if target_col and target_col in df.columns:
                    target = vehicle_data[target_col].iloc[i + sequence_length - 1]
                    targets.append(target)
        
        sequences = np.array(sequences)
        
        if targets:
            targets = np.array(targets)
            return sequences, targets
        
        return sequences, None

    def save(self, filepath: str):
        """Save time series scaler."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "window_size": self.window_size,
                "scalers": self.scalers
            }, f)

    def load(self, filepath: str):
        """Load time series scaler."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.window_size = data["window_size"]
        self.scalers = data["scalers"]
