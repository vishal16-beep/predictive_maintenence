"""Data cleaning and validation module."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and validate sensor and operational data."""

    def __init__(self):
        self.cleaning_stats: Dict[str, Any] = {}
        self.validation_rules: Dict[str, Dict] = {}

    def set_validation_rules(self, rules: Dict[str, Dict]):
        """Set custom validation rules for columns."""
        self.validation_rules = rules

    def clean_sensor_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sensor data DataFrame."""
        initial_rows = len(df)
        self.cleaning_stats = {
            "initial_rows": initial_rows,
            "operations": []
        }

        # Remove duplicates
        df = self._remove_duplicates(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Validate ranges
        df = self._validate_ranges(df)
        
        # Fix data types
        df = self._fix_data_types(df)
        
        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)

        final_rows = len(df)
        self.cleaning_stats["final_rows"] = final_rows
        self.cleaning_stats["rows_removed"] = initial_rows - final_rows
        self.cleaning_stats["removal_percentage"] = (
            (initial_rows - final_rows) / initial_rows * 100 if initial_rows > 0 else 0
        )

        logger.info(f"Cleaning complete: {initial_rows} -> {final_rows} rows "
                    f"({self.cleaning_stats['removal_percentage']:.1f}% removed)")
        
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        if removed > 0:
            self.cleaning_stats["operations"].append({
                "operation": "remove_duplicates",
                "rows_removed": removed
            })
            logger.info(f"Removed {removed} duplicate rows")
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type."""
        missing_before = df.isnull().sum().sum()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # For numeric columns: use median imputation
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns: use mode imputation
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                logger.debug(f"Filled {col} missing values with mode: {mode_val}")
        
        missing_after = df.isnull().sum().sum()
        
        self.cleaning_stats["operations"].append({
            "operation": "handle_missing_values",
            "missing_before": missing_before,
            "missing_after": missing_after,
            "values_imputed": missing_before - missing_after
        })
        
        return df

    def _remove_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method."""
        initial_rows = len(df)
        
        if method == "iqr":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ["vehicle_id", "timestamp", "rpm"]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == "zscore":
            from scipy import stats
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ["vehicle_id", "timestamp"]
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores < threshold
                df = df[mask]
        
        removed = initial_rows - len(df)
        
        if removed > 0:
            self.cleaning_stats["operations"].append({
                "operation": "remove_outliers",
                "method": method,
                "rows_removed": removed
            })
            logger.info(f"Removed {removed} outlier rows using {method}")
        
        return df

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges based on domain knowledge."""
        initial_rows = len(df)
        
        default_rules = {
            "engine_temp": {"min": 0, "max": 200},
            "vibration_level": {"min": 0, "max": 50},
            "oil_pressure": {"min": 0, "max": 100},
            "rpm": {"min": 0, "max": 10000},
            "battery_voltage": {"min": 0, "max": 20},
            "brake_wear": {"min": 0, "max": 100},
            "load_weight": {"min": 0, "max": 10000},
        }
        
        rules = {**default_rules, **self.validation_rules}
        
        for col, rule in rules.items():
            if col in df.columns:
                # Skip non-numeric columns (like tire_pressure which contains lists)
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                    
                min_val = rule.get("min")
                max_val = rule.get("max")
                
                if min_val is not None:
                    df = df[df[col] >= min_val]
                if max_val is not None:
                    df = df[df[col] <= max_val]
        
        removed = initial_rows - len(df)
        
        if removed > 0:
            self.cleaning_stats["operations"].append({
                "operation": "validate_ranges",
                "rows_removed": removed
            })
            logger.info(f"Removed {removed} rows outside valid ranges")
        
        return df

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for consistency."""
        # Convert timestamp column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Convert numeric columns
        numeric_cols = ["engine_temp", "vibration_level", "oil_pressure", 
                       "battery_voltage", "brake_wear", "load_weight",
                       "ambient_temp", "mileage"]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Convert integer columns
        int_cols = ["rpm"]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype(int, errors="ignore")
        
        # Ensure tire_pressure is a list
        if "tire_pressure" in df.columns:
            df["tire_pressure"] = df["tire_pressure"].apply(
                lambda x: x if isinstance(x, list) else [32.0, 32.0, 32.0, 32.0]
            )
        
        return df

    def clean_operational_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean operational logs DataFrame."""
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        if "event_type" in df.columns:
            df["event_type"] = df["event_type"].fillna("unknown")
        
        # Fix timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
        
        # Sort by timestamp
        df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
        
        logger.info(f"Operational logs cleaned: {initial_rows} -> {len(df)} rows")
        return df

    def clean_maintenance_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean maintenance history DataFrame."""
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        if "failure_type" in df.columns:
            df["failure_type"] = df["failure_type"].fillna("unknown")
        
        if "severity" in df.columns:
            df["severity"] = df["severity"].fillna("low")
        
        # Fix date columns
        date_cols = ["date", "next_maintenance_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        
        # Fix numeric columns
        numeric_cols = ["repair_cost", "downtime_hours"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # Ensure parts_replaced is a list
        if "parts_replaced" in df.columns:
            df["parts_replaced"] = df["parts_replaced"].apply(
                lambda x: x if isinstance(x, list) else []
            )
        
        logger.info(f"Maintenance history cleaned: {initial_rows} -> {len(df)} rows")
        return df

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get summary of cleaning operations performed."""
        return self.cleaning_stats


class DataValidator:
    """Validate data quality using Great Expectations style checks."""

    def __init__(self):
        self.validation_results: List[Dict] = []

    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate DataFrame has expected columns."""
        missing_cols = set(expected_columns) - set(df.columns)
        
        result = {
            "check": "schema_validation",
            "passed": len(missing_cols) == 0,
            "missing_columns": list(missing_cols)
        }
        self.validation_results.append(result)
        
        if not result["passed"]:
            logger.error(f"Schema validation failed: missing {missing_cols}")
        
        return result["passed"]

    def validate_not_empty(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame is not empty."""
        result = {
            "check": "not_empty",
            "passed": len(df) > 0,
            "row_count": len(df)
        }
        self.validation_results.append(result)
        
        if not result["passed"]:
            logger.error("Validation failed: DataFrame is empty")
        
        return result["passed"]

    def validate_no_nulls(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
        """Validate no null values in specified columns."""
        check_cols = columns or df.columns.tolist()
        null_counts = df[check_cols].isnull().sum()
        total_nulls = null_counts.sum()
        
        result = {
            "check": "no_nulls",
            "passed": total_nulls == 0,
            "null_counts": null_counts[null_counts > 0].to_dict()
        }
        self.validation_results.append(result)
        
        if not result["passed"]:
            logger.warning(f"Null values found: {result['null_counts']}")
        
        return result["passed"]

    def validate_value_ranges(self, df: pd.DataFrame, ranges: Dict[str, Tuple[float, float]]) -> bool:
        """Validate values are within expected ranges."""
        all_valid = True
        violations = {}
        
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                invalid_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if invalid_count > 0:
                    all_valid = False
                    violations[col] = invalid_count
        
        result = {
            "check": "value_ranges",
            "passed": all_valid,
            "violations": violations
        }
        self.validation_results.append(result)
        
        if not result["passed"]:
            logger.warning(f"Range validation violations: {violations}")
        
        return result["passed"]

    def validate_unique(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Validate uniqueness constraint on columns."""
        duplicate_count = df.duplicated(subset=columns).sum()
        
        result = {
            "check": "uniqueness",
            "passed": duplicate_count == 0,
            "duplicate_count": duplicate_count
        }
        self.validation_results.append(result)
        
        if not result["passed"]:
            logger.warning(f"Found {duplicate_count} duplicate rows")
        
        return result["passed"]

    def get_validation_report(self) -> Dict[str, Any]:
        """Get summary of all validation checks."""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r["passed"])
        
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "pass_rate": passed_checks / total_checks if total_checks > 0 else 0,
            "details": self.validation_results
        }
