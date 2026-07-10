"""MLflow experiment tracking module."""
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Track experiments and model artifacts using MLflow."""

    def __init__(self, tracking_uri: str = "http://localhost:5000",
                 experiment_name: str = "predictive-maintenance"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.current_run = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None, 
                  tags: Optional[Dict[str, str]] = None):
        """Start a new MLflow run."""
        self.current_run = mlflow.start_run(
            run_name=run_name,
            tags=tags or {}
        )
        logger.info(f"Started MLflow run: {self.current_run.info.run_name}")
        return self.current_run

    def end_run(self, status: str = "completed"):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.current_run.info.run_id}")
            self.current_run = None

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to the current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, model_name: str, 
                  artifact_path: str = "models",
                  **kwargs):
        """Log a model artifact."""
        if hasattr(model, "model") and hasattr(model.model, "save"):
            # TensorFlow/Keras model
            mlflow.tensorflow.log_model(
                model.model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
                **kwargs
            )
        elif hasattr(model, "best_model"):
            # Sklearn-style model
            mlflow.sklearn.log_model(
                model.best_model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
                **kwargs
            )
        else:
            # Try logging as generic model
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
                **kwargs
            )
        
        logger.info(f"Logged model: {model_name}")

    def log_artifact(self, local_path: str, artifact_path: str = ""):
        """Log a local file as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: Dict, filename: str, artifact_path: str = ""):
        """Log a dictionary as a JSON artifact."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(dictionary, f, indent=2, default=str)
            temp_path = f.name
        
        try:
            mlflow.log_artifact(temp_path, artifact_path)
        finally:
            os.unlink(temp_path)

    def log_rul_experiment(self, model, X_train, y_train, X_val, y_val,
                           params: Dict, metrics: Dict):
        """Log a complete RUL prediction experiment."""
        with self.start_run(run_name=f"RUL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            self.log_params({
                "model_type": "rul_lstm",
                "sequence_length": X_train.shape[1],
                "n_features": X_train.shape[2],
                **params
            })
            
            # Log training metrics
            self.log_metrics({
                "train_samples": len(X_train),
                "val_samples": len(X_val)
            })
            
            # Log validation metrics
            self.log_metrics(metrics, step=0)
            
            # Log model
            self.log_model(model, "rul_predictor")
            
            # Log data shape as artifact
            self.log_dict({
                "train_shape": X_train.shape,
                "val_shape": X_val.shape,
                "metrics": metrics
            }, "experiment_info.json")

    def log_anomaly_experiment(self, model, X_train, X_val,
                               params: Dict, metrics: Dict):
        """Log a complete anomaly detection experiment."""
        with self.start_run(run_name=f"Anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            self.log_params({
                "model_type": "anomaly_detector",
                "n_features": X_train.shape[1] if len(X_train.shape) > 1 else X_train.shape[-1],
                **params
            })
            
            # Log metrics
            self.log_metrics(metrics, step=0)
            
            # Log model
            self.log_model(model, "anomaly_detector")
            
            # Log experiment info
            self.log_dict({
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "metrics": metrics
            }, "experiment_info.json")

    def log_classification_experiment(self, model, X_train, y_train,
                                     X_val, y_val, params: Dict, metrics: Dict):
        """Log a complete classification experiment."""
        with self.start_run(run_name=f"Classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            self.log_params({
                "model_type": "failure_classifier",
                "n_features": X_train.shape[1],
                "n_classes": len(set(y_train)),
                **params
            })
            
            # Log metrics
            self.log_metrics(metrics, step=0)
            
            # Log model
            self.log_model(model, "failure_classifier")
            
            # Log classification report as artifact
            if "classification_report" in metrics:
                self.log_dict(metrics["classification_report"], "classification_report.json")

    def search_runs(self, experiment_name: Optional[str] = None,
                    filter_string: str = "", max_results: int = 10) -> List[Dict]:
        """Search for runs in the experiment."""
        exp_name = experiment_name or self.experiment_name
        
        runs = mlflow.search_runs(
            experiment_names=[exp_name],
            filter_string=filter_string,
            max_results=max_results
        )
        
        return runs.to_dict("records") if len(runs) > 0 else []

    def get_best_run(self, metric: str, experiment_name: Optional[str] = None,
                     ascending: bool = True) -> Optional[Dict]:
        """Get the best run based on a metric."""
        runs = self.search_runs(experiment_name)
        
        if not runs:
            return None
        
        # Sort by metric
        runs_with_metric = [r for r in runs if metric in r.get("metrics", {})]
        
        if not runs_with_metric:
            return None
        
        sorted_runs = sorted(
            runs_with_metric,
            key=lambda x: x["metrics"][metric],
            reverse=not ascending
        )
        
        return sorted_runs[0]

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple runs."""
        import pandas as pd
        
        runs_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            runs_data.append({
                "run_id": run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                **run.data.params,
                **run.data.metrics
            })
        
        return pd.DataFrame(runs_data)

    def register_model(self, model_uri: str, model_name: str):
        """Register a model in the MLflow Model Registry."""
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model: {model_name}, version: {result.version}")
        return result

    def transition_model_version(self, model_name: str, version: str,
                                 stage: str):
        """Transition a model version to a new stage."""
        mlflow.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")


class ExperimentLogger:
    """Simple experiment logger for when MLflow is not available."""

    def __init__(self, log_dir: str = "./experiments"):
        self.log_dir = log_dir
        self.experiments: List[Dict] = []
        self.current_experiment = None

    def start_experiment(self, name: str, params: Dict):
        """Start a new experiment."""
        self.current_experiment = {
            "name": name,
            "start_time": datetime.now().isoformat(),
            "params": params,
            "metrics": {},
            "artifacts": []
        }

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        if self.current_experiment:
            if key not in self.current_experiment["metrics"]:
                self.current_experiment["metrics"][key] = []
            
            entry = {"value": value, "timestamp": datetime.now().isoformat()}
            if step is not None:
                entry["step"] = step
            
            self.current_experiment["metrics"][key].append(entry)

    def log_artifact(self, filepath: str):
        """Log an artifact."""
        if self.current_experiment:
            self.current_experiment["artifacts"].append(filepath)

    def end_experiment(self, status: str = "completed"):
        """End the current experiment."""
        if self.current_experiment:
            self.current_experiment["end_time"] = datetime.now().isoformat()
            self.current_experiment["status"] = status
            self.experiments.append(self.current_experiment)
            
            # Save to file
            import os
            os.makedirs(self.log_dir, exist_ok=True)
            
            filename = f"{self.current_experiment['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.log_dir, filename)
            
            with open(filepath, "w") as f:
                json.dump(self.current_experiment, f, indent=2)
            
            self.current_experiment = None

    def get_experiments(self) -> List[Dict]:
        """Get all logged experiments."""
        return self.experiments
