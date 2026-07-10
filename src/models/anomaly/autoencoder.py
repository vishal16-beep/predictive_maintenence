"""Autoencoder-based anomaly detection model."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, BatchNormalization,
        Lambda
    )
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Autoencoder model will not work.")


class AnomalyDetector:
    """Autoencoder-based anomaly detection for sensor data."""

    def __init__(self, n_features: int = 10, encoding_dim: int = 32):
        self.n_features = n_features
        self.encoding_dim = encoding_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.mean_reconstruction_error = None
        self.std_reconstruction_error = None

    def build_model(self, hidden_layers: Optional[List[int]] = None) -> Model:
        """Build autoencoder model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder model")
        
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        # Encoder
        input_layer = Input(shape=(self.n_features,))
        
        x = input_layer
        for units in hidden_layers:
            x = Dense(units, activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        # Bottleneck
        encoded = Dense(self.encoding_dim, activation="relu", name="bottleneck")(x)
        
        self.encoder = Model(input_layer, encoded, name="encoder")
        
        # Decoder
        decoder_input = Input(shape=(self.encoding_dim,))
        x = decoder_input
        
        for units in reversed(hidden_layers):
            x = Dense(units, activation="relu")(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
        
        decoded = Dense(self.n_features, activation="linear")(x)
        
        self.decoder = Model(decoder_input, decoded, name="decoder")
        
        # Full autoencoder
        encoded_output = self.encoder(input_layer)
        decoded_output = self.decoder(encoded_output)
        
        self.model = Model(input_layer, decoded_output, name="autoencoder")
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse"
        )
        
        logger.info(f"Built Autoencoder: {self.n_features} -> {self.encoding_dim} -> {self.n_features}")
        return self.model

    def build_variational_autoencoder(self, latent_dim: int = 16) -> Model:
        """Build Variational Autoencoder (VAE) for better anomaly detection."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for VAE model")
        
        # Encoder
        inputs = Input(shape=(self.n_features,))
        x = Dense(64, activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Dense(32, activation="relu")(x)
        x = BatchNormalization()(x)
        
        # Mean and log variance
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        
        # Reparameterization trick
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.keras.backend.shape(z_mean)[0]
            dim = tf.keras.backend.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
        
        z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
        
        # Encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")
        
        # Decoder
        latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
        x = Dense(32, activation="relu")(latent_inputs)
        x = BatchNormalization()(x)
        x = Dense(64, activation="relu")(x)
        x = BatchNormalization()(x)
        outputs = Dense(self.n_features, activation="linear")(x)
        
        self.decoder = Model(latent_inputs, outputs, name="decoder")
        
        # Full VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.model = Model(inputs, outputs, name="vae")
        
        # VAE loss
        reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
        reconstruction_loss *= self.n_features
        
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
        kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        vae_loss = tf.keras.backend.mean(reconstruction_loss + kl_loss)
        self.model.add_loss(vae_loss)
        
        self.model.compile(optimizer=Adam(learning_rate=0.001))
        
        logger.info(f"Built VAE: {self.n_features} -> {latent_dim} -> {self.n_features}")
        return self.model

    def train(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the autoencoder on normal data."""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True
            )
        ]
        
        validation_data = (X_val, X_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, X_train,  # Autoencoder reconstructs input
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate threshold from training data
        self._calculate_threshold(X_train)
        
        return history.history

    def _calculate_threshold(self, X: np.ndarray, percentile: float = 95):
        """Calculate anomaly threshold from training data."""
        reconstructions = self.model.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        self.mean_reconstruction_error = np.mean(reconstruction_errors)
        self.std_reconstruction_error = np.std(reconstruction_errors)
        self.threshold = np.percentile(reconstruction_errors, percentile)
        
        logger.info(f"Anomaly threshold set to: {self.threshold:.4f}")
        logger.info(f"Mean reconstruction error: {self.mean_reconstruction_error:.4f}")
        logger.info(f"Std reconstruction error: {self.std_reconstruction_error:.4f}")

    def detect_anomalies(self, X: np.ndarray, threshold: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Detect anomalies in input data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if threshold is None:
            threshold = self.threshold
        
        # Get reconstructions
        reconstructions = self.model.predict(X, verbose=0)
        
        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Determine anomalies
        is_anomaly = reconstruction_errors > threshold
        
        # Calculate anomaly scores (normalized)
        anomaly_scores = reconstruction_errors / (self.threshold + 1e-8)
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Calculate per-feature anomaly contribution
        per_feature_errors = np.square(X - reconstructions)
        feature_importance = np.mean(per_feature_errors, axis=0)
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_scores": anomaly_scores,
            "reconstruction_errors": reconstruction_errors,
            "reconstructions": reconstructions,
            "per_feature_errors": per_feature_errors,
            "feature_importance": feature_importance
        }

    def get_anomaly_explanation(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Get explanation for detected anomalies."""
        results = self.detect_anomalies(X)
        
        explanations = []
        
        for i in range(len(X)):
            if results["is_anomaly"][i]:
                # Get top contributing features
                feature_errors = results["per_feature_errors"][i]
                top_features_idx = np.argsort(feature_errors)[::-1][:5]
                
                top_features = [
                    {
                        "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                        "error": float(feature_errors[idx]),
                        "value": float(X[i][idx])
                    }
                    for idx in top_features_idx
                ]
                
                explanations.append({
                    "sample_idx": i,
                    "anomaly_score": float(results["anomaly_scores"][i]),
                    "reconstruction_error": float(results["reconstruction_errors"][i]),
                    "top_contributing_features": top_features
                })
        
        return {
            "n_anomalies": int(np.sum(results["is_anomaly"])),
            "anomaly_rate": float(np.mean(results["is_anomaly"])),
            "explanations": explanations
        }

    def save_model(self, filepath: str):
        """Save autoencoder model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        
        # Save threshold and stats
        import pickle
        stats_path = filepath.replace(".keras", "_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump({
                "threshold": self.threshold,
                "mean_reconstruction_error": self.mean_reconstruction_error,
                "std_reconstruction_error": self.std_reconstruction_error,
                "n_features": self.n_features,
                "encoding_dim": self.encoding_dim
            }, f)
        
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load autoencoder model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load model")
        
        self.model = tf.keras.models.load_model(filepath)
        
        # Load threshold and stats
        import pickle
        stats_path = filepath.replace(".keras", "_stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
            self.threshold = stats["threshold"]
            self.mean_reconstruction_error = stats["mean_reconstruction_error"]
            self.std_reconstruction_error = stats["std_reconstruction_error"]
            self.n_features = stats["n_features"]
            self.encoding_dim = stats["encoding_dim"]
        
        logger.info(f"Model loaded from {filepath}")


class IsolationForestDetector:
    """Isolation Forest for anomaly detection (non-deep learning alternative)."""

    def __init__(self, n_estimators: int = 100, contamination: float = 0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = None
        self.scaler = None

    def train(self, X_train: np.ndarray):
        """Train Isolation Forest."""
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(X_scaled)
        
        logger.info("Isolation Forest trained")

    def detect_anomalies(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect anomalies using Isolation Forest."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (lower is more anomalous)
        scores = self.model.score_samples(X_scaled)
        
        is_anomaly = predictions == -1
        anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_scores": anomaly_scores,
            "raw_scores": scores
        }

    def save_model(self, filepath: str):
        """Save Isolation Forest model."""
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "n_estimators": self.n_estimators,
                "contamination": self.contamination
            }, f)

    def load_model(self, filepath: str):
        """Load Isolation Forest model."""
        import pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.n_estimators = data["n_estimators"]
            self.contamination = data["contamination"]
