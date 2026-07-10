"""LSTM model for Remaining Useful Life (RUL) prediction."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, 
        Bidirectional, Attention, Concatenate,
        BatchNormalization, Flatten
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will not work.")


class RULPredictor:
    """LSTM-based Remaining Useful Life predictor."""

    def __init__(self, sequence_length: int = 10, n_features: int = 10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        self.config = {
            "lstm_units": [64, 32],
            "dense_units": [32, 16],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }

    def build_model(self, model_type: str = "lstm") -> Any:
        """Build LSTM model architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")
        
        if model_type == "lstm":
            self.model = self._build_simple_lstm()
        elif model_type == "bidirectional":
            self.model = self._build_bidirectional_lstm()
        elif model_type == "attention":
            self.model = self._build_lstm_with_attention()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.model

    def _build_simple_lstm(self) -> Any:
        """Build simple LSTM model."""
        model = Sequential([
            LSTM(
                self.config["lstm_units"][0],
                input_shape=(self.sequence_length, self.n_features),
                return_sequences=True
            ),
            Dropout(self.config["dropout_rate"]),
            LSTM(
                self.config["lstm_units"][1],
                return_sequences=False
            ),
            Dropout(self.config["dropout_rate"]),
            Dense(self.config["dense_units"][0], activation="relu"),
            BatchNormalization(),
            Dense(self.config["dense_units"][1], activation="relu"),
            Dense(1, activation="linear")
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("Built simple LSTM model")
        return model

    def _build_bidirectional_lstm(self) -> Any:
        """Build Bidirectional LSTM model."""
        model = Sequential([
            Bidirectional(
                LSTM(self.config["lstm_units"][0], return_sequences=True),
                input_shape=(self.sequence_length, self.n_features)
            ),
            Dropout(self.config["dropout_rate"]),
            Bidirectional(LSTM(self.config["lstm_units"][1])),
            Dropout(self.config["dropout_rate"]),
            Dense(self.config["dense_units"][0], activation="relu"),
            BatchNormalization(),
            Dense(self.config["dense_units"][1], activation="relu"),
            Dense(1, activation="linear")
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("Built Bidirectional LSTM model")
        return model

    def _build_lstm_with_attention(self) -> Any:
        """Build LSTM with Attention mechanism."""
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        lstm_out = LSTM(
            self.config["lstm_units"][0],
            return_sequences=True
        )(inputs)
        lstm_out = Dropout(self.config["dropout_rate"])(lstm_out)
        
        lstm_out = LSTM(
            self.config["lstm_units"][1],
            return_sequences=True
        )(lstm_out)
        
        # Attention
        attention = Dense(1, activation="tanh")(lstm_out)
        attention = Flatten()(attention)
        attention = tf.keras.layers.Activation("softmax")(attention)
        attention = tf.keras.layers.RepeatVector(self.config["lstm_units"][1])(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        multiplied = tf.keras.layers.Multiply()([lstm_out, attention])
        context = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(multiplied)
        
        # Dense layers
        dense = Dense(self.config["dense_units"][0], activation="relu")(context)
        dense = Dropout(self.config["dropout_rate"])(dense)
        dense = Dense(self.config["dense_units"][1], activation="relu")(dense)
        output = Dense(1, activation="linear")(dense)
        
        model = Model(inputs=inputs, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("Built LSTM with Attention model")
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the RUL prediction model."""
        if self.model is None:
            self.build_model()
        
        # Update config with kwargs
        self.config.update(kwargs)
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.config["batch_size"],
            epochs=self.config["epochs"],
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL for input sequences."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def predict_with_confidence(self, X: np.ndarray, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict RUL with confidence intervals using MC Dropout."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Enable dropout for MC Dropout
        predictions = []
        
        for _ in range(n_iterations):
            pred = self.model(X, training=True)  # Enable dropout
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()
        
        # Calculate confidence intervals
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower_bound, upper_bound

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test))
        
        # R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # MAPE
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape)
        }
        
        logger.info(f"RUL Model Evaluation: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
        return metrics

    def save_model(self, filepath: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        
        # Save config
        import json
        config_path = filepath.replace(".keras", "_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load model")
        
        self.model = tf.keras.models.load_model(filepath)
        
        # Load config
        import json
        config_path = filepath.replace(".keras", "_config.json")
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found, using default config")
        
        logger.info(f"Model loaded from {filepath}")

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "No model built yet"
        
        return self.model.summary()


class TransformerRULPredictor:
    """Transformer-based RUL predictor (alternative to LSTM)."""

    def __init__(self, sequence_length: int = 10, n_features: int = 10, 
                 d_model: int = 64, n_heads: int = 4, num_layers: int = 2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.model = None

    def build_model(self) -> Any:
        """Build Transformer model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Transformer model")
        
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # Project to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embedding = tf.keras.layers.Embedding(
            input_dim=self.sequence_length, output_dim=self.d_model
        )(positions)
        x = x + position_embedding
        
        # Transformer blocks
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model
            )(x, x)
            attention_output = Dropout(0.1)(attention_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = Dense(self.d_model * 4, activation="relu")(x)
            ffn = Dense(self.d_model)(ffn)
            ffn = Dropout(0.1)(ffn)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global pooling and output
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation="linear")(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        
        logger.info("Built Transformer RUL model")
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the Transformer model."""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X, verbose=0).flatten()
