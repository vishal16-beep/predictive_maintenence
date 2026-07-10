"""Test ML models."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import tempfile
import os

from src.models.classification.xgboost_model import FailureClassifier
from src.models.anomaly.autoencoder import IsolationForestDetector


class TestFailureClassifier:
    """Tests for FailureClassifier."""

    def test_build_models(self):
        classifier = FailureClassifier(n_classes=5)
        models = classifier.build_models()
        
        assert "random_forest" in models
        assert "gradient_boosting" in models
        assert "logistic_regression" in models

    def test_train_and_predict(self):
        classifier = FailureClassifier(n_classes=3)
        classifier.build_models()
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.choice(["engine", "brake", "electrical"], 100)
        
        X_val = np.random.randn(20, 10)
        y_val = np.random.choice(["engine", "brake", "electrical"], 20)
        
        results = classifier.train(X_train, y_train, X_val, y_val)
        
        assert "random_forest" in results
        assert classifier.best_model is not None
        
        # Test prediction
        predictions, probabilities = classifier.predict(X_val)
        
        assert len(predictions) == 20
        assert probabilities.shape[1] == 3

    def test_evaluate(self):
        classifier = FailureClassifier(n_classes=3)
        classifier.build_models()
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.choice(["engine", "brake", "electrical"], 100)
        
        X_test = np.random.randn(20, 10)
        y_test = np.random.choice(["engine", "brake", "electrical"], 20)
        
        classifier.train(X_train, y_train)
        
        metrics = classifier.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_save_and_load(self):
        classifier = FailureClassifier(n_classes=3)
        classifier.build_models()
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.choice(["engine", "brake", "electrical"], 100)
        
        classifier.train(X_train, y_train)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            classifier.save_model(temp_path)
            
            # Load
            new_classifier = FailureClassifier()
            new_classifier.load_model(temp_path)
            
            assert new_classifier.best_model is not None
            assert new_classifier.class_names == classifier.class_names
        finally:
            os.unlink(temp_path)


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def test_train_and_detect(self):
        detector = IsolationForestDetector()
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        
        detector.train(X_train)
        
        # Test detection
        X_test = np.random.randn(20, 10)
        results = detector.detect_anomalies(X_test)
        
        assert "is_anomaly" in results
        assert "anomaly_scores" in results
        assert len(results["is_anomaly"]) == 20

    def test_save_and_load(self):
        detector = IsolationForestDetector()
        
        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        
        detector.train(X_train)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            detector.save_model(temp_path)
            
            # Load
            new_detector = IsolationForestDetector()
            new_detector.load_model(temp_path)
            
            assert new_detector.model is not None
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
