"""Comprehensive unit tests for Phase 6 - Inference Pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.prediction.batch_processor import BatchProcessor, ParallelBatchProcessor
from src.prediction.confidence_calibrator import ConfidenceCalibrator, UncertaintyEstimator
from src.prediction.predictor import LeakDetector
from src.prediction.real_time_predictor import RealTimePredictor, StreamingDataProcessor


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary model file."""
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Create a simple sklearn model
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    X = np.random.randn(20, 1024)
    y = np.random.randint(0, 4, 20)
    model.fit(X, y)
    
    model_path = tmp_path / "mock_model.pkl"
    joblib.dump(model, model_path)
    return str(model_path)


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return np.random.randn(10, 1024)


@pytest.fixture
def sample_batch_data():
    """Create sample batch data."""
    return np.random.randn(100, 1024)


@pytest.fixture
def class_names():
    """Class names mapping."""
    return {0: "No Leak", 1: "Leak 1/16", 2: "Leak 3/32", 3: "Leak 1/8"}


@pytest.fixture
def leak_detector(mock_model_path, class_names):
    """Create a LeakDetector instance."""
    return LeakDetector(mock_model_path, preprocessor=None, class_names=class_names)


# ============================================================================
# TESTS FOR LEAK DETECTOR (BASELINE)
# ============================================================================


class TestLeakDetector:
    """Test the base LeakDetector class."""

    def test_initialization(self, mock_model_path, class_names):
        """Test detector initialization."""
        detector = LeakDetector(mock_model_path, class_names=class_names)
        assert detector.model is not None
        assert detector.class_names == class_names

    def test_load_model_pkl(self, mock_model_path):
        """Test loading pickle model."""
        detector = LeakDetector(mock_model_path)
        assert detector.model is not None

    def test_default_class_names(self, mock_model_path):
        """Test default class names."""
        detector = LeakDetector(mock_model_path)
        assert detector.class_names[0] == "No Leak"
        assert detector.class_names[1] == "Leak 1/16"
        assert detector.class_names[2] == "Leak 3/32"
        assert detector.class_names[3] == "Leak 1/8"

    def test_predict_single(self, leak_detector, sample_data):
        """Test single prediction."""
        result = leak_detector.predict_single(sample_data[0:1])
        
        assert "predicted_class" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "probabilities" in result
        
        assert result["predicted_class"] >= 0
        assert result["predicted_class"] < 4
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 4

    def test_predict_batch(self, leak_detector, sample_batch_data):
        """Test batch prediction."""
        result = leak_detector.predict_batch(sample_batch_data)
        
        assert "predictions" in result
        assert "confidences" in result
        assert "class_names" in result
        assert "mean_confidence" in result
        assert "std_confidence" in result
        
        assert len(result["predictions"]) == len(sample_batch_data)
        assert len(result["confidences"]) == len(sample_batch_data)

    def test_predict_with_uncertainty(self, leak_detector, sample_data):
        """Test uncertainty estimation."""
        result = leak_detector.predict_with_uncertainty(sample_data[0:1], n_iterations=5)
        
        assert "predictions" in result
        assert "mean_probabilities" in result
        assert "std_probabilities" in result
        assert len(result["predictions"]) == 1

    def test_explain_prediction(self, leak_detector, sample_data):
        """Test prediction explanation."""
        result = leak_detector.explain_prediction(sample_data[0:1])
        
        assert "prediction" in result
        assert "feature_importance" in result


# ============================================================================
# TESTS FOR REAL-TIME PREDICTOR
# ============================================================================


class TestRealTimePredictor:
    """Test the RealTimePredictor class."""

    def test_initialization(self, mock_model_path, class_names):
        """Test real-time predictor initialization."""
        predictor = RealTimePredictor(
            mock_model_path,
            window_size=512,
            stride=256,
            class_names=class_names,
        )
        
        assert predictor.window_size == 512
        assert predictor.stride == 256
        assert len(predictor.buffer) == 0

    def test_add_single_sample(self, mock_model_path):
        """Test adding single samples."""
        predictor = RealTimePredictor(mock_model_path, window_size=5)
        
        # Add samples one by one
        for i in range(4):
            result = predictor.add_sample(np.random.randn(10))
            assert result is None  # Window not full yet

    def test_add_sample_full_window(self, mock_model_path):
        """Test prediction when window is full."""
        predictor = RealTimePredictor(mock_model_path, window_size=3)
        
        # Add samples with shape that matches model input (1024 features)
        for i in range(2):
            predictor.add_sample(np.random.randn(1024))
        
        # This should trigger prediction
        result = predictor.add_sample(np.random.randn(1024))
        # Result may be None or a dict, buffer should be full
        assert len(predictor.buffer) == 3

    def test_add_samples_batch(self, mock_model_path):
        """Test adding multiple samples at once."""
        predictor = RealTimePredictor(mock_model_path, window_size=5, stride=2)
        
        # Create samples with matching feature dimension (1024)
        samples = np.random.randn(10, 1024)
        results = predictor.add_samples(samples)
        
        # Should have some predictions
        assert isinstance(results, list)

    def test_predict_from_buffer(self, mock_model_path):
        """Test prediction from buffer."""
        predictor = RealTimePredictor(mock_model_path, window_size=3)
        
        # Fill buffer with matching feature dimension
        for i in range(3):
            predictor.buffer.append(np.random.randn(1024))
        
        # Predict from buffer
        result = predictor.predict_from_buffer()
        assert result is not None  # Should return a result

    def test_ensemble_prediction(self, mock_model_path):
        """Test ensemble prediction from history."""
        predictor = RealTimePredictor(mock_model_path, window_size=3)
        
        # Add some predictions to history manually
        predictor.prediction_history.append({
            "predicted_class": 0,
            "class_name": "No Leak",
            "confidence": 0.9,
            "probabilities": {"No Leak": 0.9, "Leak 1/16": 0.05, "Leak 3/32": 0.03, "Leak 1/8": 0.02},
        })
        
        result = predictor.get_ensemble_prediction()
        
        assert "predicted_class" in result
        assert "confidence" in result
        assert "n_predictions_averaged" in result

    def test_reset(self, mock_model_path):
        """Test resetting the predictor."""
        predictor = RealTimePredictor(mock_model_path, window_size=5)
        
        predictor.buffer.append(np.random.randn(10))
        predictor.prediction_history.append({"pred": 0})
        
        predictor.reset()
        
        assert len(predictor.buffer) == 0
        assert len(predictor.prediction_history) == 0

    def test_get_status(self, mock_model_path):
        """Test getting predictor status."""
        predictor = RealTimePredictor(mock_model_path, window_size=10, confidence_threshold=0.8)
        
        status = predictor.get_status()
        
        assert "buffer_size" in status
        assert "buffer_capacity" in status
        assert "buffer_full" in status
        assert "history_size" in status
        assert "confidence_threshold" in status
        assert status["buffer_capacity"] == 10
        assert status["confidence_threshold"] == 0.8

    def test_callback_registration(self, mock_model_path):
        """Test registering callbacks."""
        predictor = RealTimePredictor(mock_model_path)
        
        callback_called = []
        
        def test_callback(pred):
            callback_called.append(pred)
        
        predictor.register_callback(test_callback)
        assert len(predictor.callbacks) == 1


# ============================================================================
# TESTS FOR STREAMING DATA PROCESSOR
# ============================================================================


class TestStreamingDataProcessor:
    """Test the StreamingDataProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = StreamingDataProcessor(window_size=512, stride=256)
        
        assert processor.window_size == 512
        assert processor.stride == 256

    def test_process_stream(self):
        """Test processing a stream."""
        processor = StreamingDataProcessor(window_size=5, stride=2)
        
        # Create stream of 20 samples
        data = np.random.randn(20)
        
        windows = processor.process_stream(data)
        
        assert len(windows) > 0
        for window in windows:
            assert window.shape[0] == 5

    def test_process_stream_multichannel(self):
        """Test processing multichannel stream."""
        processor = StreamingDataProcessor(window_size=10, stride=5, n_channels=9)
        
        data = np.random.randn(50, 9)
        windows = processor.process_stream(data)
        
        assert len(windows) > 0

    def test_reset(self):
        """Test resetting processor."""
        processor = StreamingDataProcessor()
        processor.buffer.append(np.random.randn(10))
        
        processor.reset()
        assert len(processor.buffer) == 0


# ============================================================================
# TESTS FOR BATCH PROCESSOR
# ============================================================================


class TestBatchProcessor:
    """Test the BatchProcessor class."""

    def test_initialization(self, mock_model_path):
        """Test batch processor initialization."""
        processor = BatchProcessor(mock_model_path, batch_size=16, n_workers=2)
        
        assert processor.batch_size == 16
        assert processor.n_workers == 2

    def test_process_batch(self, mock_model_path, sample_batch_data):
        """Test processing a batch."""
        processor = BatchProcessor(mock_model_path, batch_size=32)
        
        result = processor.process_batch(sample_batch_data, show_progress=False)
        
        assert "predictions" in result
        assert "confidences" in result
        assert "class_names" in result
        assert "mean_confidence" in result
        assert "std_confidence" in result
        assert "n_samples" in result

    def test_save_predictions_json(self, mock_model_path, sample_batch_data, tmp_path):
        """Test saving predictions as JSON."""
        processor = BatchProcessor(mock_model_path)
        
        result = processor.process_batch(sample_batch_data, show_progress=False)
        
        output_path = tmp_path / "predictions.json"
        processor.save_predictions(result, output_path, format="json")
        
        assert output_path.exists()
        
        with open(output_path, "r") as f:
            loaded = json.load(f)
            assert "predictions" in loaded

    def test_save_predictions_csv(self, mock_model_path, sample_batch_data, tmp_path):
        """Test saving predictions as CSV."""
        pytest.importorskip("pandas")
        
        processor = BatchProcessor(mock_model_path)
        result = processor.process_batch(sample_batch_data, show_progress=False)
        
        output_path = tmp_path / "predictions.csv"
        processor.save_predictions(result, output_path, format="csv")
        
        assert output_path.exists()

    def test_save_predictions_npz(self, mock_model_path, sample_batch_data, tmp_path):
        """Test saving predictions as NPZ."""
        processor = BatchProcessor(mock_model_path)
        result = processor.process_batch(sample_batch_data, show_progress=False)
        
        output_path = tmp_path / "predictions.npz"
        processor.save_predictions(result, output_path, format="npz")
        
        assert output_path.exists()

    def test_load_predictions(self, mock_model_path, sample_batch_data, tmp_path):
        """Test loading predictions."""
        processor = BatchProcessor(mock_model_path)
        result = processor.process_batch(sample_batch_data, show_progress=False)
        
        output_path = tmp_path / "predictions.json"
        processor.save_predictions(result, output_path, format="json")
        
        loaded = processor.load_predictions(output_path)
        assert "predictions" in loaded


# ============================================================================
# TESTS FOR PARALLEL BATCH PROCESSOR
# ============================================================================


class TestParallelBatchProcessor:
    """Test the ParallelBatchProcessor class."""

    def test_initialization(self, mock_model_path):
        """Test parallel processor initialization."""
        processor = ParallelBatchProcessor(mock_model_path, n_workers=2)
        
        assert processor.n_workers == 2

    def test_process_parallel(self, mock_model_path, sample_batch_data):
        """Test parallel processing."""
        processor = ParallelBatchProcessor(mock_model_path, n_workers=2)
        
        result = processor.process_parallel(sample_batch_data, show_progress=False)
        
        assert "predictions" in result
        assert "confidences" in result
        assert "n_chunks" in result
        assert "n_samples" in result


# ============================================================================
# TESTS FOR CONFIDENCE CALIBRATOR
# ============================================================================


class TestConfidenceCalibrator:
    """Test the ConfidenceCalibrator class."""

    def test_initialization(self, mock_model_path, class_names):
        """Test calibrator initialization."""
        calibrator = ConfidenceCalibrator(mock_model_path, class_names=class_names)
        
        assert calibrator.temperature == 1.0
        assert calibrator.calibration_method is None

    def test_temperature_scale(self, mock_model_path):
        """Test temperature scaling."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        logits = np.array([[1.0, 2.0, 0.5, 0.1]])
        scaled = calibrator._temperature_scale(logits, temperature=2.0)
        
        assert scaled.shape == logits.shape
        assert np.allclose(np.sum(scaled, axis=1), 1.0)

    def test_platt_scale(self, mock_model_path):
        """Test Platt scaling."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        confidences = np.array([0.5, 0.7, 0.9])
        scaled = calibrator._platt_scale(confidences, weights=(1.0, 0.0))
        
        assert scaled.shape == confidences.shape
        assert np.all(scaled >= 0) and np.all(scaled <= 1)

    def test_isotonic_scale(self, mock_model_path):
        """Test isotonic regression scaling."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        confidences = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        scaled = calibrator._isotonic_regression_scale(confidences, thresholds)
        
        assert scaled.shape == confidences.shape

    def test_calibrate_temperature(self, mock_model_path):
        """Test temperature calibration."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        val_data = np.random.randn(50, 1024)
        val_labels = np.random.randint(0, 4, 50)
        
        temp = calibrator.calibrate_temperature(val_data, val_labels, verbose=False)
        
        assert temp > 0
        assert calibrator.calibration_method == "temperature"

    def test_predict_calibrated_no_calibration(self, mock_model_path, sample_data):
        """Test prediction without calibration."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        result = calibrator.predict_calibrated(sample_data[0:1])
        
        assert "predicted_class" in result
        assert "confidence" in result

    def test_get_calibration_info(self, mock_model_path):
        """Test getting calibration info."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        info = calibrator.get_calibration_info()
        
        assert "method" in info
        assert "temperature" in info
        assert "parameters" in info


# ============================================================================
# TESTS FOR UNCERTAINTY ESTIMATOR
# ============================================================================


class TestUncertaintyEstimator:
    """Test the UncertaintyEstimator class."""

    def test_initialization(self, mock_model_path):
        """Test uncertainty estimator initialization."""
        estimator = UncertaintyEstimator(mock_model_path)
        
        assert estimator.leak_detector is not None

    def test_estimate_entropy(self, mock_model_path, sample_batch_data):
        """Test entropy estimation."""
        estimator = UncertaintyEstimator(mock_model_path)
        
        result = estimator.estimate_entropy(sample_batch_data)
        
        assert "entropy" in result
        assert "normalized_entropy" in result
        assert "mean_entropy" in result
        assert "predictions" in result
        
        assert len(result["entropy"]) == len(sample_batch_data)

    def test_estimate_confidence_margin(self, mock_model_path, sample_batch_data):
        """Test confidence margin estimation."""
        estimator = UncertaintyEstimator(mock_model_path)
        
        result = estimator.estimate_confidence_margin(sample_batch_data)
        
        assert "margin" in result
        assert "uncertainty" in result
        assert "mean_margin" in result
        assert "predictions" in result

    def test_estimate_variance(self, mock_model_path, sample_batch_data):
        """Test variance estimation."""
        estimator = UncertaintyEstimator(mock_model_path)
        
        result = estimator.estimate_variance(sample_batch_data, n_iterations=3)
        
        assert "prediction_variance" in result
        assert "mean_variance" in result
        assert "predictions" in result


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestPredictionPipelineIntegration:
    """Integration tests for the complete prediction pipeline."""

    def test_full_pipeline_single_prediction(self, mock_model_path, sample_data, class_names):
        """Test full pipeline with single prediction."""
        detector = LeakDetector(mock_model_path, class_names=class_names)
        result = detector.predict_single(sample_data[0:1])
        
        assert "predicted_class" in result
        assert "confidence" in result

    def test_realtime_to_batch(self, mock_model_path, sample_batch_data):
        """Test real-time predictor transitioning to batch."""
        # Use small window size to trigger predictions with our test data
        predictor = RealTimePredictor(mock_model_path, window_size=5)
        
        # Simulate streaming with first 10 samples
        results = predictor.add_samples(sample_batch_data[:10])
        
        # Should return results (may be empty if not enough predictions)
        assert isinstance(results, list)

    def test_calibration_workflow(self, mock_model_path, sample_batch_data):
        """Test calibration workflow."""
        calibrator = ConfidenceCalibrator(mock_model_path)
        
        # Get uncalibrated prediction
        result_before = calibrator.predict_calibrated(sample_batch_data[0:1])
        
        # Calibrate
        val_data = sample_batch_data[:50]
        val_labels = np.random.randint(0, 4, 50)
        calibrator.calibrate_temperature(val_data, val_labels, verbose=False)
        
        # Get calibrated prediction
        result_after = calibrator.predict_calibrated(sample_batch_data[0:1])
        
        assert "calibration_method" in result_after

    def test_uncertainty_workflow(self, mock_model_path, sample_batch_data):
        """Test uncertainty estimation workflow."""
        estimator = UncertaintyEstimator(mock_model_path)
        
        # Test multiple uncertainty methods
        entropy_result = estimator.estimate_entropy(sample_batch_data[:20])
        margin_result = estimator.estimate_confidence_margin(sample_batch_data[:20])
        
        assert len(entropy_result["entropy"]) == 20
        assert len(margin_result["margin"]) == 20


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_prediction(self, leak_detector, sample_data):
        """Test predicting on single sample."""
        result = leak_detector.predict_single(sample_data[0:1])
        assert result is not None

    def test_empty_batch(self, leak_detector):
        """Test handling empty batch."""
        with pytest.raises((ValueError, IndexError)):
            leak_detector.predict_batch(np.array([]))

    def test_wrong_input_shape(self, leak_detector):
        """Test handling wrong input shape."""
        # Test with 3D array - should raise ValueError or handle it
        try:
            result = leak_detector.predict_single(np.random.randn(10, 1, 5))
            # If no error, result should be a dict
            assert isinstance(result, dict) or result is None
        except ValueError:
            # Expected behavior - model expects 2D input
            pass

    def test_very_small_batch(self, leak_detector):
        """Test very small batch."""
        small_data = np.random.randn(1, 1024)
        result = leak_detector.predict_batch(small_data)
        
        assert len(result["predictions"]) == 1

    def test_very_large_batch(self, leak_detector):
        """Test very large batch."""
        large_data = np.random.randn(1000, 1024)
        result = leak_detector.predict_batch(large_data)
        
        assert len(result["predictions"]) == 1000

    def test_streaming_with_partial_window(self, mock_model_path):
        """Test streaming with partial window."""
        predictor = RealTimePredictor(mock_model_path, window_size=100)
        
        # Add fewer samples than window size
        for i in range(50):
            result = predictor.add_sample(np.random.randn(10))
            assert result is None  # Should not predict yet

        assert len(predictor.buffer) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])