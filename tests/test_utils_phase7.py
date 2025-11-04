"""Unit tests for Phase 7 Utils module."""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
import os
import logging

from src.utils import (
    ConfigManager,
    LoggerSetup,
    setup_logging,
    get_logger,
    FileUtils,
    MATLABBridge,
    VisualizationUtils,
    MathUtils,
    set_seed,
    enable_deterministic,
    record_versions,
    get_versions,
    compute_data_hash,
    verify_data_integrity,
)


class TestConfigManager:
    """Tests for ConfigManager."""
    
    def test_load_config(self):
        """Test loading configuration from file."""
        config = ConfigManager("config.yaml")
        assert config.config is not None
        assert config.get("data.sample_rate") is not None
    
    def test_get_nested_key(self):
        """Test getting nested configuration values."""
        config = ConfigManager("config.yaml")
        assert config.get("data.raw_data_path") == "data/raw"
        assert config.get("training.batch_size") == 32
    
    def test_set_value(self):
        """Test setting configuration values."""
        config = ConfigManager("config.yaml")
        config.set("test.key", "test_value")
        assert config.get("test.key") == "test_value"
    
    def test_get_with_default(self):
        """Test getting value with default."""
        config = ConfigManager("config.yaml")
        result = config.get("nonexistent.key", default="default_value")
        assert result == "default_value"
    
    def test_get_all(self):
        """Test getting entire configuration."""
        config = ConfigManager("config.yaml")
        all_config = config.get_all()
        assert isinstance(all_config, dict)
        assert len(all_config) > 0
    
    def test_validate_success(self):
        """Test successful validation."""
        config = ConfigManager("config.yaml")
        assert config.validate(["data.raw_data_path"])
    
    def test_validate_failure(self):
        """Test validation with missing key."""
        config = ConfigManager("config.yaml")
        with pytest.raises(ValueError):
            config.validate(["nonexistent.key"])
    
    def test_merge_config(self):
        """Test merging configurations."""
        config = ConfigManager("config.yaml")
        other = {"new_section": {"key": "value"}}
        config.merge(other)
        assert config.get("new_section.key") == "value"
    
    def test_save_config(self):
        """Test saving configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ConfigManager("config.yaml")
            save_path = Path(tmpdir) / "config_save.yaml"
            config.save(str(save_path))
            assert save_path.exists()
    
    def test_override_from_env(self):
        """Test environment variable override."""
        config = ConfigManager("config.yaml")
        os.environ["ALD_TEST__VALUE"] = "env_value"
        config.override_from_env(prefix="ALD_")
        assert config.get("test.value") == "env_value"
        del os.environ["ALD_TEST__VALUE"]


class TestLoggerSetup:
    """Tests for LoggerSetup."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(log_dir=tmpdir, console_level=logging.WARNING)
            assert logger is not None
            assert isinstance(logger, logging.Logger)
    
    def test_get_logger(self):
        """Test getting logger for module."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"
    
    def test_setup_json_format(self):
        """Test JSON format logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(log_dir=tmpdir, json_format=True)
            assert logger is not None
    
    def test_module_levels(self):
        """Test setting module-specific log levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            module_levels = {"module_a": logging.DEBUG, "module_b": logging.ERROR}
            logger = setup_logging(log_dir=tmpdir, module_levels=module_levels)
            assert logger is not None


class TestFileUtils:
    """Tests for FileUtils."""
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "test" / "nested" / "dir"
            result = FileUtils.ensure_directory(test_dir)
            assert test_dir.exists()
            assert result == test_dir
    
    def test_save_and_load_pickle(self):
        """Test saving and loading pickle files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"test": "data", "array": np.array([1, 2, 3])}
            file_path = Path(tmpdir) / "test.pkl"
            
            FileUtils.safe_save_file(data, file_path, format="pickle")
            loaded = FileUtils.safe_load_file(file_path, format="pickle")
            
            assert loaded["test"] == data["test"]
            assert np.array_equal(loaded["array"], data["array"])
    
    def test_save_and_load_json(self):
        """Test saving and loading JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"test": "data", "number": 42}
            file_path = Path(tmpdir) / "test.json"
            
            FileUtils.safe_save_file(data, file_path, format="json")
            loaded = FileUtils.safe_load_file(file_path, format="json")
            
            assert loaded == data
    
    def test_safe_save_with_backup(self):
        """Test safe save with backup creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.json"
            
            # First save
            FileUtils.safe_save_file({"v": 1}, file_path, format="json", backup=True)
            assert file_path.exists()
            
            # Second save with backup
            FileUtils.safe_save_file({"v": 2}, file_path, format="json", backup=True)
            assert file_path.exists()
            backup_path = file_path.with_stem(file_path.stem + ".backup")
            assert backup_path.exists()
    
    def test_get_file_size(self):
        """Test getting file size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test content")
            size = FileUtils.get_file_size(file_path)
            assert size > 0
    
    def test_human_readable_size(self):
        """Test human readable size conversion."""
        assert "B" in FileUtils.get_human_readable_size(100)
        assert "KB" in FileUtils.get_human_readable_size(1024 * 100)
        assert "MB" in FileUtils.get_human_readable_size(1024 * 1024 * 10)
    
    def test_list_files(self):
        """Test listing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir).joinpath("test1.csv").touch()
            Path(tmpdir).joinpath("test2.csv").touch()
            Path(tmpdir).joinpath("test.txt").touch()
            
            csv_files = FileUtils.list_files(tmpdir, pattern="*.csv")
            assert len(csv_files) == 2
    
    def test_remove_file(self):
        """Test file removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("test")
            assert file_path.exists()
            
            result = FileUtils.remove_file(file_path)
            assert result is True
            assert not file_path.exists()
    
    def test_copy_file(self):
        """Test file copying."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src.txt"
            dst = Path(tmpdir) / "subdir" / "dst.txt"
            
            src.write_text("test content")
            FileUtils.copy_file(src, dst)
            
            assert dst.exists()
            assert dst.read_text() == "test content"
    
    def test_get_absolute_path(self):
        """Test absolute path conversion."""
        abs_path = FileUtils.get_absolute_path("./config.yaml")
        assert abs_path.is_absolute()
    
    def test_get_relative_path(self):
        """Test relative path conversion."""
        start = Path("/home/user/project")
        path = Path("/home/user/project/data/file.csv")
        rel_path = FileUtils.get_relative_path(path, start)
        assert str(rel_path) == "data/file.csv"


class TestMATLABBridge:
    """Tests for MATLABBridge."""
    
    def test_convert_to_numpy(self):
        """Test converting MATLAB data to NumPy."""
        mat_data = {
            "array": np.array([[1, 2], [3, 4]]),
            "value": 42,
            "__metadata": "should_be_removed"
        }
        
        converted = MATLABBridge.convert_to_numpy(mat_data, remove_metadata=True)
        assert "array" in converted
        assert "value" in converted
        assert "__metadata" not in converted
    
    def test_extract_fft(self):
        """Test extracting FFT data."""
        mat_data = {
            "fft": np.random.randn(512, 10),
        }
        
        fft_data = MATLABBridge.extract_fft(mat_data)
        assert fft_data is not None
        assert fft_data.shape[1] == 512  # After transpose: (10, 512)
    
    def test_extract_labels(self):
        """Test extracting labels."""
        mat_data = {
            "labels": np.array([[0, 1, 2, 3]])
        }
        
        labels = MATLABBridge.extract_labels(mat_data)
        assert labels is not None
        assert len(labels) == 4
    
    def test_compare_fft(self):
        """Test FFT comparison."""
        fft1 = np.random.randn(512)
        fft2 = fft1 + 0.01 * np.random.randn(512)
        
        metrics = MATLABBridge.compare_fft_with_matlab(fft1, fft2, return_correlation=True)
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "correlation" in metrics
        assert metrics["correlation"] > 0.9  # Should be highly correlated


class TestMathUtils:
    """Tests for MathUtils."""
    
    def test_normalize_minmax(self):
        """Test min-max normalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, data_min, data_max = MathUtils.normalize_minmax(data, 0, 1)
        
        assert np.min(normalized) >= 0
        assert np.max(normalized) <= 1
        assert data_min == 1
        assert data_max == 5
    
    def test_normalize_zscore(self):
        """Test z-score normalization."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        normalized, mean, std = MathUtils.normalize_zscore(data)
        
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1) < 1e-10
    
    def test_normalize_robust(self):
        """Test robust normalization."""
        data = np.array([1, 2, 3, 4, 5, 100], dtype=float)  # 100 is outlier
        normalized, median, iqr = MathUtils.normalize_robust(data)
        
        assert iqr > 0
        assert len(normalized) == len(data)
    
    def test_apply_window(self):
        """Test window application."""
        data = np.ones(100)
        
        windowed = MathUtils.apply_window(data, "hanning")
        assert windowed.shape == data.shape
        assert windowed[0] <= 0.01  # Hann window should nearly zero at edges
        assert windowed[-1] <= 0.01
        assert windowed[50] > 0.9  # Center should be near 1
    
    def test_compute_fft(self):
        """Test FFT computation."""
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        magnitude, frequencies = MathUtils.compute_fft(data, fft_size=1024)
        
        assert len(magnitude) == 512
        assert len(frequencies) == 512
        assert np.max(magnitude) > 0
    
    def test_rms(self):
        """Test RMS calculation."""
        data = np.array([1, 2, 3, 4])
        rms_val = MathUtils.rms(data)
        expected = np.sqrt(np.mean(data ** 2))
        assert abs(rms_val - expected) < 1e-10
    
    def test_peak_to_peak(self):
        """Test peak-to-peak calculation."""
        data = np.array([-5, -2, 0, 3, 8])
        ptp = MathUtils.peak_to_peak(data)
        assert ptp == 13
    
    def test_crest_factor(self):
        """Test crest factor calculation."""
        data = np.array([1, 2, 3, 4, 5])
        cf = MathUtils.crest_factor(data)
        peak = 5
        rms_val = MathUtils.rms(data)
        expected = peak / rms_val
        assert abs(cf - expected) < 1e-10
    
    def test_spectral_centroid(self):
        """Test spectral centroid calculation."""
        magnitude = np.array([1, 2, 3, 2, 1])
        frequencies = np.array([0, 100, 200, 300, 400])
        
        centroid = MathUtils.spectral_centroid(magnitude, frequencies)
        assert 0 <= centroid <= 400
    
    def test_zero_crossing_rate(self):
        """Test zero crossing rate."""
        # Signal with 10 zero crossings
        data = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        zcr = MathUtils.zero_crossing_rate(data)
        assert zcr > 0
    
    def test_kurtosis(self):
        """Test kurtosis calculation."""
        data = np.random.randn(1000)
        kurt = MathUtils.kurtosis(data)
        assert isinstance(kurt, float)
    
    def test_skewness(self):
        """Test skewness calculation."""
        data = np.random.randn(1000)
        skew = MathUtils.skewness(data)
        assert isinstance(skew, float)
    
    def test_compute_band_power(self):
        """Test band power calculation."""
        magnitude = np.random.randn(100)
        frequencies = np.linspace(0, 2000, 100)
        
        power = MathUtils.compute_band_power(magnitude, frequencies, (100, 500))
        assert power >= 0
    
    def test_correlation(self):
        """Test correlation calculation."""
        data1 = np.random.randn(100)
        data2 = data1 + 0.01 * np.random.randn(100)
        
        corr = MathUtils.correlation(data1, data2)
        assert -1 <= corr <= 1
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        data1 = np.array([1, 2, 3, 4])
        data2 = np.array([1.1, 2.1, 3.1, 4.1])
        
        mse = MathUtils.mean_squared_error(data1, data2)
        expected = np.mean((data1 - data2) ** 2)
        assert abs(mse - expected) < 1e-10


class TestVisualizationUtils:
    """Tests for VisualizationUtils."""
    
    @pytest.mark.skipif(not VisualizationUtils._matplotlib_available, 
                       reason="matplotlib not available")
    def test_plot_fft(self):
        """Test FFT plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fft_data = np.random.randn(512)
            save_path = Path(tmpdir) / "fft.png"
            
            VisualizationUtils.plot_fft(fft_data, save_path=str(save_path))
            assert save_path.exists()
    
    @pytest.mark.skipif(not VisualizationUtils._matplotlib_available, 
                       reason="matplotlib not available")
    def test_plot_time_series(self):
        """Test time series plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.randn(1000)
            save_path = Path(tmpdir) / "timeseries.png"
            
            VisualizationUtils.plot_time_series(data, save_path=str(save_path))
            assert save_path.exists()
    
    @pytest.mark.skipif(not VisualizationUtils._matplotlib_available, 
                       reason="matplotlib not available")
    def test_plot_histogram(self):
        """Test histogram plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.randn(1000)
            save_path = Path(tmpdir) / "histogram.png"
            
            VisualizationUtils.plot_histogram(data, save_path=str(save_path))
            assert save_path.exists()
    
    @pytest.mark.skipif(not VisualizationUtils._matplotlib_available, 
                       reason="matplotlib not available")
    def test_plot_comparison(self):
        """Test comparison plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data1 = np.random.randn(100)
            data2 = np.random.randn(100)
            save_path = Path(tmpdir) / "comparison.png"
            
            VisualizationUtils.plot_comparison(data1, data2, save_path=str(save_path))
            assert save_path.exists()


class TestReproducibility:
    """Tests for reproducibility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        rand1 = np.random.randn(10)
        
        set_seed(42)
        rand2 = np.random.randn(10)
        
        assert np.allclose(rand1, rand2)
    
    def test_enable_deterministic(self):
        """Test enabling deterministic mode."""
        enable_deterministic()  # Should not raise error
        assert os.environ.get('CUDA_LAUNCH_BLOCKING') == '1'
    
    def test_record_versions(self):
        """Test recording library versions."""
        versions = record_versions()
        assert isinstance(versions, dict)
        assert 'python' in versions
        assert 'numpy' in versions
    
    def test_get_versions(self):
        """Test getting recorded versions."""
        record_versions()  # Ensure versions are recorded
        versions = get_versions()
        assert isinstance(versions, dict)
    
    def test_compute_data_hash(self):
        """Test data hashing."""
        data = np.array([1, 2, 3, 4, 5])
        hash1 = compute_data_hash(data, "sha256")
        hash2 = compute_data_hash(data, "sha256")
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest length
    
    def test_verify_data_integrity(self):
        """Test data integrity verification."""
        data = np.array([1, 2, 3, 4, 5])
        data_hash = compute_data_hash(data, "sha256")
        
        assert verify_data_integrity(data, data_hash, "sha256")
    
    def test_verify_data_integrity_failure(self):
        """Test data integrity verification failure."""
        data = np.array([1, 2, 3, 4, 5])
        wrong_hash = "0" * 64
        
        assert not verify_data_integrity(data, wrong_hash, "sha256")


class TestIntegration:
    """Integration tests for Utils module."""
    
    def test_full_pipeline(self):
        """Test complete utils pipeline."""
        # Setup logging
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(log_dir=tmpdir)
            
            # Create config
            config = ConfigManager("config.yaml")
            
            # Create test data
            data = np.random.randn(1000)
            
            # Apply preprocessing
            normalized, _, _ = MathUtils.normalize_zscore(data)
            
            # Compute hash
            data_hash = compute_data_hash(normalized)
            
            # Save data
            file_path = Path(tmpdir) / "test_data.pkl"
            FileUtils.safe_save_file(normalized, file_path)
            
            # Load data
            loaded = FileUtils.safe_load_file(file_path)
            
            # Verify
            assert np.allclose(normalized, loaded)
            assert verify_data_integrity(loaded, data_hash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])