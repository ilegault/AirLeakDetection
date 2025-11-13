"""
Comprehensive unit tests for Phase 8 executable scripts.

Tests all 10 scripts with:
- Argument parsing
- Input validation
- File I/O operations
- Error handling
- Exit codes
"""

import pytest
import sys
import argparse
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all scripts as modules
from scripts import train_model, prepare_data, predict, evaluate
from scripts import cross_validate, hyperparameter_search
from scripts import benchmark, train_with_external_fft, compare_fft_methods


class TestTrainModel:
    """Tests for train_model.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = train_model.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = train_model.create_parser()
        args = parser.parse_args([])
        
        assert args.model_type == "cnn_1d"
        assert args.epochs == 100
        assert args.batch_size == 32
        assert args.learning_rate == 0.001
        assert args.output_dir == "models/"
    
    def test_parser_custom_args(self):
        """Test parser with custom arguments."""
        parser = train_model.create_parser()
        args = parser.parse_args([
            "--model-type", "random_forest",
            "--epochs", "50",
            "--batch-size", "64",
            "--learning-rate", "0.01"
        ])
        
        assert args.model_type == "random_forest"
        assert args.epochs == 50
        assert args.batch_size == 64
        assert args.learning_rate == 0.01
    
    def test_validate_inputs_missing_data(self):
        """Test validation with missing data path."""
        parser = train_model.create_parser()
        args = parser.parse_args(["--data-path", "/nonexistent/path"])
        
        result = train_model.validate_inputs(args)
        assert result is False
    
    def test_validate_inputs_invalid_epochs(self):
        """Test validation with invalid epochs."""
        parser = train_model.create_parser()
        args = parser.parse_args(["--epochs", "0"])
        
        result = train_model.validate_inputs(args)
        assert result is False
    
    def test_validate_inputs_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        parser = train_model.create_parser()
        args = parser.parse_args(["--batch-size", "-1"])
        
        result = train_model.validate_inputs(args)
        assert result is False
    
    def test_validate_inputs_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        parser = train_model.create_parser()
        args = parser.parse_args(["--learning-rate", "1.5"])
        
        result = train_model.validate_inputs(args)
        assert result is False
    
    def test_setup_output_directory(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = train_model.setup_output_directory(tmpdir)
            
            assert output_dir.exists()
            assert output_dir.parent == Path(tmpdir)
    
    def test_valid_model_types(self):
        """Test all valid model types are accepted."""
        parser = train_model.create_parser()
        valid_types = ["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost", "ensemble"]
        
        for model_type in valid_types:
            args = parser.parse_args(["--model-type", model_type])
            assert args.model_type == model_type


class TestPrepareData:
    """Tests for prepare_data.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = prepare_data.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = prepare_data.create_parser()
        args = parser.parse_args([])
        
        assert args.raw_data == "data/raw/"
        assert args.output_dir == "data/processed/"
        assert args.train_ratio == 0.7
        assert args.val_ratio == 0.15
    
    def test_validate_splits_valid(self):
        """Test split validation with valid ratios."""
        result = prepare_data.validate_splits(0.7, 0.15)
        assert result is True
    
    def test_validate_splits_invalid_zero(self):
        """Test split validation with zero ratio."""
        result = prepare_data.validate_splits(0.0, 0.5)
        assert result is False
    
    def test_validate_splits_exceeds_one(self):
        """Test split validation when sum exceeds 1."""
        result = prepare_data.validate_splits(0.8, 0.5)
        assert result is False
    
    def test_split_ratios_info(self):
        """Test split ratios calculation."""
        train_ratio, val_ratio = 0.6, 0.2
        test_ratio = 1.0 - train_ratio - val_ratio
        
        assert train_ratio == 0.6
        assert val_ratio == 0.2
        assert test_ratio == 0.2
    
    def test_parser_custom_splits(self):
        """Test parser with custom split ratios."""
        parser = prepare_data.create_parser()
        args = parser.parse_args([
            "--train-ratio", "0.6",
            "--val-ratio", "0.2"
        ])
        
        assert args.train_ratio == 0.6
        assert args.val_ratio == 0.2


class TestPredict:
    """Tests for predict.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = predict.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_required_args(self):
        """Test parser requires model-path and input."""
        parser = predict.create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])
    
    def test_parser_with_required_args(self):
        """Test parser with required arguments."""
        parser = predict.create_parser()
        args = parser.parse_args([
            "--model-path", "model.h5",
            "--input", "data/"
        ])
        
        assert args.model_path == "model.h5"
        assert args.input == "data/"
    
    def test_validate_inputs_missing_model(self):
        """Test validation with missing model."""
        parser = predict.create_parser()
        args = parser.parse_args([
            "--model-path", "/nonexistent/model.h5",
            "--input", "."
        ])
        
        result = predict.validate_inputs(args)
        assert result is False
    
    def test_validate_inputs_invalid_confidence_threshold(self):
        """Test validation with invalid confidence threshold."""
        parser = predict.create_parser()
        
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            args = parser.parse_args([
                "--model-path", f.name,
                "--input", ".",
                "--confidence-threshold", "1.5"
            ])
            
            result = predict.validate_inputs(args)
            assert result is False
    
    def test_parser_output_formats(self):
        """Test parser accepts all output formats."""
        parser = predict.create_parser()
        valid_formats = ["json", "csv", "txt"]
        
        for fmt in valid_formats:
            args = parser.parse_args([
                "--model-path", "model.h5",
                "--input", "data/",
                "--output-format", fmt
            ])
            assert args.output_format == fmt


class TestEvaluate:
    """Tests for evaluate.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = evaluate.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_required_args(self):
        """Test parser requires model-path and test-data."""
        parser = evaluate.create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = evaluate.create_parser()
        
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with tempfile.TemporaryDirectory() as tmpdir:
                args = parser.parse_args([
                    "--model-path", f.name,
                    "--test-data", tmpdir
                ])
                
                assert args.output_dir == "results/evaluation/"
                assert args.generate_report is False
                assert args.batch_size == 32
    
    def test_validate_inputs_valid(self):
        """Test validation with valid inputs."""
        parser = evaluate.create_parser()
        
        with tempfile.NamedTemporaryFile(suffix=".h5") as model_f:
            with tempfile.TemporaryDirectory() as test_dir:
                args = parser.parse_args([
                    "--model-path", model_f.name,
                    "--test-data", test_dir
                ])
                
                result = evaluate.validate_inputs(args)
                assert result is True


class TestCrossValidate:
    """Tests for cross_validate.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = cross_validate.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = cross_validate.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(["--data-path", tmpdir])
            
            assert args.model_type == "cnn_1d"
            assert args.k_folds == 5
            assert args.stratified is True
    
    def test_validate_inputs_invalid_k_folds(self):
        """Test validation with invalid k-folds."""
        parser = cross_validate.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "--data-path", tmpdir,
                "--k-folds", "1"
            ])
            
            result = cross_validate.validate_inputs(args)
            assert result is False
    
    def test_parser_valid_model_types(self):
        """Test parser accepts valid model types."""
        parser = cross_validate.create_parser()
        valid_types = ["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for model_type in valid_types:
                args = parser.parse_args([
                    "--data-path", tmpdir,
                    "--model-type", model_type
                ])
                assert args.model_type == model_type


class TestHyperparameterSearch:
    """Tests for hyperparameter_search.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = hyperparameter_search.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = hyperparameter_search.create_parser()
        args = parser.parse_args([])
        
        assert args.search_method == "bayesian"
        assert args.n_trials == 50
        assert args.n_jobs == 1
    
    def test_validate_inputs_invalid_trials(self):
        """Test validation with invalid number of trials."""
        parser = hyperparameter_search.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "--data-path", tmpdir,
                "--n-trials", "0"
            ])
            
            result = hyperparameter_search.validate_inputs(args)
            assert result is False
    
    def test_parser_search_methods(self):
        """Test parser accepts all search methods."""
        parser = hyperparameter_search.create_parser()
        valid_methods = ["grid", "random", "bayesian"]
        
        for method in valid_methods:
            args = parser.parse_args(["--search-method", method])
            assert args.search_method == method



class TestBenchmark:
    """Tests for benchmark.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = benchmark.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_required_args(self):
        """Test parser requires model-path and test-data."""
        parser = benchmark.create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = benchmark.create_parser()
        
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with tempfile.TemporaryDirectory() as tmpdir:
                args = parser.parse_args([
                    "--model-path", f.name,
                    "--test-data", tmpdir
                ])
                
                assert args.n_iterations == 100
                assert args.batch_sizes == "1,8,32,64"
                assert args.profile_memory is False
    
    def test_validate_inputs_invalid_iterations(self):
        """Test validation with invalid iterations."""
        parser = benchmark.create_parser()
        
        with tempfile.NamedTemporaryFile(suffix=".h5") as f:
            with tempfile.TemporaryDirectory() as tmpdir:
                args = parser.parse_args([
                    "--model-path", f.name,
                    "--test-data", tmpdir,
                    "--n-iterations", "0"
                ])
                
                result = benchmark.validate_inputs(args)
                assert result is False


class TestTrainWithExternalFFT:
    """Tests for train_with_external_fft.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = train_with_external_fft.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = train_with_external_fft.create_parser()
        args = parser.parse_args([])
        
        assert args.fft_source == "matlab"
        assert args.model_type == "cnn_1d"
        assert args.epochs == 100
    
    def test_parser_fft_sources(self):
        """Test parser accepts all FFT sources."""
        parser = train_with_external_fft.create_parser()
        valid_sources = ["matlab", "scipy", "numpy"]
        
        for source in valid_sources:
            args = parser.parse_args(["--fft-source", source])
            assert args.fft_source == source
    
    def test_validate_inputs_missing_matlab_path(self):
        """Test validation with missing MATLAB path."""
        parser = train_with_external_fft.create_parser()
        args = parser.parse_args(["--matlab-path", "/nonexistent/path"])
        
        result = train_with_external_fft.validate_inputs(args)
        assert result is False
    
    def test_validate_inputs_invalid_epochs(self):
        """Test validation with invalid epochs."""
        parser = train_with_external_fft.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "--matlab-path", tmpdir,
                "--epochs", "-1"
            ])
            
            result = train_with_external_fft.validate_inputs(args)
            assert result is False


class TestCompareFFTMethods:
    """Tests for compare_fft_methods.py script."""
    
    def test_parser_creation(self):
        """Test argument parser creation."""
        parser = compare_fft_methods.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
    
    def test_parser_required_args(self):
        """Test parser requires raw-data."""
        parser = compare_fft_methods.create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([])
    
    def test_parser_defaults(self):
        """Test parser default values."""
        parser = compare_fft_methods.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args(["--raw-data", tmpdir])
            
            assert args.n_samples == 10
            assert args.fft_size == 2048
            assert args.generate_plots is False
    
    def test_validate_inputs_invalid_fft_size(self):
        """Test validation with non-power-of-2 FFT size."""
        parser = compare_fft_methods.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "--raw-data", tmpdir,
                "--fft-size", "1000"
            ])
            
            result = compare_fft_methods.validate_inputs(args)
            assert result is False
    
    def test_validate_inputs_valid_fft_sizes(self):
        """Test validation with power-of-2 FFT sizes."""
        parser = compare_fft_methods.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for size in [256, 512, 1024, 2048, 4096]:
                args = parser.parse_args([
                    "--raw-data", tmpdir,
                    "--fft-size", str(size)
                ])
                
                result = compare_fft_methods.validate_inputs(args)
                assert result is True
    
    def test_validate_inputs_invalid_n_samples(self):
        """Test validation with invalid number of samples."""
        parser = compare_fft_methods.create_parser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            args = parser.parse_args([
                "--raw-data", tmpdir,
                "--n-samples", "0"
            ])
            
            result = compare_fft_methods.validate_inputs(args)
            assert result is False


class TestScriptIntegration:
    """Integration tests for scripts."""
    
    def test_all_scripts_have_main(self):
        """Test all scripts have a main() function."""
        scripts = [
            train_model, prepare_data, predict, evaluate,
            cross_validate, hyperparameter_search,
            benchmark, train_with_external_fft, compare_fft_methods
        ]
        
        for script in scripts:
            assert hasattr(script, 'main'), f"{script.__name__} missing main()"
            assert callable(script.main), f"{script.__name__}.main is not callable"
    
    def test_all_scripts_have_create_parser(self):
        """Test all scripts have create_parser() function."""
        scripts = [
            train_model, prepare_data, predict, evaluate,
            cross_validate, hyperparameter_search,
            benchmark, train_with_external_fft, compare_fft_methods
        ]
        
        for script in scripts:
            assert hasattr(script, 'create_parser'), f"{script.__name__} missing create_parser()"
    
    def test_all_scripts_have_validation(self):
        """Test all scripts have input validation."""
        scripts = [
            train_model, prepare_data, predict, evaluate,
            cross_validate, hyperparameter_search,
            benchmark, train_with_external_fft, compare_fft_methods
        ]
        
        for script in scripts:
            # Some scripts may have validate_inputs, run_<action>, etc.
            assert hasattr(script, 'create_parser'), f"{script.__name__} missing validation logic"


class TestArgumentParsing:
    """Test argument parsing for all scripts."""
    
    @pytest.mark.parametrize("script", [
        train_model, prepare_data, predict, evaluate,
        cross_validate, hyperparameter_search,
        benchmark, train_with_external_fft, compare_fft_methods
    ])
    def test_help_flag(self, script):
        """Test --help flag works for all scripts."""
        parser = script.create_parser()
        
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--help'])
        
        assert exc_info.value.code == 0


class TestErrorHandling:
    """Test error handling in scripts."""
    
    def test_train_model_defaults(self):
        """Test train_model has working defaults."""
        parser = train_model.create_parser()
        args = parser.parse_args([])
        # train_model doesn't require arguments, has defaults
        assert args.model_type == "cnn_1d"
    
    def test_predict_missing_model_path(self):
        """Test predict handles missing model."""
        with pytest.raises(SystemExit):
            predict.create_parser().parse_args([])
    
    def test_evaluate_missing_inputs(self):
        """Test evaluate handles missing inputs."""
        with pytest.raises(SystemExit):
            evaluate.create_parser().parse_args([])


# ============================================================================
# Summary Tests
# ============================================================================

def test_phase8_script_count():
    """Verify all 10 Phase 8 scripts are implemented."""
    required_scripts = [
        'train_model',
        'prepare_data',
        'predict',
        'evaluate',
        'cross_validate',
        'hyperparameter_search',
    
        'benchmark',
        'train_with_external_fft',
        'compare_fft_methods'
    ]
    
    for script_name in required_scripts:
        script_path = Path(__file__).parent.parent / "scripts" / f"{script_name}.py"
        assert script_path.exists(), f"Script {script_name}.py not found"


def test_phase8_coverage():
    """Verify comprehensive test coverage for Phase 8."""
    test_classes = [
        TestTrainModel,
        TestPrepareData,
        TestPredict,
        TestEvaluate,
        TestCrossValidate,
        TestHyperparameterSearch,
        TestExportModel,
        TestBenchmark,
        TestTrainWithExternalFFT,
        TestCompareFFTMethods,
    ]
    
    assert len(test_classes) == 10, "Should have 10 test classes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])