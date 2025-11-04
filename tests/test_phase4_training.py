"""
Unit tests for Phase 4 training pipeline.

Tests loss functions, optimizers, callbacks, cross-validation, and trainer.
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil

from src.training import (
    # Losses
    FocalLoss,
    WeightedCategoricalCrossentropy,
    LeakAwareLoss,
    get_loss_function,
    # Optimizers
    WarmupSchedule,
    CosineDecaySchedule,
    get_optimizer,
    get_learning_rate_schedule,
    # Callbacks
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    ReduceLROnPlateauCallback,
    MetricsLoggingCallback,
    get_callbacks,
    # Cross-validation
    CrossValidator,
    KFoldValidator,
    TimeSeriesValidator,
    LeaveOneOutValidator,
    evaluate_with_cross_validation,
    # Hyperparameter tuning
    GridSearchTuner,
    RandomSearchTuner,
    HyperparameterTuner,
    suggest_hyperparameter_space,
    # Trainer
    ModelTrainer,
    MultiModelTrainer
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample training data."""
    X_train = np.random.randn(100, 32).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 4, 100), 4)
    
    X_val = np.random.randn(20, 32).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(np.random.randint(0, 4, 20), 4)
    
    X_test = np.random.randn(20, 32).astype(np.float32)
    y_test = np.random.randint(0, 4, 20)
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }


@pytest.fixture
def simple_model():
    """Create a simple CNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(32,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation="softmax")
    ])
    return model


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


# ============================================================================
# LOSS FUNCTIONS TESTS
# ============================================================================

class TestFocalLoss:
    """Tests for FocalLoss."""
    
    def test_focal_loss_initialization(self):
        """Test FocalLoss initialization."""
        loss = FocalLoss(alpha=0.25, gamma=2.0)
        assert loss.alpha == 0.25
        assert loss.gamma == 2.0
    
    def test_focal_loss_output_shape(self, sample_data):
        """Test FocalLoss output shape."""
        loss = FocalLoss()
        y_true = sample_data["y_train"][:10]
        y_pred = np.random.rand(10, 4).astype(np.float32)
        y_pred /= y_pred.sum(axis=1, keepdims=True)
        
        loss_value = loss(tf.constant(y_true), tf.constant(y_pred))
        assert loss_value.shape == ()
        assert loss_value > 0
    
    def test_focal_loss_from_logits(self, sample_data):
        """Test FocalLoss with logits."""
        loss = FocalLoss(from_logits=True)
        y_true = sample_data["y_train"][:10]
        y_pred = np.random.randn(10, 4).astype(np.float32)
        
        loss_value = loss(tf.constant(y_true), tf.constant(y_pred))
        assert loss_value > 0


class TestWeightedCategoricalCrossentropy:
    """Tests for WeightedCategoricalCrossentropy."""
    
    def test_weighted_cce_initialization(self):
        """Test WeightedCategoricalCrossentropy initialization."""
        class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.5}
        loss = WeightedCategoricalCrossentropy(class_weights=class_weights)
        assert loss.class_weights == class_weights
    
    def test_weighted_cce_no_weights(self, sample_data):
        """Test WeightedCategoricalCrossentropy without weights."""
        loss = WeightedCategoricalCrossentropy()
        y_true = sample_data["y_train"][:10]
        y_pred = np.random.rand(10, 4).astype(np.float32)
        y_pred /= y_pred.sum(axis=1, keepdims=True)
        
        loss_value = loss(tf.constant(y_true), tf.constant(y_pred))
        assert loss_value > 0


class TestLeakAwareLoss:
    """Tests for LeakAwareLoss."""
    
    def test_leak_aware_loss_initialization(self):
        """Test LeakAwareLoss initialization."""
        loss = LeakAwareLoss(leak_penalty=2.0, non_leak_penalty=1.0)
        assert loss.leak_penalty == 2.0
        assert loss.non_leak_penalty == 1.0
    
    def test_leak_aware_loss_output(self, sample_data):
        """Test LeakAwareLoss output."""
        loss = LeakAwareLoss()
        y_true = sample_data["y_train"][:10]
        y_pred = np.random.rand(10, 4).astype(np.float32)
        y_pred /= y_pred.sum(axis=1, keepdims=True)
        
        loss_value = loss(tf.constant(y_true), tf.constant(y_pred))
        assert loss_value > 0


class TestGetLossFunction:
    """Tests for get_loss_function."""
    
    def test_get_focal_loss(self):
        """Test getting focal loss."""
        loss = get_loss_function("focal_loss", config={"alpha": 0.25, "gamma": 2.0})
        assert isinstance(loss, FocalLoss)
    
    def test_get_weighted_cce(self):
        """Test getting weighted CCE loss."""
        loss = get_loss_function("weighted_categorical_crossentropy")
        assert isinstance(loss, WeightedCategoricalCrossentropy)
    
    def test_get_leak_aware_loss(self):
        """Test getting leak aware loss."""
        loss = get_loss_function("leak_aware_loss")
        assert isinstance(loss, LeakAwareLoss)
    
    def test_get_categorical_crossentropy(self):
        """Test getting standard CCE loss."""
        loss = get_loss_function("categorical_crossentropy")
        assert isinstance(loss, tf.keras.losses.CategoricalCrossentropy)
    
    def test_get_invalid_loss(self):
        """Test getting invalid loss raises error."""
        with pytest.raises(ValueError):
            get_loss_function("invalid_loss")


# ============================================================================
# OPTIMIZERS TESTS
# ============================================================================

class TestWarmupSchedule:
    """Tests for WarmupSchedule."""
    
    def test_warmup_schedule_initialization(self):
        """Test WarmupSchedule initialization."""
        schedule = WarmupSchedule(
            initial_learning_rate=1e-4,
            max_learning_rate=1e-3,
            warmup_steps=1000
        )
        assert schedule.initial_learning_rate == 1e-4
        assert schedule.max_learning_rate == 1e-3
        assert schedule.warmup_steps == 1000
    
    def test_warmup_schedule_warmup_phase(self):
        """Test learning rate during warmup phase."""
        schedule = WarmupSchedule(
            initial_learning_rate=0.0,
            max_learning_rate=0.001,
            warmup_steps=1000
        )
        
        # At step 500 (halfway through warmup)
        lr = schedule(500)
        assert 0.0 < lr < 0.001
        assert abs(lr - 0.0005) < 1e-6
    
    def test_warmup_schedule_after_warmup(self):
        """Test learning rate after warmup."""
        schedule = WarmupSchedule(
            initial_learning_rate=0.0,
            max_learning_rate=0.001,
            warmup_steps=1000
        )
        
        # After warmup
        lr = schedule(1500)
        assert abs(lr - 0.001) < 1e-6
    
    def test_warmup_schedule_with_decay(self):
        """Test WarmupSchedule with decay."""
        schedule = WarmupSchedule(
            initial_learning_rate=0.0,
            max_learning_rate=0.001,
            warmup_steps=100,
            decay_steps=100,
            decay_rate=0.9
        )
        
        lr_at_150 = schedule(150)
        lr_at_200 = schedule(200)
        
        # Learning rate should decrease after warmup
        assert lr_at_200 < lr_at_150


class TestCosineDecaySchedule:
    """Tests for CosineDecaySchedule."""
    
    def test_cosine_decay_initialization(self):
        """Test CosineDecaySchedule initialization."""
        schedule = CosineDecaySchedule(
            initial_learning_rate=0.001,
            min_learning_rate=1e-5,
            total_steps=1000
        )
        assert schedule.initial_learning_rate == 0.001
        assert schedule.min_learning_rate == 1e-5
    
    def test_cosine_decay_values(self):
        """Test CosineDecaySchedule values."""
        schedule = CosineDecaySchedule(
            initial_learning_rate=0.001,
            min_learning_rate=0.0,
            total_steps=100
        )
        
        lr_0 = schedule(0)
        lr_50 = schedule(50)
        lr_100 = schedule(100)
        
        assert lr_0 > lr_50 > lr_100


class TestGetOptimizer:
    """Tests for get_optimizer."""
    
    def test_get_adam_optimizer(self):
        """Test getting Adam optimizer."""
        opt = get_optimizer("adam", learning_rate=0.001)
        assert isinstance(opt, tf.keras.optimizers.Adam)
    
    def test_get_sgd_optimizer(self):
        """Test getting SGD optimizer."""
        opt = get_optimizer("sgd", learning_rate=0.01)
        assert isinstance(opt, tf.keras.optimizers.SGD)
    
    def test_get_rmsprop_optimizer(self):
        """Test getting RMSprop optimizer."""
        opt = get_optimizer("rmsprop", learning_rate=0.001)
        assert isinstance(opt, tf.keras.optimizers.RMSprop)
    
    def test_get_invalid_optimizer(self):
        """Test getting invalid optimizer raises error."""
        with pytest.raises(ValueError):
            get_optimizer("invalid_optimizer")


class TestGetLearningRateSchedule:
    """Tests for get_learning_rate_schedule."""
    
    def test_get_constant_schedule(self):
        """Test getting constant schedule."""
        schedule = get_learning_rate_schedule("constant", 0.001)
        assert schedule == 0.001
    
    def test_get_warmup_schedule(self):
        """Test getting warmup schedule."""
        schedule = get_learning_rate_schedule(
            "warmup",
            base_learning_rate=0.001,
            config={"warmup_steps": 100}
        )
        assert isinstance(schedule, WarmupSchedule)
    
    def test_get_cosine_schedule(self):
        """Test getting cosine schedule."""
        schedule = get_learning_rate_schedule(
            "cosine",
            base_learning_rate=0.001,
            config={"total_steps": 1000}
        )
        assert isinstance(schedule, CosineDecaySchedule)


# ============================================================================
# CALLBACKS TESTS
# ============================================================================

class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""
    
    def test_early_stopping_initialization(self):
        """Test EarlyStoppingCallback initialization."""
        callback = EarlyStoppingCallback(patience=15)
        assert callback.patience == 15
        assert callback.wait_count == 0
    
    def test_early_stopping_on_improvement(self, simple_model):
        """Test early stopping tracks improvement."""
        callback = EarlyStoppingCallback(patience=2, restore_best_weights=False)
        callback.set_model(simple_model)

        logs_1 = {"val_loss": 0.5}
        callback.on_epoch_end(0, logs_1)
        assert callback.best_value == 0.5

        logs_2 = {"val_loss": 0.4}
        callback.on_epoch_end(1, logs_2)
        assert callback.best_value == 0.4
        assert callback.wait_count == 0
    
    def test_early_stopping_on_plateau(self, simple_model):
        """Test early stopping on plateau."""
        callback = EarlyStoppingCallback(patience=2, restore_best_weights=False)
        callback.set_model(simple_model)

        logs_1 = {"val_loss": 0.5}
        callback.on_epoch_end(0, logs_1)

        logs_2 = {"val_loss": 0.5}
        callback.on_epoch_end(1, logs_2)
        assert callback.wait_count == 1

        logs_3 = {"val_loss": 0.5}
        callback.on_epoch_end(2, logs_3)
        assert callback.wait_count == 2


class TestModelCheckpointCallback:
    """Tests for ModelCheckpointCallback."""
    
    def test_checkpoint_callback_initialization(self, temp_dir):
        """Test ModelCheckpointCallback initialization."""
        callback = ModelCheckpointCallback(
            filepath=str(Path(temp_dir) / "model.keras")
        )
        assert callback.save_best_only
        assert callback.best_value is None


class TestReduceLROnPlateauCallback:
    """Tests for ReduceLROnPlateauCallback."""
    
    def test_reduce_lr_initialization(self):
        """Test ReduceLROnPlateauCallback initialization."""
        callback = ReduceLROnPlateauCallback(factor=0.5, patience=5)
        assert callback.factor == 0.5
        assert callback.patience == 5


class TestGetCallbacks:
    """Tests for get_callbacks."""
    
    def test_get_callbacks_default(self, temp_dir):
        """Test getting default callbacks."""
        callbacks = get_callbacks(checkpoint_dir=temp_dir)
        assert len(callbacks) > 0
    
    def test_get_callbacks_with_config(self, temp_dir):
        """Test getting callbacks with configuration."""
        config = {
            "early_stopping": True,
            "model_checkpoint": True,
            "reduce_lr_on_plateau": True
        }
        callbacks = get_callbacks(config, temp_dir)
        assert len(callbacks) >= 3


# ============================================================================
# CROSS-VALIDATION TESTS
# ============================================================================

class TestCrossValidator:
    """Tests for CrossValidator."""
    
    def test_cross_validator_initialization(self):
        """Test CrossValidator initialization."""
        validator = CrossValidator(method="stratified_kfold", n_splits=5)
        assert validator.method == "stratified_kfold"
        assert validator.n_splits == 5
    
    def test_cross_validator_split(self, sample_data):
        """Test CrossValidator split."""
        validator = CrossValidator(method="stratified_kfold", n_splits=3)
        
        # Convert one-hot to class indices for stratified k-fold
        y = np.argmax(sample_data["y_train"], axis=1)
        splits = list(validator.split(sample_data["X_train"], y))
        assert len(splits) == 3
        
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


class TestKFoldValidator:
    """Tests for KFoldValidator."""
    
    def test_kfold_validator_split(self, sample_data):
        """Test KFoldValidator split."""
        validator = KFoldValidator(n_splits=5)
        splits = list(validator.split(sample_data["X_train"], 
                                     np.argmax(sample_data["y_train"], axis=1)))
        assert len(splits) == 5
    
    def test_kfold_validator_statistics(self, sample_data):
        """Test KFoldValidator statistics."""
        validator = KFoldValidator(n_splits=3)
        stats = validator.get_fold_statistics(
            sample_data["X_train"],
            np.argmax(sample_data["y_train"], axis=1)
        )
        
        assert stats["n_folds"] == 3
        assert len(stats["fold_info"]) == 3


class TestTimeSeriesValidator:
    """Tests for TimeSeriesValidator."""
    
    def test_time_series_validator_split(self):
        """Test TimeSeriesValidator split."""
        X = np.random.randn(100, 10)
        validator = TimeSeriesValidator(n_splits=3)
        
        splits = list(validator.split(X))
        assert len(splits) == 3
        
        # Check that training is always before testing
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx)


class TestLeaveOneOutValidator:
    """Tests for LeaveOneOutValidator."""
    
    def test_leave_one_out_validator(self):
        """Test LeaveOneOutValidator."""
        X = np.random.randn(10, 5)
        validator = LeaveOneOutValidator()
        
        n_splits = validator.get_n_splits(X)
        assert n_splits == 10


# ============================================================================
# HYPERPARAMETER TUNER TESTS
# ============================================================================

class TestGridSearchTuner:
    """Tests for GridSearchTuner."""
    
    def test_grid_search_initialization(self):
        """Test GridSearchTuner initialization."""
        param_grid = {"n_estimators": [10, 20], "max_depth": [5, 10]}
        tuner = GridSearchTuner(param_grid=param_grid)
        assert tuner.param_grid == param_grid
    
    def test_suggest_hyperparameter_space_rf(self):
        """Test suggested hyperparameter space for Random Forest."""
        space = suggest_hyperparameter_space("rf")
        assert "n_estimators" in space
        assert "max_depth" in space


class TestRandomSearchTuner:
    """Tests for RandomSearchTuner."""
    
    def test_random_search_initialization(self):
        """Test RandomSearchTuner initialization."""
        param_dist = {"n_estimators": [10, 20, 30]}
        tuner = RandomSearchTuner(param_distributions=param_dist, n_iter=5)
        assert tuner.n_iter == 5


class TestHyperparameterTuner:
    """Tests for HyperparameterTuner."""
    
    def test_hyperparameter_tuner_initialization(self):
        """Test HyperparameterTuner initialization."""
        tuner = HyperparameterTuner(method="grid", param_grid={"C": [0.1, 1.0]})
        assert tuner.method == "grid"


# ============================================================================
# TRAINER TESTS
# ============================================================================

class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    def test_trainer_initialization(self, simple_model, temp_dir):
        """Test ModelTrainer initialization."""
        config = {"experiment_dir": temp_dir}
        trainer = ModelTrainer(simple_model, config, "test_experiment")
        
        assert trainer.experiment_name == "test_experiment"
        assert trainer.checkpoint_dir.exists()
    
    def test_trainer_compile(self, simple_model):
        """Test ModelTrainer compile."""
        config = {
            "training": {
                "loss": "categorical_crossentropy",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "metrics": ["accuracy"]
            }
        }
        trainer = ModelTrainer(simple_model, config)
        trainer.compile()
        
        assert trainer.model.optimizer is not None
        assert trainer.model.loss is not None
    
    def test_trainer_compile_with_focal_loss(self, simple_model):
        """Test ModelTrainer compile with focal loss."""
        config = {
            "training": {
                "loss": "focal_loss",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss_config": {"alpha": 0.25, "gamma": 2.0}
            }
        }
        trainer = ModelTrainer(simple_model, config)
        trainer.compile()
        assert trainer.model.optimizer is not None
        assert isinstance(trainer.model.loss, FocalLoss)
    
    def test_trainer_train(self, simple_model, sample_data, temp_dir):
        """Test ModelTrainer train."""
        config = {
            "experiment_dir": temp_dir,
            "training": {
                "loss": "categorical_crossentropy",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "epochs": 2,
                "batch_size": 32,
                "callbacks": {
                    "early_stopping": False,
                    "model_checkpoint": False,
                    "reduce_lr_on_plateau": False
                }
            }
        }
        trainer = ModelTrainer(simple_model, config)
        trainer.compile()
        
        history = trainer.train(
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2
        )
        
        assert "loss" in history
        assert len(history["loss"]) == 2
    
    def test_trainer_evaluate(self, simple_model, sample_data, temp_dir):
        """Test ModelTrainer evaluate."""
        config = {
            "experiment_dir": temp_dir,
            "training": {
                "loss": "categorical_crossentropy",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "callbacks": {
                    "early_stopping": False,
                    "model_checkpoint": False,
                    "reduce_lr_on_plateau": False
                }
            }
        }
        trainer = ModelTrainer(simple_model, config)
        trainer.compile()
        
        # Quick training
        trainer.train(sample_data["X_train"], sample_data["y_train"], epochs=1)
        
        # Evaluate (convert test labels to one-hot)
        y_test_onehot = tf.keras.utils.to_categorical(sample_data["y_test"], 4)
        results = trainer.evaluate(sample_data["X_test"], y_test_onehot)
        assert "loss" in results or len(results) > 0
    
    def test_trainer_predict(self, simple_model, sample_data, temp_dir):
        """Test ModelTrainer predict."""
        config = {"experiment_dir": temp_dir}
        trainer = ModelTrainer(simple_model, config)
        trainer.compile()
        
        predictions = trainer.predict(sample_data["X_test"])
        assert predictions.shape == (len(sample_data["X_test"]), 4)
    
    def test_trainer_save_model(self, simple_model, temp_dir):
        """Test ModelTrainer save_model."""
        config = {"experiment_dir": temp_dir}
        trainer = ModelTrainer(simple_model, config)
        trainer.compile()
        
        model_path = str(Path(temp_dir) / "model.keras")
        trainer.save_model(model_path)
        
        assert Path(model_path).exists()
    
    def test_trainer_get_model_summary(self, simple_model):
        """Test ModelTrainer get_model_summary."""
        trainer = ModelTrainer(simple_model)
        summary = trainer.get_model_summary()
        
        assert isinstance(summary, str)
        assert "Dense" in summary


class TestMultiModelTrainer:
    """Tests for MultiModelTrainer."""
    
    def test_multi_model_trainer_initialization(self):
        """Test MultiModelTrainer initialization."""
        trainer = MultiModelTrainer()
        assert len(trainer.trainers) == 0
    
    def test_multi_model_trainer_add_model(self, simple_model):
        """Test adding model to MultiModelTrainer."""
        trainer = MultiModelTrainer()
        trainer.add_model("model_1", simple_model)
        
        assert "model_1" in trainer.trainers
    
    def test_multi_model_trainer_compare_models(self):
        """Test comparing models."""
        trainer = MultiModelTrainer()
        trainer.results = {
            "model_1": {"val_accuracy": [0.8, 0.85, 0.87]},
            "model_2": {"val_accuracy": [0.75, 0.82, 0.84]},
        }
        
        comparison = trainer.compare_models(metric="accuracy")
        assert "model_1" in comparison
        assert comparison["model_1"] > comparison["model_2"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_full_training_workflow(self, simple_model, sample_data, temp_dir):
        """Test complete training workflow."""
        config = {
            "experiment_dir": temp_dir,
            "training": {
                "loss": "focal_loss",
                "optimizer": "adam",
                "learning_rate": 0.001,
                "lr_schedule": "constant",
                "epochs": 2,
                "batch_size": 16,
                "callbacks": {
                    "early_stopping": True,
                    "model_checkpoint": True,
                    "reduce_lr_on_plateau": False
                }
            }
        }
        
        # Initialize trainer
        trainer = ModelTrainer(simple_model, config, "integration_test")
        
        # Compile
        trainer.compile()
        assert trainer.model.optimizer is not None
        
        # Train
        history = trainer.train(
            sample_data["X_train"],
            sample_data["y_train"],
            sample_data["X_val"],
            sample_data["y_val"],
            epochs=2
        )
        assert history is not None
        
        # Evaluate (convert test labels to one-hot)
        y_test_onehot = tf.keras.utils.to_categorical(sample_data["y_test"], 4)
        results = trainer.evaluate(sample_data["X_test"], y_test_onehot)
        assert results is not None
        
        # Save
        trainer.save_model()
        assert (trainer.experiment_dir / "final_model.keras").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])