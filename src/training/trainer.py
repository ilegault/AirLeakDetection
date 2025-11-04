"""
Main training orchestrator for model training.

Provides ModelTrainer class for unified training, evaluation, and model management
with support for all model types, callbacks, and advanced features.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
from datetime import datetime

from .callbacks import get_callbacks
from .losses import get_loss_function
from .optimizers import get_optimizer, get_learning_rate_schedule


class ModelTrainer:
    """Main training orchestrator supporting all model types.
    
    Handles training workflow including:
    - Model compilation with custom losses and optimizers
    - Training with callbacks and monitoring
    - Evaluation and checkpoint management
    - Results logging and reporting
    
    Args:
        model: Keras model to train
        config: Configuration dictionary
        experiment_name: Name for this training experiment
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        config: Dict[str, Any] = None,
        experiment_name: str = "default"
    ):
        self.model = model
        self.config = config or {}
        self.experiment_name = experiment_name
        
        # Setup directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = Path(
            self.config.get("experiment_dir", "./experiments")
        ) / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.experiment_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.training_history = None
        self.best_model = None
        self.best_epoch = None
        self.best_score = None
    
    def compile(self, loss: str = None, optimizer: str = None, metrics: List[str] = None):
        """
        Compile the model with loss, optimizer, and metrics.
        
        Args:
            loss: Loss function name
            optimizer: Optimizer name
            metrics: List of metric names
        """
        loss = loss or self.config.get("training", {}).get("loss", "categorical_crossentropy")
        optimizer = optimizer or self.config.get("training", {}).get("optimizer", "adam")
        metrics = metrics or self.config.get("training", {}).get("metrics", ["accuracy"])
        
        # Get loss function
        loss_config = self.config.get("training", {}).get("loss_config", {})
        class_weights = self.config.get("training", {}).get("class_weights")
        loss_fn = get_loss_function(loss, loss_config, class_weights)
        
        # Get optimizer with learning rate schedule
        lr = self.config.get("training", {}).get("learning_rate", 0.001)
        lr_schedule_name = self.config.get("training", {}).get("lr_schedule", "constant")
        lr_schedule_config = self.config.get("training", {}).get("lr_schedule_config", {})
        
        learning_rate = get_learning_rate_schedule(
            lr_schedule_name,
            lr,
            lr_schedule_config
        )
        
        optimizer_config = self.config.get("training", {}).get("optimizer_config", {})
        optimizer_fn = get_optimizer(optimizer, learning_rate, optimizer_config)
        
        # Compile
        self.model.compile(
            loss=loss_fn,
            optimizer=optimizer_fn,
            metrics=metrics
        )
        
        # Log configuration
        self._log_config({
            "loss": loss,
            "optimizer": optimizer,
            "metrics": metrics,
            "loss_config": loss_config,
            "optimizer_config": optimizer_config
        })
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = None,
        batch_size: int = None,
        verbose: int = 1
    ) -> dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.get("training", {}).get("epochs", 50)
        batch_size = batch_size or self.config.get("training", {}).get("batch_size", 32)
        
        # Prepare data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Get callbacks
        callbacks_config = self.config.get("training", {}).get("callbacks", {})
        callbacks = get_callbacks(callbacks_config, str(self.checkpoint_dir))
        
        # Train
        print(f"\nTraining {self.experiment_name}...")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"- Validation samples: {len(X_val)}")
        
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history = history.history
        
        # Save training history
        self._save_history()
        
        return self.training_history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = None,
        verbose: int = 1
    ) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size for evaluation
            verbose: Verbosity level
            
        Returns:
            Dictionary with evaluation metrics
        """
        batch_size = batch_size or self.config.get("training", {}).get("batch_size", 32)
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
        
        # Format results
        if isinstance(results, list):
            metric_names = ["loss"] + self.model.metrics_names
            eval_dict = {name: float(val) for name, val in zip(metric_names, results)}
        else:
            eval_dict = {"loss": float(results)}
        
        print(f"\nTest Evaluation Results:")
        for metric, value in eval_dict.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save results
        self._save_evaluation_results(eval_dict)
        
        return eval_dict
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = None,
        verbose: int = 0
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            batch_size: Batch size for prediction
            verbose: Verbosity level
            
        Returns:
            Predictions array
        """
        batch_size = batch_size or self.config.get("training", {}).get("batch_size", 32)
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)
    
    def save_model(self, filepath: str = None):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save model. If None, uses default location.
        """
        if filepath is None:
            filepath = str(self.experiment_dir / "final_model.keras")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_best_model(self, checkpoint_path: str = None):
        """
        Load best model from checkpoints.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, uses default.
        """
        if checkpoint_path is None:
            checkpoint_path = str(self.checkpoint_dir / "best_model.keras")
        
        if Path(checkpoint_path).exists():
            self.model = tf.keras.models.load_model(checkpoint_path)
            print(f"Loaded best model from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        summary_str = []
        self.model.summary(print_fn=summary_str.append)
        return "\n".join(summary_str)
    
    def _log_config(self, training_config: dict):
        """Log training configuration."""
        config_path = self.experiment_dir / "training_config.json"
        config_to_save = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "training_config": training_config
        }
        
        with open(config_path, "w") as f:
            json.dump(config_to_save, f, indent=2)
    
    def _save_history(self):
        """Save training history."""
        history_path = self.experiment_dir / "training_history.json"
        
        # Convert numpy arrays to lists
        history_dict = {}
        for key, values in self.training_history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=2)
    
    def _save_evaluation_results(self, results: dict):
        """Save evaluation results."""
        results_path = self.experiment_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)


class MultiModelTrainer:
    """Trainer for multiple models with comparison.
    
    Trains multiple models and compares their performance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.trainers = {}
        self.results = {}
    
    def add_model(
        self,
        name: str,
        model: tf.keras.Model,
        config: Dict[str, Any] = None
    ):
        """Add a model to train."""
        trainer_config = config or self.config
        trainer = ModelTrainer(model, trainer_config, name)
        self.trainers[name] = trainer
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> dict:
        """Train all added models."""
        results = {}
        
        for name, trainer in self.trainers.items():
            print(f"\n{'='*60}")
            print(f"Training model: {name}")
            print(f"{'='*60}")
            
            trainer.compile()
            history = trainer.train(X_train, y_train, X_val, y_val, **kwargs)
            results[name] = history
        
        self.results = results
        return results
    
    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> dict:
        """Evaluate all trained models."""
        results = {}
        
        for name, trainer in self.trainers.items():
            print(f"\n{'='*60}")
            print(f"Evaluating model: {name}")
            print(f"{'='*60}")
            
            eval_results = trainer.evaluate(X_test, y_test, **kwargs)
            results[name] = eval_results
        
        return results
    
    def compare_models(self, metric: str = "accuracy") -> dict:
        """
        Compare models on a specific metric.
        
        Args:
            metric: Metric to compare
            
        Returns:
            Comparison dictionary sorted by metric
        """
        comparison = {}
        
        for name, history in self.results.items():
            if f"val_{metric}" in history:
                comparison[name] = history[f"val_{metric}"][-1]
            elif metric in history:
                comparison[name] = history[metric][-1]
        
        # Sort by metric value
        sorted_comparison = dict(sorted(
            comparison.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_comparison