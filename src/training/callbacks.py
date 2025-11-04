"""
Custom training callbacks for model training.

Provides callbacks for early stopping, model checkpointing, learning rate reduction,
and custom metrics logging.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """Early stopping with patience and optional restoration of best weights.
    
    Args:
        monitor: Metric to monitor (e.g., 'val_loss')
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore weights from best epoch
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 15,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: int = 1
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait_count = 0
        self.best_epoch = 0
        self.best_value = None
        self.best_weights = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Check for improvement after each epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose > 0:
                print(f"EarlyStoppingCallback: metric {self.monitor} not available")
            return
        
        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = [w.numpy().copy() for w in self.model.weights]
            return
        
        # Check if improved
        if "loss" in self.monitor:
            improved = (self.best_value - current) > self.min_delta
        else:
            improved = (current - self.best_value) > self.min_delta
        
        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.wait_count = 0
            if self.restore_best_weights:
                self.best_weights = [w.numpy().copy() for w in self.model.weights]
            if self.verbose > 0:
                print(f"Epoch {epoch}: {self.monitor} improved to {current:.4f}")
        else:
            self.wait_count += 1
            if self.verbose > 0 and self.wait_count % 5 == 0:
                print(f"Epoch {epoch}: {self.monitor} did not improve. "
                      f"Wait {self.wait_count}/{self.patience}")
            
            if self.wait_count >= self.patience:
                if self.verbose > 0:
                    print(f"Epoch {epoch}: Early stopping triggered. "
                          f"Best epoch: {self.best_epoch}")
                self.model.stop_training = True
                
                if self.restore_best_weights and self.best_weights:
                    for w, best_w in zip(self.model.weights, self.best_weights):
                        w.assign(best_w)


class ModelCheckpointCallback(tf.keras.callbacks.Callback):
    """Save model checkpoint when metric improves.
    
    Args:
        filepath: Path to save model
        monitor: Metric to monitor
        save_best_only: Only save if metric improves
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        save_best_only: bool = True,
        verbose: int = 1
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.best_value = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Save model if metric improves."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif self.best_value is None:
            should_save = True
            self.best_value = current
        else:
            if "loss" in self.monitor:
                should_save = current < self.best_value
            else:
                should_save = current > self.best_value
            
            if should_save:
                self.best_value = current
        
        if should_save:
            if self.verbose > 0:
                print(f"Epoch {epoch}: Saving model to {self.filepath}")
            self.model.save(str(self.filepath))


class ReduceLROnPlateauCallback(tf.keras.callbacks.Callback):
    """Reduce learning rate when metric plateaus.
    
    Args:
        monitor: Metric to monitor
        factor: Multiplicative factor for learning rate reduction
        patience: Number of epochs with no improvement before reducing
        min_lr: Minimum learning rate
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-7,
        verbose: int = 1
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.wait_count = 0
        self.best_value = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Reduce learning rate if no improvement."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        if self.best_value is None:
            self.best_value = current
            return
        
        if "loss" in self.monitor:
            improved = current < self.best_value
        else:
            improved = current > self.best_value
        
        if improved:
            self.best_value = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                old_lr = float(self.model.optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                self.model.optimizer.learning_rate = new_lr
                self.wait_count = 0
                
                if self.verbose > 0:
                    print(f"Epoch {epoch}: Reducing learning rate to {new_lr:.2e}")


class MetricsLoggingCallback(tf.keras.callbacks.Callback):
    """Log custom metrics during training.
    
    Args:
        metrics_fn: Function that computes custom metrics
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        metrics_fn: callable = None,
        verbose: int = 1
    ):
        super().__init__()
        self.metrics_fn = metrics_fn
        self.verbose = verbose
        self.history = {
            "epochs": [],
            "train_metrics": [],
            "val_metrics": []
        }
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Log metrics after each epoch."""
        logs = logs or {}
        
        self.history["epochs"].append(epoch)
        
        if self.metrics_fn:
            try:
                custom_metrics = self.metrics_fn(epoch, logs)
                self.history["val_metrics"].append(custom_metrics)
                
                if self.verbose > 0:
                    metrics_str = ", ".join(
                        f"{k}: {v:.4f}" for k, v in custom_metrics.items()
                    )
                    print(f"Epoch {epoch}: {metrics_str}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Error computing custom metrics: {e}")
    
    def get_history(self):
        """Get accumulated metrics history."""
        return self.history


class TensorBoardCallback(tf.keras.callbacks.TensorBoard):
    """Enhanced TensorBoard callback with custom logging.
    
    Args:
        log_dir: Directory to save logs
        histogram_freq: Frequency for histogram logging
        update_freq: Update frequency
        profile_batch: Batch range for profiling
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        histogram_freq: int = 0,
        update_freq: str = "epoch",
        profile_batch: str = "500,520"
    ):
        super().__init__(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            update_freq=update_freq,
            profile_batch=profile_batch
        )
        Path(log_dir).mkdir(parents=True, exist_ok=True)


def get_callbacks(
    config: Dict[str, Any] = None,
    checkpoint_dir: str = "./checkpoints"
) -> list:
    """
    Create a list of callbacks based on configuration.
    
    Args:
        config: Configuration dictionary with callback settings
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        List of callback instances
    """
    config = config or {}
    callbacks = []
    
    # Early stopping
    if config.get("early_stopping", True):
        callbacks.append(
            EarlyStoppingCallback(
                monitor=config.get("early_stopping_monitor", "val_loss"),
                patience=config.get("early_stopping_patience", 15),
                min_delta=config.get("early_stopping_min_delta", 1e-4),
                restore_best_weights=True,
                verbose=1
            )
        )
    
    # Model checkpointing
    if config.get("model_checkpoint", True):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpointCallback(
                filepath=str(Path(checkpoint_dir) / "best_model.keras"),
                monitor=config.get("checkpoint_monitor", "val_loss"),
                save_best_only=True,
                verbose=1
            )
        )
    
    # Reduce learning rate on plateau
    if config.get("reduce_lr_on_plateau", True):
        callbacks.append(
            ReduceLROnPlateauCallback(
                monitor=config.get("reduce_lr_monitor", "val_loss"),
                factor=config.get("reduce_lr_factor", 0.5),
                patience=config.get("reduce_lr_patience", 5),
                min_lr=config.get("reduce_lr_min", 1e-7),
                verbose=1
            )
        )
    
    # TensorBoard
    if config.get("tensorboard", False):
        callbacks.append(
            TensorBoardCallback(
                log_dir=config.get("tensorboard_log_dir", "./logs")
            )
        )
    
    return callbacks