"""
Optimizer configurations and learning rate schedules.

Provides pre-configured optimizers and flexible learning rate scheduling
strategies for training deep learning models.
"""

import tensorflow as tf
from typing import Callable, Optional, Dict, Any
import math


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup followed by decay.
    
    The learning rate linearly increases from initial_learning_rate to
    max_learning_rate over warmup_steps, then optionally decays.
    
    Args:
        initial_learning_rate: Starting learning rate
        max_learning_rate: Peak learning rate after warmup
        warmup_steps: Number of steps to reach max learning rate
        decay_steps: Number of steps for decay phase
        decay_rate: Rate of decay (ignored if decay_steps=0)
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        max_learning_rate: float,
        warmup_steps: int,
        decay_steps: int = 0,
        decay_rate: float = 0.96
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
    
    def __call__(self, step) -> float:
        """Calculate learning rate for current step."""
        # Handle Tensor input
        try:
            step = float(step)
        except (TypeError, ValueError):
            import numpy as np
            if isinstance(step, np.ndarray):
                step = float(step)
            else:
                step = float(step)
        
        warmup_steps = float(self.warmup_steps)
        
        if step < warmup_steps:
            # Linear warmup
            return self.initial_learning_rate + (
                self.max_learning_rate - self.initial_learning_rate
            ) * (step / warmup_steps)
        elif self.decay_steps > 0:
            # Exponential decay after warmup
            decay_step = step - warmup_steps
            return self.max_learning_rate * (
                self.decay_rate ** (decay_step / self.decay_steps)
            )
        else:
            # Constant after warmup
            return self.max_learning_rate
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "max_learning_rate": self.max_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
        }


class CosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing learning rate schedule.
    
    Learning rate decays following a cosine curve from initial to minimum value
    over total_steps.
    
    Args:
        initial_learning_rate: Starting learning rate
        min_learning_rate: Minimum learning rate
        total_steps: Total training steps
        warmup_steps: Number of warmup steps (optional)
    """
    
    def __init__(
        self,
        initial_learning_rate: float,
        min_learning_rate: float,
        total_steps: int,
        warmup_steps: int = 0
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
    
    def __call__(self, step) -> float:
        """Calculate learning rate for current step."""
        # Handle Tensor input
        try:
            step = float(step)
        except (TypeError, ValueError):
            import numpy as np
            if isinstance(step, np.ndarray):
                step = float(step)
            else:
                step = float(step)
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.initial_learning_rate * (step / self.warmup_steps)
        
        # Cosine annealing
        progress = (step - self.warmup_steps) / (
            self.total_steps - self.warmup_steps
        )
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr_range = self.initial_learning_rate - self.min_learning_rate
        
        return self.min_learning_rate + lr_range * cosine_decay
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        }


def get_optimizer(
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    config: Dict[str, Any] = None
) -> tf.keras.optimizers.Optimizer:
    """
    Get an optimizer with specified configuration.
    
    Args:
        optimizer_name: Name of optimizer (adam, sgd, rmsprop)
        learning_rate: Base learning rate
        config: Configuration dictionary
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If optimizer name is not recognized
    """
    config = config or {}
    
    if optimizer_name.lower() == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=config.get("beta_1", 0.9),
            beta_2=config.get("beta_2", 0.999),
            epsilon=config.get("epsilon", 1e-7),
            weight_decay=config.get("weight_decay", 0.0)
        )
    
    elif optimizer_name.lower() == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=config.get("momentum", 0.9),
            nesterov=config.get("nesterov", True),
            weight_decay=config.get("weight_decay", 0.0)
        )
    
    elif optimizer_name.lower() == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=config.get("rho", 0.9),
            momentum=config.get("momentum", 0.0),
            epsilon=config.get("epsilon", 1e-7),
            weight_decay=config.get("weight_decay", 0.0)
        )
    
    elif optimizer_name.lower() == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            beta_1=config.get("beta_1", 0.9),
            beta_2=config.get("beta_2", 0.999),
            epsilon=config.get("epsilon", 1e-7),
            weight_decay=config.get("weight_decay", 0.01)
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_learning_rate_schedule(
    schedule_name: str = "constant",
    base_learning_rate: float = 0.001,
    config: Dict[str, Any] = None
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
    """
    Get a learning rate schedule.
    
    Args:
        schedule_name: Name of schedule (constant, warmup, cosine, exponential)
        base_learning_rate: Base learning rate
        config: Configuration dictionary
        
    Returns:
        Learning rate schedule instance
        
    Raises:
        ValueError: If schedule name is not recognized
    """
    config = config or {}
    
    if schedule_name.lower() == "constant":
        return base_learning_rate
    
    elif schedule_name.lower() == "warmup":
        return WarmupSchedule(
            initial_learning_rate=config.get("initial_lr", base_learning_rate * 0.1),
            max_learning_rate=base_learning_rate,
            warmup_steps=config.get("warmup_steps", 1000),
            decay_steps=config.get("decay_steps", 0),
            decay_rate=config.get("decay_rate", 0.96)
        )
    
    elif schedule_name.lower() == "cosine":
        return CosineDecaySchedule(
            initial_learning_rate=base_learning_rate,
            min_learning_rate=config.get("min_lr", base_learning_rate * 0.01),
            total_steps=config.get("total_steps", 10000),
            warmup_steps=config.get("warmup_steps", 500)
        )
    
    elif schedule_name.lower() == "exponential":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=base_learning_rate,
            decay_steps=config.get("decay_steps", 1000),
            decay_rate=config.get("decay_rate", 0.96),
            staircase=config.get("staircase", False)
        )
    
    elif schedule_name.lower() == "polynomial":
        return tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=base_learning_rate,
            decay_steps=config.get("decay_steps", 1000),
            end_learning_rate=config.get("end_lr", base_learning_rate * 0.01),
            power=config.get("power", 1.0),
            cycle=config.get("cycle", False)
        )
    
    else:
        raise ValueError(f"Unknown learning rate schedule: {schedule_name}")