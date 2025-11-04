"""
Custom loss functions for air leak detection.

Provides specialized loss functions optimized for imbalanced multi-class
classification of air leak detection tasks.
"""

import tensorflow as tf
import numpy as np
from typing import Optional


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss for handling class imbalance.
    
    Focal loss applies a modulation term to the cross-entropy loss which helps
    focus learning on hard negative examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    
    Args:
        alpha: Weighting factor in [0, 1] to balance classes
        gamma: Focusing parameter for modulating loss (typically 2.0)
        from_logits: Whether predictions are logits or probabilities
        name: Loss name
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        name: str = "focal_loss"
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate focal loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities or logits
            
        Returns:
            Scalar loss value
        """
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        
        # Calculate cross entropy
        ce_loss = -y_true * tf.math.log(y_pred)
        
        # Get predicted class (for each sample, get max prob class)
        y_pred_max = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        
        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - y_pred_max) ** self.gamma
        
        # Apply alpha weighting
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


class WeightedCategoricalCrossentropy(tf.keras.losses.Loss):
    """Weighted categorical cross-entropy for class imbalance.
    
    Args:
        class_weights: Dictionary mapping class index to weight or array of weights
        name: Loss name
    """
    
    def __init__(
        self,
        class_weights: dict = None,
        name: str = "weighted_categorical_crossentropy"
    ):
        super().__init__(name=name)
        self.class_weights = class_weights or {}
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate weighted categorical cross-entropy loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Scalar loss value
        """
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        
        # Standard cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Apply class weights if provided
        if self.class_weights:
            # Convert dict to weights tensor
            n_classes = tf.shape(y_true)[-1]
            weights = tf.ones([n_classes])
            
            for class_idx, weight in self.class_weights.items():
                weights = tf.tensor_scatter_nd_update(
                    weights,
                    [[class_idx]],
                    [weight]
                )
            
            # Broadcast weights across batch
            weighted_ce = ce * weights
        else:
            weighted_ce = ce
        
        return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


class LeakAwareLoss(tf.keras.losses.Loss):
    """Custom loss penalizing leak misclassification more than non-leak.
    
    Assumes class 0 = No Leak, classes 1-3 = Leak types.
    Penalizes predicting "No Leak" when leak exists more heavily.
    
    Args:
        leak_penalty: Multiplier for false negative (predicting no leak when leak exists)
        non_leak_penalty: Multiplier for false positive (predicting leak when no leak)
        name: Loss name
    """
    
    def __init__(
        self,
        leak_penalty: float = 2.0,
        non_leak_penalty: float = 1.0,
        name: str = "leak_aware_loss"
    ):
        super().__init__(name=name)
        self.leak_penalty = leak_penalty
        self.non_leak_penalty = non_leak_penalty
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate leak-aware loss.
        
        Args:
            y_true: Ground truth labels (one-hot encoded)
            y_pred: Predicted probabilities
            
        Returns:
            Scalar loss value
        """
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        
        # Base cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Get true class indices (0 = no leak, 1-3 = leak)
        true_class = tf.argmax(y_true, axis=-1)
        is_leak = tf.cast(true_class > 0, tf.float32)  # 1 if leak, 0 if no leak
        
        # Apply penalty based on sample type
        penalty = (
            is_leak * self.leak_penalty +
            (1.0 - is_leak) * self.non_leak_penalty
        )
        
        # Broadcast penalty across classes
        penalty = tf.expand_dims(penalty, axis=-1)
        
        # Apply penalty
        weighted_ce = ce * penalty
        
        return tf.reduce_mean(tf.reduce_sum(weighted_ce, axis=-1))


def get_loss_function(
    loss_name: str = "categorical_crossentropy",
    config: dict = None,
    class_weights: dict = None
) -> tf.keras.losses.Loss:
    """
    Get a loss function by name with optional configuration.
    
    Args:
        loss_name: Name of loss function
        config: Configuration dictionary with hyperparameters
        class_weights: Class weights for imbalanced data
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss_name is not recognized
    """
    config = config or {}
    
    if loss_name == "focal_loss":
        return FocalLoss(
            alpha=config.get("alpha", 0.25),
            gamma=config.get("gamma", 2.0),
            from_logits=config.get("from_logits", False)
        )
    elif loss_name == "weighted_categorical_crossentropy":
        return WeightedCategoricalCrossentropy(class_weights=class_weights)
    elif loss_name == "leak_aware_loss":
        return LeakAwareLoss(
            leak_penalty=config.get("leak_penalty", 2.0),
            non_leak_penalty=config.get("non_leak_penalty", 1.0)
        )
    elif loss_name == "categorical_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    elif loss_name == "sparse_categorical_crossentropy":
        return tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")