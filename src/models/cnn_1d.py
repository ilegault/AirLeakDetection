"""1D convolutional neural network architectures for vibration classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Dropout,
                                     GlobalAveragePooling1D, MaxPooling1D)


@dataclass
class CNNHyperParams:
    """Hyper-parameters controlling the CNN topology."""

    conv_filters: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    dense_units: tuple[int, ...]
    dropout_rates: tuple[float, ...]
    learning_rate: float


class CNN1DBuilder:
    """Factory for building configurable 1D CNN models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})

        self.hyperparams = CNNHyperParams(
            conv_filters=tuple(model_cfg.get("conv_filters", (32, 64, 128))),
            kernel_sizes=tuple(model_cfg.get("kernel_sizes", (7, 5, 3))),
            dense_units=tuple(model_cfg.get("dense_units", (256, 128))),
            dropout_rates=tuple(model_cfg.get("dropout_rates", (0.3, 0.3, 0.4, 0.3))),
            learning_rate=float(training_cfg.get("learning_rate", 0.001)),
        )

        if len(self.hyperparams.conv_filters) != len(self.hyperparams.kernel_sizes):
            raise ValueError("`conv_filters` and `kernel_sizes` must have the same length")

        if len(self.hyperparams.dropout_rates) < len(self.hyperparams.conv_filters) + len(self.hyperparams.dense_units):
            raise ValueError("Insufficient dropout rates provided for the CNN blocks")

    def build(self, input_shape: tuple[int, ...], n_classes: int) -> Model:
        """Construct the CNN model according to the configured hyperparameters."""
        inputs = Input(shape=input_shape, name="signals")
        x = inputs

        # Convolutional feature extractor
        for idx, (filters, kernel_size) in enumerate(zip(self.hyperparams.conv_filters, self.hyperparams.kernel_sizes)):
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation="relu", name=f"conv_{idx+1}")(x)
            x = BatchNormalization(name=f"bn_{idx+1}")(x)
            x = MaxPooling1D(pool_size=2, name=f"pool_{idx+1}")(x)
            x = Dropout(rate=self.hyperparams.dropout_rates[idx], name=f"dropout_conv_{idx+1}")(x)

        x = GlobalAveragePooling1D(name="global_avg_pool")(x)

        # Dense classifier
        dense_offset = len(self.hyperparams.conv_filters)
        for idx, units in enumerate(self.hyperparams.dense_units):
            x = Dense(units, activation="relu", name=f"dense_{idx+1}")(x)
            dropout_rate = self.hyperparams.dropout_rates[dense_offset + idx]
            x = Dropout(rate=dropout_rate, name=f"dropout_dense_{idx+1}")(x)

        outputs = Dense(n_classes, activation="softmax", name="predictions")(x)
        model = Model(inputs=inputs, outputs=outputs, name="cnn_1d_classifier")

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparams.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
        )

        return model