"""LSTM-based model for sequential vibration data classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (LSTM, Bidirectional, Dense, Dropout,
                                     GlobalAveragePooling1D)


@dataclass
class LSTMHyperParams:
    """Hyperparameters for LSTM architecture."""

    lstm_units: tuple[int, ...]
    dense_units: tuple[int, ...]
    dropout_rates: tuple[float, ...]
    learning_rate: float
    bidirectional: bool


class LSTMBuilder:
    """Factory for building configurable LSTM models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {}).get("lstm", {})

        self.hyperparams = LSTMHyperParams(
            lstm_units=tuple(model_cfg.get("lstm_units", (64, 32))),
            dense_units=tuple(model_cfg.get("dense_units", (128, 64))),
            dropout_rates=tuple(model_cfg.get("dropout_rates", (0.2, 0.3, 0.3, 0.3))),
            learning_rate=float(training_cfg.get("learning_rate", 0.001)),
            bidirectional=bool(model_cfg.get("bidirectional", True)),
        )

        if len(self.hyperparams.dropout_rates) < len(self.hyperparams.lstm_units) + len(
            self.hyperparams.dense_units
        ):
            raise ValueError("Insufficient dropout rates provided for LSTM blocks")

    def build(self, input_shape: tuple[int, ...], n_classes: int) -> Model:
        """Construct the LSTM model."""
        inputs = Input(shape=input_shape, name="sequences")
        x = inputs

        # LSTM layers
        for idx, units in enumerate(self.hyperparams.lstm_units):
            return_sequences = idx < len(self.hyperparams.lstm_units) - 1
            if self.hyperparams.bidirectional:
                x = Bidirectional(
                    LSTM(units, return_sequences=return_sequences, activation="relu"),
                    name=f"bilstm_{idx+1}",
                )(x)
            else:
                x = LSTM(units, return_sequences=return_sequences, activation="relu", name=f"lstm_{idx+1}")(x)

            x = Dropout(rate=self.hyperparams.dropout_rates[idx], name=f"dropout_lstm_{idx+1}")(x)

        # Global pooling if needed (if output is 2D)
        if len(x.shape) > 2:
            x = GlobalAveragePooling1D(name="global_avg_pool")(x)

        # Dense classifier
        dense_offset = len(self.hyperparams.lstm_units)
        for idx, units in enumerate(self.hyperparams.dense_units):
            x = Dense(units, activation="relu", name=f"dense_{idx+1}")(x)
            dropout_rate = self.hyperparams.dropout_rates[dense_offset + idx]
            x = Dropout(rate=dropout_rate, name=f"dropout_dense_{idx+1}")(x)

        outputs = Dense(n_classes, activation="softmax", name="predictions")(x)
        model = Model(inputs=inputs, outputs=outputs, name="lstm_classifier")

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.hyperparams.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
        )

        return model