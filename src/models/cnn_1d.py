import tensorflow as tf
from tensorflow.keras import layers, models


class CNN1D:
    """1D CNN for leak detection"""

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def build_model(self, input_shape: tuple, n_classes: int):
        """Build 1D CNN architecture"""

        model = models.Sequential([
            # Input layer
            layers.Input(shape=input_shape),

            # Conv Block 1
            layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Conv Block 2
            layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Conv Block 3
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),

            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),

            # Output layer
            layers.Dense(n_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        )

        self.model = model
        return model