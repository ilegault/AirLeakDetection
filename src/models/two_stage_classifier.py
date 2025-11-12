"""
Two-stage classifier that combines accelerometer identification and hole size detection.

Stage 1: Identifies which accelerometer (0, 1, 2) the sample is from
Stage 2: Uses accelerometer-specific hole size classifier to predict leak size

This enables the system to:
1. Determine which sensor the measurement came from
2. Apply the optimal leak detection model for that specific sensor position
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class TwoStageClassifier:
    """Two-stage classifier for accelerometer identification and leak detection.

    Architecture:
        Input Signal
            ↓
        [Stage 1: Accelerometer Classifier]
            ↓
        Accelerometer ID (0, 1, or 2)
            ↓
        [Stage 2: Hole Size Classifier (per accelerometer)]
            ↓
        Hole Size Prediction (NOLEAK, 1_16, 3_32, 1_8)

    Attributes:
        accelerometer_classifier: Model to identify accelerometer (0, 1, 2)
        hole_size_classifiers: Dict mapping accelerometer_id -> hole size classifier
        class_names: Dict mapping class id -> class name for hole sizes
        accelerometer_names: Dict mapping accelerometer id -> name
    """

    def __init__(
        self,
        accelerometer_classifier_path: Optional[str] = None,
        hole_size_classifier_paths: Optional[Dict[int, str]] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        """Initialize the two-stage classifier.

        Args:
            accelerometer_classifier_path: Path to saved accelerometer classifier (.pkl)
            hole_size_classifier_paths: Dict mapping accelerometer_id -> classifier path
                Example: {0: "models/accel_0_rf.pkl", 1: "models/accel_1_rf.pkl", ...}
            class_names: Dict mapping hole size class id -> name
                Example: {0: "NOLEAK", 1: "1_16", 2: "3_32", 3: "1_8"}
        """
        self.accelerometer_classifier = None
        self.hole_size_classifiers: Dict[int, any] = {}
        self.class_names = class_names or {
            0: "NOLEAK",
            1: "1_16",
            2: "3_32",
            3: "1_8"
        }
        self.accelerometer_names = {
            0: "Accelerometer 0 (Closest)",
            1: "Accelerometer 1 (Middle)",
            2: "Accelerometer 2 (Farthest)"
        }

        # Load models if paths provided
        if accelerometer_classifier_path:
            self.load_accelerometer_classifier(accelerometer_classifier_path)

        if hole_size_classifier_paths:
            self.load_hole_size_classifiers(hole_size_classifier_paths)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_accelerometer_classifier(self, model_path: str) -> None:
        """Load the accelerometer identification classifier.

        Args:
            model_path: Path to saved accelerometer classifier (.pkl)
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Accelerometer classifier not found: {model_path}")

        logger.info(f"Loading accelerometer classifier from {model_path}")
        self.accelerometer_classifier = joblib.load(model_path)
        logger.info("Accelerometer classifier loaded successfully")

    def load_hole_size_classifiers(self, classifier_paths: Dict[int, str]) -> None:
        """Load hole size classifiers for each accelerometer.

        Args:
            classifier_paths: Dict mapping accelerometer_id -> classifier path
                Example: {0: "models/accel_0_rf.pkl", 1: "models/accel_1_rf.pkl", ...}
        """
        for accel_id, model_path in classifier_paths.items():
            # Convert string keys to integers (handles JSON loading which converts keys to strings)
            accel_id_int = int(accel_id)
            
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                logger.warning(f"Hole size classifier not found for accelerometer {accel_id_int}: {model_path}")
                continue

            logger.info(f"Loading hole size classifier for accelerometer {accel_id_int} from {model_path}")
            self.hole_size_classifiers[accel_id_int] = joblib.load(model_path)
            logger.info(f"  Loaded classifier for {self.accelerometer_names[accel_id_int]}")

        if not self.hole_size_classifiers:
            logger.warning("No hole size classifiers loaded!")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_single(self, features: np.ndarray) -> Dict:
        """Predict accelerometer ID and hole size for a single sample.

        Args:
            features: Feature array of shape (n_features,)

        Returns:
            Dictionary with:
            - 'accelerometer_id': Predicted accelerometer ID (0, 1, 2)
            - 'accelerometer_name': Accelerometer name
            - 'accelerometer_confidence': Confidence scores [0, 1, 2]
            - 'hole_size_id': Predicted hole size class ID
            - 'hole_size_name': Predicted hole size class name
            - 'hole_size_confidence': Confidence scores for hole sizes
            - 'combined_confidence': Overall confidence in the prediction
        """
        if self.accelerometer_classifier is None:
            raise ValueError("Accelerometer classifier not loaded")

        # Ensure 2D shape for sklearn
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Stage 1: Predict accelerometer ID
        accel_id = self.accelerometer_classifier.predict(features)[0]
        accel_proba = self.accelerometer_classifier.predict_proba(features)[0]

        result = {
            'accelerometer_id': int(accel_id),
            'accelerometer_name': self.accelerometer_names.get(int(accel_id), f"Unknown ({accel_id})"),
            'accelerometer_confidence': accel_proba.tolist(),
        }

        # Stage 2: Predict hole size using accelerometer-specific classifier
        if accel_id in self.hole_size_classifiers:
            hole_classifier = self.hole_size_classifiers[accel_id]
            hole_id = hole_classifier.predict(features)[0]
            hole_proba = hole_classifier.predict_proba(features)[0]

            result['hole_size_id'] = int(hole_id)
            result['hole_size_name'] = self.class_names.get(int(hole_id), f"Unknown ({hole_id})")
            result['hole_size_confidence'] = hole_proba.tolist()

            # Combined confidence: product of both stage confidences
            result['combined_confidence'] = float(accel_proba[accel_id] * hole_proba[hole_id])
        else:
            logger.warning(f"No hole size classifier for accelerometer {accel_id}")
            result['hole_size_id'] = -1
            result['hole_size_name'] = "Unknown"
            result['hole_size_confidence'] = []
            result['combined_confidence'] = 0.0

        return result

    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """Predict accelerometer ID and hole size for multiple samples.

        Args:
            features: Feature array of shape (n_samples, n_features)

        Returns:
            List of prediction dictionaries (one per sample)
        """
        if self.accelerometer_classifier is None:
            raise ValueError("Accelerometer classifier not loaded")

        # Ensure 2D shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        results = []
        for i in range(features.shape[0]):
            result = self.predict_single(features[i])
            results.append(result)

        return results

    def predict(
        self,
        features: np.ndarray,
        return_confidence: bool = True
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Predict accelerometer IDs and hole sizes for batch input.

        Args:
            features: Feature array of shape (n_samples, n_features)
            return_confidence: If True, return confidence scores

        Returns:
            If return_confidence=False:
                (accelerometer_ids, hole_size_ids)
            If return_confidence=True:
                (accelerometer_ids, hole_size_ids, combined_confidences)
        """
        predictions = self.predict_batch(features)

        accel_ids = np.array([p['accelerometer_id'] for p in predictions])
        hole_ids = np.array([p['hole_size_id'] for p in predictions])
        confidences = np.array([p['combined_confidence'] for p in predictions])

        if return_confidence:
            return accel_ids, hole_ids, confidences
        else:
            return accel_ids, hole_ids

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        features: np.ndarray,
        true_accel_ids: np.ndarray,
        true_hole_ids: np.ndarray
    ) -> Dict:
        """Evaluate the two-stage classifier.

        Args:
            features: Feature array of shape (n_samples, n_features)
            true_accel_ids: True accelerometer IDs
            true_hole_ids: True hole size class IDs

        Returns:
            Dictionary with evaluation metrics:
            - 'stage1_accuracy': Accelerometer identification accuracy
            - 'stage2_accuracy': Hole size prediction accuracy (given correct accelerometer)
            - 'overall_accuracy': End-to-end accuracy
            - 'per_accelerometer_accuracy': Dict of accuracies per accelerometer
        """
        pred_accel_ids, pred_hole_ids, confidences = self.predict(features, return_confidence=True)

        # Stage 1 accuracy: accelerometer identification
        stage1_accuracy = np.mean(pred_accel_ids == true_accel_ids)

        # Stage 2 accuracy: hole size prediction (only for correctly identified accelerometers)
        correct_accel_mask = pred_accel_ids == true_accel_ids
        if np.sum(correct_accel_mask) > 0:
            stage2_accuracy = np.mean(
                pred_hole_ids[correct_accel_mask] == true_hole_ids[correct_accel_mask]
            )
        else:
            stage2_accuracy = 0.0

        # Overall accuracy: both stages correct
        overall_accuracy = np.mean(
            (pred_accel_ids == true_accel_ids) & (pred_hole_ids == true_hole_ids)
        )

        # Per-accelerometer accuracy
        per_accel_accuracy = {}
        for accel_id in np.unique(true_accel_ids):
            mask = true_accel_ids == accel_id
            if np.sum(mask) > 0:
                accel_correct = np.mean(
                    (pred_accel_ids[mask] == true_accel_ids[mask]) &
                    (pred_hole_ids[mask] == true_hole_ids[mask])
                )
                per_accel_accuracy[int(accel_id)] = float(accel_correct)

        results = {
            'stage1_accuracy': float(stage1_accuracy),
            'stage2_accuracy': float(stage2_accuracy),
            'overall_accuracy': float(overall_accuracy),
            'per_accelerometer_accuracy': per_accel_accuracy,
            'mean_confidence': float(np.mean(confidences)),
        }

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(self, output_path: str) -> None:
        """Save configuration (model paths, class names, etc.) to JSON.

        Args:
            output_path: Path to save configuration (.json)
        """
        config = {
            'accelerometer_names': self.accelerometer_names,
            'class_names': self.class_names,
            'hole_size_classifier_accelerometers': list(self.hole_size_classifiers.keys()),
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved to {output_path}")

    @classmethod
    def from_config(
        cls,
        config_path: str,
        accelerometer_classifier_path: str,
        hole_size_classifier_dir: str
    ) -> TwoStageClassifier:
        """Load a two-stage classifier from configuration.

        Args:
            config_path: Path to configuration JSON
            accelerometer_classifier_path: Path to accelerometer classifier
            hole_size_classifier_dir: Directory containing hole size classifiers
                Expected files: accel_0_classifier.pkl, accel_1_classifier.pkl, accel_2_classifier.pkl

        Returns:
            Loaded TwoStageClassifier instance
        """
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Build hole size classifier paths
        hole_classifier_dir = Path(hole_size_classifier_dir)
        hole_classifier_paths = {}

        for accel_id in config.get('hole_size_classifier_accelerometers', [0, 1, 2]):
            model_path = hole_classifier_dir / f"accel_{accel_id}_classifier.pkl"
            if model_path.exists():
                hole_classifier_paths[accel_id] = str(model_path)

        # Create classifier
        classifier = cls(
            accelerometer_classifier_path=accelerometer_classifier_path,
            hole_size_classifier_paths=hole_classifier_paths,
            class_names=config.get('class_names')
        )

        return classifier
