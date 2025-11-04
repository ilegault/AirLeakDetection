"""Large-scale batch processing with parallel prediction support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.prediction.predictor import LeakDetector


class BatchProcessor:
    """Process large batches of data with parallel prediction."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor: Optional[Any] = None,
        class_names: Optional[Dict[int, str]] = None,
        batch_size: int = 32,
        n_workers: int = 4,
    ) -> None:
        """Initialize batch processor.

        Args:
            model_path: Path to trained model file
            preprocessor: Preprocessor instance
            class_names: Dictionary mapping class indices to names
            batch_size: Size of mini-batches for processing
            n_workers: Number of parallel workers
        """
        self.leak_detector = LeakDetector(model_path, preprocessor, class_names)
        self.batch_size = batch_size
        self.n_workers = n_workers

    def process_batch(
        self, data: np.ndarray, show_progress: bool = True
    ) -> Dict[str, Any]:
        """Process a batch of samples.

        Args:
            data: Batch data (n_samples, features) or (n_samples, time, channels)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with batch predictions and statistics
        """
        n_samples = data.shape[0]
        all_predictions = []
        all_confidences = []
        all_probabilities = []

        # Process in mini-batches
        iterator = range(0, n_samples, self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=(n_samples + self.batch_size - 1) // self.batch_size,
                desc="Processing batch",
            )

        for i in iterator:
            end_idx = min(i + self.batch_size, n_samples)
            mini_batch = data[i:end_idx]

            result = self.leak_detector.predict_batch(mini_batch)

            all_predictions.extend(result["predictions"])
            all_confidences.extend(result["confidences"])
            all_probabilities.extend(result["probabilities"])

        return {
            "predictions": all_predictions,
            "confidences": all_confidences,
            "class_names": [self.leak_detector.class_names.get(int(p), "Unknown") for p in all_predictions],
            "probabilities": all_probabilities,
            "mean_confidence": float(np.mean(all_confidences)),
            "std_confidence": float(np.std(all_confidences)),
            "min_confidence": float(np.min(all_confidences)),
            "max_confidence": float(np.max(all_confidences)),
            "n_samples": len(all_predictions),
        }

    def process_files(
        self,
        file_paths: list[str | Path],
        loader_func,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Process multiple files in parallel.

        Args:
            file_paths: List of file paths to process
            loader_func: Function to load data from file (takes file_path, returns data)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with aggregated results
        """
        all_predictions = []
        all_confidences = []
        all_class_names = []
        file_results = []

        iterator = file_paths
        if show_progress:
            iterator = tqdm(file_paths, desc="Processing files")

        for file_path in iterator:
            try:
                # Load data
                data = loader_func(file_path)

                # Predict
                result = self.leak_detector.predict_batch(data)

                all_predictions.extend(result["predictions"])
                all_confidences.extend(result["confidences"])
                all_class_names.extend(result["class_names"])

                file_results.append({
                    "file": str(file_path),
                    "n_samples": result["n_samples"] if "n_samples" in result else len(result["predictions"]),
                    "predictions": result["predictions"],
                    "mean_confidence": result["mean_confidence"],
                })
            except Exception as e:
                file_results.append({
                    "file": str(file_path),
                    "error": str(e),
                })

        return {
            "total_predictions": len(all_predictions),
            "predictions": all_predictions,
            "confidences": all_confidences,
            "class_names": all_class_names,
            "mean_confidence": float(np.mean(all_confidences)) if all_confidences else 0.0,
            "std_confidence": float(np.std(all_confidences)) if all_confidences else 0.0,
            "file_results": file_results,
        }

    def save_predictions(
        self,
        predictions: Dict[str, Any],
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """Save predictions to file.

        Args:
            predictions: Predictions dictionary from process_batch or process_files
            output_path: Path to save predictions
            format: Output format ('json', 'csv', 'npz')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            import json
            with open(output_path, "w") as f:
                json.dump(predictions, f, indent=2)
        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame({
                "predictions": predictions["predictions"],
                "confidences": predictions["confidences"],
                "class_names": predictions["class_names"],
            })
            df.to_csv(output_path, index=False)
        elif format == "npz":
            np.savez(
                output_path,
                predictions=predictions["predictions"],
                confidences=predictions["confidences"],
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_predictions(self, file_path: str | Path) -> Dict[str, Any]:
        """Load predictions from file.

        Args:
            file_path: Path to predictions file

        Returns:
            Predictions dictionary
        """
        file_path = Path(file_path)

        if file_path.suffix == ".json":
            import json
            with open(file_path, "r") as f:
                return json.load(f)
        elif file_path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_dict("list")
        elif file_path.suffix == ".npz":
            data = np.load(file_path)
            return {key: data[key].tolist() for key in data.files}
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")


class ParallelBatchProcessor:
    """Process batches in parallel using thread pool."""

    def __init__(
        self,
        model_path: str | Path,
        preprocessor: Optional[Any] = None,
        class_names: Optional[Dict[int, str]] = None,
        n_workers: int = 4,
    ) -> None:
        """Initialize parallel batch processor.

        Args:
            model_path: Path to trained model file
            preprocessor: Preprocessor instance
            class_names: Dictionary mapping class indices to names
            n_workers: Number of parallel workers
        """
        self.model_path = model_path
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.n_workers = n_workers

    def _predict_chunk(self, chunk: tuple[int, np.ndarray]) -> tuple[int, Dict[str, Any]]:
        """Predict a single chunk of data.

        Args:
            chunk: Tuple of (chunk_id, data)

        Returns:
            Tuple of (chunk_id, predictions)
        """
        chunk_id, data = chunk
        detector = LeakDetector(self.model_path, self.preprocessor, self.class_names)
        return chunk_id, detector.predict_batch(data)

    def process_parallel(
        self,
        data: np.ndarray,
        chunk_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """Process data in parallel chunks.

        Args:
            data: Full dataset (n_samples, features)
            chunk_size: Size of each chunk (default: auto)
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with aggregated predictions
        """
        if chunk_size is None:
            chunk_size = max(1, len(data) // (self.n_workers * 4))

        # Create chunks
        chunks = []
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk_id = len(chunks)
            chunks.append((chunk_id, data[i:end_idx]))

        # Process chunks in parallel
        all_results = {}
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(self._predict_chunk, chunk): chunk[0] for chunk in chunks}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Processing chunks")

            for future in iterator:
                chunk_id, result = future.result()
                all_results[chunk_id] = result

        # Aggregate results
        all_predictions = []
        all_confidences = []
        all_class_names = []
        all_probabilities = []

        for chunk_id in sorted(all_results.keys()):
            result = all_results[chunk_id]
            all_predictions.extend(result["predictions"])
            all_confidences.extend(result["confidences"])
            all_class_names.extend(result["class_names"])
            all_probabilities.extend(result["probabilities"])

        return {
            "predictions": all_predictions,
            "confidences": all_confidences,
            "class_names": all_class_names,
            "probabilities": all_probabilities,
            "mean_confidence": float(np.mean(all_confidences)),
            "std_confidence": float(np.std(all_confidences)),
            "n_chunks": len(chunks),
            "n_samples": len(all_predictions),
        }