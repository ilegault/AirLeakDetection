#!/usr/bin/env python3
"""
Run inference with a trained model.

Performs predictions on single files or batch data with optional
confidence filtering and multiple output formats.

Usage:
    python scripts/predict.py --model-path models/best_model.h5 --input data/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for prediction script."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on single file
  python scripts/predict.py --model-path models/best.h5 --input data/sample.csv
  
  # Batch prediction
  python scripts/predict.py --model-path models/best.h5 --input data/test/
  
  # With confidence threshold
  python scripts/predict.py --model-path models/best.h5 --input data/test/ --confidence-threshold 0.8
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model file"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file or directory"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/predictions.json",
        help="Output file for predictions (default: results/predictions.json)"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "txt"],
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (default: 0.0)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    return parser


def validate_inputs(args) -> bool:
    """Validate command-line arguments."""
    logger = get_logger(__name__)
    
    # Check model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {args.model_path}")
        return False
    
    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        return False
    
    # Validate confidence threshold
    if not (0 <= args.confidence_threshold <= 1):
        logger.error(f"Confidence threshold must be between 0 and 1: {args.confidence_threshold}")
        return False
    
    # Validate batch size
    if args.batch_size <= 0:
        logger.error(f"Batch size must be positive: {args.batch_size}")
        return False
    
    return True


def run_predictions(args):
    """Run predictions with trained model."""
    logger = get_logger(__name__)
    
    try:
        # Validate inputs
        if not validate_inputs(args):
            return 1
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Confidence threshold: {args.confidence_threshold}")
        logger.info(f"Output format: {args.output_format}")

        # Create output directory
        output_path = Path(args.output)
        FileUtils.ensure_directory(str(output_path.parent))

        # Import necessary modules
        import numpy as np
        import json
        import csv
        from src.prediction.predictor import LeakDetector
        from src.prediction.batch_processor import BatchProcessor

        # Initialize predictor
        logger.info("Loading model...")
        predictor = LeakDetector(model_path=args.model_path)

        input_path = Path(args.input)
        predictions = []

        # Check if input is a file or directory
        if input_path.is_file():
            logger.info(f"Running prediction on single file: {input_path}")

            # Load single file
            if input_path.suffix == '.npy':
                data = np.load(str(input_path))
            elif input_path.suffix == '.csv':
                data = np.loadtxt(str(input_path), delimiter=',')
            else:
                logger.error(f"Unsupported file format: {input_path.suffix}")
                return 1

            # Run prediction
            result = predictor.predict_single(data)

            predictions.append({
                'file': str(input_path.name),
                'prediction': int(result['prediction']),
                'confidence': float(result['confidence']),
                'probabilities': result['probabilities'].tolist()
            })

            logger.info(f"Prediction: Class {result['prediction']}")
            logger.info(f"Confidence: {result['confidence']:.4f}")

        elif input_path.is_dir():
            logger.info(f"Running batch predictions on directory: {input_path}")

            # Find all data files in directory
            data_files = list(input_path.glob('*.npy')) + list(input_path.glob('*.csv'))

            if not data_files:
                logger.warning(f"No data files found in {input_path}")
                return 1

            logger.info(f"Found {len(data_files)} files to process")

            # Process files in batches
            batch_processor = BatchProcessor(predictor=predictor, batch_size=args.batch_size)

            for i, file_path in enumerate(data_files):
                try:
                    # Load file
                    if file_path.suffix == '.npy':
                        data = np.load(str(file_path))
                    elif file_path.suffix == '.csv':
                        data = np.loadtxt(str(file_path), delimiter=',')
                    else:
                        logger.warning(f"Skipping unsupported file: {file_path}")
                        continue

                    # Run prediction
                    result = predictor.predict_single(data)

                    # Apply confidence threshold
                    if result['confidence'] >= args.confidence_threshold:
                        predictions.append({
                            'file': str(file_path.name),
                            'prediction': int(result['prediction']),
                            'confidence': float(result['confidence']),
                            'probabilities': result['probabilities'].tolist()
                        })

                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(data_files)} files")

                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    continue

            logger.info(f"Successfully processed {len(predictions)} files")

        else:
            logger.error(f"Input path not found: {args.input}")
            return 1

        # Save predictions based on output format
        logger.info(f"Saving predictions to {output_path} in {args.output_format} format...")

        if args.output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)

        elif args.output_format == 'csv':
            with open(output_path, 'w', newline='') as f:
                if predictions:
                    fieldnames = ['file', 'prediction', 'confidence']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for pred in predictions:
                        writer.writerow({
                            'file': pred['file'],
                            'prediction': pred['prediction'],
                            'confidence': pred['confidence']
                        })

        elif args.output_format == 'txt':
            with open(output_path, 'w') as f:
                for pred in predictions:
                    f.write(f"{pred['file']}: Class {pred['prediction']} (confidence: {pred['confidence']:.4f})\n")

        logger.info(f"Saved {len(predictions)} predictions to {output_path}")
        logger.info("Predictions completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("PREDICTION - Run Inference with Trained Model")
    logger.info("=" * 60)
    
    return run_predictions(args)


if __name__ == "__main__":
    sys.exit(main())