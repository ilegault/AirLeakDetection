#!/usr/bin/env python3
"""
Perform k-fold cross-validation.

Evaluates model stability and generalization across multiple folds.

Usage:
    python scripts/cross_validate.py --model-type cnn_1d --data-path data/processed/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for cross-validation script."""
    parser = argparse.ArgumentParser(
        description="Perform k-fold cross-validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5-fold cross-validation
  python scripts/cross_validate.py --model-type cnn_1d --data-path data/processed/
  
  # With custom k-folds
  python scripts/cross_validate.py \\
      --model-type cnn_1d \\
      --data-path data/processed/ \\
      --k-folds 10
        """
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_1d",
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost"],
        help="Model type to cross-validate"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to processed data"
    )
    
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cross_validation/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified k-fold"
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
    
    if not Path(args.data_path).exists():
        logger.error(f"Data path not found: {args.data_path}")
        return False
    
    if args.k_folds < 2:
        logger.error(f"k-folds must be at least 2: {args.k_folds}")
        return False
    
    return True


def run_cross_validation(args):
    """Run k-fold cross-validation."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"K-folds: {args.k_folds}")
        logger.info(f"Stratified: {args.stratified}")

        # Import necessary modules
        import numpy as np
        import json
        from src.training.cross_validator import CrossValidator, evaluate_with_cross_validation
        from src.models.cnn_1d import CNN1DBuilder
        from src.models.cnn_2d import CNN2DBuilder
        from src.models.lstm_model import LSTMBuilder
        from src.models.random_forest import RandomForestModel
        from src.models.svm_classifier import SVMClassifier

        # Load data
        logger.info("Loading data...")
        data_path_obj = Path(args.data_path)

        # Load training and validation data (combine for cross-validation)
        X_train = np.load(data_path_obj / "train" / "signals.npy")
        y_train = np.load(data_path_obj / "train" / "labels.npy")
        X_val = np.load(data_path_obj / "val" / "signals.npy")
        y_val = np.load(data_path_obj / "val" / "labels.npy")

        # Combine train and validation for cross-validation
        X = np.concatenate([X_train, X_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)

        logger.info(f"Total data shape: {X.shape}")
        logger.info(f"Number of classes: {len(np.unique(y))}")

        num_classes = len(np.unique(y))

        # Create model builder function
        def create_model():
            """Factory function to create model instances."""
            if args.model_type == "cnn_1d":
                builder = CNN1DBuilder()
                model = builder.build(
                    input_shape=X.shape[1:],
                    num_classes=num_classes,
                    conv_filters=[64, 128, 256],
                    kernel_sizes=[3, 3, 3],
                    dense_units=[128, 64]
                )
                return model

            elif args.model_type == "cnn_2d":
                builder = CNN2DBuilder()
                X_reshaped = X.reshape(X.shape[0], X.shape[1], 1, 1) if len(X.shape) == 2 else X
                model = builder.build(
                    input_shape=X_reshaped.shape[1:],
                    num_classes=num_classes,
                    conv_filters=[32, 64, 128],
                    kernel_sizes=[(3, 3), (3, 3), (3, 3)],
                    dense_units=[128, 64]
                )
                return model

            elif args.model_type == "lstm":
                builder = LSTMBuilder()
                X_reshaped = X.reshape(X.shape[0], X.shape[1], 1) if len(X.shape) == 2 else X
                model = builder.build(
                    input_shape=X_reshaped.shape[1:],
                    num_classes=num_classes,
                    lstm_units=[128, 64],
                    dense_units=[64]
                )
                return model

            elif args.model_type == "random_forest":
                return RandomForestModel(n_estimators=300, max_depth=None, n_jobs=-1)

            elif args.model_type == "svm":
                return SVMClassifier(kernel='rbf', C=1.0)

            else:
                raise ValueError(f"Unsupported model type: {args.model_type}")

        # Initialize cross-validator
        logger.info("Initializing cross-validator...")
        cv_method = 'stratified_kfold' if args.stratified else 'kfold'
        cross_validator = CrossValidator(
            n_splits=args.k_folds,
            method=cv_method,
            shuffle=True,
            random_state=42
        )

        # Run cross-validation
        logger.info(f"Running {args.k_folds}-fold cross-validation...")

        results = evaluate_with_cross_validation(
            model_fn=create_model,
            X=X,
            y=y,
            cv=cross_validator,
            epochs=50,  # Reduced epochs for cross-validation
            batch_size=32,
            verbose=1
        )

        # Log results
        logger.info("\nCross-validation results:")
        logger.info(f"Mean accuracy: {results['mean_accuracy']:.4f} (+/- {results['std_accuracy']:.4f})")
        logger.info(f"Mean precision: {results['mean_precision']:.4f} (+/- {results['std_precision']:.4f})")
        logger.info(f"Mean recall: {results['mean_recall']:.4f} (+/- {results['std_recall']:.4f})")
        logger.info(f"Mean F1 score: {results['mean_f1']:.4f} (+/- {results['std_f1']:.4f})")

        logger.info("\nFold-wise results:")
        for i, fold_result in enumerate(results['fold_results']):
            logger.info(f"  Fold {i + 1}:")
            logger.info(f"    Accuracy: {fold_result['accuracy']:.4f}")
            logger.info(f"    Precision: {fold_result['precision']:.4f}")
            logger.info(f"    Recall: {fold_result['recall']:.4f}")
            logger.info(f"    F1: {fold_result['f1']:.4f}")

        # Save results
        logger.info("Saving cross-validation results...")
        results_file = output_path / "cross_validation_results.json"

        # Convert numpy types to Python types for JSON serialization
        results_to_save = {
            'model_type': args.model_type,
            'k_folds': args.k_folds,
            'stratified': args.stratified,
            'mean_accuracy': float(results['mean_accuracy']),
            'std_accuracy': float(results['std_accuracy']),
            'mean_precision': float(results['mean_precision']),
            'std_precision': float(results['std_precision']),
            'mean_recall': float(results['mean_recall']),
            'std_recall': float(results['std_recall']),
            'mean_f1': float(results['mean_f1']),
            'std_f1': float(results['std_f1']),
            'fold_results': [
                {k: float(v) for k, v in fold.items()}
                for fold in results['fold_results']
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        logger.info("Cross-validation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Cross-validation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION - K-Fold Cross-Validation")
    logger.info("=" * 60)
    
    return run_cross_validation(args)


if __name__ == "__main__":
    sys.exit(main())