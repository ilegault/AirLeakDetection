#!/usr/bin/env python3
"""
Hyperparameter optimization using various search strategies.

Supports grid search, random search, and Bayesian optimization.

Usage:
    python scripts/hyperparameter_search.py --model-type cnn_1d --search-method bayesian
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bayesian optimization
  python scripts/hyperparameter_search.py \\
      --model-type cnn_1d \\
      --search-method bayesian \\
      --n-trials 50
  
  # Grid search
  python scripts/hyperparameter_search.py \\
      --model-type cnn_1d \\
      --search-method grid
        """
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_1d",
        choices=["cnn_1d", "cnn_2d", "lstm", "random_forest", "svm", "xgboost"],
        help="Model type"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--search-method",
        type=str,
        default="bayesian",
        choices=["grid", "random", "bayesian"],
        help="Search method"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials/iterations"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparameter_search/",
        help="Output directory"
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
    
    if args.n_trials <= 0:
        logger.error(f"Number of trials must be positive: {args.n_trials}")
        return False
    
    if args.n_jobs <= 0:
        logger.error(f"Number of jobs must be positive: {args.n_jobs}")
        return False
    
    return True


def run_hyperparameter_search(args):
    """Run hyperparameter search."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Search method: {args.search_method}")
        logger.info(f"Number of trials: {args.n_trials}")
        logger.info(f"Parallel jobs: {args.n_jobs}")

        # Import necessary modules
        import numpy as np
        import json
        from src.training.hyperparameter_tuning import (
            GridSearchTuner, RandomSearchTuner, BayesianOptimizationTuner,
            suggest_hyperparameter_space
        )
        from src.models.cnn_1d import CNN1DBuilder
        from src.models.cnn_2d import CNN2DBuilder
        from src.models.lstm_model import LSTMBuilder
        from src.models.random_forest import RandomForestModel
        from src.models.svm_classifier import SVMClassifier

        # Load data
        logger.info("Loading data...")
        data_path_obj = Path(args.data_path)

        X_train = np.load(data_path_obj / "train" / "signals.npy")
        y_train = np.load(data_path_obj / "train" / "labels.npy")
        X_val = np.load(data_path_obj / "val" / "signals.npy")
        y_val = np.load(data_path_obj / "val" / "labels.npy")

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

        num_classes = len(np.unique(y_train))

        # Get suggested hyperparameter space for the model type
        param_space = suggest_hyperparameter_space(args.model_type)

        # Create model builder function
        def create_model(**hyperparams):
            """Factory function to create model with hyperparameters."""
            if args.model_type == "cnn_1d":
                builder = CNN1DBuilder()
                model = builder.build(
                    input_shape=X_train.shape[1:],
                    num_classes=num_classes,
                    conv_filters=hyperparams.get('conv_filters', [64, 128, 256]),
                    kernel_sizes=hyperparams.get('kernel_sizes', [3, 3, 3]),
                    dense_units=hyperparams.get('dense_units', [128, 64]),
                    dropout_rate=hyperparams.get('dropout_rate', 0.3)
                )
                return model

            elif args.model_type == "cnn_2d":
                builder = CNN2DBuilder()
                X_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1) if len(X_train.shape) == 2 else X_train
                model = builder.build(
                    input_shape=X_reshaped.shape[1:],
                    num_classes=num_classes,
                    conv_filters=hyperparams.get('conv_filters', [32, 64, 128]),
                    kernel_sizes=hyperparams.get('kernel_sizes', [(3, 3), (3, 3), (3, 3)]),
                    dense_units=hyperparams.get('dense_units', [128, 64]),
                    dropout_rate=hyperparams.get('dropout_rate', 0.3)
                )
                return model

            elif args.model_type == "lstm":
                builder = LSTMBuilder()
                X_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) if len(X_train.shape) == 2 else X_train
                model = builder.build(
                    input_shape=X_reshaped.shape[1:],
                    num_classes=num_classes,
                    lstm_units=hyperparams.get('lstm_units', [128, 64]),
                    dense_units=hyperparams.get('dense_units', [64]),
                    dropout_rate=hyperparams.get('dropout_rate', 0.3)
                )
                return model

            elif args.model_type == "random_forest":
                return RandomForestModel(
                    n_estimators=hyperparams.get('n_estimators', 300),
                    max_depth=hyperparams.get('max_depth', None),
                    min_samples_split=hyperparams.get('min_samples_split', 2),
                    n_jobs=args.n_jobs
                )

            elif args.model_type == "svm":
                return SVMClassifier(
                    kernel=hyperparams.get('kernel', 'rbf'),
                    C=hyperparams.get('C', 1.0),
                    gamma=hyperparams.get('gamma', 'scale')
                )

            else:
                raise ValueError(f"Unsupported model type: {args.model_type}")

        # Initialize tuner based on search method
        logger.info(f"Initializing {args.search_method} tuner...")

        if args.search_method == "grid":
            tuner = GridSearchTuner(
                model_fn=create_model,
                param_grid=param_space,
                cv=3,
                n_jobs=args.n_jobs
            )

        elif args.search_method == "random":
            tuner = RandomSearchTuner(
                model_fn=create_model,
                param_distributions=param_space,
                n_iter=args.n_trials,
                cv=3,
                n_jobs=args.n_jobs
            )

        elif args.search_method == "bayesian":
            tuner = BayesianOptimizationTuner(
                model_fn=create_model,
                param_space=param_space,
                n_trials=args.n_trials,
                cv=3
            )

        else:
            logger.error(f"Unsupported search method: {args.search_method}")
            return 1

        # Run hyperparameter search
        logger.info(f"Running hyperparameter search with {args.n_trials} trials...")

        best_params, best_score, all_results = tuner.search(
            X_train, y_train,
            X_val, y_val
        )

        # Log results
        logger.info("\nHyperparameter search results:")
        logger.info(f"Best score: {best_score:.4f}")
        logger.info("Best hyperparameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")

        # Save results
        logger.info("Saving hyperparameter search results...")
        results_file = output_path / "hyperparameter_search_results.json"

        results_to_save = {
            'model_type': args.model_type,
            'search_method': args.search_method,
            'n_trials': args.n_trials,
            'best_score': float(best_score),
            'best_params': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                           for k, v in best_params.items()},
            'all_results': [
                {
                    'params': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v
                              for k, v in result['params'].items()},
                    'score': float(result['score'])
                }
                for result in all_results
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Save best model with best hyperparameters
        logger.info("Training final model with best hyperparameters...")
        best_model = create_model(**best_params)

        if args.model_type in ["cnn_1d", "cnn_2d", "lstm"]:
            # Deep learning model
            from src.training.trainer import ModelTrainer
            trainer = ModelTrainer(model=best_model, model_name=args.model_type)
            trainer.compile(
                optimizer='adam',
                learning_rate=0.001,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            trainer.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32
            )
            model_path = output_path / f"best_{args.model_type}_model.h5"
            trainer.save_checkpoint(str(model_path))
        else:
            # Traditional ML model
            X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
            best_model.fit(X_train_flat, y_train)
            model_path = output_path / f"best_{args.model_type}_model.pkl"
            best_model.save(str(model_path))

        logger.info(f"Best model saved to {model_path}")
        logger.info("Hyperparameter search completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Hyperparameter search failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER SEARCH - Find Optimal Hyperparameters")
    logger.info("=" * 60)
    
    return run_hyperparameter_search(args)


if __name__ == "__main__":
    sys.exit(main())