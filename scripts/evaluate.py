#!/usr/bin/env python3
"""
Evaluate a trained model.

Computes comprehensive evaluation metrics, generates visualizations,
and produces detailed reports.

Usage:
    python scripts/evaluate.py --model-path models/best_model.h5 --test-data data/processed/test/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate.py --model-path models/best.h5 --test-data data/processed/test/
  
  # With detailed report
  python scripts/evaluate.py \\
      --model-path models/best.h5 \\
      --test-data data/processed/test/ \\
      --output-dir results/evaluation/ \\
      --generate-report
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
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
    
    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        return False
    
    # Check test data exists
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data not found: {args.test_data}")
        return False
    
    return True


def evaluate_model(args):
    """Evaluate trained model."""
    logger = get_logger(__name__)
    
    try:
        # Validate inputs
        if not validate_inputs(args):
            return 1
        
        # Setup output directory
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Test data: {args.test_data}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Generate report: {args.generate_report}")
        logger.info(f"Generate plots: {args.generate_plots}")

        # Import necessary modules
        import numpy as np
        from pathlib import Path
        from src.evaluation.metrics import ModelMetrics
        from src.evaluation.visualizer import ResultVisualizer
        from src.evaluation.report_generator import ReportGenerator
        import tensorflow as tf
        import joblib

        # Load test data
        logger.info("Loading test data...")
        test_path = Path(args.test_data)

        # Try to load both signal and FFT data if available
        test_files = list(test_path.glob("*.npy"))
        if test_path.is_dir():
            X_test = np.load(test_path / "signals.npy")
            y_test = np.load(test_path / "labels.npy")
        else:
            logger.error(f"Test data directory not found: {args.test_data}")
            return 1

        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Test labels shape: {y_test.shape}")

        # Load model
        logger.info("Loading model...")
        model_path_obj = Path(args.model_path)

        if model_path_obj.suffix in ['.h5', '.keras']:
            # Deep learning model
            model = tf.keras.models.load_model(str(model_path_obj))
            is_deep_learning = True
        elif model_path_obj.suffix in ['.pkl', '.joblib']:
            # Traditional ML model
            model = joblib.load(str(model_path_obj))
            is_deep_learning = False
            # Flatten features for traditional ML
            if len(X_test.shape) > 2:
                X_test = X_test.reshape(X_test.shape[0], -1)
        else:
            logger.error(f"Unsupported model format: {model_path_obj.suffix}")
            return 1

        # Run predictions
        logger.info("Running predictions on test set...")
        if is_deep_learning:
            y_pred_proba = model.predict(X_test, batch_size=args.batch_size, verbose=1)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            try:
                y_pred_proba = model.predict_proba(X_test)
            except:
                y_pred_proba = None

        logger.info("Computing evaluation metrics...")

        # Compute metrics
        metrics = ModelMetrics(y_test, y_pred, y_pred_proba)

        accuracy = metrics.accuracy()
        precision = metrics.precision(average='weighted')
        recall = metrics.recall(average='weighted')
        f1 = metrics.f1(average='weighted')
        conf_matrix = metrics.confusion_matrix()
        per_class = metrics.per_class_metrics()

        # Log metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision (weighted): {precision:.4f}")
        logger.info(f"Recall (weighted): {recall:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")

        logger.info("\nPer-class metrics:")
        for cls, cls_metrics in per_class.items():
            logger.info(f"  Class {cls}:")
            logger.info(f"    Precision: {cls_metrics['precision']:.4f}")
            logger.info(f"    Recall: {cls_metrics['recall']:.4f}")
            logger.info(f"    F1: {cls_metrics['f1']:.4f}")

        # Save metrics to JSON
        metrics_dict = {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'per_class_metrics': {str(k): {mk: float(mv) for mk, mv in v.items()}
                                  for k, v in per_class.items()},
            'confusion_matrix': conf_matrix.tolist()
        }

        import json
        metrics_file = output_path / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        # Generate visualizations if requested
        if args.generate_plots:
            logger.info("Generating visualization plots...")
            visualizer = ResultVisualizer()

            # Confusion matrix plot
            cm_path = output_path / "confusion_matrix.png"
            visualizer.plot_confusion_matrix(
                conf_matrix,
                class_names=[f"Class {i}" for i in range(len(conf_matrix))],
                save_path=str(cm_path)
            )
            logger.info(f"Confusion matrix saved to {cm_path}")

            # Class distribution plot
            dist_path = output_path / "class_distribution.png"
            visualizer.plot_class_distribution(
                y_test,
                class_names=[f"Class {i}" for i in range(len(np.unique(y_test)))],
                save_path=str(dist_path)
            )
            logger.info(f"Class distribution saved to {dist_path}")

        # Generate report if requested
        if args.generate_report:
            logger.info("Generating evaluation report...")
            report_gen = ReportGenerator()

            results = {
                'model_path': str(args.model_path),
                'test_data': str(args.test_data),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist(),
                'per_class_metrics': per_class
            }

            # Generate HTML report
            html_path = output_path / "evaluation_report.html"
            report_gen.generate_html_report(results, str(html_path))
            logger.info(f"HTML report saved to {html_path}")

            # Generate markdown report
            md_path = output_path / "evaluation_report.md"
            report_gen.generate_markdown_report(results, str(md_path))
            logger.info(f"Markdown report saved to {md_path}")

        logger.info("Evaluation completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("EVALUATION - Evaluate Trained Model")
    logger.info("=" * 60)
    
    return evaluate_model(args)


if __name__ == "__main__":
    sys.exit(main())