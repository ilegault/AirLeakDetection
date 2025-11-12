#!/usr/bin/env python3
"""
Evaluate all trained models and generate comparison report.

This script evaluates all 4 models (CNN-1D, LSTM, Random Forest, SVM)
and generates a comprehensive comparison report.

Usage:
    python scripts/evaluate_all_models.py
    python scripts/evaluate_all_models.py --output-dir results/model_comparison/
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models with default settings
  python scripts/evaluate_all_models.py
  
  # Custom output directory
  python scripts/evaluate_all_models.py --output-dir results/comparison/
  
  # Generate plots and reports
  python scripts/evaluate_all_models.py --generate-plots --generate-report
        """
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test/",
        help="Path to test data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison/",
        help="Output directory for comparison results"
    )
    
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots for each model"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML report for each model"
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


def evaluate_single_model(model_info: dict, args, logger) -> dict:
    """Evaluate a single model and return metrics."""
    import subprocess
    
    model_name = model_info['name']
    model_path = model_info['path']
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Model path: {model_path}")
    
    # Create output directory for this model
    output_dir = Path(args.output_dir) / model_name.lower().replace(' ', '_')
    FileUtils.ensure_directory(str(output_dir))
    
    # Build evaluation command
    cmd = [
        "python",
        "scripts/evaluate.py",
        "--model-path", str(model_path),
        "--test-data", args.test_data,
        "--output-dir", str(output_dir),
        "--batch-size", str(args.batch_size),
    ]
    
    if args.generate_plots:
        cmd.append("--generate-plots")
    
    if args.generate_report:
        cmd.append("--generate-report")
    
    if args.verbose:
        cmd.append("--verbose")
    
    try:
        # Run evaluation
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=False
        )
        
        if result.returncode != 0:
            logger.error(f"✗ {model_name} evaluation failed")
            return None
        
        # Load metrics
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            logger.info(f"✓ {model_name} evaluation completed")
            return {
                'name': model_name,
                'path': str(model_path),
                'metrics': metrics,
                'output_dir': str(output_dir)
            }
        else:
            logger.error(f"✗ Metrics file not found for {model_name}")
            return None
            
    except Exception as e:
        logger.error(f"✗ Failed to evaluate {model_name}: {e}")
        return None


def generate_comparison_report(results: list, output_dir: Path, logger):
    """Generate comparison report for all models."""
    logger.info("\n" + "="*60)
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("="*60)
    
    # Create comparison table
    comparison_data = []
    for result in results:
        if result is None:
            continue
        
        metrics = result['metrics']
        comparison_data.append({
            'Model': result['name'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision_weighted'],
            'Recall': metrics['recall_weighted'],
            'F1 Score': metrics['f1_weighted']
        })
    
    # Sort by F1 score
    comparison_data.sort(key=lambda x: x['F1 Score'], reverse=True)
    
    # Print comparison table
    logger.info("\nModel Performance Comparison:")
    logger.info("-" * 80)
    logger.info(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10}")
    logger.info("-" * 80)
    
    for data in comparison_data:
        logger.info(
            f"{data['Model']:<20} "
            f"{data['Accuracy']:>10.4f} "
            f"{data['Precision']:>10.4f} "
            f"{data['Recall']:>10.4f} "
            f"{data['F1 Score']:>10.4f}"
        )
    logger.info("-" * 80)
    
    # Save comparison to JSON
    comparison_file = output_dir / "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models': comparison_data,
            'best_model': comparison_data[0]['Model'] if comparison_data else None
        }, f, indent=2)
    
    logger.info(f"\nComparison saved to: {comparison_file}")
    
    # Generate markdown report
    md_file = output_dir / "model_comparison.md"
    with open(md_file, 'w') as f:
        f.write("# Model Performance Comparison\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        if comparison_data:
            f.write(f"**Best Model:** {comparison_data[0]['Model']} "
                   f"(F1 Score: {comparison_data[0]['F1 Score']:.4f})\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        
        for data in comparison_data:
            f.write(
                f"| {data['Model']} | "
                f"{data['Accuracy']:.4f} | "
                f"{data['Precision']:.4f} | "
                f"{data['Recall']:.4f} | "
                f"{data['F1 Score']:.4f} |\n"
            )
        
        f.write("\n## Detailed Results\n\n")
        for result in results:
            if result is None:
                continue
            f.write(f"### {result['name']}\n\n")
            f.write(f"- **Model Path:** `{result['path']}`\n")
            f.write(f"- **Results Directory:** `{result['output_dir']}`\n")
            f.write(f"- **Accuracy:** {result['metrics']['accuracy']:.4f}\n")
            f.write(f"- **Precision (weighted):** {result['metrics']['precision_weighted']:.4f}\n")
            f.write(f"- **Recall (weighted):** {result['metrics']['recall_weighted']:.4f}\n")
            f.write(f"- **F1 Score (weighted):** {result['metrics']['f1_weighted']:.4f}\n\n")
    
    logger.info(f"Markdown report saved to: {md_file}")
    
    # Generate HTML report
    html_file = output_dir / "model_comparison.html"
    with open(html_file, 'w') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n")
        f.write("<title>Model Performance Comparison</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 40px; }\n")
        f.write("h1 { color: #333; }\n")
        f.write("table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n")
        f.write("th { background-color: #4CAF50; color: white; }\n")
        f.write("tr:nth-child(even) { background-color: #f2f2f2; }\n")
        f.write(".best { background-color: #d4edda !important; }\n")
        f.write(".metric { text-align: right; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        
        f.write("<h1>Model Performance Comparison</h1>\n")
        f.write(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        
        if comparison_data:
            f.write(f"<p><strong>Best Model:</strong> {comparison_data[0]['Model']} "
                   f"(F1 Score: {comparison_data[0]['F1 Score']:.4f})</p>\n")
        
        f.write("<h2>Performance Metrics</h2>\n")
        f.write("<table>\n")
        f.write("<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr>\n")
        
        for i, data in enumerate(comparison_data):
            row_class = ' class="best"' if i == 0 else ''
            f.write(f"<tr{row_class}>")
            f.write(f"<td>{data['Model']}</td>")
            f.write(f"<td class='metric'>{data['Accuracy']:.4f}</td>")
            f.write(f"<td class='metric'>{data['Precision']:.4f}</td>")
            f.write(f"<td class='metric'>{data['Recall']:.4f}</td>")
            f.write(f"<td class='metric'>{data['F1 Score']:.4f}</td>")
            f.write("</tr>\n")
        
        f.write("</table>\n")
        
        f.write("<h2>Detailed Results</h2>\n")
        for result in results:
            if result is None:
                continue
            f.write(f"<h3>{result['name']}</h3>\n")
            f.write("<ul>\n")
            f.write(f"<li><strong>Model Path:</strong> <code>{result['path']}</code></li>\n")
            f.write(f"<li><strong>Results Directory:</strong> <code>{result['output_dir']}</code></li>\n")
            f.write(f"<li><strong>Accuracy:</strong> {result['metrics']['accuracy']:.4f}</li>\n")
            f.write(f"<li><strong>Precision (weighted):</strong> {result['metrics']['precision_weighted']:.4f}</li>\n")
            f.write(f"<li><strong>Recall (weighted):</strong> {result['metrics']['recall_weighted']:.4f}</li>\n")
            f.write(f"<li><strong>F1 Score (weighted):</strong> {result['metrics']['f1_weighted']:.4f}</li>\n")
            f.write("</ul>\n")
        
        f.write("</body>\n</html>\n")
    
    logger.info(f"HTML report saved to: {html_file}")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("EVALUATE ALL MODELS")
    logger.info("=" * 60)
    
    # Setup output directory
    output_path = Path(args.output_dir)
    FileUtils.ensure_directory(str(output_path))
    
    # Define models to evaluate
    models = [
        {
            'name': 'CNN-1D',
            'path': Path('models/model_20251111_163055/cnn_1d_model.h5')
        },
        {
            'name': 'LSTM',
            'path': Path('models/model_20251111_163230/lstm_model.h5')
        },
        {
            'name': 'Random Forest',
            'path': Path('models/model_20251111_164809/random_forest_model.pkl')
        },
        {
            'name': 'SVM',
            'path': Path('models/model_20251111_164813/svm_model.pkl')
        }
    ]
    
    # Check which models exist
    available_models = []
    for model in models:
        if model['path'].exists():
            available_models.append(model)
            logger.info(f"✓ Found {model['name']}: {model['path']}")
        else:
            logger.warning(f"✗ Model not found: {model['name']} at {model['path']}")
    
    if not available_models:
        logger.error("No models found to evaluate!")
        return 1
    
    logger.info(f"\nEvaluating {len(available_models)} models...")
    
    # Evaluate each model
    results = []
    for model_info in available_models:
        result = evaluate_single_model(model_info, args, logger)
        results.append(result)
    
    # Generate comparison report
    valid_results = [r for r in results if r is not None]
    if valid_results:
        generate_comparison_report(valid_results, output_path, logger)
    else:
        logger.error("No successful evaluations to compare!")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())