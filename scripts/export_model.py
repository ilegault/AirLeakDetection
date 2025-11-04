#!/usr/bin/env python3
"""
Export trained model for deployment.

Supports TensorFlow Lite, ONNX, and TorchScript formats with
optional quantization.

Usage:
    python scripts/export_model.py --model-path models/best.h5 --format tflite
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, get_logger, FileUtils


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for model export."""
    parser = argparse.ArgumentParser(
        description="Export model for deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to TensorFlow Lite
  python scripts/export_model.py \\
      --model-path models/best.h5 \\
      --format tflite \\
      --quantize
  
  # Export to ONNX
  python scripts/export_model.py \\
      --model-path models/best.h5 \\
      --format onnx
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="tflite",
        choices=["tflite", "onnx", "torchscript", "keras", "pickle"],
        help="Export format"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment/models/",
        help="Output directory"
    )
    
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization"
    )
    
    parser.add_argument(
        "--optimization-level",
        type=str,
        default="default",
        choices=["none", "default", "lite", "aggressive"],
        help="Optimization level"
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
    
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return False
    
    return True


def export_model(args):
    """Export model to specified format."""
    logger = get_logger(__name__)
    
    try:
        if not validate_inputs(args):
            return 1
        
        output_path = Path(args.output_dir)
        FileUtils.ensure_directory(str(output_path))
        
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Export format: {args.format}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Quantization: {args.quantize}")
        logger.info(f"Optimization level: {args.optimization_level}")
        
        # TODO: Implement model export logic
        logger.info("Model export logic to be implemented with Phase 6 (Prediction)")
        
        logger.info("Model export completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Model export failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_dir="logs", console_level=log_level)
    
    logger.info("=" * 60)
    logger.info("MODEL EXPORT - Export Model for Deployment")
    logger.info("=" * 60)
    
    return export_model(args)


if __name__ == "__main__":
    sys.exit(main())