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

        # Import necessary modules
        import tensorflow as tf
        import numpy as np
        import joblib
        from pathlib import Path

        model_path_obj = Path(args.model_path)
        output_dir = Path(args.output_dir)

        # Load model
        logger.info("Loading model...")
        if model_path_obj.suffix in ['.h5', '.keras']:
            model = tf.keras.models.load_model(str(model_path_obj))
            is_keras_model = True
        elif model_path_obj.suffix in ['.pkl', '.joblib']:
            model = joblib.load(str(model_path_obj))
            is_keras_model = False
        else:
            logger.error(f"Unsupported model format: {model_path_obj.suffix}")
            return 1

        # Export based on format
        if args.format == "tflite":
            if not is_keras_model:
                logger.error("TFLite export only supports Keras models")
                return 1

            logger.info("Converting model to TensorFlow Lite...")
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Apply optimizations
            if args.optimization_level == "default":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif args.optimization_level == "lite":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            elif args.optimization_level == "aggressive":
                converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]

            # Apply quantization if requested
            if args.quantize:
                logger.info("Applying quantization...")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()

            # Save TFLite model
            output_file = output_dir / f"{model_path_obj.stem}.tflite"
            with open(output_file, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"TFLite model saved to {output_file}")

        elif args.format == "onnx":
            if not is_keras_model:
                logger.error("ONNX export only supports Keras models")
                return 1

            try:
                import tf2onnx
                logger.info("Converting model to ONNX...")

                # Convert to ONNX
                model_proto, _ = tf2onnx.convert.from_keras(model)

                output_file = output_dir / f"{model_path_obj.stem}.onnx"
                with open(output_file, 'wb') as f:
                    f.write(model_proto.SerializeToString())

                logger.info(f"ONNX model saved to {output_file}")

            except ImportError:
                logger.error("tf2onnx is not installed. Install with: pip install tf2onnx")
                return 1

        elif args.format == "torchscript":
            logger.error("TorchScript export requires PyTorch model, not currently supported")
            return 1

        elif args.format == "keras":
            if not is_keras_model:
                logger.error("Keras export only supports Keras models")
                return 1

            logger.info("Saving model in Keras format...")
            output_file = output_dir / f"{model_path_obj.stem}.keras"
            model.save(str(output_file))
            logger.info(f"Keras model saved to {output_file}")

        elif args.format == "pickle":
            if is_keras_model:
                logger.warning("Pickle export for Keras models is not recommended, use .h5 or .keras format")

            logger.info("Saving model in pickle format...")
            output_file = output_dir / f"{model_path_obj.stem}.pkl"
            joblib.dump(model, str(output_file))
            logger.info(f"Pickle model saved to {output_file}")

        else:
            logger.error(f"Unsupported export format: {args.format}")
            return 1

        # Save model metadata
        metadata = {
            'original_model': str(args.model_path),
            'export_format': args.format,
            'quantized': args.quantize,
            'optimization_level': args.optimization_level,
            'model_type': 'keras' if is_keras_model else 'sklearn'
        }

        if is_keras_model:
            metadata['input_shape'] = [int(d) for d in model.input_shape[1:]]
            metadata['output_shape'] = [int(d) for d in model.output_shape[1:]]

        import json
        metadata_file = output_dir / f"{model_path_obj.stem}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_file}")
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