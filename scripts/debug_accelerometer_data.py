#!/usr/bin/env python3
"""
Comprehensive diagnostic script for accelerometer classifier data.

This script performs Steps 3-7 of the debugging process:
- Step 3: Fix the data flow and verify accelerometer labels
- Step 4: Restructure training data check
- Step 5: Verify feature differences
- Step 6: Check the pipeline
- Step 7: Random Forest specific issues

Usage:
    python scripts/debug_accelerometer_data.py \\
        --accelerometer-data data/accelerometer_classifier/ \\
        --processed-data data/processed/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, FileUtils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Debug accelerometer classifier data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--accelerometer-data",
        type=str,
        required=True,
        help="Path to accelerometer classification data directory"
    )

    parser.add_argument(
        "--processed-data",
        type=str,
        required=True,
        help="Path to processed NPZ data directory"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug_output/",
        help="Output directory for diagnostic plots"
    )

    return parser


def step3_check_data_flow(accel_data_path: Path) -> Dict:
    """STEP 3: Fix the Data Flow - Verify accelerometer labels correspond to positions."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("STEP 3: CHECKING DATA FLOW AND ACCELEROMETER LABELS")
    LOGGER.info("="*80)

    results = {}

    # Load training data
    train_dir = accel_data_path / "train"
    if not train_dir.exists():
        LOGGER.error(f"Training directory not found: {train_dir}")
        return results

    X_train = np.load(train_dir / "features.npy")
    y_train = np.load(train_dir / "labels.npy")

    # Also check if hole_size_labels exist to verify data structure
    if (train_dir / "hole_size_labels.npy").exists():
        hole_labels = np.load(train_dir / "hole_size_labels.npy")
        LOGGER.info(f"Hole size labels found: {np.unique(hole_labels)}")
        LOGGER.info(f"This confirms we have one sample per accelerometer per recording")

    LOGGER.info(f"\nData structure verification:")
    LOGGER.info(f"  Total samples: {len(X_train)}")
    LOGGER.info(f"  Unique accelerometer labels: {np.unique(y_train)}")

    # Count samples per accelerometer
    unique, counts = np.unique(y_train, return_counts=True)
    LOGGER.info(f"\nSamples per accelerometer:")
    for accel_id, count in zip(unique, counts):
        LOGGER.info(f"  Accelerometer {accel_id}: {count} samples ({100*count/len(y_train):.2f}%)")

    results['samples_per_accel'] = dict(zip(unique.tolist(), counts.tolist()))
    results['total_samples'] = len(X_train)

    return results


def step4_restructure_check(accel_data_path: Path) -> Dict:
    """STEP 4: Restructure Your Training Data - Verify data organization."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("STEP 4: CHECKING TRAINING DATA STRUCTURE")
    LOGGER.info("="*80)

    results = {}

    train_dir = accel_data_path / "train"
    X_train = np.load(train_dir / "features.npy")
    y_train = np.load(train_dir / "labels.npy")

    LOGGER.info(f"\nVerifying data structure requirements:")
    LOGGER.info(f"  ✓ Each sample represents ONE accelerometer's reading: {X_train.shape}")
    LOGGER.info(f"  ✓ Labels indicate which accelerometer (0, 1, or 2): {np.unique(y_train)}")

    # Check if we have expected number of samples (should be 3x number of recordings)
    n_recordings_estimate = len(X_train) // 3
    LOGGER.info(f"  ✓ Estimated number of recordings: ~{n_recordings_estimate}")
    LOGGER.info(f"  ✓ Samples per recording: {len(X_train) / n_recordings_estimate:.1f} (should be ~3)")

    # Verify label distribution is roughly balanced
    unique, counts = np.unique(y_train, return_counts=True)
    balance_ratio = counts.max() / counts.min()
    LOGGER.info(f"\nClass balance check:")
    LOGGER.info(f"  Max/Min ratio: {balance_ratio:.2f} (should be close to 1.0)")
    if balance_ratio > 1.2:
        LOGGER.warning(f"  ⚠ Classes are imbalanced! This may affect training.")
    else:
        LOGGER.info(f"  ✓ Classes are well balanced")

    results['n_recordings_estimate'] = n_recordings_estimate
    results['balance_ratio'] = float(balance_ratio)

    return results


def step5_verify_feature_differences(accel_data_path: Path, output_dir: Path) -> Dict:
    """STEP 5: Verify Feature Differences - Check if features differ by accelerometer."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("STEP 5: VERIFYING FEATURE DIFFERENCES BETWEEN ACCELEROMETERS")
    LOGGER.info("="*80)

    results = {}

    train_dir = accel_data_path / "train"
    X_train = np.load(train_dir / "features.npy")
    y_train = np.load(train_dir / "labels.npy")

    # Flatten features if multi-dimensional
    if len(X_train.shape) > 2:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_flat = X_train

    LOGGER.info(f"\nGrouping samples by accelerometer label...")

    stats = {}
    for accel_id in range(3):
        mask = y_train == accel_id
        samples = X_train_flat[mask]

        stats[accel_id] = {
            'count': int(np.sum(mask)),
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'min': float(np.min(samples)),
            'max': float(np.max(samples)),
        }

        LOGGER.info(f"\nAccelerometer {accel_id}:")
        LOGGER.info(f"  Count: {stats[accel_id]['count']}")
        LOGGER.info(f"  Mean: {stats[accel_id]['mean']:.6f}")
        LOGGER.info(f"  Std: {stats[accel_id]['std']:.6f}")
        LOGGER.info(f"  Min: {stats[accel_id]['min']:.6f}")
        LOGGER.info(f"  Max: {stats[accel_id]['max']:.6f}")

    # Check if statistics are different
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("CRITICAL: Comparing accelerometer statistics...")
    LOGGER.info("="*80)

    all_means_same = True
    all_stds_same = True

    for i in range(3):
        for j in range(i+1, 3):
            mean_diff = abs(stats[i]['mean'] - stats[j]['mean'])
            std_diff = abs(stats[i]['std'] - stats[j]['std'])

            LOGGER.info(f"\nAccelerometer {i} vs Accelerometer {j}:")
            LOGGER.info(f"  Mean difference: {mean_diff:.6f}")
            LOGGER.info(f"  Std difference: {std_diff:.6f}")

            if mean_diff > 1e-6:
                all_means_same = False
                LOGGER.info(f"  ✓ Means are DIFFERENT (good!)")
            else:
                LOGGER.warning(f"  ✗ Means are IDENTICAL (bad!)")

            if std_diff > 1e-6:
                all_stds_same = False
                LOGGER.info(f"  ✓ Stds are DIFFERENT (good!)")
            else:
                LOGGER.warning(f"  ✗ Stds are IDENTICAL (bad!)")

    if all_means_same or all_stds_same:
        LOGGER.error("\n" + "!"*80)
        LOGGER.error("PROBLEM FOUND: Features are IDENTICAL across accelerometers!")
        LOGGER.error("This means the distance-based assumption is NOT captured in the data.")
        LOGGER.error("!"*80)
        results['features_different'] = False
    else:
        LOGGER.info("\n" + "="*80)
        LOGGER.info("✓ GOOD: Features ARE different between accelerometers")
        LOGGER.info("="*80)
        results['features_different'] = True

    results['stats'] = stats

    # Create visualization
    try:
        FileUtils.ensure_directory(str(output_dir))

        # Plot 1: Feature distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Feature Distributions by Accelerometer', fontsize=16)

        for accel_id in range(3):
            mask = y_train == accel_id
            samples = X_train_flat[mask].flatten()

            axes[accel_id].hist(samples, bins=50, alpha=0.7, edgecolor='black')
            axes[accel_id].set_title(f'Accelerometer {accel_id}')
            axes[accel_id].set_xlabel('Feature Value')
            axes[accel_id].set_ylabel('Frequency')
            axes[accel_id].axvline(stats[accel_id]['mean'], color='r', linestyle='--',
                                   label=f"Mean: {stats[accel_id]['mean']:.4f}")
            axes[accel_id].legend()

        plt.tight_layout()
        plt.savefig(output_dir / "step5_feature_distributions.png", dpi=150)
        LOGGER.info(f"\nSaved feature distribution plot to {output_dir / 'step5_feature_distributions.png'}")
        plt.close()

        # Plot 2: PCA visualization
        if X_train_flat.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_train_flat)

            plt.figure(figsize=(10, 8))
            for accel_id in range(3):
                mask = y_train == accel_id
                plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                          label=f'Accelerometer {accel_id}', alpha=0.6, s=50)

            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            plt.title('PCA: Accelerometer Features in 2D')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "step5_pca_visualization.png", dpi=150)
            LOGGER.info(f"Saved PCA visualization to {output_dir / 'step5_pca_visualization.png'}")
            plt.close()

            results['pca_variance'] = pca.explained_variance_ratio_.tolist()

    except Exception as e:
        LOGGER.error(f"Failed to create visualization: {e}")

    return results


def step6_check_pipeline(processed_data_path: Path, accel_data_path: Path) -> Dict:
    """STEP 6: Check Your Pipeline - Verify data creation process."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("STEP 6: CHECKING DATA PIPELINE")
    LOGGER.info("="*80)

    results = {}

    # Check processed NPZ files
    LOGGER.info("\nChecking source NPZ files...")
    train_dir = processed_data_path / "train"

    if not train_dir.exists():
        LOGGER.error(f"Processed data directory not found: {train_dir}")
        return results

    npz_files = sorted(train_dir.glob("*.npz"))
    LOGGER.info(f"Found {len(npz_files)} NPZ files in processed/train/")

    if npz_files:
        # Inspect first NPZ file
        first_file = npz_files[0]
        LOGGER.info(f"\nInspecting first NPZ file: {first_file.name}")

        data = np.load(first_file, allow_pickle=True)
        LOGGER.info(f"Keys in NPZ file: {list(data.keys())}")

        for key in data.keys():
            value = data[key]
            if key == "label":
                LOGGER.info(f"  {key}: {value} (hole size label)")
            else:
                LOGGER.info(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, dtype={getattr(value, 'dtype', type(value))}")

                # Check if we have data for 3 accelerometers
                if hasattr(value, 'shape') and len(value.shape) >= 2:
                    if value.shape[1] == 3:
                        LOGGER.info(f"    ✓ Has 3 columns (3 accelerometers)")
                        LOGGER.info(f"    Values for each accelerometer:")
                        for i in range(3):
                            col_data = value[:, i]
                            LOGGER.info(f"      Accel {i}: mean={np.mean(col_data):.6f}, std={np.std(col_data):.6f}")
                    else:
                        LOGGER.warning(f"    ✗ Expected 3 columns, got {value.shape[1]}")

                # Special check for welch_bandpower
                if key == "welch_bandpower" and hasattr(value, '__len__'):
                    if len(value) == 3:
                        LOGGER.info(f"    ✓ Has 3 values (one per accelerometer)")
                        for i in range(3):
                            LOGGER.info(f"      Accel {i}: {value[i]:.6f}")

                        # Check if values are different
                        if len(set(value)) == 1:
                            LOGGER.error(f"    ✗ ✗ ✗ ALL VALUES ARE IDENTICAL! This is the problem!")
                        else:
                            LOGGER.info(f"    ✓ Values are different (good!)")
                    else:
                        LOGGER.warning(f"    ✗ Expected 3 values, got {len(value)}")

        results['npz_keys'] = list(data.keys())

    # Check accelerometer classification data
    LOGGER.info("\n" + "-"*80)
    LOGGER.info("Checking accelerometer classification data...")

    accel_train_dir = accel_data_path / "train"
    if accel_train_dir.exists():
        X = np.load(accel_train_dir / "features.npy")
        y = np.load(accel_train_dir / "labels.npy")

        LOGGER.info(f"Accelerometer classification data:")
        LOGGER.info(f"  Features shape: {X.shape}")
        LOGGER.info(f"  Labels shape: {y.shape}")
        LOGGER.info(f"  Unique labels: {np.unique(y)}")

        # Expected: 3x as many samples as NPZ files (one per accelerometer)
        expected_samples = len(npz_files) * 3
        actual_samples = len(X)

        LOGGER.info(f"\nSample count verification:")
        LOGGER.info(f"  NPZ files: {len(npz_files)}")
        LOGGER.info(f"  Expected samples (3 per file): {expected_samples}")
        LOGGER.info(f"  Actual samples: {actual_samples}")

        if actual_samples == expected_samples:
            LOGGER.info(f"  ✓ Sample count matches (3 per recording)")
        else:
            LOGGER.warning(f"  ✗ Sample count mismatch!")

        results['expected_samples'] = expected_samples
        results['actual_samples'] = actual_samples

    return results


def step7_random_forest_checks(accel_data_path: Path) -> Dict:
    """STEP 7: Random Forest Specific Issues - Check for RF-specific problems."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("STEP 7: RANDOM FOREST SPECIFIC CHECKS")
    LOGGER.info("="*80)

    results = {}

    train_dir = accel_data_path / "train"
    X_train = np.load(train_dir / "features.npy")
    y_train = np.load(train_dir / "labels.npy")

    # Flatten features
    if len(X_train.shape) > 2:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_flat = X_train

    LOGGER.info(f"\nFeature scaling check:")
    LOGGER.info(f"  Feature shape: {X_train_flat.shape}")
    LOGGER.info(f"  Global mean: {np.mean(X_train_flat):.6f}")
    LOGGER.info(f"  Global std: {np.std(X_train_flat):.6f}")
    LOGGER.info(f"  Global min: {np.min(X_train_flat):.6f}")
    LOGGER.info(f"  Global max: {np.max(X_train_flat):.6f}")

    # Check for extreme values
    value_range = np.max(X_train_flat) - np.min(X_train_flat)
    LOGGER.info(f"  Value range: {value_range:.6f}")

    if value_range < 1e-10:
        LOGGER.error("  ✗ ✗ ✗ Feature values have almost no variation!")
        LOGGER.error("  This explains why Random Forest cannot learn!")
        results['has_variation'] = False
    else:
        LOGGER.info("  ✓ Features have reasonable variation")
        results['has_variation'] = True

    # Check for constant features
    LOGGER.info(f"\nChecking for constant features...")
    n_features = X_train_flat.shape[1]
    constant_features = 0

    for i in range(min(n_features, 10)):  # Check first 10 features
        unique_vals = len(np.unique(X_train_flat[:, i]))
        if unique_vals == 1:
            constant_features += 1
            LOGGER.warning(f"  Feature {i}: constant (only 1 unique value)")
        else:
            LOGGER.info(f"  Feature {i}: {unique_vals} unique values")

    if constant_features > 0:
        LOGGER.warning(f"  Found {constant_features} constant features!")
        results['constant_features'] = constant_features
    else:
        LOGGER.info(f"  ✓ No constant features found")
        results['constant_features'] = 0

    # Feature importance simulation
    LOGGER.info(f"\nSimulating feature importance...")
    try:
        from sklearn.ensemble import RandomForestClassifier

        # Quick train to check if it can learn anything
        LOGGER.info("  Training quick Random Forest (10 trees)...")
        rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
        rf.fit(X_train_flat, y_train)

        train_acc = rf.score(X_train_flat, y_train)
        LOGGER.info(f"  Quick RF training accuracy: {train_acc:.4f} ({100*train_acc:.2f}%)")

        if train_acc < 0.4:
            LOGGER.error("  ✗ ✗ ✗ Cannot learn even on training data!")
            LOGGER.error("  This confirms features are not discriminative!")
            results['can_learn'] = False
        else:
            LOGGER.info("  ✓ RF can learn from the data")
            results['can_learn'] = True

            # Show top features
            importances = rf.feature_importances_
            top_5_idx = np.argsort(importances)[-5:][::-1]
            LOGGER.info(f"\n  Top 5 feature importances:")
            for rank, idx in enumerate(top_5_idx, 1):
                LOGGER.info(f"    {rank}. Feature {idx}: {importances[idx]:.4f}")

    except Exception as e:
        LOGGER.error(f"Failed to train quick RF: {e}")
        results['can_learn'] = None

    return results


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    accel_data_path = Path(args.accelerometer_data)
    processed_data_path = Path(args.processed_data)
    output_dir = Path(args.output_dir)

    if not accel_data_path.exists():
        LOGGER.error(f"Accelerometer data path not found: {accel_data_path}")
        return 1

    if not processed_data_path.exists():
        LOGGER.error(f"Processed data path not found: {processed_data_path}")
        return 1

    FileUtils.ensure_directory(str(output_dir))

    LOGGER.info("="*80)
    LOGGER.info("COMPREHENSIVE ACCELEROMETER DATA DIAGNOSTICS")
    LOGGER.info("="*80)
    LOGGER.info(f"Accelerometer data: {accel_data_path}")
    LOGGER.info(f"Processed data: {processed_data_path}")
    LOGGER.info(f"Output directory: {output_dir}")

    # Run all diagnostic steps
    results = {}

    try:
        results['step3'] = step3_check_data_flow(accel_data_path)
        results['step4'] = step4_restructure_check(accel_data_path)
        results['step5'] = step5_verify_feature_differences(accel_data_path, output_dir)
        results['step6'] = step6_check_pipeline(processed_data_path, accel_data_path)
        results['step7'] = step7_random_forest_checks(accel_data_path)

        # Summary
        LOGGER.info("\n" + "="*80)
        LOGGER.info("DIAGNOSTIC SUMMARY")
        LOGGER.info("="*80)

        issues_found = []

        if not results.get('step5', {}).get('features_different', True):
            issues_found.append("Features are IDENTICAL across accelerometers")

        if not results.get('step7', {}).get('has_variation', True):
            issues_found.append("Features have almost no variation")

        if not results.get('step7', {}).get('can_learn', True):
            issues_found.append("Random Forest cannot learn from the data")

        if results.get('step7', {}).get('constant_features', 0) > 0:
            issues_found.append(f"{results['step7']['constant_features']} constant features")

        if issues_found:
            LOGGER.error("\n❌ ISSUES FOUND:")
            for issue in issues_found:
                LOGGER.error(f"  - {issue}")
            LOGGER.error("\nROOT CAUSE: The data preparation or feature extraction is not capturing")
            LOGGER.error("the differences between accelerometers. Check the NPZ file generation!")
        else:
            LOGGER.info("\n✓ No critical issues found. Data appears to be structured correctly.")

        return 0

    except Exception as e:
        LOGGER.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
