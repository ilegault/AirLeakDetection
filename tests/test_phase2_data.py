"""Tests covering Phase 2 data loading, preprocessing, and feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.data_loader import WebDAQDataLoader
from src.data.preprocessor import SignalPreprocessor
from src.data.feature_extractor import FeatureExtractor


@pytest.fixture()
def synthetic_config(tmp_path: Path) -> tuple[Path, dict]:
    """Create a temporary configuration file and directories with synthetic CSV data."""
    config = {
        "data": {
            "raw_data_path": "data/raw",
            "processed_data_path": "data/processed",
            "sample_rate": 100,
            "duration": 1,
            "n_channels": 3,
        },
        "preprocessing": {
            "fft_size": 128,
            "window": "hann",
            "overlap": 0.5,
            "freq_max": 50.0,
            "normalize": True,
        },
        "classes": {0: "NOLEAK"},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    raw_dir = tmp_path / "data" / "raw" / "NOLEAK"
    raw_dir.mkdir(parents=True)
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)

    # Create two synthetic CSV files with varying lengths to exercise padding and truncation.
    shorter_signal = np.stack([
        np.linspace(0, 1, 80, dtype=np.float32),
        np.linspace(1, 0, 80, dtype=np.float32),
        np.sin(np.linspace(0, 4 * np.pi, 80, dtype=np.float32)),
    ], axis=1)
    longer_signal = np.stack([
        np.linspace(0, 1, 120, dtype=np.float32),
        np.linspace(1, 0, 120, dtype=np.float32),
        np.cos(np.linspace(0, 6 * np.pi, 120, dtype=np.float32)),
    ], axis=1)

    pd.DataFrame(shorter_signal, columns=["ch0", "ch1", "ch2"]).to_csv(raw_dir / "sample_a.csv", index=False)
    pd.DataFrame(longer_signal, columns=["ch0", "ch1", "ch2"]).to_csv(raw_dir / "sample_b.csv", index=False)

    return config_path, config


def test_data_loader_handles_padding_and_truncation(synthetic_config: tuple[Path, dict]) -> None:
    """Verify the loader pads short samples and truncates long ones to the configured duration."""
    config_path, config_dict = synthetic_config
    loader = WebDAQDataLoader(config_path)

    signals, labels, file_paths = loader.load_dataset()

    expected_samples = config_dict["data"]["sample_rate"] * config_dict["data"]["duration"]
    assert signals.shape == (2, expected_samples, 3)
    assert labels.tolist() == [0, 0]
    assert {path.name for path in file_paths} == {"sample_a.csv", "sample_b.csv"}

    # First file should be padded with zeros at the end to reach the expected length.
    padded_tail = signals[0, -5:, :]
    assert np.allclose(padded_tail, 0.0)

    # Second file should be truncated, so its first few samples should remain intact.
    # Confirm truncation preserves the earliest samples.
    assert np.allclose(signals[1, :3, 0], [0.0, 1.0 / expected_samples, 2.0 / expected_samples], atol=1e-6)

    summary = loader.dataset_summary()
    assert summary["class_distribution"]["NOLEAK"] == 2
    assert summary["expected_samples"] == expected_samples


def test_preprocessor_fft_and_segmentation(synthetic_config: tuple[Path, dict]) -> None:
    """Ensure FFT computation respects frequency limits and segmentation yields expected windows."""
    _, config_dict = synthetic_config
    preprocessor = SignalPreprocessor(config_dict)

    # Generate a multi-channel sinusoidal signal with two distinct frequencies.
    timesteps = 100
    time_axis = np.linspace(0.0, 1.0, timesteps, endpoint=False, dtype=np.float32)
    signal = np.stack([
        np.sin(2 * np.pi * 10 * time_axis),
        np.sin(2 * np.pi * 20 * time_axis),
        np.sin(2 * np.pi * 30 * time_axis),
    ], axis=1)

    frequencies, magnitudes = preprocessor.compute_fft(signal)
    assert magnitudes.shape[0] == config_dict["data"]["n_channels"]
    assert np.all(frequencies <= config_dict["preprocessing"]["freq_max"] + 1e-6)
    assert np.all((magnitudes >= 0.0) & (magnitudes <= 1.0))

    segments = preprocessor.segment_signal(signal, window_size=20, step_size=10)
    assert segments.shape == (9, 20, 3)


def test_feature_extractor_combines_features_with_labels() -> None:
    """Check that feature extractor merges statistics and attaches label metadata."""
    config = {"classes": {0: "NOLEAK"}}
    extractor = FeatureExtractor(config)

    signal = np.linspace(-1.0, 1.0, 100, dtype=np.float32)
    time_features = extractor.extract_time_features(signal)

    frequencies = np.linspace(0.0, 50.0, 32, dtype=np.float32)
    magnitude = np.abs(np.sin(frequencies)) + 1.0  # ensure strictly positive magnitudes
    freq_features = extractor.extract_frequency_features(frequencies, magnitude)

    combined = extractor.combine_features(time_features, freq_features, label=0)

    assert "label" in combined and combined["label"] == 0.0
    assert combined["label_name"] == "NOLEAK"
    # Ensure representative keys from both domains are present
    assert any(key.startswith("band_power") for key in combined)
    assert any(key.startswith("zero_crossing_rate") or key.endswith("peak_to_peak") for key in combined)