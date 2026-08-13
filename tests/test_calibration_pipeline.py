"""
tests/test_calibration_pipeline.py

Tests for the calibration pipeline:
  1. src/config.py — load_K_from_file(), set_camera_type(K_file=...)
  2. Synthetic calibration fixture validation
  3. k_calibration/ CLI smoke tests (--help)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "calibration"
SYNTHETIC_CSV = FIXTURE_DIR / "synthetic_fit_results_with_p.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_K_json(tmp_path):
    """Write a valid {pair}_K.json and return its path."""
    data = {
        "pair": "Raw2DVS346",
        "K": [2.388, 4.166e-7, 1.541e-6, 9.768e-8, 1.466e-11, 9.824e-6],
    }
    p = tmp_path / "Raw2DVS346_K.json"
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# 1. src/config.py
# ---------------------------------------------------------------------------


class TestConfigLoadKFromFile:
    def test_load_valid_json_updates_K_MAP(self, valid_K_json):
        # Import fresh to avoid global state pollution
        import importlib
        import src.config as cfg_mod

        importlib.reload(cfg_mod)

        pair = cfg_mod.load_K_from_file(str(valid_K_json))
        assert pair == "Raw2DVS346"
        # R1 (2026-04-30): 6D K is auto-padded to 8D with k_on=k_off=1.0.
        assert cfg_mod.K_MAP["Raw2DVS346"] == [
            2.388,
            4.166e-7,
            1.541e-6,
            9.768e-8,
            1.466e-11,
            9.824e-6,
            1.0,
            1.0,
        ]

    def test_load_missing_pair_raises(self, tmp_path):
        import src.config as cfg_mod

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"K": [1, 2, 3, 4, 5, 6]}))
        with pytest.raises(ValueError, match="'pair'"):
            cfg_mod.load_K_from_file(str(bad))

    def test_load_missing_K_raises(self, tmp_path):
        import src.config as cfg_mod

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"pair": "Raw2DVS346"}))
        with pytest.raises(ValueError, match="'K'"):
            cfg_mod.load_K_from_file(str(bad))

    def test_load_wrong_K_length_raises(self, tmp_path):
        import src.config as cfg_mod

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"pair": "Raw2DVS346", "K": [1, 2, 3]}))
        # R1: 6 (legacy) or 8 (polarity-aware) are both accepted; 3 is not.
        with pytest.raises(ValueError, match="6 .legacy. or 8 .polarity"):
            cfg_mod.load_K_from_file(str(bad))

    def test_load_nonexistent_file_raises(self, tmp_path):
        import src.config as cfg_mod

        with pytest.raises((FileNotFoundError, OSError)):
            cfg_mod.load_K_from_file(str(tmp_path / "nonexistent.json"))


class TestSetCameraTypeWithKFile:
    def test_set_camera_type_with_K_file_overrides_default(self, tmp_path):
        import importlib
        import src.config as cfg_mod

        importlib.reload(cfg_mod)

        custom_K = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        p = tmp_path / "Raw2DVS346_K.json"
        p.write_text(json.dumps({"pair": "Raw2DVS346", "K": custom_K}))

        cfg_mod.set_camera_type("Raw2DVS346", K_file=str(p))
        # R1: 6D K is auto-padded to 8D with k_on=k_off=1.0.
        assert cfg_mod.cfg.SENSOR.K == custom_K + [1.0, 1.0]
        assert cfg_mod.cfg.SENSOR.CAMERA_TYPE == "Raw2DVS346"

    def test_set_camera_type_mismatched_pair_raises(self, tmp_path):
        import src.config as cfg_mod

        p = tmp_path / "DVS346_K.json"
        p.write_text(json.dumps({"pair": "DVS346", "K": [1, 2, 3, 4, 5, 6]}))

        with pytest.raises(ValueError, match="does not match"):
            cfg_mod.set_camera_type("Raw2DVS346", K_file=str(p))

    def test_set_camera_type_unknown_raises_without_K_file(self):
        import src.config as cfg_mod

        with pytest.raises(ValueError, match="Unknown camera type"):
            cfg_mod.set_camera_type("NonExistentCamera")


# ---------------------------------------------------------------------------
# 2. Synthetic calibration fixture validation
# ---------------------------------------------------------------------------


class TestSyntheticFixture:
    """Validate the checked-in synthetic calibration fixture."""

    def test_fixture_file_exists(self):
        assert SYNTHETIC_CSV.exists(), f"Fixture missing: {SYNTHETIC_CSV}"

    def test_fixture_has_required_columns(self):
        df = pd.read_csv(SYNTHETIC_CSV)
        required = {"MeanMin", "MeanMax", "MuHat", "Count", "P"}
        assert required.issubset(df.columns), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_fixture_has_both_polarities(self):
        df = pd.read_csv(SYNTHETIC_CSV)
        assert set(df["P"].unique()) == {0, 1}

    def test_fixture_has_multiple_luminance_bins(self):
        df = pd.read_csv(SYNTHETIC_CSV)
        n_bins = len(df[["MeanMin", "MeanMax"]].drop_duplicates())
        assert n_bins >= 3, f"Expected >= 3 luminance bins, got {n_bins}"

    def test_fixture_readme_exists(self):
        readme = FIXTURE_DIR / "README.md"
        assert readme.exists(), f"README missing: {readme}"


# ---------------------------------------------------------------------------
# 3. k_calibration/ CLI smoke tests
# ---------------------------------------------------------------------------


class TestKCalibrationCliHelp:
    """Verify each k_calibration script is importable and its CLI is parseable."""

    @pytest.mark.parametrize(
        "script",
        [
            "k_calibration/k_preprocess.py",
            "k_calibration/k_estimate.py",
            "k_calibration/k_optimize.py",
        ],
    )
    def test_help_exits_zero(self, script):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / script), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{script} --help failed:\n{result.stderr}"
        )
