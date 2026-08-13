"""Unit tests for k_calibration/k_estimate.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ensure k_calibration is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_calibration.k_estimate import (
    _ig_mle,
    load_csv,
    load_pt_data,
    step1_per_lbar_regression,
    step2_cross_lbar_regression,
    step3_global_regression,
    validate_global_fit,
    wls,
)


# ---------------------------------------------------------------------------
# WLS
# ---------------------------------------------------------------------------


class TestWLS:
    def test_perfect_linear_fit(self):
        """y = 2*x + 3 should give exact coefficients and R2=1."""
        x = np.arange(10, dtype=float)
        y = 2.0 * x + 3.0
        beta, r2, rmse = wls(x, y)
        assert beta == pytest.approx([2.0, 3.0], abs=1e-10)
        assert r2 == pytest.approx(1.0, abs=1e-10)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_uniform_weights_same_as_none(self):
        """Uniform weights should match unweighted fit."""
        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 50)
        y = 1.5 * x - 2.0 + rng.normal(0, 0.5, 50)
        beta1, r2_1, _ = wls(x, y, w=None)
        beta2, r2_2, _ = wls(x, y, w=np.ones(50))
        assert beta1 == pytest.approx(beta2, abs=1e-10)
        assert r2_1 == pytest.approx(r2_2, abs=1e-10)

    def test_multivariate(self):
        """y = 2*x1 + 3*x2 + 1."""
        rng = np.random.default_rng(7)
        x1 = rng.uniform(0, 5, 100)
        x2 = rng.uniform(0, 5, 100)
        y = 2.0 * x1 + 3.0 * x2 + 1.0
        X = np.column_stack([x1, x2])
        beta, r2, _ = wls(X, y)
        assert beta == pytest.approx([2.0, 3.0, 1.0], abs=1e-8)
        assert r2 == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# IG MLE
# ---------------------------------------------------------------------------


class TestIGMLE:
    def test_constant_intervals(self):
        """All intervals equal -> mu_hat = Theta/tau, lambda = n (denom=0)."""
        tau = np.full(100, 500.0)
        mu, lam = _ig_mle(tau)
        assert mu == pytest.approx(1.0 / 500.0)
        assert lam == 100.0  # fallback when denom=0

    def test_known_distribution(self):
        """Check mu_hat = Theta / mean(tau) for random data."""
        rng = np.random.default_rng(123)
        tau = rng.exponential(scale=200.0, size=10000)
        mu, lam = _ig_mle(tau)
        expected_mu = 1.0 / tau.mean()
        assert mu == pytest.approx(expected_mu, rel=1e-10)
        assert lam > 0


# ---------------------------------------------------------------------------
# Cross-scene interval isolation
# ---------------------------------------------------------------------------


class TestCrossSceneIsolation:
    """Codex review C1: multi-scene .pt files must not produce cross-file intervals."""

    def _make_pt_files(self, tmp_path: Path) -> Path:
        """Create two .pt files with overlapping timestamps at the same pixels."""
        import torch

        data_dir = tmp_path / "calib"
        for i, scene_name in enumerate(["scene_a", "scene_b"]):
            scene_dir = data_dir / scene_name / "frames_analysis_full"
            scene_dir.mkdir(parents=True)

            n = 200
            # both scenes use the same pixel (100, 100) and overlapping timestamps
            ts = np.linspace(1000 + i * 10, 50000 + i * 10, n)
            x = np.full(n, 100.0)
            y = np.full(n, 100.0)
            pol = np.ones(n)
            prev_lum = np.full(n, 150.0)
            next_lum = np.full(n, 155.0)
            dt_frame = np.full(n, 16667.0)

            tensor = torch.tensor(
                np.column_stack([ts, x, y, pol, prev_lum, next_lum, dt_frame]),
                dtype=torch.float32,
            )
            torch.save(tensor, scene_dir / "events_with_luminance_raw.pt")

        return data_dir

    def test_no_cross_scene_intervals(self, tmp_path):
        """Intervals must only be computed within the same scene."""
        data_dir = self._make_pt_files(tmp_path)

        # Each scene has 200 events at one pixel -> 199 intervals per scene
        # Without scene isolation, we'd get many more due to interleaving
        # after sorting by (x, y, ts).
        df = load_pt_data(
            str(data_dir),
            "raw",
            n_lbar_bins=1,
            n_dl_bins=1,
            min_events=1,
        )
        # With 2 scenes of 200 events each at the same pixel,
        # we expect at most 199 + 199 = 398 intervals, NOT 399
        total_count = df["Count"].sum()
        assert total_count <= 398, (
            f"Expected <= 398 intervals (199 per scene), got {total_count}. "
            "Cross-scene intervals are leaking."
        )

    def test_single_scene_produces_intervals(self, tmp_path):
        """A single .pt file should produce valid intervals."""
        import torch

        scene_dir = tmp_path / "single" / "frames_analysis_full"
        scene_dir.mkdir(parents=True)

        n = 100
        ts = np.linspace(1000, 50000, n)
        x = np.full(n, 50.0)
        y = np.full(n, 50.0)
        pol = np.ones(n)
        prev_lum = np.full(n, 200.0)
        next_lum = np.full(n, 210.0)
        dt_frame = np.full(n, 16667.0)

        tensor = torch.tensor(
            np.column_stack([ts, x, y, pol, prev_lum, next_lum, dt_frame]),
            dtype=torch.float32,
        )
        torch.save(tensor, scene_dir / "events_with_luminance_raw.pt")

        df = load_pt_data(
            str(tmp_path / "single"),
            "raw",
            n_lbar_bins=1,
            n_dl_bins=1,
            min_events=1,
        )
        assert df["Count"].sum() == 99  # n-1 intervals from 1 pixel


# ---------------------------------------------------------------------------
# Polarity sign convention
# ---------------------------------------------------------------------------


class TestPolaritySign:
    """Codex review C2: polarity should determine drift rate sign."""

    def test_on_events_positive_mu(self, tmp_path):
        """Bin dominated by ON (P=1) events should have positive Mu_signed."""
        import torch

        scene_dir = tmp_path / "pol_test" / "frames_analysis_full"
        scene_dir.mkdir(parents=True)

        n = 200
        ts = np.linspace(1000, 100000, n)
        x = np.full(n, 50.0)
        y = np.full(n, 50.0)
        pol = np.ones(n)  # all ON events
        prev_lum = np.full(n, 150.0)
        next_lum = np.full(n, 160.0)  # brightness increasing -> ON
        dt_frame = np.full(n, 16667.0)

        tensor = torch.tensor(
            np.column_stack([ts, x, y, pol, prev_lum, next_lum, dt_frame]),
            dtype=torch.float32,
        )
        torch.save(tensor, scene_dir / "events_with_luminance_raw.pt")

        df = load_pt_data(
            str(tmp_path / "pol_test"),
            "raw",
            n_lbar_bins=1,
            n_dl_bins=1,
            min_events=1,
        )
        assert df["Mu_signed"].iloc[0] > 0, "ON-dominated bin should have positive mu"

    def test_off_events_negative_mu(self, tmp_path):
        """Bin dominated by OFF (P=0) events should have negative Mu_signed."""
        import torch

        scene_dir = tmp_path / "pol_off" / "frames_analysis_full"
        scene_dir.mkdir(parents=True)

        n = 200
        ts = np.linspace(1000, 100000, n)
        x = np.full(n, 50.0)
        y = np.full(n, 50.0)
        pol = np.zeros(n)  # all OFF events
        prev_lum = np.full(n, 150.0)
        next_lum = np.full(n, 140.0)  # brightness decreasing -> OFF
        dt_frame = np.full(n, 16667.0)

        tensor = torch.tensor(
            np.column_stack([ts, x, y, pol, prev_lum, next_lum, dt_frame]),
            dtype=torch.float32,
        )
        torch.save(tensor, scene_dir / "events_with_luminance_raw.pt")

        df = load_pt_data(
            str(tmp_path / "pol_off"),
            "raw",
            n_lbar_bins=1,
            n_dl_bins=1,
            min_events=1,
        )
        assert df["Mu_signed"].iloc[0] < 0, "OFF-dominated bin should have negative mu"


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def _make_csv(self, tmp_path: Path, fmt: str = "old") -> Path:
        """Create a minimal fit_results CSV."""
        path = tmp_path / "fit_results.csv"
        n = 50
        rng = np.random.default_rng(99)
        lbar_lo = rng.uniform(50, 200, n)
        lbar_hi = lbar_lo + 20
        dl_lo = rng.uniform(-50, 50, n)
        dl_hi = dl_lo + 10

        if fmt == "old":
            # Old format: MuHat = IG mean (large), Mu = signed drift (small, inverted)
            mu = rng.uniform(-5e-5, 5e-5, n)
            muhat = 1.0 / np.abs(mu)
            df = pd.DataFrame(
                {
                    "P": rng.choice([0, 1], n),
                    "MeanMin": lbar_lo,
                    "MeanMax": lbar_hi,
                    "DiffMin": dl_lo,
                    "DiffMax": dl_hi,
                    "MuHat": muhat,
                    "Mu": mu,
                    "LambdaHat": rng.uniform(1, 10, n),
                    "Sigma": rng.uniform(0.001, 0.01, n),
                    "Count": rng.integers(100, 1000, n),
                }
            )
        else:
            # New format: MuHat = drift rate magnitude
            df = pd.DataFrame(
                {
                    "P": rng.choice([0, 1], n),
                    "MeanMin": lbar_lo,
                    "MeanMax": lbar_hi,
                    "DiffMin": dl_lo,
                    "DiffMax": dl_hi,
                    "MuHat": rng.uniform(1e-5, 1e-4, n),
                    "Count": rng.integers(100, 1000, n),
                }
            )

        df.to_csv(path, index=False)
        return path

    def test_old_format_loads(self, tmp_path):
        path = self._make_csv(tmp_path, "old")
        df = load_csv(str(path), min_count=1, dt=16666.7)
        assert "Mu_signed" in df.columns
        assert "Lbar" in df.columns
        assert "kdL" in df.columns
        assert len(df) > 0

    def test_old_format_sign_convention(self, tmp_path):
        """Old format: Mu_signed = -Mu."""
        path = self._make_csv(tmp_path, "old")
        raw = pd.read_csv(path)
        df = load_csv(str(path), min_count=1, dt=16666.7)
        # merge back to check sign
        for _, row in df.iterrows():
            # find matching raw row
            mask = (raw["MeanMin"] == row["MeanMin"]) & (
                raw["MeanMax"] == row["MeanMax"]
            )
            if mask.sum() > 0:
                orig_mu = raw.loc[mask, "Mu"].iloc[0]
                assert row["Mu_signed"] == pytest.approx(-orig_mu)
                break

    def test_new_format_loads(self, tmp_path):
        path = self._make_csv(tmp_path, "new")
        df = load_csv(str(path), min_count=1, dt=16666.7)
        assert "Mu_signed" in df.columns
        assert len(df) > 0

    def test_min_count_filters(self, tmp_path):
        path = self._make_csv(tmp_path, "old")
        df_all = load_csv(str(path), min_count=1, dt=16666.7)
        df_high = load_csv(str(path), min_count=500, dt=16666.7)
        assert len(df_high) <= len(df_all)


# ---------------------------------------------------------------------------
# Three-step regression on synthetic data
# ---------------------------------------------------------------------------


class TestThreeStepRegression:
    """Test the full regression pipeline with known K values."""

    @pytest.fixture()
    def synthetic_df(self) -> pd.DataFrame:
        """Generate synthetic data from known K parameters.

        mu = k1 * kdL / (Lbar + k2) + k4 + k5 * Lbar
        """
        k1_true, k2_true, k4_true, k5_true = 2.5, 10.0, 1e-6, 1e-9

        rng = np.random.default_rng(42)
        rows = []
        for lbar in np.linspace(80, 400, 15):
            for kdl in np.linspace(-0.003, 0.003, 15):
                mu_true = k1_true * kdl / (lbar + k2_true) + k4_true + k5_true * lbar
                # add small noise
                mu_noisy = mu_true + rng.normal(0, abs(mu_true) * 0.05 + 1e-8)
                rows.append(
                    {
                        "Lbar": lbar,
                        "dLbar": kdl * 16667,
                        "kdL": kdl,
                        "MeanMin": lbar - 10,
                        "MeanMax": lbar + 10,
                        "DiffMin": kdl * 16667 - 5,
                        "DiffMax": kdl * 16667 + 5,
                        "Mu_signed": mu_noisy,
                        "Count": 500,
                    }
                )
        return pd.DataFrame(rows)

    def test_step1_produces_valid_bins(self, synthetic_df):
        df, n_valid = step1_per_lbar_regression(synthetic_df, n_lbar_bins=8)
        assert n_valid >= 2
        assert "a_n" in df.columns
        assert "lbar_bin" in df.columns
        assert df["a_n"].notna().sum() > 0

    def test_step2_recovers_k1_k2(self, synthetic_df):
        df, _ = step1_per_lbar_regression(synthetic_df, n_lbar_bins=8)
        k1, k2, r2, _, _, _ = step2_cross_lbar_regression(df)

        # k1 should be in the right ballpark (within 50% for noisy synthetic)
        assert 1.0 < k1 < 5.0, f"k1={k1} out of expected range [1, 5]"
        assert r2 > 0.5, f"Step 2 R2={r2} too low"

    def test_step3_recovers_k1_prime(self, synthetic_df):
        df, _ = step1_per_lbar_regression(synthetic_df, n_lbar_bins=8)
        k1, k2, _, _, _, _ = step2_cross_lbar_regression(df)
        k1_prime, k4, k5, r2, _, _, _, _ = step3_global_regression(df, k2)

        # k1' should be close to k1
        k1_err = abs(k1_prime - k1) / abs(k1) * 100
        assert k1_err < 30, f"k1/k1' consistency {k1_err:.1f}% too high"
        assert r2 > 0.5, f"Step 3 R2={r2} too low"

    def test_global_validation_reasonable(self, synthetic_df):
        k1_g, k2_g, k4_g, k5_g, r2_g = validate_global_fit(synthetic_df)
        assert 1.0 < k1_g < 5.0, f"Global k1={k1_g} out of range"
        assert r2_g > 0.5, f"Global R2={r2_g} too low"


# ---------------------------------------------------------------------------
# End-to-end output
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_save_results_json(self, tmp_path):
        from k_calibration.k_estimate import save_results

        out = save_results(
            tmp_path,
            "TestPair",
            "raw",
            k1=2.0,
            k2=10.0,
            k4=1e-6,
            k5=1e-9,
            k1_prime=2.05,
        )
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert data["pair"] == "TestPair"
        assert data["regression_consistency"] == "pass"
        assert data["k1_error_percent"] < 5.0

    def test_save_results_fail_consistency(self, tmp_path, capsys):
        from k_calibration.k_estimate import save_results

        save_results(
            tmp_path,
            "TestPair",
            "raw",
            k1=2.0,
            k2=10.0,
            k4=1e-6,
            k5=1e-9,
            k1_prime=3.0,
        )
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
