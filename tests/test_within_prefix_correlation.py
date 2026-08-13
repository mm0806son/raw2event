"""Unit tests for tools/v2e_baseline/within_prefix_correlation.py and
tools/v2e_baseline/dump_per_prefix_correctness.py.

These tests use synthetic JSON fixtures only — no GPU, no torch, no real
dataset access — so they run in any dev environment with numpy + scipy.

Run:
    python -m pytest tests/test_within_prefix_correlation.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.v2e_baseline import within_prefix_correlation as wpc
from tools.v2e_baseline import dump_per_prefix_correctness as dump


# ── helpers ──────────────────────────────────────────────────────────────


def _write_emd_result_json(out_dir: Path, variant: str, prefixes: list[str],
                            emd_values: list[float],
                            n_dv_zero: list[str] | None = None) -> Path:
    """Mimic result_V<NN>.json structure consumed by load_emd_per_prefix."""
    n_dv_zero = set(n_dv_zero or [])
    raw = []
    for p, v in zip(prefixes, emd_values):
        raw.append({
            "prefix": p,
            "n_sim": 100,
            "n_dv": 0 if p in n_dv_zero else 100,
            "per_pixel_count_emd": v,
            "sinkhorn_dt_emd": v * 2.0,
            "count_ratio": 1.0,
            "polarity_delta": 0.1,
        })
    path = out_dir / f"result_{variant}.json"
    path.write_text(json.dumps({"raw": {variant: raw}}))
    return path


def _make_correctness_payload(prefixes: list[str], variant: str,
                               correct_arrays_per_seed: list[list[int]]) -> dict:
    """Build a synthetic per_prefix_correctness.json payload."""
    per_run = {}
    for s, arr in enumerate(correct_arrays_per_seed):
        assert len(arr) == len(prefixes)
        per_run[f"qkformer.{variant}.seed{s}"] = {
            "label": f"variant {variant}", "kind": "test",
            "correct": list(arr),
            "acc": float(sum(arr)) / len(arr),
            "n": len(arr),
        }
    return {
        "test_modality": "dv",
        "split_source_run": "/fake/split",
        "test_data_dir": "/fake/data",
        "n_test": len(prefixes),
        "test_indices": list(range(len(prefixes))),
        "prefixes": prefixes,
        "per_run": per_run,
    }


# ── load_emd_per_prefix ──────────────────────────────────────────────────


def test_load_emd_drops_n_dv_zero(tmp_path: Path):
    prefixes = [f"p{i:03d}" for i in range(10)]
    emd = [0.01 + 0.005 * i for i in range(10)]
    _write_emd_result_json(tmp_path, "V01", prefixes, emd,
                            n_dv_zero=["p002", "p007"])
    out = wpc.load_emd_per_prefix(tmp_path, "V01", "per_pixel_count_emd")
    assert "p002" not in out and "p007" not in out
    assert len(out) == 8
    np.testing.assert_allclose(out["p000"], 0.01)
    np.testing.assert_allclose(out["p005"], 0.01 + 0.005 * 5)


def test_load_emd_alternative_metric(tmp_path: Path):
    prefixes = ["p0", "p1"]
    _write_emd_result_json(tmp_path, "V03", prefixes, [0.1, 0.2])
    out = wpc.load_emd_per_prefix(tmp_path, "V03", "sinkhorn_dt_emd")
    np.testing.assert_allclose(out["p0"], 0.2)
    np.testing.assert_allclose(out["p1"], 0.4)


# ── average_correctness_per_prefix ───────────────────────────────────────


def test_average_correctness_three_seeds():
    prefixes = ["a", "b", "c", "d"]
    payload = _make_correctness_payload(
        prefixes, "V01",
        correct_arrays_per_seed=[
            [1, 0, 1, 0],   # seed 0
            [1, 1, 0, 0],   # seed 1
            [0, 1, 1, 0],   # seed 2
        ],
    )
    out = wpc.average_correctness_per_prefix(payload, "V01")
    expected = {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3, "d": 0.0}
    for k, v in expected.items():
        assert out[k] == pytest.approx(v, abs=1e-6)


def test_average_correctness_no_seeds_raises():
    payload = _make_correctness_payload(["a"], "V01", [[1]])
    with pytest.raises(KeyError):
        wpc.average_correctness_per_prefix(payload, "V99")


# ── join_for_variant ─────────────────────────────────────────────────────


def test_join_intersection_and_canonical_filter():
    emd = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    cor = {"a": 0.0, "b": 0.5, "c": 1.0}   # missing 'd'
    canonical = {"a", "b", "c", "extra"}    # 'extra' not in others
    keys, x, y = wpc.join_for_variant(emd, cor, canonical)
    assert keys == ["a", "b", "c"]
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y, [0.0, 0.5, 1.0])


# ── bootstrap_spearman_ci ────────────────────────────────────────────────


def test_bootstrap_returns_finite_ci_for_strong_correlation():
    rng = np.random.default_rng(42)
    x = np.linspace(0, 1, 50)
    y = x + rng.normal(0, 0.05, size=50)
    lo, hi, n_valid = wpc.bootstrap_spearman_ci(x, y, n_bootstrap=200, rng_seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert lo > 0.5     # strong positive — CI should clear 0.5
    assert n_valid >= 195


def test_bootstrap_nan_safe_when_n_too_small():
    lo, hi, n_valid = wpc.bootstrap_spearman_ci(np.array([1.0, 2.0, 3.0]),
                                                np.array([1.0, 2.0, 3.0]),
                                                n_bootstrap=100, rng_seed=0)
    assert np.isnan(lo) and np.isnan(hi)
    assert n_valid == 0


def test_bootstrap_handles_constant_arm():
    """When y is constant, every resample's spearman is undefined; CI returns NaN."""
    x = np.arange(10, dtype=float)
    y = np.zeros(10)
    lo, hi, n_valid = wpc.bootstrap_spearman_ci(x, y, n_bootstrap=50, rng_seed=0)
    assert np.isnan(lo) and np.isnan(hi)
    assert n_valid == 0


# ── correlation_for_variant ──────────────────────────────────────────────


def test_correlation_for_variant_negative_rho():
    """Synthetic case: correctness decreases as EMD increases (anti-coupled).
    Spearman should be strongly negative."""
    n = 100
    prefixes = [f"p{i:03d}" for i in range(n)]
    emd_map = {p: i / n for i, p in enumerate(prefixes)}
    # correctness inversely correlated with EMD; quantize to {0, 1/3, 2/3, 1}
    cor_levels = np.array([1.0, 2 / 3, 1 / 3, 0.0])
    cor_map = {p: float(cor_levels[(i * 4) // n]) for i, p in enumerate(prefixes)}
    canonical = set(prefixes)
    res = wpc.correlation_for_variant(
        "V01", emd_map, cor_map, canonical,
        n_bootstrap=200, rng_seed=0,
    )
    assert res["n_used"] == n
    assert res["rho"] < -0.9
    assert res["rho_ci_high"] < 0   # CI strictly below 0


def test_correlation_for_variant_too_few_samples():
    res = wpc.correlation_for_variant(
        "V01",
        emd_map={"a": 1.0, "b": 2.0},
        correctness_map={"a": 0.0, "b": 1.0},
        canonical={"a", "b"},
        n_bootstrap=10, rng_seed=0,
    )
    assert res["n_used"] == 2
    assert np.isnan(res["rho"])


# ── summarize_across_variants ────────────────────────────────────────────


def test_summary_median_iqr():
    per_variant = [
        {"rho": -0.3}, {"rho": -0.1}, {"rho": 0.0}, {"rho": 0.1}, {"rho": 0.5},
    ]
    s = wpc.summarize_across_variants(per_variant)
    assert s["n_variants_used"] == 5
    assert s["rho_median"] == pytest.approx(0.0)
    assert s["rho_min"] == pytest.approx(-0.3)
    assert s["rho_max"] == pytest.approx(0.5)
    # IQR should bracket median
    assert s["rho_iqr_low"] <= s["rho_median"] <= s["rho_iqr_high"]


def test_summary_skips_nan():
    per_variant = [{"rho": float("nan")}, {"rho": 0.5}, {"rho": -0.5}]
    s = wpc.summarize_across_variants(per_variant)
    assert s["n_variants_used"] == 2
    assert s["rho_median"] == pytest.approx(0.0)


# ── end-to-end main(): write JSON + MD ───────────────────────────────────


def test_main_end_to_end(tmp_path: Path):
    """Run main() with synthetic inputs across 3 variants; verify output files."""
    n = 60
    prefixes = [f"p{i:03d}" for i in range(n)]
    fail_set = {"p010", "p020", "p030"}    # 3 fail-mode prefixes
    emd_dir = tmp_path / "emd"
    emd_dir.mkdir()

    # Build EMD JSON for V01..V03 with different relationships
    rng = np.random.default_rng(0)
    emd_per_variant = {
        "V01": np.linspace(0.005, 0.05, n).tolist(),       # well-correlated
        "V02": rng.uniform(0.005, 0.05, n).tolist(),       # uncorrelated noise
        "V03": np.linspace(0.05, 0.005, n).tolist(),       # anti-correlated
    }
    for v, vals in emd_per_variant.items():
        _write_emd_result_json(emd_dir, v, prefixes, vals,
                                n_dv_zero=list(fail_set))

    # Correctness payload — 3 seeds
    # For V01: prefer prefixes with high index (anti-correlated to EMD ordering of V01)
    # For V03: prefer prefixes with low index (correlated with V03's reversed EMD)
    payload = {
        "test_modality": "dv",
        "split_source_run": "/fake",
        "test_data_dir": "/fake",
        "n_test": n,
        "test_indices": list(range(n)),
        "prefixes": prefixes,
        "per_run": {},
    }

    def correctness_for(v: str, seed: int) -> list[int]:
        if v == "V01":
            # high i ⇒ likely correct ⇒ inverse to V01 EMD (which grows with i)
            return [1 if (i + seed) % 3 != 0 and i >= n // 4 else 0 for i in range(n)]
        if v == "V03":
            # high i ⇒ likely wrong; V03 EMD decreases with i
            return [1 if i < n // 2 else 0 for i in range(n)]
        # V02: random
        return [int(x) for x in rng.integers(0, 2, size=n)]

    for v in ("V01", "V02", "V03"):
        for s in (0, 1, 2):
            payload["per_run"][f"qkformer.{v}.seed{s}"] = {
                "label": v, "kind": "test",
                "correct": correctness_for(v, s),
                "acc": 0.0, "n": n,
            }
    correctness_path = tmp_path / "per_prefix_correctness.json"
    correctness_path.write_text(json.dumps(payload))

    canonical_path = tmp_path / "canonical.txt"
    canonical_path.write_text("\n".join(prefixes) + "\n")

    out_dir = tmp_path / "out"
    rc = wpc.main([
        "--emd_results_dir", str(emd_dir),
        "--correctness_json", str(correctness_path),
        "--canonical_prefix_list", str(canonical_path),
        "--variants", "V01,V02,V03",
        "--bootstrap", "200",
        "--rng_seed", "0",
        "--output_dir", str(out_dir),
    ])
    assert rc == 0
    results_json = json.loads((out_dir / "results.json").read_text())
    md_text = (out_dir / "results.md").read_text()

    assert len(results_json["per_variant"]) == 3
    n_used_each = [r["n_used"] for r in results_json["per_variant"]]
    assert all(x == n - len(fail_set) for x in n_used_each)
    summary = results_json["across_variants"]
    assert summary["n_variants_used"] == 3
    assert "rho_median" in summary
    # MD should contain table headers + summary block
    assert "| Variant |" in md_text
    assert "Across-variants summary" in md_text


# ── dump_per_prefix_correctness helpers ──────────────────────────────────


def test_list_dataset_prefixes_ordering(tmp_path: Path):
    """Verify the dumper's prefix listing matches EventNpzDataset's
    sorted(glob, key=leading_int) order, including for non-zero-padded ints."""
    # Filenames designed to expose lexical-vs-numeric sort difference:
    # "10_..." sorts before "9_..." lexically but after numerically.
    leading = [9, 10, 100, 540, 5803]
    for n in leading:
        (tmp_path / f"{n}_class_1_x_ts_filtered_dv.npz").touch()
        (tmp_path / f"{n}_class_1_x_ts_filtered_raw.npz").touch()  # noise
    out = dump.list_dataset_prefixes(tmp_path, "dv")
    assert out == [f"{n}_class_1_x_ts" for n in sorted(leading)]


def test_list_dataset_prefixes_filters_modality(tmp_path: Path):
    (tmp_path / "1_a_1_x_ts_filtered_dv.npz").touch()
    (tmp_path / "2_b_1_y_ts_filtered_raw.npz").touch()
    out_dv = dump.list_dataset_prefixes(tmp_path, "dv")
    out_raw = dump.list_dataset_prefixes(tmp_path, "raw")
    assert out_dv == ["1_a_1_x_ts"]
    assert out_raw == ["2_b_1_y_ts"]


def test_list_dataset_prefixes_empty_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        dump.list_dataset_prefixes(tmp_path, "dv")


def test_select_run_keys_by_variant_and_keys():
    runs = {
        "qkformer.V01.seed0": {}, "qkformer.V01.seed1": {},
        "qkformer.V02.seed0": {}, "qkformer.V08.seed2": {},
    }
    assert dump.select_run_keys(runs, ["V01", "V02"], None) == [
        "qkformer.V01.seed0", "qkformer.V01.seed1", "qkformer.V02.seed0"
    ]
    assert dump.select_run_keys(runs, None, ["qkformer.V08.seed2"]) == [
        "qkformer.V08.seed2"
    ]
    # Both filters compose (intersect)
    assert dump.select_run_keys(runs, ["V01"], ["qkformer.V01.seed1"]) == [
        "qkformer.V01.seed1"
    ]
