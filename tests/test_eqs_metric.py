"""Unit tests for the EQS metric core (LFS block), perturbations, and the
downstream rank-correlation analysis.

These cover the pure-NumPy/SciPy scientific core only; the torch + RVT adapter
(``tools/eqs/rvt_extractor.py``) needs the RVT weights and is exercised on
Leonardo, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.eqs.eqs_downstream_correlation import rank_correlation
from tools.eqs.eqs_score import latent_feature_similarity, patch_pool
from tools.eqs.perturbations import coord_shuffle, polarity_flip, time_shuffle


def _rand_acts(rng, shapes):
    return [rng.standard_normal(s) for s in shapes]


# --------------------------------------------------------------------------- #
# patch_pool
# --------------------------------------------------------------------------- #
def test_patch_pool_shape_and_padding():
    # (C=2, H=4, W=5), patch 3 -> 2x2 patch grid (padded to 6x6) -> 4 patches.
    act = np.ones((2, 4, 5))
    pooled = patch_pool(act, patch=3)
    assert pooled.shape == (4, 2)
    # Top-left patch is fully inside the data (all ones) -> mean over 3*3 = 1.0.
    assert pooled[0, 0] == pytest.approx(1.0)
    # Bottom-right patch overlaps padding -> its mean is < 1 (zeros counted).
    assert pooled[-1, 0] < 1.0


def test_patch_pool_known_values():
    act = np.arange(9, dtype=float).reshape(1, 3, 3)  # single 3x3 patch
    pooled = patch_pool(act, patch=3)
    assert pooled.shape == (1, 1)
    assert pooled[0, 0] == pytest.approx(np.arange(9).mean())


def test_patch_pool_accepts_batch_dim():
    a = np.ones((1, 2, 3, 3))
    assert patch_pool(a, 3).shape == (1, 2)


# --------------------------------------------------------------------------- #
# latent_feature_similarity
# --------------------------------------------------------------------------- #
def test_identical_streams_similarity_one_distance_zero():
    rng = np.random.default_rng(0)
    acts = _rand_acts(rng, [(8, 16, 16), (16, 8, 8), (32, 4, 4)])
    out = latent_feature_similarity(acts, acts, patch=3)
    assert out["eqs_similarity_mean"] == pytest.approx(1.0, abs=1e-9)
    assert out["eqs_distance_mean"] == pytest.approx(0.0, abs=1e-9)
    assert out["n_scales"] == 3
    assert len(out["eqs_similarity_per_scale"]) == 3


def test_different_streams_lower_similarity_and_bounded():
    rng = np.random.default_rng(1)
    a = _rand_acts(rng, [(8, 16, 16), (16, 8, 8), (32, 4, 4)])
    b = _rand_acts(rng, [(8, 16, 16), (16, 8, 8), (32, 4, 4)])
    same = latent_feature_similarity(a, a, patch=3)["eqs_similarity_mean"]
    diff = latent_feature_similarity(a, b, patch=3)["eqs_similarity_mean"]
    assert diff < same
    assert -1.0 - 1e-9 <= diff <= 1.0 + 1e-9


def test_determinism():
    rng = np.random.default_rng(2)
    a = _rand_acts(rng, [(8, 16, 16), (16, 8, 8)])
    b = _rand_acts(rng, [(8, 16, 16), (16, 8, 8)])
    r1 = latent_feature_similarity(a, b, patch=3)
    r2 = latent_feature_similarity(a, b, patch=3)
    assert r1["eqs_similarity_mean"] == r2["eqs_similarity_mean"]


def test_zero_norm_fraction_and_drop_policy():
    # A scale that is entirely zeros -> every patch is zero-norm.
    zeros = [np.zeros((4, 6, 6))]
    out = latent_feature_similarity(zeros, zeros, patch=3, zero_norm_policy="drop")
    assert out["zero_norm_patch_fraction"] == pytest.approx(1.0)
    # All patches dropped -> per-scale similarity is NaN, not a crash.
    assert np.isnan(out["eqs_similarity_mean"])


def test_partial_zero_background_does_not_break_identity():
    # Half-empty map: identical streams must still give similarity 1 (drop policy
    # excludes the empty patches rather than scoring them as dissimilar).
    a = np.zeros((4, 6, 6))
    a[:, :3, :3] = np.random.default_rng(3).standard_normal((4, 3, 3))
    out = latent_feature_similarity([a], [a], patch=3, zero_norm_policy="drop")
    assert out["eqs_similarity_mean"] == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < out["zero_norm_patch_fraction"] < 1.0


def test_scale_count_mismatch_raises():
    with pytest.raises(ValueError):
        latent_feature_similarity([np.ones((2, 3, 3))], [np.ones((2, 3, 3))] * 2)


# --------------------------------------------------------------------------- #
# perturbations
# --------------------------------------------------------------------------- #
def _events(rng, n=200):
    t = np.sort(rng.integers(0, 100000, size=n)).astype(float)
    x = rng.integers(0, 346, size=n).astype(float)
    y = rng.integers(0, 260, size=n).astype(float)
    p = rng.integers(0, 2, size=n).astype(float)
    return np.stack([t, x, y, p], axis=1)


@pytest.mark.parametrize("fn", [polarity_flip, time_shuffle, coord_shuffle])
def test_perturbation_preserves_shape_and_is_deterministic(fn):
    ev = _events(np.random.default_rng(4))
    out1 = fn(ev, seed=7)
    out2 = fn(ev, seed=7)
    assert out1.shape == ev.shape
    np.testing.assert_array_equal(out1, out2)
    # Input not mutated in place.
    assert not np.shares_memory(out1, ev)


def test_polarity_flip_swaps_zero_one():
    ev = _events(np.random.default_rng(5))
    flipped = polarity_flip(ev)
    np.testing.assert_array_equal(flipped[:, 3], 1.0 - ev[:, 3])
    # Non-polarity columns untouched.
    np.testing.assert_array_equal(flipped[:, :3], ev[:, :3])


def test_shuffles_preserve_value_multiset_but_change_pairing():
    ev = _events(np.random.default_rng(6))
    ts = time_shuffle(ev, seed=1)
    np.testing.assert_array_equal(np.sort(ts[:, 0]), np.sort(ev[:, 0]))
    np.testing.assert_array_equal(ts[:, 1:], ev[:, 1:])  # only t permuted
    assert not np.array_equal(ts[:, 0], ev[:, 0])

    cs = coord_shuffle(ev, seed=1)
    np.testing.assert_array_equal(np.sort(cs[:, 1]), np.sort(ev[:, 1]))
    np.testing.assert_array_equal(np.sort(cs[:, 2]), np.sort(ev[:, 2]))
    np.testing.assert_array_equal(cs[:, [0, 3]], ev[:, [0, 3]])  # t,p untouched


# --------------------------------------------------------------------------- #
# rank_correlation
# --------------------------------------------------------------------------- #
def test_monotone_increasing_gives_rho_near_one():
    up = {f"V{i:02d}": float(i) for i in range(12)}
    down = {f"V{i:02d}": float(i) ** 2 for i in range(12)}  # strictly increasing
    r = rank_correlation(up, down, n_perm=2000, n_boot=2000, seed=0)
    assert r["n"] == 12
    assert r["spearman_rho"] == pytest.approx(1.0)
    assert r["permutation_p"] < 0.05


def test_reversed_gives_rho_near_minus_one():
    up = {f"V{i:02d}": float(i) for i in range(12)}
    down = {f"V{i:02d}": float(-i) for i in range(12)}
    r = rank_correlation(up, down, n_perm=2000, n_boot=2000, seed=0)
    assert r["spearman_rho"] == pytest.approx(-1.0)


def test_uncorrelated_has_high_permutation_p():
    up = {f"V{i:02d}": float(i) for i in range(12)}
    # A deliberately scrambled, near-uncorrelated downstream ordering.
    down = {
        "V00": 5,
        "V01": 2,
        "V02": 8,
        "V03": 1,
        "V04": 7,
        "V05": 3,
        "V06": 6,
        "V07": 0,
        "V08": 9,
        "V09": 4,
        "V10": 11,
        "V11": 10,
    }
    down = {k: float(v) for k, v in down.items()}
    r = rank_correlation(up, down, n_perm=2000, n_boot=2000, seed=0)
    assert r["permutation_p"] > 0.1
    assert -1.0 <= r["spearman_rho"] <= 1.0


def test_degenerate_inputs_return_nan_not_raise():
    up = {"V00": 1.0, "V01": 2.0}
    down = {"V00": 1.0, "V01": 2.0}
    r = rank_correlation(up, down)
    assert np.isnan(r["spearman_rho"])
    assert r["n"] == 2


def test_alignment_drops_nan_and_missing():
    up = {"V00": 1.0, "V01": 2.0, "V02": np.nan, "V03": 4.0}
    down = {"V00": 1.0, "V01": 2.0, "V02": 3.0, "V99": 9.0}  # V03 missing, V02 nan
    r = rank_correlation(up, down, n_perm=500, n_boot=500)
    assert r["variants"] == ["V00", "V01"]
    assert r["n"] == 2


def test_constant_input_is_degenerate_even_with_enough_n():
    up = {f"V{i:02d}": 3.0 for i in range(5)}  # constant -> ptp == 0
    down = {f"V{i:02d}": float(i) for i in range(5)}
    r = rank_correlation(up, down, n_perm=500, n_boot=500)
    assert r["n"] == 5
    assert np.isnan(r["spearman_rho"])


# --------------------------------------------------------------------------- #
# zero-norm policy variants (eps / ones)
# --------------------------------------------------------------------------- #
def test_zero_norm_policies_differ_on_empty_scale():
    zeros = [np.zeros((4, 6, 6))]
    drop = latent_feature_similarity(zeros, zeros, patch=3, zero_norm_policy="drop")
    ones = latent_feature_similarity(zeros, zeros, patch=3, zero_norm_policy="ones")
    eps = latent_feature_similarity(zeros, zeros, patch=3, zero_norm_policy="eps")
    assert np.isnan(drop["eqs_similarity_mean"])  # all patches dropped
    assert ones["eqs_similarity_mean"] == pytest.approx(1.0)  # empty == empty -> 1
    assert eps["eqs_similarity_mean"] == pytest.approx(0.0)  # dot 0 / (0+eps) -> 0


def test_coord_shuffle_preserves_xy_pair_multiset():
    ev = _events(np.random.default_rng(11))
    cs = coord_shuffle(ev, seed=2)
    a = np.array(sorted(map(tuple, ev[:, [1, 2]])))
    b = np.array(sorted(map(tuple, cs[:, [1, 2]])))
    np.testing.assert_array_equal(a, b)  # the (x,y) PAIRS move together


# --------------------------------------------------------------------------- #
# eqs_compare.compute_eqs_table (with a stub extractor — no torch needed)
# --------------------------------------------------------------------------- #
class _FakeExtractor:
    """Deterministic stand-in for the RVT adapter: identical events -> identical
    activations (so identity control == 1), different events -> different maps."""

    stages = (1, 2, 3)
    pad_multiple = 32

    def __call__(self, events):
        seed = abs(hash(np.asarray(events).tobytes())) % (2**32)
        rng = np.random.default_rng(seed)
        return [rng.standard_normal(s) for s in ((4, 8, 8), (8, 4, 4), (16, 2, 2))]


def test_compute_eqs_table_reuse_finite_and_skip(tmp_path):
    from tools.eqs.eqs_compare import compute_eqs_table, summarise, summarise_controls

    dv_dir = tmp_path / "dv"
    sim_dir = tmp_path / "sim"
    dv_dir.mkdir()
    sim_dir.mkdir()
    rng = np.random.default_rng(20)
    prefixes = ["p0", "p1", "p2"]
    for pf in prefixes:
        np.savez(dv_dir / f"{pf}_filtered_dv.npz", events=_events(rng))
    # p0,p1 get a valid sim; p2 sim is missing (-> fewer rows, not a crash);
    # also add a corrupt sim for p1 under a second variant to hit fail-soft.
    for pf in ["p0", "p1"]:
        np.savez(sim_dir / f"{pf}_filtered_rgb.npz", events=_events(rng))
    (sim_dir / "p0_filtered_bad.npz").write_bytes(b"not an npz")

    cfg = {
        "V01": {"npz_dir": str(sim_dir), "npz_suffix": "rgb", "label": "good"},
        "V02": {"npz_dir": str(sim_dir), "npz_suffix": "bad", "label": "corrupt"},
    }
    per_variant, controls, skipped = compute_eqs_table(
        _FakeExtractor(), cfg, dv_dir, prefixes, {"patch": 3}, run_controls=True
    )

    # V01 has sims for p0,p1 only (p2 missing) -> 2 rows.
    assert len(per_variant["V01"]) == 2
    # V02's only file (p0) is corrupt -> 0 rows + 1 skipped record.
    assert len(per_variant["V02"]) == 0
    assert any(s["variant"] == "V02" for s in skipped)

    summary = {s["variant"]: s for s in summarise(per_variant, cfg)}
    assert summary["V01"]["n_used"] == 2
    assert summary["V01"]["n_eqs_used"] == 2  # all finite (non-empty maps)
    assert np.isfinite(summary["V01"]["eqs_similarity_mean_mean"])
    assert summary["V02"]["n_used"] == 0

    ctrl = summarise_controls(controls)
    # Identity upper bound must be exactly 1 (same events -> same activations).
    assert ctrl["real_identity"]["mean"] == pytest.approx(1.0)
    assert ctrl["real_identity"]["n"] == 3
