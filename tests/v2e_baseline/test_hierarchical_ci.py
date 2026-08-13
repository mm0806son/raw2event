"""Tests for hierarchical seed x prefix bootstrap CI (Codex Top-1).

The existing paired_bootstrap_ci resamples only the test prefixes while holding
the 3 training seeds fixed, which understates uncertainty: it cannot answer
"would V01 still beat V08 if we retrained?". The hierarchical version resamples
BOTH the seed-level units and the within-seed prefixes, so between-seed variance
inflates the CI as it should.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.v2e_baseline.hierarchical_ci import (  # noqa: E402
    hierarchical_paired_bootstrap_ci,
    seed_level_paired_t,
    simultaneous_vs_reference,
)


def _const_delta(n_seeds, n, value, rng):
    """Per-seed correctness pairs whose delta is exactly `value` everywhere."""
    left, right = {}, {}
    for s in range(n_seeds):
        r = (rng.random(n) < 0.5).astype(np.int8)
        left[s] = np.clip(r + value, 0, 1).astype(np.int8) if value > 0 else r
        right[s] = r if value > 0 else np.clip(r - (-value), 0, 1).astype(np.int8)
    return left, right


def test_identical_distributions_ci_contains_zero():
    rng = np.random.default_rng(0)
    n = 600
    left = {s: (rng.random(n) < 0.3).astype(np.int8) for s in range(3)}
    right = {s: (rng.random(n) < 0.3).astype(np.int8) for s in range(3)}
    out = hierarchical_paired_bootstrap_ci(left, right, n_bootstrap=2000, rng_seed=1)
    assert out["ci_low_pp"] <= 0 <= out["ci_high_pp"]
    assert abs(out["mean_pp"]) < 3.0


def test_clear_positive_delta_recovered():
    # left correct everywhere, right wrong everywhere -> delta = +100 pp.
    n, S = 200, 3
    left = {s: np.ones(n, dtype=np.int8) for s in range(S)}
    right = {s: np.zeros(n, dtype=np.int8) for s in range(S)}
    out = hierarchical_paired_bootstrap_ci(left, right, n_bootstrap=1000, rng_seed=2)
    assert out["mean_pp"] == pytest.approx(100.0, abs=1e-6)
    assert out["ci_low_pp"] == pytest.approx(100.0, abs=1e-6)


def test_hierarchical_wider_than_fixed_seed_when_between_seed_variance_high():
    """With big per-seed offsets, hierarchical CI must be wider than a
    prefix-only bootstrap that averages the fixed seeds first."""
    from tools.v2e_baseline.cross_modal_eval_with_ci import paired_bootstrap_ci

    rng = np.random.default_rng(7)
    n = 600
    # Three seeds with very different deltas: ~ -1, +8, +10 pp (mirrors QKF V01-V08).
    seed_means = [-0.01, 0.08, 0.10]
    left, right, deltas = {}, {}, []
    for s, dm in enumerate(seed_means):
        base = (rng.random(n) < 0.10).astype(np.int8)
        flip = rng.random(n) < abs(dm)
        if dm >= 0:
            r_left = np.clip(base + flip, 0, 1).astype(np.int8)
            r_right = base
        else:
            r_left = base
            r_right = np.clip(base + flip, 0, 1).astype(np.int8)
        left[s], right[s] = r_left, r_right
        deltas.append((r_left.astype(np.int8) - r_right.astype(np.int8)))

    hier = hierarchical_paired_bootstrap_ci(left, right, n_bootstrap=4000, rng_seed=3)
    fixed = paired_bootstrap_ci(deltas, n_bootstrap=4000, rng_seed=3)
    hier_w = hier["ci_high_pp"] - hier["ci_low_pp"]
    fixed_w = fixed["ci_high_pp"] - fixed["ci_low_pp"]
    assert hier_w > fixed_w, f"hierarchical {hier_w:.2f} should exceed fixed {fixed_w:.2f}"


def test_single_seed_reduces_to_prefix_bootstrap():
    from tools.v2e_baseline.cross_modal_eval_with_ci import paired_bootstrap_ci

    rng = np.random.default_rng(11)
    n = 500
    a = (rng.random(n) < 0.2).astype(np.int8)
    b = (rng.random(n) < 0.25).astype(np.int8)
    hier = hierarchical_paired_bootstrap_ci({0: a}, {0: b}, n_bootstrap=3000, rng_seed=5)
    fixed = paired_bootstrap_ci([(a.astype(np.int8) - b.astype(np.int8))],
                                n_bootstrap=3000, rng_seed=5)
    # Same point estimate; CI widths within a small tolerance (different RNG paths).
    assert hier["mean_pp"] == pytest.approx(fixed["mean_pp"], abs=1e-6)
    hw = hier["ci_high_pp"] - hier["ci_low_pp"]
    fw = fixed["ci_high_pp"] - fixed["ci_low_pp"]
    assert abs(hw - fw) < 1.5


def test_string_seed_labels_supported():
    """Real dump keys carry string seed labels (e.g. 'seed2024'); the CI must
    not assume int-convertible seed keys."""
    rng = np.random.default_rng(0)
    n = 300
    left = {f"seed{2024 + i}": (rng.random(n) < 0.3).astype(np.int8) for i in range(3)}
    right = {f"seed{2024 + i}": (rng.random(n) < 0.3).astype(np.int8) for i in range(3)}
    out = hierarchical_paired_bootstrap_ci(left, right, n_bootstrap=500, rng_seed=1)
    assert out["n_seeds"] == 3
    assert out["seeds"] == ["seed2024", "seed2025", "seed2026"]


def test_deterministic_with_fixed_seed():
    rng = np.random.default_rng(0)
    n = 400
    left = {s: (rng.random(n) < 0.3).astype(np.int8) for s in range(3)}
    right = {s: (rng.random(n) < 0.3).astype(np.int8) for s in range(3)}
    a = hierarchical_paired_bootstrap_ci(left, right, n_bootstrap=500, rng_seed=42)
    b = hierarchical_paired_bootstrap_ci(left, right, n_bootstrap=500, rng_seed=42)
    assert a == b


def test_seed_level_paired_t_matches_known_interval():
    # Three per-seed deltas of +7.5/+10.3/+7.0 pp -> mean 8.27, t(df=2)=4.303.
    # Build correctness so per-seed mean deltas hit those values on N=1000.
    n = 1000
    vals = [0.075, 0.103, 0.070]
    left, right = {}, {}
    for s, v in enumerate(vals):
        k = round(v * n)
        d = np.zeros(n, dtype=np.int8)
        d[:k] = 1                      # exactly k more correct on the left
        left[s] = d
        right[s] = np.zeros(n, dtype=np.int8)
    out = seed_level_paired_t(left, right)
    assert out["mean_pp"] == pytest.approx(8.2667, abs=0.05)
    # Hand-computed: sd≈1.now, t*se -> CI well above 0 but wide.
    assert out["ci_low_pp"] > 0
    assert out["ci_high_pp"] - out["ci_low_pp"] > 5  # only 3 seeds -> wide


def test_seed_level_paired_t_crosses_zero_when_one_seed_null():
    # QKF-like deltas +0.3/+8.0/+10.2 -> paired-t must cross zero (uncertain).
    n = 1000
    vals = [0.003, 0.080, 0.102]
    left, right = {}, {}
    for s, v in enumerate(vals):
        k = round(v * n)
        d = np.zeros(n, dtype=np.int8)
        d[:k] = 1
        left[s] = d
        right[s] = np.zeros(n, dtype=np.int8)
    out = seed_level_paired_t(left, right)
    assert out["ci_low_pp"] < 0 < out["ci_high_pp"]


def test_simultaneous_vs_reference_controls_family():
    """max-T simultaneous intervals must be wider than per-comparison ones."""
    rng = np.random.default_rng(9)
    n = 600
    ref = {s: (rng.random(n) < 0.28).astype(np.int8) for s in range(3)}
    others = {
        "V03": {s: (rng.random(n) < 0.11).astype(np.int8) for s in range(3)},
        "V07": {s: (rng.random(n) < 0.10).astype(np.int8) for s in range(3)},
        "V08": {s: (rng.random(n) < 0.12).astype(np.int8) for s in range(3)},
    }
    out = simultaneous_vs_reference(ref, others, n_bootstrap=2000, rng_seed=4)
    assert set(out["per_comparison"].keys()) == {"V03", "V07", "V08"}
    # Simultaneous half-width >= the max marginal half-width.
    marg = max(
        (c["ci_high_pp"] - c["ci_low_pp"]) / 2 for c in out["per_comparison"].values()
    )
    sim = max(
        (c["sim_ci_high_pp"] - c["sim_ci_low_pp"]) / 2
        for c in out["per_comparison"].values()
    )
    assert sim >= marg - 1e-9
