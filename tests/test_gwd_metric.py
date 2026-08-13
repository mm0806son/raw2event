"""Unit tests for the GWD (Gromov-Wasserstein Discrepancy) upstream metric.

GWD measures the structural fidelity between two event point clouds in
normalised (x, y, t) space, independent of absolute position/scale. It is
added to the upstream event-statistics diagnostic alongside the existing
count/EMD metrics (see .planning/20260601_gwd_upstream_metric.md).

These tests run on synthetic point clouds only — no real NPZ data needed —
so they are stable and reproducible on any machine with POT installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.v2e_baseline.k_health_v2e_compare import HAVE_POT, gwd_one, render_md_table

pytestmark = pytest.mark.skipif(not HAVE_POT, reason="POT (ot) not installed")


def _make_events(n: int, seed: int, *, t_span_us: int = 50_000,
                 w: int = 346, h: int = 260, spread: float = 1.0) -> np.ndarray:
    """Build a synthetic [t, x, y, p] event array of shape (n, 4).

    ``spread`` < 1 concentrates events into a tighter spatial region. Note GWD
    is invariant to such pure-scale changes (per-axis z-score + max-normalised
    distances), so this knob is for fixtures, not for forcing a GWD difference —
    see test_structural_difference_increases_gwd for what GWD actually detects.
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, t_span_us, size=n)).astype(np.float64)
    cx, cy = w / 2.0, h / 2.0
    x = np.clip(cx + (rng.random(n) - 0.5) * w * spread, 0, w - 1)
    y = np.clip(cy + (rng.random(n) - 0.5) * h * spread, 0, h - 1)
    p = rng.integers(0, 2, size=n).astype(np.float64)
    return np.column_stack([t, x, y, p])


def test_identical_clouds_gwd_near_zero() -> None:
    """A cloud compared with itself has (near) zero structural discrepancy."""
    ev = _make_events(64, seed=1)
    # n_sample >= n so all points are used → fully deterministic, no subsample.
    val = gwd_one(ev, ev.copy(), n_sample=64)
    assert math.isfinite(val)
    assert val < 1e-6


def test_deterministic_under_subsampling() -> None:
    """Same input + same seed → identical result, even when subsampling."""
    a = _make_events(200, seed=2)
    b = _make_events(200, seed=3)
    v1 = gwd_one(a, b, n_sample=64, seed=7)
    v2 = gwd_one(a, b, n_sample=64, seed=7)
    assert v1 == v2


def test_structural_difference_increases_gwd() -> None:
    """A cloud with different *intrinsic* geometry is farther than a same-family one.

    GWD is deliberately invariant to per-axis scale (z-score) and global scale
    (max-normalised distances), so a merely tighter/looser cloud reads as
    equally close — that extent axis is covered by active_pixel/entropy ratios.
    What GWD *does* see is intrinsic structure: here a degenerate 1D line cloud
    (all events collinear) has a fundamentally different distance structure
    than the 3D base, and must score farther than a same-family random cloud.
    """
    base = _make_events(128, seed=4, spread=1.0)
    near = _make_events(128, seed=5, spread=1.0)  # same generative structure (3D blob)
    # Degenerate 1D structure: x sweeps a line, y and t collapsed to constants.
    n = 128
    line = np.column_stack([
        np.linspace(0, 50_000, n),          # t
        np.linspace(0, 345, n),             # x
        np.full(n, 130.0),                  # y constant
        np.zeros(n),                        # p
    ])
    gwd_near = gwd_one(base, near, n_sample=128)
    gwd_line = gwd_one(base, line, n_sample=128)
    assert gwd_line > gwd_near


def test_empty_or_too_few_events_returns_nan() -> None:
    """Degenerate inputs return NaN instead of raising (matches EMD metrics)."""
    ev = _make_events(64, seed=8)
    empty = np.empty((0, 4), dtype=np.float64)
    assert math.isnan(gwd_one(empty, ev))
    assert math.isnan(gwd_one(ev, empty))
    assert math.isnan(gwd_one(ev[:3], ev))  # below min_points


def test_shuffle_invariance() -> None:
    """GWD is a function of the point *set*, not its serialisation order.

    Subsampling canonicalises order first, so permuting the input rows (even
    when subsampling kicks in) must not change the result.
    """
    a = _make_events(300, seed=9)
    b = _make_events(300, seed=10)
    rng = np.random.default_rng(11)
    a_shuf = a[rng.permutation(a.shape[0])]
    b_shuf = b[rng.permutation(b.shape[0])]
    assert gwd_one(a, b, n_sample=64) == gwd_one(a_shuf, b_shuf, n_sample=64)


def test_render_md_table_schema() -> None:
    """Guard the table schema (the fan-out merge once drifted on these keys).

    A summary dict must render the GWD column, and an empty-variant row
    (n_prefix_used == 0) must not raise.
    """
    summary = [
        {"variant": "V01", "label": "Raw2Event-RAW", "n_prefix_total": 559,
         "n_prefix_used": 557, "n_skipped": 2, "n_gwd_used": 555,
         "count_ratio_mean": 1.14, "count_ratio_median": 1.10, "tv_delta_t_mean": 0.20,
         "spatial_entropy_ratio_mean": 0.98, "active_pixel_ratio_mean": 0.95,
         "polarity_delta_mean": 0.07, "pass_count_mean": 4.2,
         "per_pixel_count_emd_median": 0.0095, "sinkhorn_dt_emd_median": 0.012,
         "gwd_median": 0.0083},
        {"variant": "V09", "label": "empty", "n_prefix_total": 559,
         "n_prefix_used": 0, "n_skipped": 559},
    ]
    md = render_md_table(summary)
    assert "GWD med" in md
    assert "0.0083" in md
    assert "| V09 | empty | 0 |" in md
