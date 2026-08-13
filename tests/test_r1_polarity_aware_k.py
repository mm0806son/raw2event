"""Tests for the R1 polarity-aware K patch (8D K [k1..k6, k_on, k_off]).

Verifies:
- EventSim accepts both 6D and 8D K and the byte-identical contract holds
  when k_on=k_off=1.0 (the legacy default).
- load_K_from_file accepts 6D and 8D K and pads the 6D case to 8D.
- generate_events_tensor accepts both 6D and 8D K vectors.
- per_pixel_probs_pol / pos_ratio behave correctly on canonical inputs.

Refs design: output/diagnostics_20260430_polarity_patch/r1_patch_design.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.config as _cfg_mod  # noqa: E402
from src.config import load_K_from_file  # noqa: E402
from src.simulator import EventSim  # noqa: E402
from k_calibration.k_optimize import (  # noqa: E402
    per_pixel_probs,
    per_pixel_probs_pol,
    pos_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(K):
    """Build a minimal cfg-like object compatible with EventSim."""
    from easydict import EasyDict as edict

    cfg = edict()
    cfg.SENSOR = edict()
    cfg.SENSOR.K = list(K)
    return cfg


def _ramp_frames(num_frames: int = 4, h: int = 12, w: int = 16, seed: int = 42):
    """Synthesize a deterministic luminance ramp sequence."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(20.0, 200.0, size=(h, w)).astype(np.float64)
    frames = []
    for i in range(num_frames):
        # gentle linear drift + small noise to ensure events fire on most pixels.
        delta = (i + 1) * 3.0 + rng.normal(0.0, 0.5, size=base.shape)
        frames.append(base + delta)
    timestamps = [int(1_000 * (i + 1)) for i in range(num_frames)]
    return frames, timestamps


def _seed_all(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_simulation(K, seed: int = 42, sim_backend: str = "numpy"):
    """Run EventSim on a fixed deterministic input. Returns concatenated events."""
    _seed_all(seed)
    cfg = _make_cfg(K)
    sim = EventSim(cfg=cfg, output_folder="", sim_backend=sim_backend)
    frames, timestamps = _ramp_frames()
    all_events = []
    for frame, ts in zip(frames, timestamps, strict=False):
        ev = sim.generate_events(frame, ts)
        if ev is not None and len(ev) > 0:
            all_events.append(np.asarray(ev))
    if not all_events:
        return np.zeros((0, 4), dtype=np.int32)
    return np.concatenate(all_events, axis=0)


# ---------------------------------------------------------------------------
# 1. EventSim 6D / 8D K back-compat
# ---------------------------------------------------------------------------


class TestEventSimKDimension:
    """Ensure 6D K and 8D K with k_on=k_off=1.0 produce identical events."""

    K_BASE = [1.66, -35.56, 1e-4, 1e-7, -2.0e-9, 1e-5]

    def test_eventsim_unpacks_6d_k_and_defaults_thresholds_to_one(self):
        cfg = _make_cfg(self.K_BASE)
        sim = EventSim(cfg=cfg, sim_backend="numpy")
        assert sim.k_on == 1.0
        assert sim.k_off == 1.0

    def test_eventsim_unpacks_8d_k_with_custom_thresholds(self):
        K8 = list(self.K_BASE) + [1.5, 0.7]
        cfg = _make_cfg(K8)
        sim = EventSim(cfg=cfg, sim_backend="numpy")
        assert sim.k_on == 1.5
        assert sim.k_off == 0.7

    def test_6d_and_8d_unit_thresholds_produce_identical_events_numpy(self):
        """Sanity gate: 8D K with k_on=k_off=1.0 must equal 6D K bit-for-bit."""
        events_6d = _run_simulation(self.K_BASE, sim_backend="numpy")
        events_8d = _run_simulation(self.K_BASE + [1.0, 1.0], sim_backend="numpy")
        assert events_6d.shape == events_8d.shape
        assert np.array_equal(events_6d, events_8d), (
            "8D K with unit thresholds must reproduce 6D K events bit-for-bit "
            "in the numpy backend."
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA backend smoke test"
    )
    def test_6d_and_8d_unit_thresholds_produce_identical_events_cuda(self):
        events_6d = _run_simulation(self.K_BASE, sim_backend="cuda")
        events_8d = _run_simulation(self.K_BASE + [1.0, 1.0], sim_backend="cuda")
        assert events_6d.shape == events_8d.shape
        assert np.array_equal(events_6d, events_8d)

    def test_asymmetric_thresholds_change_polarity_balance(self):
        """When k_on >> k_off, ON events should become rarer."""
        events_sym = _run_simulation(self.K_BASE + [1.0, 1.0], sim_backend="numpy")
        events_high_on = _run_simulation(
            self.K_BASE + [3.0, 1.0], sim_backend="numpy"
        )
        if events_sym.shape[0] == 0 or events_high_on.shape[0] == 0:
            pytest.skip("No events generated; ramp parameters too gentle.")
        ratio_sym = float((events_sym[:, 3] > 0).mean())
        ratio_high_on = float((events_high_on[:, 3] > 0).mean())
        assert ratio_high_on < ratio_sym + 1e-6


# ---------------------------------------------------------------------------
# 2. load_K_from_file 6D / 8D
# ---------------------------------------------------------------------------


class TestLoadKFromFile:
    K_BASE = [1.66, -35.56, 1e-4, 1e-7, -2.0e-9, 1e-5]

    def test_loads_6d_k_and_pads_with_unit_thresholds(self, tmp_path: Path):
        path = tmp_path / "Test6D_K.json"
        path.write_text(json.dumps({"pair": "Test6D", "K": self.K_BASE}))
        try:
            pair = load_K_from_file(str(path))
            assert pair == "Test6D"
            assert len(_cfg_mod.K_MAP[pair]) == 8
            assert _cfg_mod.K_MAP[pair][:6] == self.K_BASE
            assert _cfg_mod.K_MAP[pair][6:] == [1.0, 1.0]
        finally:
            _cfg_mod.K_MAP.pop("Test6D", None)

    def test_loads_8d_k_unchanged(self, tmp_path: Path):
        K8 = list(self.K_BASE) + [1.2, 0.85]
        path = tmp_path / "Test8D_K.json"
        path.write_text(json.dumps({"pair": "Test8D", "K": K8, "version": 2}))
        try:
            pair = load_K_from_file(str(path))
            assert pair == "Test8D"
            assert len(_cfg_mod.K_MAP[pair]) == 8
            assert _cfg_mod.K_MAP[pair][6] == pytest.approx(1.2)
            assert _cfg_mod.K_MAP[pair][7] == pytest.approx(0.85)
        finally:
            _cfg_mod.K_MAP.pop("Test8D", None)

    def test_rejects_invalid_k_length(self, tmp_path: Path):
        path = tmp_path / "Bad_K.json"
        path.write_text(json.dumps({"pair": "Bad", "K": [1.0, 2.0, 3.0]}))
        with pytest.raises(ValueError, match="6 .legacy. or 8 .polarity"):
            load_K_from_file(str(path))


# ---------------------------------------------------------------------------
# 3. dvs_generate accepts 6D / 8D
# ---------------------------------------------------------------------------


class TestDvsGenerateKDimension:
    K_BASE = [1.66, -35.56, 1e-4, 1e-7, -2.0e-9, 1e-5]

    def test_generate_events_tensor_accepts_6d_k(self):
        from src.process_data import dvs_generate

        timestamps = torch.tensor([0, 1000, 2000], dtype=torch.int64)
        rng = np.random.default_rng(0)
        frames_np = rng.uniform(50.0, 200.0, size=(3, 8, 8)).astype(np.float32)
        frames = torch.from_numpy(frames_np)

        out = dvs_generate.generate_events_tensor(
            timestamps,
            frames,
            is_rgb=False,
            raw_is_luminance=True,
            k_values=list(self.K_BASE),
            sim_backend="numpy",
        )
        assert out.shape[1] == 4
        # cfg.SENSOR.K must now be 8D (auto-padded).
        from src.config import cfg

        assert len(cfg.SENSOR.K) == 8
        assert cfg.SENSOR.K[6] == 1.0
        assert cfg.SENSOR.K[7] == 1.0

    def test_generate_events_tensor_accepts_8d_k(self):
        from src.process_data import dvs_generate

        timestamps = torch.tensor([0, 1000, 2000], dtype=torch.int64)
        rng = np.random.default_rng(0)
        frames_np = rng.uniform(50.0, 200.0, size=(3, 8, 8)).astype(np.float32)
        frames = torch.from_numpy(frames_np)

        K8 = list(self.K_BASE) + [1.4, 0.8]
        dvs_generate.generate_events_tensor(
            timestamps,
            frames,
            is_rgb=False,
            raw_is_luminance=True,
            k_values=K8,
            sim_backend="numpy",
        )
        from src.config import cfg

        assert list(cfg.SENSOR.K) == K8


# ---------------------------------------------------------------------------
# 4. per_pixel_probs_pol / pos_ratio
# ---------------------------------------------------------------------------


class TestPolarityAwareLossHelpers:
    def test_pos_ratio_empty(self):
        empty = torch.zeros((0, 4))
        assert pos_ratio(empty) == pytest.approx(0.5)

    def test_pos_ratio_basic(self):
        ev = torch.tensor(
            [
                [0, 0, 0, 1],
                [1, 1, 0, 1],
                [2, 0, 1, 0],
                [3, 1, 1, 0],
                [4, 2, 1, 1],
            ],
            dtype=torch.float64,
        )
        assert pos_ratio(ev) == pytest.approx(3.0 / 5.0)

    def test_per_pixel_probs_pol_separates_polarities(self):
        # 4 events at distinct pixels, two ON two OFF.
        ev = torch.tensor(
            [
                [0, 0, 0, 1],
                [1, 1, 0, 1],
                [2, 2, 1, 0],
                [3, 3, 1, 0],
            ],
            dtype=torch.float64,
        )
        h, w = 4, 4
        p_pos, p_neg = per_pixel_probs_pol(ev, height=h, width=w)
        # Two POS events split evenly.
        assert p_pos.sum().item() == pytest.approx(1.0)
        assert p_neg.sum().item() == pytest.approx(1.0)
        # POS pixels are flat indices 0 and 1.
        assert p_pos[0].item() == pytest.approx(0.5)
        assert p_pos[1].item() == pytest.approx(0.5)
        # NEG pixels are flat indices y=1, x=2 -> 1*4+2=6 and y=1, x=3 -> 7.
        assert p_neg[6].item() == pytest.approx(0.5)
        assert p_neg[7].item() == pytest.approx(0.5)

    def test_per_pixel_probs_pol_falls_back_to_uniform_for_empty_branch(self):
        ev = torch.tensor(
            [
                [0, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.float64,
        )
        h, w = 2, 2
        p_pos, p_neg = per_pixel_probs_pol(ev, height=h, width=w)
        # POS branch normalized over the two events (split evenly here).
        assert p_pos.sum().item() == pytest.approx(1.0)
        # NEG branch is empty -> uniform.
        assert p_neg.sum().item() == pytest.approx(1.0)
        assert torch.allclose(p_neg, torch.full((h * w,), 1.0 / (h * w), dtype=p_neg.dtype))

    def test_per_pixel_probs_collapsed_equals_sum_of_branches_when_normalized(self):
        ev = torch.tensor(
            [
                [0, 0, 0, 1],
                [1, 1, 0, 1],
                [2, 0, 1, 0],
                [3, 1, 1, 0],
            ],
            dtype=torch.float64,
        )
        h, w = 4, 4
        collapsed = per_pixel_probs(ev, height=h, width=w)
        p_pos, p_neg = per_pixel_probs_pol(ev, height=h, width=w)
        # 0.5 weight each branch reproduces the collapsed distribution
        # when each branch contributes the same number of events.
        recombined = 0.5 * p_pos + 0.5 * p_neg
        assert torch.allclose(collapsed, recombined)
