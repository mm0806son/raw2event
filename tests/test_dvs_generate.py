"""Tests for ``src.process_data.dvs_generate.generate_events_tensor`` guards."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.process_data import dvs_generate


class _DummyEventSim:
    def __init__(self, cfg=None, output_folder=None, sim_backend="auto"):
        del cfg, output_folder, sim_backend

    def generate_events(self, frame, timestamp):
        del frame, timestamp
        return None


@pytest.fixture(autouse=True)
def _patch_eventsim(monkeypatch):
    monkeypatch.setattr("src.simulator.EventSim", _DummyEventSim)


def test_generate_events_tensor_skips_bayer_conversion_when_raw_is_luminance(monkeypatch):
    call_count = {"count": 0}

    def fake_bayer_to_y(frame):
        call_count["count"] += 1
        return frame

    monkeypatch.setattr(
        "src.process_data.file_read.bayer_mosaic_to_luminance",
        fake_bayer_to_y,
    )

    timestamps = torch.tensor([0, 1000], dtype=torch.int64)
    frames = torch.from_numpy(np.ones((2, 4, 4), dtype=np.int32))

    dvs_generate.generate_events_tensor(
        timestamps,
        frames,
        is_rgb=False,
        raw_is_luminance=True,
        sim_backend="numpy",
    )

    assert call_count["count"] == 0


def test_generate_events_tensor_keeps_bayer_conversion_as_default(monkeypatch):
    call_count = {"count": 0}

    def fake_bayer_to_y(frame):
        call_count["count"] += 1
        return frame

    monkeypatch.setattr(
        "src.process_data.file_read.bayer_mosaic_to_luminance",
        fake_bayer_to_y,
    )

    timestamps = torch.tensor([0, 1000], dtype=torch.int64)
    frames = torch.from_numpy(np.ones((2, 4, 4), dtype=np.int32))

    dvs_generate.generate_events_tensor(
        timestamps,
        frames,
        is_rgb=False,
        sim_backend="numpy",
    )

    assert call_count["count"] == 2


def test_generate_events_tensor_rgb_path_ignores_raw_is_luminance(monkeypatch):
    call_count = {"count": 0}

    def fake_bayer_to_y(frame):
        call_count["count"] += 1
        return frame

    monkeypatch.setattr(
        "src.process_data.file_read.bayer_mosaic_to_luminance",
        fake_bayer_to_y,
    )

    timestamps = torch.tensor([0, 1000], dtype=torch.int64)
    frames = torch.from_numpy(np.ones((2, 4, 4, 3), dtype=np.uint8))

    dvs_generate.generate_events_tensor(
        timestamps,
        frames,
        is_rgb=True,
        raw_is_luminance=True,
        sim_backend="numpy",
    )

    assert call_count["count"] == 0
