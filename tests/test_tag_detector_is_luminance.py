"""Regression tests for the ``is_luminance`` guard in ``tag_detector``.

Once ``process_single_batch._raw_bayer_to_y_downsampled`` debayers a RAW frame
to luminance, the downstream AprilTag detection path must not repeat the
Bayer->Gray conversion; otherwise ROI/crop results are silently corrupted.
These tests pin that behaviour.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.process_data import tag_detector


class _NoDetectionDetector:
    """Stub detector so ``process_frame`` exits after the preprocessing gates."""

    def detect(self, _frame):
        return []


def _patch_cvtcolor(monkeypatch):
    calls = []
    real_cvt = cv2.cvtColor

    def spy(frame, code, *args, **kwargs):
        calls.append(code)
        return real_cvt(frame, code, *args, **kwargs)

    monkeypatch.setattr(tag_detector.cv2, "cvtColor", spy)
    return calls


def test_is_luminance_true_skips_bayer_debayer(monkeypatch):
    calls = _patch_cvtcolor(monkeypatch)
    frame = np.full((260, 346), 200, dtype=np.uint16)

    result = tag_detector.process_frame(
        frame,
        timestamp=0,
        detector=_NoDetectionDetector(),
        is_raw=True,
        is_luminance=True,
    )

    assert result == (None, None, None)
    assert cv2.COLOR_BayerRG2GRAY not in calls


def test_is_luminance_false_still_triggers_bayer_debayer(monkeypatch):
    calls = _patch_cvtcolor(monkeypatch)
    frame = np.full((260, 346), 200, dtype=np.uint16)

    tag_detector.process_frame(
        frame,
        timestamp=0,
        detector=_NoDetectionDetector(),
        is_raw=True,
        is_luminance=False,
    )

    assert cv2.COLOR_BayerRG2GRAY in calls


def test_is_luminance_true_still_stretches_uint16_to_uint8(monkeypatch):
    """uint16 luminance frames still need the bit-depth stretch for AprilTag."""
    observed = {}

    class _CaptureDetector:
        def detect(self, frame):
            observed["dtype"] = frame.dtype
            return []

    frame = np.full((260, 346), 500, dtype=np.uint16)

    tag_detector.process_frame(
        frame,
        timestamp=0,
        detector=_CaptureDetector(),
        is_raw=True,
        is_luminance=True,
    )

    assert observed["dtype"] == np.uint8


def test_is_raw_false_neither_debayers_nor_stretches(monkeypatch):
    calls = _patch_cvtcolor(monkeypatch)
    observed = {}

    class _CaptureDetector:
        def detect(self, frame):
            observed["dtype"] = frame.dtype
            return []

    frame = np.full((260, 346), 128, dtype=np.uint8)

    tag_detector.process_frame(
        frame,
        timestamp=0,
        detector=_CaptureDetector(),
        is_raw=False,
    )

    assert cv2.COLOR_BayerRG2GRAY not in calls
    assert observed["dtype"] == np.uint8


@pytest.mark.parametrize("is_luminance", [False, True])
def test_process_batch_forwards_is_luminance(monkeypatch, is_luminance):
    captured = []

    def fake_process_frame(frame, ts, detector, **kwargs):
        captured.append(kwargs.get("is_luminance"))
        return None, None, None

    monkeypatch.setattr(tag_detector, "process_frame", fake_process_frame)

    frames = np.zeros((3, 10, 10), dtype=np.uint16)
    timestamps = [1, 2, 3]

    tag_detector.process_batch(
        frames, timestamps, is_raw=True, is_luminance=is_luminance
    )

    assert captured == [is_luminance] * 3
