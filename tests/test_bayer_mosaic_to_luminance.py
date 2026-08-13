"""Unit tests for ``src.process_data.file_read.bayer_mosaic_to_luminance``.

These lock in the RAW→Y conversion that replaced the previous
mosaic-passthrough behavior in the K calibration and event-generation
pipelines.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.process_data.file_read import bayer_mosaic_to_luminance


def _uniform_rggb_mosaic(h: int, w: int, r: int, g: int, b: int) -> np.ndarray:
    """Build a (h, w) RGGB mosaic with constant R/G/B sub-channel values.

    Layout (top-left 2x2): R G / G B → repeated over the frame.
    """
    assert h % 2 == 0 and w % 2 == 0, "Mosaic dims must be even for RGGB tiling"
    frame = np.empty((h, w), dtype=np.uint16)
    frame[0::2, 0::2] = r  # R
    frame[0::2, 1::2] = g  # Gr
    frame[1::2, 0::2] = g  # Gb
    frame[1::2, 1::2] = b  # B
    return frame


class TestBayerMosaicToLuminance:
    def test_uint16_dtype_preserved(self):
        mosaic = _uniform_rggb_mosaic(8, 8, r=100, g=200, b=50)
        y = bayer_mosaic_to_luminance(mosaic)
        assert y.dtype == np.uint16

    def test_2d_shape_preserved(self):
        mosaic = _uniform_rggb_mosaic(10, 12, r=500, g=500, b=500)
        y = bayer_mosaic_to_luminance(mosaic)
        assert y.shape == (10, 12)

    def test_3d_batch_shape_preserved(self):
        mosaic = np.stack(
            [_uniform_rggb_mosaic(8, 8, r=i, g=i, b=i) for i in (100, 300, 900)]
        )
        y = bayer_mosaic_to_luminance(mosaic)
        assert y.shape == (3, 8, 8)
        assert y.dtype == np.uint16

    def test_uniform_gray_roundtrips_to_same_value(self):
        """When R == G == B, BT.601 Y equals that common value."""
        mosaic = _uniform_rggb_mosaic(8, 8, r=400, g=400, b=400)
        y = bayer_mosaic_to_luminance(mosaic)
        # Central region avoids boundary demosaic artifacts.
        assert np.all(np.abs(y[2:-2, 2:-2].astype(int) - 400) <= 1)

    def test_pure_green_matches_bt601_weight(self):
        """Pure G (no R, no B) should yield ~0.587 * G in the interior."""
        g_val = 1000
        mosaic = _uniform_rggb_mosaic(12, 12, r=0, g=g_val, b=0)
        y = bayer_mosaic_to_luminance(mosaic)
        expected = 0.587 * g_val
        center = y[4:-4, 4:-4].astype(float)
        assert abs(center.mean() - expected) < 5  # ±5 LSB tolerance

    def test_torch_tensor_input_accepted(self):
        mosaic = _uniform_rggb_mosaic(8, 8, r=200, g=400, b=100)
        y_np = bayer_mosaic_to_luminance(mosaic)
        y_torch = bayer_mosaic_to_luminance(torch.from_numpy(mosaic.astype(np.int32)))
        assert y_torch.shape == y_np.shape
        # cv2 demosaic result is deterministic for identical integer inputs
        assert np.array_equal(y_torch, y_np)

    def test_int32_input_is_cast_to_uint16(self):
        """Upstream code wraps uint16 mosaics as int32 tensors for torch
        compatibility; the utility must accept that without raising."""
        mosaic = _uniform_rggb_mosaic(8, 8, r=100, g=200, b=50).astype(np.int32)
        y = bayer_mosaic_to_luminance(mosaic)
        assert y.dtype == np.uint16

    def test_trailing_singleton_channel_is_collapsed(self):
        mosaic = _uniform_rggb_mosaic(8, 8, r=100, g=200, b=50)[..., None]  # (H, W, 1)
        # Wrap into batch form (N, H, W, 1) which some call sites produce.
        batch = mosaic[None]
        y = bayer_mosaic_to_luminance(batch)
        assert y.shape == (1, 8, 8)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="expects"):
            bayer_mosaic_to_luminance(np.zeros((3, 4, 8, 8), dtype=np.uint16))
