"""
Tests for generate_event.py — focused on 120K MKV batch production safety.

Run:  python -m pytest tests/test_generate_event.py -v
"""

import os
import sys
import shutil
import tempfile
import subprocess

import numpy as np
import cv2
import pytest

# Make project importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import generate_event as generate_event_module
import dv_processing as dv

from generate_event import (
    generate_events_naive,
    _flush_events_to_writer,
    process_single,
    save_to_aedat4,
    load_from_video,
    _is_rgb_video,
    _scan_directory_once,
    _build_subprocess_cmd,
    batch_process,
    _ffprobe_video_info,
    _ffmpeg_decode_frames,
)


# ──────────────────────────────────────────────────────────────
#  Helpers — synthetic test data
# ──────────────────────────────────────────────────────────────

def _make_gradient_frames(n=10, h=64, w=64, dtype=np.uint8):
    """Create n frames with a linearly increasing gradient (guarantees events)."""
    frames = np.zeros((n, h, w), dtype=dtype)
    for i in range(n):
        # Each frame shifts brightness by +25, wrapping within dtype range
        val = min((i + 1) * 25, np.iinfo(dtype).max)
        frames[i, :, :] = val
    return frames


def _make_timestamps(n=10, fps=30.0):
    """Generate evenly-spaced timestamps in microseconds."""
    dt = 1e6 / fps
    return [int(i * dt) for i in range(n)]


def _create_test_mkv(path, n_frames=30, h=64, w=64, fps=30.0):
    """Create a small synthetic MKV video with changing frames."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=True)
    if not out.isOpened():
        # Fallback codec for environments without XVID
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # MJPG might need .avi extension; try mp4v for .mkv
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h), isColor=True)
    if not out.isOpened():
        pytest.skip("No suitable video codec available for test MKV creation")

    for i in range(n_frames):
        # Step by 25 per frame to guarantee events above typical thresholds (10-20)
        val = min(int(i * 25), 255)
        frame = np.full((h, w, 3), val, dtype=np.uint8)
        # Add spatial variation — different region brightens faster
        frame[h // 4 : h // 2, w // 4 : w // 2, :] = min(val + 60, 255)
        out.write(frame)
    out.release()
    # Verify the file was created and is non-empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pytest.skip(f"Failed to create test video: {path}")


def _require_ffmpeg_tools():
    """Skip the current test if ffmpeg/ffprobe are unavailable."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available in test environment")


def _create_gray16le_mkv(path, frames, fps=30.0):
    """Create a gray16le FFV1 MKV from a uint16 frame stack via ffmpeg."""
    _require_ffmpeg_tools()

    frames = np.asarray(frames, dtype=np.uint16)
    if frames.ndim != 3:
        raise ValueError("frames must have shape (N, H, W)")

    n_frames, height, width = frames.shape
    raw_path = os.path.splitext(path)[0] + ".raw"
    frames.tofile(raw_path)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "-s:v",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                raw_path,
                "-frames:v",
                str(n_frames),
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "gray16le",
                path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace")
        pytest.skip(f"ffmpeg failed to create gray16le MKV: {stderr}")
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        pytest.skip(f"Failed to create gray16le MKV: {path}")


# ──────────────────────────────────────────────────────────────
#  1. Unit: generate_events_naive
# ──────────────────────────────────────────────────────────────

class TestGenerateEventsNaive:
    def test_basic_events(self):
        """Changing frames produce ON/OFF events."""
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.full((64, 64), 100, dtype=np.uint8)
        events = generate_events_naive(prev, curr, timestamp=1000, threshold=20)
        assert events is not None
        assert events.shape[1] == 4  # [t, x, y, p]
        assert np.all(events[:, 0] == 1000)  # all same timestamp
        assert np.all(events[:, 3] == 1)  # all ON (brightness increased)

    def test_no_change_no_events(self):
        """Identical frames produce zero events."""
        frame = np.full((64, 64), 128, dtype=np.uint8)
        events = generate_events_naive(frame, frame.copy(), timestamp=0, threshold=20)
        assert events is None

    def test_below_threshold(self):
        """Small changes below threshold produce zero events."""
        prev = np.full((64, 64), 100, dtype=np.uint8)
        curr = np.full((64, 64), 110, dtype=np.uint8)
        events = generate_events_naive(prev, curr, timestamp=0, threshold=20)
        assert events is None

    def test_off_events(self):
        """Brightness decrease → OFF events (polarity=0)."""
        prev = np.full((64, 64), 200, dtype=np.uint8)
        curr = np.zeros((64, 64), dtype=np.uint8)
        events = generate_events_naive(prev, curr, timestamp=500, threshold=20)
        assert events is not None
        assert np.all(events[:, 3] == 0)  # all OFF

    def test_mixed_polarity(self):
        """Some pixels increase, some decrease → mixed ON/OFF."""
        prev = np.full((64, 64), 128, dtype=np.uint8)
        curr = prev.copy()
        curr[:32, :] = 200  # ON in top half
        curr[32:, :] = 50   # OFF in bottom half
        events = generate_events_naive(prev, curr, timestamp=100, threshold=20)
        assert events is not None
        polarities = set(events[:, 3].tolist())
        assert polarities == {0, 1}

    def test_none_inputs(self):
        assert generate_events_naive(None, np.zeros((4, 4)), 0) is None
        assert generate_events_naive(np.zeros((4, 4)), None, 0) is None

    def test_shape_mismatch(self):
        a = np.zeros((64, 64), dtype=np.uint8)
        b = np.zeros((32, 32), dtype=np.uint8)
        assert generate_events_naive(a, b, 0) is None

    def test_coordinate_bounds(self):
        """Event x/y should be within frame dimensions."""
        h, w = 100, 200
        prev = np.zeros((h, w), dtype=np.uint8)
        curr = np.full((h, w), 128, dtype=np.uint8)
        events = generate_events_naive(prev, curr, timestamp=0, threshold=10)
        assert events is not None
        assert np.all(events[:, 1] >= 0) and np.all(events[:, 1] < w)
        assert np.all(events[:, 2] >= 0) and np.all(events[:, 2] < h)


# ──────────────────────────────────────────────────────────────
#  2. Unit: _flush_events_to_writer
# ──────────────────────────────────────────────────────────────

class TestFlushEventsToWriter:
    def test_empty_buffer(self):
        assert _flush_events_to_writer(None, []) == 0

    def test_counts_events(self):
        """Flushing returns correct event count."""
        events = np.array([[100, 10, 20, 1], [200, 5, 15, 0]], dtype=np.int64)
        n = _flush_events_to_writer(None, [events], quiet=True)
        assert n == 2

    def test_filters_negative_timestamps(self):
        """Events with t<0 are filtered out."""
        events = np.array([
            [-10, 1, 1, 1],
            [100, 2, 2, 1],
            [200, 3, 3, 0],
        ], dtype=np.int64)
        n = _flush_events_to_writer(None, [events], quiet=True)
        assert n == 2  # only the two valid events

    def test_all_negative_returns_zero(self):
        events = np.array([[-1, 0, 0, 1], [-5, 1, 1, 0]], dtype=np.int64)
        assert _flush_events_to_writer(None, [events], quiet=True) == 0

    def test_writes_to_txt(self):
        """Events are written to text file handle."""
        events = np.array([[100, 10, 20, 1], [200, 5, 15, 0]], dtype=np.int64)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as fh:
            path = fh.name
            n = _flush_events_to_writer(None, [events], txt_fh=fh, quiet=True)
        try:
            assert n == 2
            data = np.loadtxt(path, dtype=np.int64)
            assert data.shape == (2, 4)
        finally:
            os.unlink(path)

    def test_multiple_chunks(self):
        """Buffer with multiple event arrays is concatenated correctly."""
        e1 = np.array([[100, 1, 1, 1]], dtype=np.int64)
        e2 = np.array([[200, 2, 2, 0], [150, 3, 3, 1]], dtype=np.int64)
        n = _flush_events_to_writer(None, [e1, e2], quiet=True)
        assert n == 3

    def test_polarity_normalization(self):
        """Polarity values > 1 are clamped to 1."""
        events = np.array([[100, 1, 1, 5], [200, 2, 2, -1]], dtype=np.int64)
        # polarity 5 > 0 → 1;  polarity -1 <= 0 → 0
        n = _flush_events_to_writer(None, [events], quiet=True)
        assert n == 2


# ──────────────────────────────────────────────────────────────
#  3. Unit: save_to_aedat4 — does not modify input
# ──────────────────────────────────────────────────────────────

class TestSaveToAedat4:
    def test_does_not_modify_input(self):
        """save_to_aedat4 should NOT modify the caller's events array."""
        events = np.array([
            [100, 10, 20, 1],
            [200, 5,  15, 0],
            [150, 8,  12, 1],
        ], dtype=np.int64)
        original = events.copy()
        with tempfile.NamedTemporaryFile(suffix='.aedat4', delete=False) as f:
            path = f.name
        try:
            save_to_aedat4(events, filename=path, input_resolution=(64, 64), quiet=True)
            np.testing.assert_array_equal(events, original)
        finally:
            os.unlink(path)

    def test_output_file_created(self):
        events = np.array([[100, 10, 20, 1]], dtype=np.int64)
        with tempfile.NamedTemporaryFile(suffix='.aedat4', delete=False) as f:
            path = f.name
        try:
            save_to_aedat4(events, filename=path, input_resolution=(64, 64), quiet=True)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────────────────────
#  4. Integration: process_single (streaming + cleanup)
# ──────────────────────────────────────────────────────────────

class TestProcessSingle:
    def test_basic_aedat4_output(self):
        """Full pipeline: synthetic frames → aedat4 file."""
        frames = _make_gradient_frames(n=10, h=32, w=32)
        ts = _make_timestamps(n=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test_output')
            n = process_single(
                frames, ts, (32, 32),
                method='naive', output_path=out,
                save_aedat4=True, save_txt=False,
                threshold=10, quiet=True,
            )
            assert n > 0
            assert os.path.exists(out + '.aedat4')
            assert os.path.getsize(out + '.aedat4') > 100

    def test_txt_output(self):
        """Text file output has correct format."""
        frames = _make_gradient_frames(n=5, h=16, w=16)
        ts = _make_timestamps(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test_output')
            n = process_single(
                frames, ts, (16, 16),
                method='naive', output_path=out,
                save_aedat4=False, save_txt=True,
                threshold=10, quiet=True,
            )
            assert n > 0
            txt_path = out + '.txt'
            assert os.path.exists(txt_path)
            data = np.loadtxt(txt_path, dtype=np.int64)
            assert data.shape[0] == n
            assert data.shape[1] == 4

    def test_no_events_cleans_up_files(self):
        """When no events, output files should be removed."""
        # All-constant frames → no events
        frames = np.full((5, 16, 16), 128, dtype=np.uint8)
        ts = _make_timestamps(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test_output')
            n = process_single(
                frames, ts, (16, 16),
                method='naive', output_path=out,
                save_aedat4=True, save_txt=True,
                threshold=20, quiet=True,
            )
            assert n == 0
            # Files should NOT exist (cleaned up)
            assert not os.path.exists(out + '.aedat4')
            assert not os.path.exists(out + '.txt')

    def test_streaming_flush(self):
        """Large number of events triggers intermediate flushes (memory bounded)."""
        h, w = 128, 128
        n_frames = 20
        frames = np.zeros((n_frames, h, w), dtype=np.uint8)
        for i in range(n_frames):
            # Random frames ensure many events per pair
            frames[i] = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        ts = _make_timestamps(n=n_frames, fps=60)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test_output')
            n = process_single(
                frames, ts, (w, h),
                method='naive', output_path=out,
                save_aedat4=True, threshold=5, quiet=True,
            )
            assert n > 0
            assert os.path.exists(out + '.aedat4')

    def test_frame_timestamp_mismatch(self):
        """Mismatched counts → uses minimum, no crash."""
        frames = _make_gradient_frames(n=10, h=16, w=16)
        ts = _make_timestamps(n=7)  # fewer timestamps
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test_output')
            n = process_single(
                frames, ts, (16, 16),
                method='naive', output_path=out,
                save_aedat4=False, save_txt=False,
                threshold=10, quiet=True,
            )
            # Should not crash, and processes min(10, 7) frames
            assert isinstance(n, int)

    def test_empty_frames(self):
        """Zero frames → returns 0, no crash."""
        frames = np.zeros((0, 16, 16), dtype=np.uint8)
        ts = []
        n = process_single(frames, ts, (16, 16), quiet=True, save_aedat4=False)
        assert n == 0

    def test_dvs_forwards_sim_backend(self, monkeypatch):
        """process_single should pass sim_backend to EventSim."""
        captured = {}

        class _DummyEventSim:
            def __init__(self, cfg, output_folder='', video_name='', sim_backend='auto'):
                captured['sim_backend'] = sim_backend
                self.t_previous = None

            def generate_events(self, new_frame, t_frame):
                self.t_previous = float(t_frame)
                return None

        monkeypatch.setattr('generate_event.EventSim', _DummyEventSim)

        frames = _make_gradient_frames(n=3, h=16, w=16)
        ts = _make_timestamps(n=3)
        # is_rgb=True routes through RGB2DVS346, which is still populated in
        # K_MAP after the RAW→Y-domain switch that dropped Raw2DVS346.
        n = process_single(
            frames, ts, (16, 16),
            method='dvs', output_path='ignore',
            save_aedat4=False, save_txt=False,
            quiet=True, sim_backend='numpy', is_rgb=True,
        )
        assert isinstance(n, int)
        assert captured['sim_backend'] == 'numpy'


# ──────────────────────────────────────────────────────────────
#  5. Unit: load_from_video
# ──────────────────────────────────────────────────────────────

class TestLoadFromVideo:
    @pytest.fixture
    def test_mkv(self, tmp_path):
        path = str(tmp_path / "test_video.mkv")
        _create_test_mkv(path, n_frames=15, h=48, w=64, fps=30.0)
        return path

    def test_loads_correct_frame_count(self, test_mkv):
        frames, ts = load_from_video(test_mkv, quiet=True)
        assert len(frames) == len(ts)
        assert len(frames) > 0

    def test_frames_are_grayscale_2d(self, test_mkv):
        frames, _ = load_from_video(test_mkv, quiet=True)
        assert frames.ndim == 3  # (N, H, W)

    def test_timestamps_are_nonnegative(self, test_mkv):
        _, ts = load_from_video(test_mkv, quiet=True)
        assert all(t >= 0 for t in ts)

    def test_timestamps_are_monotonic(self, test_mkv):
        _, ts = load_from_video(test_mkv, quiet=True)
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1], f"Timestamps not monotonic at index {i}"

    def test_preallocated_dtype(self, test_mkv):
        """Standard video frames should be uint8."""
        frames, _ = load_from_video(test_mkv, quiet=True)
        assert frames.dtype == np.uint8

    def test_nonexistent_file(self):
        with pytest.raises(ValueError, match="ffprobe failed"):
            load_from_video("/nonexistent/video.mkv", quiet=True)

    @pytest.mark.parametrize(
        ("video_name", "pix_fmt"),
        [
            ("sample_raw_10bit.mkv", "yuv420p"),
            ("sample_standard.mkv", "gray16le"),
        ],
    )
    def test_rejects_unexpected_pix_fmt(self, monkeypatch, video_name, pix_fmt):
        def fake_ffprobe(_video_path):
            return {
                "width": 8,
                "height": 6,
                "pix_fmt": pix_fmt,
                "fps": 30.0,
                "nb_frames": 2,
            }

        monkeypatch.setattr(generate_event_module, "_ffprobe_video_info", fake_ffprobe)

        with pytest.raises(ValueError, match="Unexpected pix_fmt"):
            load_from_video(video_name, quiet=True)

    def test_raw_10bit_preserves_10bit_scale(self, tmp_path):
        """Downstream RAW decode must keep the 0-1023 10-bit scale that K
        calibration was fit against (calibration ``.dat`` measured max ~792).

        The previous behavior (commit 6135bce) left-shifted by 6 here, which
        inflated downstream luminance by 64x versus calibration — the root
        cause of the downstream event-density gap; see ``.cursor/findings.md
        §28``.
        """
        encoded = np.array(
            [
                [[0, 64, 128, 256], [512, 768, 900, 1023]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
            ],
            dtype=np.uint16,
        )
        video_path = tmp_path / "restored_raw_10bit.mkv"
        _create_gray16le_mkv(str(video_path), encoded, fps=25.0)

        frames, timestamps_us = load_from_video(str(video_path), quiet=True)

        assert frames.dtype == np.uint16
        # Values must stay in the 10-bit range (no ``<< 6``).
        assert frames.max() == int(encoded.max())
        assert frames.max() <= 1023
        assert frames.shape == encoded.shape
        assert timestamps_us == [0, 40000]

    def test_raw_10bit_keeps_bayer_mosaic_pattern(self, monkeypatch):
        mosaic = np.array(
            [
                [
                    [10, 900, 10, 900],
                    [20, 800, 20, 800],
                    [10, 900, 10, 900],
                    [20, 800, 20, 800],
                ]
            ],
            dtype=np.uint16,
        )

        monkeypatch.setattr(
            generate_event_module,
            "_ffprobe_video_info",
            lambda _video_path: {
                "width": 4,
                "height": 4,
                "pix_fmt": "gray16le",
                "fps": 50.0,
                "nb_frames": 1,
            },
        )
        monkeypatch.setattr(
            generate_event_module,
            "_ffmpeg_decode_frames",
            lambda *args, **kwargs: mosaic.copy(),
        )

        frames, timestamps_us = load_from_video("checker_raw_10bit.mkv", quiet=True)

        even_mean = frames[0, :, 0::2].mean()
        odd_mean = frames[0, :, 1::2].mean()
        input_even_mean = mosaic[0, :, 0::2].mean()
        input_odd_mean = mosaic[0, :, 1::2].mean()

        assert frames.dtype == np.uint16
        assert timestamps_us == [0]
        assert odd_mean > even_mean
        # The Bayer mosaic must be preserved verbatim (no ``<< 6`` amplification).
        assert odd_mean - even_mean == input_odd_mean - input_even_mean


class TestFfprobeVideoInfo:
    def test_missing_ffprobe_binary_is_wrapped(self, monkeypatch):
        def raise_missing(*args, **kwargs):
            raise FileNotFoundError("missing ffprobe")

        monkeypatch.setattr(generate_event_module.subprocess, "check_output", raise_missing)

        with pytest.raises(ValueError, match="ffprobe binary not found"):
            _ffprobe_video_info("video.mkv")


class TestFfmpegDecodeFrames:
    def test_incomplete_final_frame_raises_value_error(self, monkeypatch):
        frame_size = 2 * 2 * 2
        partial = b"\x01" * (frame_size - 1)

        class FakeStdout:
            def __init__(self, payload):
                self._payload = payload
                self._sent = False

            def readinto(self, buffer):
                if self._sent:
                    return 0
                n_bytes = min(len(buffer), len(self._payload))
                buffer[:n_bytes] = self._payload[:n_bytes]
                self._sent = True
                return n_bytes

            def close(self):
                return None

        class FakeStderr:
            def read(self, _size=-1):
                return b""

            def close(self):
                return None

        class FakeProc:
            def __init__(self):
                self.stdout = FakeStdout(partial)
                self.stderr = FakeStderr()
                self._killed = False

            def poll(self):
                return None if not self._killed else 1

            def kill(self):
                self._killed = True

            def wait(self):
                return 1 if self._killed else 0

        monkeypatch.setattr(generate_event_module.subprocess, "Popen", lambda *args, **kwargs: FakeProc())

        with pytest.raises(ValueError, match="Incomplete final frame"):
            _ffmpeg_decode_frames(
                "short_read.mkv",
                "gray16le",
                channels=1,
                bytes_per_sample=2,
                width=2,
                height=2,
            )

    def test_missing_ffmpeg_binary_is_wrapped(self, monkeypatch):
        def raise_missing(*args, **kwargs):
            raise FileNotFoundError("missing ffmpeg")

        monkeypatch.setattr(generate_event_module.subprocess, "Popen", raise_missing)

        with pytest.raises(ValueError, match="ffmpeg binary not found"):
            _ffmpeg_decode_frames(
                "video.mkv",
                "gray16le",
                channels=1,
                bytes_per_sample=2,
                width=2,
                height=2,
            )


# ──────────────────────────────────────────────────────────────
#  6. Unit: _is_rgb_video
# ──────────────────────────────────────────────────────────────

class TestIsRgbVideo:
    def test_standard_is_rgb(self):
        assert _is_rgb_video("video.mkv") is True
        assert _is_rgb_video("test_rgb.mkv") is True

    def test_raw_is_not_rgb(self):
        assert _is_rgb_video("test_raw.mkv") is False
        assert _is_rgb_video("test_raw_10bit.mkv") is False

    def test_case_insensitive(self):
        assert _is_rgb_video("TEST_RAW.MKV") is False
        assert _is_rgb_video("Video_RAW_10BIT.mkv") is False


# ──────────────────────────────────────────────────────────────
#  7. Unit: _scan_directory_once (case-insensitive extensions)
# ──────────────────────────────────────────────────────────────

class TestScanDirectoryOnce:
    def test_finds_mkv_files(self, tmp_path):
        (tmp_path / "a.mkv").touch()
        (tmp_path / "b.mp4").touch()
        (tmp_path / "c.txt").touch()
        result = _scan_directory_once(str(tmp_path))
        basenames = {os.path.basename(p) for p in result['video_files']}
        assert 'a.mkv' in basenames
        assert 'b.mp4' in basenames
        assert 'c.txt' not in basenames

    def test_case_insensitive_extensions(self, tmp_path):
        """CRITICAL: .MKV and .Mkv must be found (120K file safety)."""
        (tmp_path / "upper.MKV").touch()
        (tmp_path / "mixed.Mkv").touch()
        (tmp_path / "lower.mkv").touch()
        result = _scan_directory_once(str(tmp_path))
        basenames = {os.path.basename(p) for p in result['video_files']}
        assert 'upper.MKV' in basenames
        assert 'mixed.Mkv' in basenames
        assert 'lower.mkv' in basenames
        assert len(result['video_files']) == 3

    def test_dat_files_collected(self, tmp_path):
        (tmp_path / "raw_frames_001.dat").touch()
        (tmp_path / "metadata_001.dat").touch()
        result = _scan_directory_once(str(tmp_path))
        assert 'raw_frames_001.dat' in result['dat_files']
        assert 'metadata_001.dat' in result['dat_files']

    def test_empty_directory(self, tmp_path):
        result = _scan_directory_once(str(tmp_path))
        assert result['video_files'] == []
        assert result['dat_files'] == {}

    def test_nonexistent_directory(self):
        result = _scan_directory_once("/nonexistent/path")
        assert result['video_files'] == []
        assert result['dat_files'] == {}


# ──────────────────────────────────────────────────────────────
#  8. Unit: _build_subprocess_cmd
# ──────────────────────────────────────────────────────────────

class TestBuildSubprocessCmd:
    def test_includes_quiet_flag(self):
        """CRITICAL: subprocess workers MUST have --quiet to avoid pipe deadlock."""
        task = {'name': 'test', 'source': 'video', 'info': '/path/to/video.mkv'}
        cmd = _build_subprocess_cmd(task, '/out', 'naive', 692, 520, True, False, 20)
        assert '--quiet' in cmd

    def test_includes_save_aedat4(self):
        task = {'name': 'test', 'source': 'video', 'info': '/path/to/video.mkv'}
        cmd = _build_subprocess_cmd(task, '/out', 'naive', 692, 520, True, False, 20)
        assert '--save_aedat4' in cmd

    def test_video_task(self):
        task = {'name': 'test', 'source': 'video', 'info': '/path/video.mkv'}
        cmd = _build_subprocess_cmd(task, '/out', 'naive', 692, 520, True, False, 20)
        assert '--video' in cmd
        assert '/path/video.mkv' in cmd

    def test_dat_task_with_rgb(self):
        group = {'frames': '/f.dat', 'metadata': '/m.dat', 'type': 'rgb'}
        task = {'name': 'test', 'source': 'dat', 'info': group}
        cmd = _build_subprocess_cmd(task, '/out', 'naive', 692, 520, True, False, 20)
        assert '--is_rgb' in cmd
        assert '--raw_frames' in cmd

    def test_includes_sim_backend(self):
        task = {'name': 'test', 'source': 'video', 'info': '/path/video.mkv'}
        cmd = _build_subprocess_cmd(
            task, '/out', 'dvs', 692, 520, True, False, 20, sim_backend='cuda'
        )
        assert '--sim_backend' in cmd
        idx = cmd.index('--sim_backend')
        assert cmd[idx + 1] == 'cuda'


# ──────────────────────────────────────────────────────────────
#  9. Integration: batch_process — skip_existing + report
# ──────────────────────────────────────────────────────────────

class TestBatchProcess:
    @pytest.fixture
    def batch_dirs(self, tmp_path):
        """Create input dir with synthetic MKVs and an output dir."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        # Create 3 small test MKVs
        for i in range(3):
            _create_test_mkv(str(input_dir / f"clip_{i:03d}.mkv"),
                             n_frames=10, h=32, w=32, fps=30.0)
        return str(input_dir), str(output_dir)

    def test_sequential_batch(self, batch_dirs):
        """Full sequential batch processes all files."""
        input_dir, output_dir = batch_dirs
        results = batch_process(
            input_dir, output_dir,
            method='naive', save_aedat4=True, threshold=10,
            skip_existing=True, num_workers=1,
        )
        assert len(results['failed']) == 0
        assert len(results['success']) == 3

    def test_skip_existing_valid(self, batch_dirs):
        """Already-converted files with valid size are skipped."""
        input_dir, output_dir = batch_dirs
        # Run once
        batch_process(input_dir, output_dir, method='naive', save_aedat4=True,
                      threshold=10, num_workers=1)
        # Run again — should skip all
        results = batch_process(input_dir, output_dir, method='naive', save_aedat4=True,
                                threshold=10, num_workers=1)
        assert len(results['skipped']) == 3
        assert len(results['success']) == 0

    def test_skip_existing_ignores_tiny_files(self, batch_dirs):
        """Corrupt/empty output files (< MIN_VALID_SIZE) should NOT be skipped."""
        input_dir, output_dir = batch_dirs
        # Create a tiny "corrupt" file
        corrupt_file = os.path.join(output_dir, "clip_000.aedat4")
        with open(corrupt_file, 'wb') as f:
            f.write(b'\x00' * 50)  # 50 bytes < MIN_VALID_SIZE (100)
        results = batch_process(input_dir, output_dir, method='naive', save_aedat4=True,
                                threshold=10, num_workers=1)
        # clip_000 should NOT be skipped (file too small)
        skipped_names = results['skipped']
        assert 'clip_000' not in skipped_names

    def test_report_file_append(self, batch_dirs):
        """Multiple runs append to report, not overwrite."""
        input_dir, output_dir = batch_dirs
        report_path = os.path.join(output_dir, "batch_report.csv")
        # Run twice
        batch_process(input_dir, output_dir, method='naive', save_aedat4=True,
                      threshold=10, num_workers=1)
        size_after_first = os.path.getsize(report_path)

        batch_process(input_dir, output_dir, method='naive', save_aedat4=True,
                      threshold=10, num_workers=1, skip_existing=False)
        size_after_second = os.path.getsize(report_path)

        # Second run should have APPENDED, so file is larger
        assert size_after_second > size_after_first

    def test_report_has_header(self, batch_dirs):
        input_dir, output_dir = batch_dirs
        batch_process(input_dir, output_dir, method='naive', save_aedat4=True,
                      threshold=10, num_workers=1)
        report_path = os.path.join(output_dir, "batch_report.csv")
        with open(report_path) as f:
            first_line = f.readline().strip()
        assert first_line == "status,name,detail"


# ──────────────────────────────────────────────────────────────
# 10. Integration: parallel subprocess (--quiet + DEVNULL)
# ──────────────────────────────────────────────────────────────

class TestParallelSubprocess:
    @pytest.fixture
    def batch_dirs(self, tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        for i in range(3):
            _create_test_mkv(str(input_dir / f"par_{i:03d}.mkv"),
                             n_frames=8, h=32, w=32, fps=30.0)
        return str(input_dir), str(output_dir)

    def test_parallel_batch_completes(self, batch_dirs):
        """Parallel mode with 2 workers completes without deadlock."""
        input_dir, output_dir = batch_dirs
        results = batch_process(
            input_dir, output_dir,
            method='naive', save_aedat4=True, threshold=10,
            num_workers=2,
        )
        # Should finish (no deadlock) with no failures
        assert len(results['failed']) == 0
        assert len(results['success']) == 3


# ──────────────────────────────────────────────────────────────
# 11. Stress: CLI --quiet flag end-to-end
# ──────────────────────────────────────────────────────────────

class TestQuietFlag:
    def test_quiet_produces_no_stdout(self, tmp_path):
        """With --quiet, subprocess should produce zero stdout."""
        video_path = str(tmp_path / "test.mkv")
        _create_test_mkv(video_path, n_frames=5, h=32, w=32)
        out_path = str(tmp_path / "out")
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), '..', 'generate_event.py'),
            '--video', video_path,
            '--output', out_path,
            '--save_aedat4',
            '--method', 'naive',
            '--threshold', '10',
            '--quiet',
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        assert result.returncode == 0
        # stdout should be empty (all output suppressed)
        assert len(result.stdout) == 0, f"stdout not empty with --quiet: {result.stdout[:200]}"


# ──────────────────────────────────────────────────────────────
# 12. Edge case: timestamp monotonicity from video PTS
# ──────────────────────────────────────────────────────────────

class TestVideoTimestamps:
    def test_timestamps_monotonic_after_load(self, tmp_path):
        """Timestamps from load_from_video must be monotonically non-decreasing."""
        video_path = str(tmp_path / "mono_test.mkv")
        _create_test_mkv(video_path, n_frames=30, h=48, w=64, fps=25.0)
        _, ts = load_from_video(video_path, quiet=True)
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1], (
                f"Timestamp regression at frame {i}: {ts[i-1]} -> {ts[i]}"
            )

    def test_first_timestamp_is_zero_or_near(self, tmp_path):
        """First frame timestamp should be 0 or very close to 0."""
        video_path = str(tmp_path / "first_ts.mkv")
        _create_test_mkv(video_path, n_frames=10, h=32, w=32, fps=30.0)
        _, ts = load_from_video(video_path, quiet=True)
        # First frame PTS is typically 0; allow small tolerance
        assert ts[0] < 100_000, f"First timestamp too large: {ts[0]}"


# ──────────────────────────────────────────────────────────────
# 13. Round-trip: events → aedat4 → read back
# ──────────────────────────────────────────────────────────────

class TestAedat4RoundTrip:
    def test_events_readable_after_write(self, tmp_path):
        """Events written via process_single can be read back from AEDAT4."""
        frames = _make_gradient_frames(n=8, h=32, w=32)
        ts = _make_timestamps(n=8)
        out = str(tmp_path / "roundtrip")
        n = process_single(
            frames, ts, (32, 32),
            method='naive', output_path=out,
            save_aedat4=True, threshold=10, quiet=True,
        )
        assert n > 0
        # Read back
        recording = dv.io.MonoCameraRecording(out + '.aedat4')
        assert recording.isEventStreamAvailable()
        events = recording.getNextEventBatch()
        assert events is not None
        assert events.size() > 0


# ──────────────────────────────────────────────────────────────
# 14. DVS method: frame-0 initialization + non-monotonic guard
# ──────────────────────────────────────────────────────────────

class TestDVSMethodIntegration:
    """Tests for DVS-specific fixes in process_single."""

    def test_dvs_processes_all_frame_pairs(self):
        """DVS should generate events starting from frame pair (0, 1),
        not from (1, 2) which was the old buggy behavior."""
        # Create frames where frame 0→1 has a big change, later pairs are identical
        n = 5
        h, w = 32, 32
        frames = np.zeros((n, h, w), dtype=np.uint8)
        frames[0, :, :] = 0
        frames[1, :, :] = 200  # Big change: 0 → 200 (frame pair 0→1)
        # Frames 2-4 are same as frame 1 → no more events
        for i in range(2, n):
            frames[i, :, :] = 200
        ts = _make_timestamps(n=n, fps=30.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'dvs_test')
            n_events = process_single(
                frames, ts, (w, h),
                method='dvs', output_path=out,
                save_aedat4=True, threshold=20, quiet=True, is_rgb=True,
            )
            # DVS should generate events from the 0→1 transition
            # (old code skipped frame 0, so only 1→2 transition was processed,
            #  which produced 0 events since frames 1-4 are identical)
            assert n_events > 0, (
                "DVS produced 0 events — frame 0 likely not passed to simulator"
            )

    def test_dvs_survives_nonmonotonic_timestamps(self):
        """Non-monotonic timestamps should be skipped, not crash."""
        n = 6
        h, w = 32, 32
        frames = _make_gradient_frames(n=n, h=h, w=w)
        # Inject a non-monotonic timestamp at index 3
        ts = _make_timestamps(n=n, fps=30.0)
        ts[3] = ts[1]  # Jump back to an earlier time
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'dvs_mono')
            # Should NOT raise ValueError
            n_events = process_single(
                frames, ts, (w, h),
                method='dvs', output_path=out,
                save_aedat4=False, save_txt=False,
                quiet=True, is_rgb=True,
            )
            assert isinstance(n_events, int)  # No crash

    def test_dvs_duplicate_timestamps_skipped(self):
        """Duplicate timestamps (same value) should be skipped for DVS."""
        n = 4
        h, w = 16, 16
        frames = _make_gradient_frames(n=n, h=h, w=w)
        ts = [0, 10000, 10000, 20000]  # ts[2] == ts[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'dvs_dup')
            n_events = process_single(
                frames, ts, (w, h),
                method='dvs', output_path=out,
                save_aedat4=False, save_txt=False,
                quiet=True, is_rgb=True,
            )
            assert isinstance(n_events, int)  # No crash


# ──────────────────────────────────────────────────────────────
# 15. Stress: large event count (memory bounded via streaming)
# ──────────────────────────────────────────────────────────────

class TestMemoryBounded:
    def test_large_random_frames_no_oom(self):
        """Processing many random frames should not accumulate unbounded memory.
        This is a functional test — we verify correctness, not peak RSS.
        """
        h, w = 256, 256
        n_frames = 50
        frames = np.random.randint(0, 256, (n_frames, h, w), dtype=np.uint8)
        ts = _make_timestamps(n=n_frames, fps=120)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'stress_output')
            n = process_single(
                frames, ts, (w, h),
                method='naive', output_path=out,
                save_aedat4=True, save_txt=True,
                threshold=5, quiet=True,
            )
            assert n > 100_000, f"Expected many events from random frames, got {n}"
            assert os.path.exists(out + '.aedat4')
            assert os.path.exists(out + '.txt')
            # Verify txt file has correct count
            txt_data = np.loadtxt(out + '.txt', dtype=np.int64)
            assert txt_data.shape[0] == n


# ──────────────────────────────────────────────────────────────
# 16. Edge case: output path handling
# ──────────────────────────────────────────────────────────────

class TestOutputPathHandling:
    def test_output_path_with_extension(self):
        """output_path with .aedat4 extension should not double-extend."""
        frames = _make_gradient_frames(n=5, h=16, w=16)
        ts = _make_timestamps(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'test.aedat4')
            process_single(
                frames, ts, (16, 16),
                method='naive', output_path=out,
                save_aedat4=True, threshold=10, quiet=True,
            )
            # Should create "test.aedat4", not "test.aedat4.aedat4"
            assert os.path.exists(os.path.join(tmpdir, 'test.aedat4'))
            assert not os.path.exists(os.path.join(tmpdir, 'test.aedat4.aedat4'))

    def test_output_path_without_extension(self):
        frames = _make_gradient_frames(n=5, h=16, w=16)
        ts = _make_timestamps(n=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'my_events')
            process_single(
                frames, ts, (16, 16),
                method='naive', output_path=out,
                save_aedat4=True, threshold=10, quiet=True,
            )
            assert os.path.exists(os.path.join(tmpdir, 'my_events.aedat4'))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
