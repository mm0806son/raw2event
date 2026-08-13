"""
Event Generation from Frames

This script converts frame sequences into event data.

Supported input formats:
- Video files (.mp4, etc.)
- Raw frame files (.npy, .dat) with metadata
- AEDAT4 files (for reading existing events)

Event generation methods:
- Naive: Simple frame difference thresholding
- DVS: Dynamic Vision Sensor simulation

Output formats:
- Text files (.txt) with event data
- AEDAT4 files (.aedat4) for compatibility with event camera tools
"""

import numpy as np
from src.simulator import EventSim
from src.config import cfg, set_camera_type
from src.process_data.file_read import bayer_mosaic_to_luminance
import os
import sys
import gc
import glob
import time
import subprocess
from datetime import datetime
from tqdm import tqdm
import argparse
import cv2
import dv_processing as dv


# Optionally import pandas if you need CSV support.
import pandas as pd

def load_metadata(file_path):
    """
    Loads metadata from a file by inspecting the file extension.
    Supports: .npy, .dat, .txt, .csv.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.npy':
        return np.load(file_path, allow_pickle=True)
    elif ext == '.dat':
        # Define metadata data type
        metadata_dtype = np.dtype([
            ('SensorTimestamp', 'float64'),
            ('RealTime', 'S30')  # String type, maximum length 30
        ])
        # Get file size and calculate frame count
        file_size = os.path.getsize(file_path)
        frame_count = file_size // metadata_dtype.itemsize
        # Use memmap to read data
        return np.memmap(file_path, dtype=metadata_dtype, mode='r', shape=(frame_count,))
    elif ext in ['.txt']:
        try:
            metadata = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Could not load {file_path} using np.genfromtxt: {e}")
        return metadata

    elif ext == '.csv':
        # For CSV files, use pandas for robust parsing.
        try:
            metadata = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not load {file_path} using pandas.read_csv: {e}")
        return metadata

    else:
        raise ValueError(f"Unsupported metadata file type: {ext}")

def extract_timestamps(metadata):
    """
    Extracts timestamps from the loaded metadata.
    Supports NumPy structured arrays and pandas DataFrames.
    Looks for 'SensorTimestamp' or 'timestamp' field/column.
    """
    # Case 1: NumPy structured array (e.g., from np.genfromtxt)
    if isinstance(metadata, np.ndarray) and metadata.dtype.names is not None:
        if 'SensorTimestamp' in metadata.dtype.names:
            return [int(t) for t in metadata['SensorTimestamp']]
        elif 'timestamp' in metadata.dtype.names:
            return [int(t) for t in metadata['timestamp']]
        else:
            raise KeyError("No 'SensorTimestamp' or 'timestamp' field in metadata.")

    # Case 2: pandas DataFrame (e.g., from CSV)
    elif isinstance(metadata, pd.DataFrame):
        if 'SensorTimestamp' in metadata.columns:
            return metadata['SensorTimestamp'].astype(int).tolist()
        elif 'timestamp' in metadata.columns:
            return metadata['timestamp'].astype(int).tolist()
        else:
            raise KeyError("No 'SensorTimestamp' or 'timestamp' column in metadata.")

    # Otherwise, unsupported metadata type.
    raise TypeError("Unsupported metadata type. Please use .npy, .dat, .txt, or .csv with an appropriate format.")

def load_from_file(raw_frames_path, metadata_path, frame_width=None, frame_height=None,
                   is_rgb=False, quiet=False):
    """
    Loads raw frames and timestamps from the provided file paths.
    Supports both .npy and .dat formats for raw frames.
    
    Args:
        raw_frames_path: Path to the raw frames file
        metadata_path: Path to the metadata file
        frame_width: Width of each frame (required for .dat files)
        frame_height: Height of each frame (required for .dat files)
        is_rgb: Whether the input is RGB data (4 channels)
        quiet: if True, suppress verbose output (for batch mode)
    """
    # Select loading method based on file extension
    if raw_frames_path.endswith('.npy'):
        raw_frames = np.load(raw_frames_path)
        if not quiet:
            print(f"Input resolution: {raw_frames.shape[2]}x{raw_frames.shape[1]} (npy)")
    elif raw_frames_path.endswith('.dat'):
        if frame_width is None or frame_height is None:
            raise ValueError("frame_width and frame_height must be specified for .dat files")
            
        # Select dtype based on data type
        if is_rgb:
            dtype = np.dtype('uint8')
            channels = 4  # BGRA
        else:
            dtype = np.dtype('uint16')
            channels = 1
            
        # Get file size and calculate frame count
        file_size = os.path.getsize(raw_frames_path)
        frame_size = frame_width * frame_height * channels * dtype.itemsize
        frame_count = file_size // frame_size
        
        # Use memmap to read data
        if is_rgb:
            raw_frames = np.memmap(raw_frames_path, dtype=dtype, mode='r',
                                 shape=(frame_count, frame_height, frame_width, channels))
            if not quiet:
                print(f"Input resolution: {frame_width}x{frame_height} (RGB)")
            # Convert to grayscale
            gray_frames = []
            for frame in raw_frames:
                # Remove alpha channel
                rgb_frame = frame[:, :, :3]
                # Convert to grayscale
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                gray_frames.append(gray_frame)
            raw_frames = np.array(gray_frames, dtype=np.uint16)
        else:
            raw_frames = np.memmap(raw_frames_path, dtype=dtype, mode='r',
                                 shape=(frame_count, frame_height, frame_width))
            if not quiet:
                print(f"Input resolution: {frame_width}x{frame_height} (dat)")
    else:
        raise ValueError(f"Unsupported raw frames format: {raw_frames_path}")
        
    metadata = load_metadata(metadata_path)
    timestamps_us = extract_timestamps(metadata)
    return raw_frames, timestamps_us

def _ffprobe_video_info(video_path):
    """Query video stream properties via ffprobe.

    Returns dict with width, height, pix_fmt, fps, nb_frames (may be None).
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=0",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE).decode("utf-8", "replace")
    except FileNotFoundError as exc:
        raise ValueError(
            f"ffprobe binary not found (required for {video_path}): {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
        raise ValueError(
            f"ffprobe failed (rc={exc.returncode}) for {video_path}: {stderr.strip()}"
        ) from exc

    info = {}
    for line in out.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    try:
        width = int(info["width"])
        height = int(info["height"])
        pix_fmt = info["pix_fmt"]
    except KeyError as exc:
        raise ValueError(f"ffprobe missing core field for {video_path}: {info}") from exc

    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = info.get(key, "0/0")
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                num_f, den_f = float(num), float(den)
                if den_f > 0 and num_f > 0:
                    fps = num_f / den_f
                    break
            except ValueError:
                continue
    if fps <= 0:
        raise ValueError(f"Could not determine FPS for {video_path}: {info}")

    nb_frames = None
    for key in ("nb_frames", "nb_read_frames"):
        raw = info.get(key, "")
        if raw and raw.isdigit():
            val = int(raw)
            if val > 0:
                nb_frames = val
                break

    return {
        "width": width, "height": height, "pix_fmt": pix_fmt,
        "fps": fps, "nb_frames": nb_frames,
    }


# pix_fmt families we accept as "compatible" for each decode target. This
# prevents silently demosaicing an accidentally-mislabeled RGB file as raw
# Bayer data, or vice versa.
_PIX_FMT_COMPAT = {
    # dat2mkv.py always writes gray16le for RAW; keep the whitelist tight so
    # we fail fast on anything else rather than silently `<< 6`-ing data that
    # may not actually be 10-bit values.
    "gray16le": {"gray16le"},
    "bgr24": {"bgr0", "bgra", "rgba", "bgr24", "rgb24", "yuv420p", "yuvj420p",
              "yuv422p", "yuv444p", "nv12"},
}


def _ffmpeg_decode_frames(video_path, out_pix_fmt, channels, bytes_per_sample,
                          width, height, n_frames_hint=None):
    """Stream-decode a video into a pre-allocated numpy array via ffmpeg pipe.

    Reads one frame_size worth of bytes at a time from ffmpeg's stdout, so
    peak memory is one decoded frame plus the final output array — not the
    entire decoded stream as with ``communicate()``.

    Returns an array shaped (N, H, W) for channels==1, else (N, H, W, C).
    Dtype is uint16 for bytes_per_sample==2, uint8 for ==1.
    """
    import threading

    dtype = np.uint16 if bytes_per_sample == 2 else np.uint8
    frame_elems = height * width * channels
    frame_size = frame_elems * bytes_per_sample
    if frame_size <= 0:
        raise ValueError(f"Invalid frame_size={frame_size} for {video_path}")

    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", out_pix_fmt, "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"ffmpeg binary not found (required for {video_path}): {exc}"
        ) from exc

    # Drain stderr in a background thread so ffmpeg never blocks on a full
    # stderr pipe while we're busy reading stdout.
    stderr_chunks = []

    def _drain_stderr():
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    per_frame_shape = (height, width) if channels == 1 else (height, width, channels)
    # Pre-allocate with hint when available; otherwise start with a modest
    # chunk and grow by concatenation (which is O(total bytes copied) but
    # bounded — no single allocation exceeds 2x current size).
    alloc = max(n_frames_hint or 512, 512)
    chunks = []
    current = np.empty((alloc,) + per_frame_shape, dtype=dtype)
    current_fill = 0
    frame_buf = bytearray(frame_size)
    mv = memoryview(frame_buf)
    total = 0

    try:
        while True:
            # Fill frame_buf fully — stdout.read may return short reads.
            filled = 0
            while filled < frame_size:
                n = proc.stdout.readinto(mv[filled:])
                if not n:
                    break
                filled += n
            if filled == 0:
                break  # clean EOF at a frame boundary
            if filled != frame_size:
                raise ValueError(
                    f"Incomplete final frame from ffmpeg for {video_path}: "
                    f"got {filled}/{frame_size} bytes"
                )

            if current_fill >= current.shape[0]:
                chunks.append(current[:current_fill])
                next_cap = max(1024, current.shape[0])
                current = np.empty((next_cap,) + per_frame_shape, dtype=dtype)
                current_fill = 0

            np.copyto(
                current[current_fill],
                np.frombuffer(frame_buf, dtype=dtype).reshape(per_frame_shape),
            )
            current_fill += 1
            total += 1
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        ret = proc.wait()
        stderr_thread.join(timeout=5.0)
        try:
            proc.stderr.close()
        except Exception:
            pass

    if ret != 0:
        err_msg = b"".join(stderr_chunks).decode("utf-8", "replace").strip()
        raise ValueError(
            f"ffmpeg decode failed (rc={ret}) for {video_path}: "
            f"{err_msg or '(no stderr)'}"
        )
    if total == 0:
        raise ValueError(f"No frames decoded from {video_path}")

    if not chunks:
        # Happy path: hint was accurate or exceeded actual frame count.
        return current[:current_fill].copy()

    chunks.append(current[:current_fill])
    return np.concatenate(chunks, axis=0)


def load_from_video(video_path, quiet=False):
    """Load raw frames and generate timestamps from a video file via ffmpeg.

    Uses ffmpeg subprocess uniformly for all video types so that 16-bit
    pixel formats (e.g. gray16le) preserve full precision — cv2.VideoCapture
    silently downcasts such streams to 8-bit.

    Handles different MKV types based on filename convention:
      - *_raw_10bit.mkv: 16-bit-container gray16le whose values already sit in
        the 0-1023 10-bit range (``dat2mkv.py`` applies ``(v >> 6) & 0x3FF``
        before encoding).  We pass them through *as-is* so that downstream
        frames share the same luminance scale as ``read_raw_frames`` on the
        calibration ``.dat`` (max ~792, well within 10-bit) — the domain K
        was fit against.  The Bayer mosaic is not demosaiced here because
        the calibration pipeline (k_preprocess.py / k_optimize.py) and the
        DVS simulator both operate directly on the raw mosaic; downstream
        tag detection applies its own demosaic when needed.
      - *_raw.mkv:       16-bit raw grayscale (gray16le) -> uint16 as-is.
      - *_rgb.mkv / other: 8-bit color -> grayscale uint8.
    """
    basename_lower = os.path.basename(video_path).lower()
    if '_raw_10bit' in basename_lower:
        video_type = 'raw_10bit'
    elif '_raw' in basename_lower:
        video_type = 'raw'
    else:
        video_type = 'standard'

    info = _ffprobe_video_info(video_path)
    width = info["width"]
    height = info["height"]
    pix_fmt = info["pix_fmt"]
    fps = info["fps"]
    nb_frames = info["nb_frames"]

    # Pix_fmt sanity check: fail fast on mislabeled files rather than
    # silently producing semantically wrong data.
    if video_type in ('raw_10bit', 'raw'):
        expected = _PIX_FMT_COMPAT["gray16le"]
        target = "gray16le"
    else:
        expected = _PIX_FMT_COMPAT["bgr24"]
        target = "bgr24"
    if pix_fmt not in expected:
        raise ValueError(
            f"Unexpected pix_fmt '{pix_fmt}' for video_type='{video_type}' "
            f"({video_path}). Expected one of {sorted(expected)} to decode as "
            f"'{target}'."
        )

    if not quiet:
        print(
            f"Input resolution: {width}x{height} "
            f"(video, type={video_type}, pix_fmt={pix_fmt}, fps={fps:.2f})"
        )

    if video_type == 'raw_10bit':
        raw_frames = _ffmpeg_decode_frames(
            video_path, "gray16le", 1, 2, width, height, n_frames_hint=nb_frames,
        )
        # MKV values from dat2mkv.py are already 10-bit (0-1023), matching
        # the calibration ``.dat`` domain (measured global max ~792). Do NOT
        # left-shift here — the previous ``np.left_shift(raw_frames, 6)``
        # inflated downstream luminance by 64x versus calibration and was the
        # root cause of the order-of-magnitude downstream event-density gap.
        # See the docstring above for details.
        # Previous (incorrect) behavior:
        #     np.left_shift(raw_frames, 6, out=raw_frames)
    elif video_type == 'raw':
        raw_frames = _ffmpeg_decode_frames(
            video_path, "gray16le", 1, 2, width, height, n_frames_hint=nb_frames,
        )
    else:
        bgr = _ffmpeg_decode_frames(
            video_path, "bgr24", 3, 1, width, height, n_frames_hint=nb_frames,
        )
        raw_frames = np.empty((bgr.shape[0], height, width), dtype=np.uint8)
        for i in range(bgr.shape[0]):
            raw_frames[i] = cv2.cvtColor(bgr[i], cv2.COLOR_BGR2GRAY)
        del bgr

    dt_us = 1e6 / fps
    n = raw_frames.shape[0]
    timestamps_us = [int(i * dt_us) for i in range(n)]

    return raw_frames, timestamps_us

def _is_rgb_video(video_path):
    """Detect if a video file is RGB (not raw) based on filename convention.

    Raw videos contain '_raw_10bit' or '_raw' in the filename;
    everything else (including '_rgb') is treated as RGB/standard.
    """
    basename_lower = os.path.basename(video_path).lower()
    return '_raw_10bit' not in basename_lower and '_raw' not in basename_lower


def save_to_aedat4(events, filename="events_output.aedat4", input_resolution=None, quiet=False):
    """
    Save events to AEDAT4 format using dv package.
    events: NumPy array of shape (N, 4) with columns: timestamp, x, y, polarity
    input_resolution: tuple of (width, height) for the input frames
    quiet: if True, suppress verbose output (for batch mode)
    """
    # Work on a copy to avoid modifying the caller's array
    events = events.copy()
    events[:, 3] = (events[:, 3] > 0).astype(events.dtype)

    # Sort events by timestamp
    events = events[events[:, 0].argsort()]

    # Vectorized timestamp validation
    valid_mask = events[:, 0] >= 0
    n_invalid = int(np.sum(~valid_mask))
    if n_invalid > 0:
        if not quiet:
            print(f"Warning: Skipping {n_invalid} events with negative timestamps")
        events = events[valid_mask]

    if len(events) == 0:
        raise ValueError("No valid events to save")

    if not quiet:
        print(f"Valid events count: {len(events)}")
        print(f"Timestamp range: {events[0, 0]} - {events[-1, 0]}")

    # Define resolution
    if input_resolution is not None:
        resolution = input_resolution
    else:
        resolution = (int(np.max(events[:, 1])) + 1, int(np.max(events[:, 2])) + 1)

    # Setup writer config and write
    config = dv.io.MonoCameraWriter.Config("DVXplorer_sample")
    config.addEventStream(resolution)
    writer = dv.io.MonoCameraWriter(filename, config)

    store = dv.EventStore()
    # Convert to Python lists for faster iteration (avoids numpy scalar overhead)
    ts = events[:, 0].tolist()
    xs = events[:, 1].tolist()
    ys = events[:, 2].tolist()
    ps = events[:, 3].tolist()
    for t, x, y, p in zip(ts, xs, ys, ps):
        store.push_back(t, x, y, bool(p))
    writer.writeEvents(store)

    if not quiet:
        print(f"Resolution: {resolution}")
        print(f"Saved {len(events)} events to {filename} (AEDAT 4)")


def _flush_events_to_writer(writer, event_buffer, txt_fh=None, quiet=False):
    """Flush accumulated event chunks to AEDAT4 writer and/or text file.

    Events are concatenated, validated, sorted per-chunk, and written.
    Used by process_single() for memory-bounded streaming.

    Args:
        writer: dv.io.MonoCameraWriter instance (or None to skip AEDAT4)
        event_buffer: list of numpy arrays, each shape (N_i, 4)
        txt_fh: open file handle for text output (or None)
        quiet: suppress warnings

    Returns:
        Number of valid events flushed
    """
    if not event_buffer:
        return 0

    events = np.concatenate(event_buffer, axis=0)

    # Polarity: ensure 0/1 (events is a new concat, so no aliasing issues)
    events[:, 3] = (events[:, 3] > 0).astype(events.dtype)

    # Filter negative timestamps
    valid_mask = events[:, 0] >= 0
    n_invalid = int(np.sum(~valid_mask))
    if n_invalid > 0:
        if not quiet:
            print(f"Warning: Skipping {n_invalid} events with negative timestamps")
        events = events[valid_mask]

    if len(events) == 0:
        return 0

    # Sort by timestamp (within chunk; globally OK because frames are sequential)
    events = events[events[:, 0].argsort()]

    # Write to AEDAT4
    if writer is not None:
        store = dv.EventStore()
        ts = events[:, 0].tolist()
        xs = events[:, 1].tolist()
        ys = events[:, 2].tolist()
        ps = events[:, 3].tolist()
        for t, x, y, p in zip(ts, xs, ys, ps):
            store.push_back(t, x, y, bool(p))
        writer.writeEvents(store)
        del store

    # Write to text file
    if txt_fh is not None:
        np.savetxt(txt_fh, events, fmt="%d")

    n = len(events)
    del events
    return n


def generate_events_naive(prev_frame, curr_frame, timestamp, threshold=50):
    """
    Naive method: Generates events when the change is bigger than threshold.
    Fully vectorized with numpy (no Python loops).
    """
    if prev_frame is None or curr_frame is None:
        return None

    # Ensure frames have the same shape
    if prev_frame.shape != curr_frame.shape:
        return None

    diff = curr_frame.astype(np.int32) - prev_frame.astype(np.int32)

    # ON and OFF events
    on_coords = np.argwhere(diff > threshold)    # shape (N_on, 2) with [y, x]
    off_coords = np.argwhere(diff < -threshold)  # shape (N_off, 2) with [y, x]

    n_on = len(on_coords)
    n_off = len(off_coords)
    n_total = n_on + n_off

    if n_total == 0:
        return None

    # Build event array: [timestamp, x, y, polarity] - fully vectorized
    events = np.empty((n_total, 4), dtype=np.int64)
    events[:, 0] = timestamp

    if n_on > 0:
        events[:n_on, 1] = on_coords[:, 1]   # x
        events[:n_on, 2] = on_coords[:, 0]   # y
        events[:n_on, 3] = 1

    if n_off > 0:
        events[n_on:, 1] = off_coords[:, 1]  # x
        events[n_on:, 2] = off_coords[:, 0]  # y
        events[n_on:, 3] = 0

    return events

def load_from_aedat4(aedat4_path):
    """
    Loads raw frames and events from an AEDAT4 file.
    Returns raw_frames, timestamps_us, and events if available.
    """
    if not os.path.exists(aedat4_path):
        raise FileNotFoundError(f"AEDAT4 file not found: {aedat4_path}")
    
    recording = dv.io.MonoCameraRecording(aedat4_path)
    
    # Load frames if available
    raw_frames = []
    timestamps_us = []
    if recording.isFrameStreamAvailable():
        while True:
            frame = recording.getNextFrame()
            if frame is None:
                break
            # Convert frame to grayscale if it's not already
            if len(frame.image.shape) == 3:
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.image
            raw_frames.append(gray)
            timestamps_us.append(frame.timestamp)
    
    # Load events if available
    events = None
    if recording.isEventStreamAvailable():
        events_packets = []
        while True:
            event_batch = recording.getNextEventBatch()
            if event_batch is None:
                break
            events_packets.append(event_batch.numpy())
        if events_packets:
            events = np.concatenate(events_packets, axis=0)
    
    return np.array(raw_frames), timestamps_us, events

def process_single(raw_frames, timestamps_us, input_resolution, method='naive',
                   output_path='events_output', save_aedat4=True, save_txt=False,
                   threshold=20, quiet=False, is_rgb=False, sim_backend='auto'):
    """
    Core event generation logic for a single set of frames.
    Events are streamed to disk in chunks to avoid accumulating all in memory.

    Args:
        raw_frames: numpy array of shape (N, H, W) containing frames
        timestamps_us: list of timestamps in microseconds
        input_resolution: tuple (width, height)
        method: 'naive' or 'dvs'
        output_path: output file path (extension will be added automatically)
        save_aedat4: whether to save as .aedat4
        save_txt: whether to save as .txt
        threshold: threshold for naive method
        quiet: if True, suppress per-frame progress (for batch mode)
        is_rgb: whether the input is RGB (affects DVS camera type selection)
        sim_backend: simulator backend for DVS method ('auto', 'cuda', 'cpu', 'numpy')

    Returns:
        Number of events generated, or 0 if none
    """
    # Validate frame/timestamp count
    n_frames = len(raw_frames)
    n_ts = len(timestamps_us)
    if n_frames == 0:
        if not quiet:
            print("Warning: No frames provided.")
        return 0
    if n_ts == 0:
        if not quiet:
            print("Warning: No timestamps provided.")
        return 0
    if n_frames != n_ts:
        # Truncate BOTH to avoid memory waste and index errors
        n_use = min(n_frames, n_ts)
        if not quiet:
            print(f"Warning: Frame/timestamp count mismatch ({n_frames} frames, {n_ts} timestamps). "
                  f"Using first {n_use}.")
        timestamps_us = timestamps_us[:n_use]
        raw_frames = raw_frames[:n_use]

    if method == 'dvs':
        camera_type = 'RGB2DVS346' if is_rgb else 'Raw2DVS346'
        set_camera_type(camera_type)
        if not quiet:
            print(f"Using camera type: {camera_type}")
        sim = EventSim(cfg=cfg, output_folder='.', sim_backend=sim_backend)
        if not quiet:
            print(f"Using simulator backend: {sim.sim_backend}")

        # RAW input is a Bayer mosaic; the Raw2DVS346 K is calibrated in
        # the BT.601 luminance domain, so convert before simulation.
        if not is_rgb:
            raw_frames = bayer_mosaic_to_luminance(raw_frames)

    # --- Setup output writers upfront (streaming) ---
    output_base = os.path.splitext(output_path)[0]
    writer = None
    txt_fh = None

    if save_aedat4:
        aedat4_path = output_base + '.aedat4'
        config = dv.io.MonoCameraWriter.Config("DVXplorer_sample")
        config.addEventStream(input_resolution)
        writer = dv.io.MonoCameraWriter(aedat4_path, config)

    if save_txt:
        txt_path = output_base + '.txt'
        txt_fh = open(txt_path, 'w')

    # --- Generate events, flushing periodically to limit memory ---
    FLUSH_THRESHOLD = 500_000  # events per chunk
    event_buffer = []
    buffer_count = 0
    total_events = 0
    prev_frame = None

    frame_iter = range(len(timestamps_us))
    if not quiet:
        frame_iter = tqdm(frame_iter, desc="Generating events", unit="frame")

    try:
        for i in frame_iter:
            frame = raw_frames[i]
            timestamp = timestamps_us[i]

            if method == 'naive':
                # Naive needs a frame pair: skip first frame (initialize prev_frame)
                if prev_frame is None:
                    prev_frame = frame
                    continue
                events = generate_events_naive(prev_frame, frame, timestamp, threshold=threshold)
                prev_frame = frame
            elif method == 'dvs':
                # DVS simulator handles initialization internally (first call → returns None).
                # Guard: DVS model requires strictly increasing timestamps.
                if sim.t_previous is not None and timestamp <= sim.t_previous:
                    if not quiet:
                        print(f"Warning: Skipping frame {i} — non-monotonic timestamp "
                              f"({timestamp} <= {int(sim.t_previous)})")
                    continue
                events = sim.generate_events(frame, timestamp)
            else:
                raise ValueError(f"Unknown method: {method}")

            if events is not None:
                event_buffer.append(events)
                buffer_count += len(events)

                if buffer_count >= FLUSH_THRESHOLD:
                    total_events += _flush_events_to_writer(writer, event_buffer, txt_fh, quiet)
                    event_buffer = []
                    buffer_count = 0

        # Flush remaining events
        if event_buffer:
            total_events += _flush_events_to_writer(writer, event_buffer, txt_fh, quiet)
            event_buffer = []

    finally:
        # Always close file handles, even on error
        if txt_fh is not None:
            txt_fh.close()
        del writer  # triggers destructor → closes AEDAT4 file

    # Remove empty output files (e.g. no events generated)
    if total_events == 0:
        for f in [output_base + '.aedat4', output_base + '.txt']:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        if not quiet:
            print("No events detected.")
    elif not quiet:
        print(f"Generated {total_events} events")
        if save_aedat4:
            print(f"Saved AEDAT4: {aedat4_path}")
        if save_txt:
            print(f"Saved TXT: {txt_path}")

    return total_events


def _scan_directory_once(input_dir):
    """Scan input directory once and classify files by type.

    Uses os.scandir() for a single-pass traversal — much faster than
    multiple glob.glob() calls when the directory contains 100K+ files.

    Returns:
        dict with keys:
            'video_files': sorted list of video paths (.mkv, .mp4, .avi, .mov)
            'dat_files':   dict mapping basename -> full path for all .dat files
    """
    VIDEO_EXTS = {'.mkv', '.mp4', '.avi', '.mov'}
    video_files = []
    dat_files = {}  # basename -> full path

    try:
        with os.scandir(input_dir) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=True):
                    continue
                name = entry.name
                ext = os.path.splitext(name)[1].lower()
                if ext in VIDEO_EXTS:
                    video_files.append(entry.path)
                elif ext == '.dat':
                    dat_files[name] = entry.path
    except OSError as e:
        print(f"Error scanning directory {input_dir}: {e}")
        return {'video_files': [], 'dat_files': {}}

    video_files.sort()
    return {'video_files': video_files, 'dat_files': dat_files}


def find_dat_groups(input_dir, dat_files=None):
    """
    Find matching groups of (raw_frames|rgb_frames) + metadata .dat files.

    Naming convention:
        raw_frames_{suffix}.dat + metadata_{suffix}.dat -> raw2event_{suffix}
        rgb_frames_{suffix}.dat + metadata_{suffix}.dat -> rgb2event_{suffix}

    Args:
        input_dir: directory path (used only if dat_files is None)
        dat_files: optional pre-scanned dict {basename: full_path} from
                   _scan_directory_once(). If None, falls back to glob.

    Returns:
        List of dicts: {type, frames, metadata, suffix, output_name}
    """
    groups = []

    if dat_files is not None:
        # Fast path: use pre-scanned file list (single-pass)
        for prefix, out_prefix, ftype in [
            ("raw_frames_", "raw2event_", "raw"),
            ("rgb_frames_", "rgb2event_", "rgb"),
        ]:
            matching = sorted(
                (name, path) for name, path in dat_files.items()
                if name.startswith(prefix)
            )
            for basename, frames_path in matching:
                suffix = basename[len(prefix):-len(".dat")]
                meta_basename = f"metadata_{suffix}.dat"
                if meta_basename in dat_files:
                    groups.append({
                        'type': ftype,
                        'frames': frames_path,
                        'metadata': dat_files[meta_basename],
                        'suffix': suffix,
                        'output_name': f"{out_prefix}{suffix}"
                    })
                else:
                    print(f"⚠️ No matching metadata for: {basename}")
    else:
        # Fallback: glob-based scanning (kept for backward compatibility)
        for prefix, out_prefix in [("raw_frames_", "raw2event_"), ("rgb_frames_", "rgb2event_")]:
            pattern = os.path.join(input_dir, f"{prefix}*.dat")
            for frames_file in sorted(glob.glob(pattern)):
                basename = os.path.basename(frames_file)
                suffix = basename[len(prefix):-len(".dat")]
                metadata_file = os.path.join(input_dir, f"metadata_{suffix}.dat")
                if os.path.exists(metadata_file):
                    groups.append({
                        'type': 'raw' if prefix.startswith('raw') else 'rgb',
                        'frames': frames_file,
                        'metadata': metadata_file,
                        'suffix': suffix,
                        'output_name': f"{out_prefix}{suffix}"
                    })
                else:
                    print(f"⚠️ No matching metadata for: {basename}")
    return groups


def find_video_files(input_dir, pre_scanned=None):
    """Find all video files (.mkv, .mp4, .avi, .mov) in the directory.

    Args:
        input_dir: directory path (used only if pre_scanned is None)
        pre_scanned: optional pre-sorted list from _scan_directory_once().
                     If None, falls back to glob.
    """
    if pre_scanned is not None:
        return pre_scanned
    videos = []
    for ext in ['*.mkv', '*.mp4', '*.avi', '*.mov']:
        videos.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(videos)


def _build_subprocess_cmd(task, output_dir, method, frame_width, frame_height,
                          save_aedat4, save_txt, threshold, sim_backend='auto'):
    """Build a command line to process a single task as an independent subprocess.

    Each worker runs:  python generate_event.py --video <file> --output <path> ...
    This avoids all multiprocessing IPC issues with C++ extensions.
    """
    output_path = os.path.join(output_dir, task['name'])
    cmd = [sys.executable, os.path.abspath(__file__)]

    if task['source'] == 'video':
        cmd += ['--video', task['info']]
    else:
        group = task['info']
        cmd += ['--raw_frames', group['frames'],
                '--metadata', group['metadata'],
                '--frame_width', str(frame_width),
                '--frame_height', str(frame_height)]
        if group['type'] == 'rgb':
            cmd.append('--is_rgb')

    cmd += ['--method', method,
            '--threshold', str(threshold),
            '--sim_backend', sim_backend,
            '--output', output_path,
            '--quiet']  # suppress output in subprocess workers
    if save_aedat4:
        cmd.append('--save_aedat4')
    if save_txt:
        cmd.append('--save_txt')
    return cmd


def batch_process(input_dir, output_dir, method='naive', frame_width=692, frame_height=520,
                  save_aedat4=True, save_txt=False, threshold=20, skip_existing=True,
                  num_workers=1, sim_backend='auto'):
    """
    Batch process all matching files in input_dir.

    Discovers:
        1. .dat file groups (raw_frames/rgb_frames + metadata pairs)
        2. Video files (.mkv, .mp4, .avi, .mov)

    Args:
        input_dir: directory containing input files
        output_dir: directory for output files
        method: 'naive' or 'dvs'
        frame_width: width for .dat files
        frame_height: height for .dat files
        save_aedat4: save as .aedat4
        save_txt: save as .txt
        threshold: threshold for naive method
        skip_existing: skip files whose output already exists
        num_workers: number of parallel workers (1 = sequential, >1 = subprocess parallelism)
        sim_backend: simulator backend for DVS method ('auto', 'cuda', 'cpu', 'numpy')

    Returns:
        Dict with 'success', 'failed', 'skipped' lists
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {'success': [], 'failed': [], 'skipped': []}

    # --- Discover files (single directory scan for 100K+ file performance) ---
    print("Scanning input directory (single-pass)...")
    scan = _scan_directory_once(input_dir)
    dat_groups = find_dat_groups(input_dir, dat_files=scan['dat_files'])
    video_files = find_video_files(input_dir, pre_scanned=scan['video_files'])
    del scan  # free the intermediate dict

    # Build unified task list: (name, type, source_info)
    tasks = []
    for group in dat_groups:
        tasks.append({
            'name': group['output_name'],
            'source': 'dat',
            'info': group,
        })
    for vpath in video_files:
        basename = os.path.splitext(os.path.basename(vpath))[0]
        tasks.append({
            'name': basename,
            'source': 'video',
            'info': vpath,
        })

    total = len(tasks)
    if total == 0:
        print(f"No processable files found in: {input_dir}")
        print("  Expected: raw_frames_*.dat + metadata_*.dat pairs, or .mkv/.mp4 videos")
        return results

    # --- Pre-filter: count skippable (scan output dir once instead of N os.path.exists calls) ---
    n_skip_pre = 0
    MIN_VALID_SIZE = 100  # bytes — skip only non-trivially-sized outputs (detect corrupt/empty files)
    if skip_existing:
        existing_outputs = set()
        try:
            with os.scandir(output_dir) as entries:
                for e in entries:
                    if e.is_file():
                        try:
                            if e.stat().st_size > MIN_VALID_SIZE:
                                existing_outputs.add(e.name)
                        except OSError:
                            pass
        except OSError:
            pass
        for task in tasks:
            if task['name'] + '.aedat4' in existing_outputs:
                n_skip_pre += 1
    else:
        existing_outputs = set()

    # Clamp num_workers
    max_workers = os.cpu_count() or 1
    if num_workers <= 0:
        num_workers = max(1, max_workers - 2)  # Auto: leave 2 cores for system
    num_workers = min(num_workers, max_workers, total)

    start_time = time.time()
    print(f"{'='*60}")
    print(f"Batch processing: {input_dir}")
    print(f"  Total files: {total} ({len(dat_groups)} .dat groups + {len(video_files)} videos)")
    if skip_existing and n_skip_pre > 0:
        print(f"  Will skip: ~{n_skip_pre} already converted (skip_existing=True)")
        print(f"  To process: ~{total - n_skip_pre}")
    print(f"  Output: {output_dir}")
    print(f"  Method: {method}, Threshold: {threshold}")
    if method == 'dvs':
        print(f"  Simulator backend: {sim_backend}")
    print(f"  Workers: {num_workers}" + (" (parallel)" if num_workers > 1 else " (sequential)"))
    if dat_groups:
        print(f"  Frame size (for .dat): {frame_width}x{frame_height}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Report file path
    report_path = os.path.join(output_dir, "batch_report.csv")

    if num_workers > 1:
        # ===== Parallel processing via independent subprocesses =====
        # Each worker is a completely separate Python process (subprocess.Popen).
        # This avoids ALL issues with Python multiprocessing + C++ extensions
        # (dv_processing, PyTorch, OpenCV FFMPEG) — no pickling, no fork/spawn
        # deadlocks, no shared state.
        FILE_TIMEOUT = 600  # 10 min per file

        # Pre-filter: separate skipped vs to-process tasks (uses pre-scanned set)
        tasks_to_run = []
        for task in tasks:
            if skip_existing and task['name'] + '.aedat4' in existing_outputs:
                results['skipped'].append(task['name'])
            else:
                tasks_to_run.append(task)

        n_todo = len(tasks_to_run)
        desc = f"Batch ({num_workers}w)"

        try:
            with tqdm(total=total, desc=desc, unit="file", dynamic_ncols=True,
                      initial=len(results['skipped'])) as pbar:
                # Active subprocess pool: {proc: (task_name, start_time)}
                active = {}
                next_idx = 0  # Index into tasks_to_run

                def _launch_next():
                    """Launch the next pending task. Returns True if launched."""
                    nonlocal next_idx
                    if next_idx >= n_todo:
                        return False
                    task = tasks_to_run[next_idx]
                    next_idx += 1
                    cmd = _build_subprocess_cmd(
                        task, output_dir, method, frame_width, frame_height,
                        save_aedat4, save_txt, threshold, sim_backend
                    )
                    env = os.environ.copy()
                    if sim_backend in {'cpu', 'numpy'}:
                        # Prevent third-party CUDA auto-init when user explicitly requests non-CUDA backends.
                        env['CUDA_VISIBLE_DEVICES'] = ''
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env
                    )
                    active[proc] = (task['name'], time.time())
                    return True

                # Fill the initial worker pool
                for _ in range(min(num_workers, n_todo)):
                    _launch_next()

                # Poll loop: check for finished workers, launch replacements
                while active:
                    finished = []
                    for proc, (name, t_start) in list(active.items()):
                        ret = proc.poll()
                        if ret is not None:
                            # Process finished
                            finished.append(proc)
                            if ret == 0:
                                results['success'].append((name, 'ok'))
                            else:
                                stderr_text = proc.stderr.read().decode(
                                    'utf-8', errors='replace').strip()
                                short_err = stderr_text[-200:] if len(stderr_text) > 200 else stderr_text
                                results['failed'].append((name, f"exit {ret}: {short_err}"))
                        elif time.time() - t_start > FILE_TIMEOUT:
                            # Timeout — kill hung worker
                            proc.kill()
                            proc.wait()
                            finished.append(proc)
                            results['failed'].append((name, f"timeout ({FILE_TIMEOUT}s)"))

                    for proc in finished:
                        # Close stderr pipe to prevent FD leak over 120K files
                        if proc.stderr:
                            proc.stderr.close()
                        del active[proc]
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"ok:{len(results['success'])} fail:{len(results['failed'])}"
                        )
                        # Launch replacement worker
                        _launch_next()

                    if not finished:
                        time.sleep(0.2)  # Avoid busy-wait

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted! Killing workers...")
            for proc in list(active.keys()):
                proc.kill()
                proc.wait()
                if proc.stderr:
                    proc.stderr.close()
            print("Saving partial report...")

    else:
        # ===== Sequential processing (single worker) =====
        GC_INTERVAL = 100

        with tqdm(total=total, desc="Batch", unit="file", dynamic_ncols=True) as pbar:
            for idx, task in enumerate(tasks):
                output_path = os.path.join(output_dir, task['name'])

                # Skip existing (uses pre-scanned output set)
                if skip_existing and task['name'] + '.aedat4' in existing_outputs:
                    results['skipped'].append(task['name'])
                    pbar.update(1)
                    continue

                raw_frames = None
                timestamps_us = None
                try:
                    if task['source'] == 'dat':
                        group = task['info']
                        is_rgb = (group['type'] == 'rgb')
                        raw_frames, timestamps_us = load_from_file(
                            group['frames'], group['metadata'],
                            frame_width=frame_width, frame_height=frame_height,
                            is_rgb=is_rgb, quiet=True
                        )
                        input_resolution = (frame_width, frame_height)
                    else:
                        video_path = task['info']
                        raw_frames, timestamps_us = load_from_video(video_path, quiet=True)
                        input_resolution = (raw_frames.shape[2], raw_frames.shape[1])
                        is_rgb = _is_rgb_video(video_path)

                    n_events = process_single(
                        raw_frames, timestamps_us, input_resolution,
                        method=method, output_path=output_path,
                        save_aedat4=save_aedat4, save_txt=save_txt,
                        threshold=threshold, quiet=True, is_rgb=is_rgb, sim_backend=sim_backend
                    )

                    results['success'].append((task['name'], n_events))
                    pbar.set_postfix_str(f"ok:{len(results['success'])} fail:{len(results['failed'])}")

                except Exception as e:
                    results['failed'].append((task['name'], str(e)))
                    pbar.set_postfix_str(f"ok:{len(results['success'])} fail:{len(results['failed'])}")

                # Always free memory, even after failures
                del raw_frames, timestamps_us

                pbar.update(1)

                # Periodic garbage collection to prevent memory buildup
                if (idx + 1) % GC_INTERVAL == 0:
                    gc.collect()

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    # --- Save report to disk (append mode — preserves previous runs on restart) ---
    try:
        report_is_new = not os.path.exists(report_path)
        with open(report_path, 'a') as f:
            if report_is_new:
                f.write("status,name,detail\n")
            f.write(f"# Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"| elapsed: {hours}h{minutes}m{seconds}s\n")
            for name, n_events in results['success']:
                f.write(f"success,{name},{n_events} events\n")
            for name, err in results['failed']:
                # Escape commas/newlines in error messages
                safe_err = err.replace(',', ';').replace('\n', ' ')
                f.write(f"failed,{name},{safe_err}\n")
            for name in results['skipped']:
                f.write(f"skipped,{name},already exists\n")
    except Exception as e:
        print(f"Warning: Could not write report: {e}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"  ✅ Success: {len(results['success'])}")
    print(f"  ❌ Failed:  {len(results['failed'])}")
    print(f"  ⏭️  Skipped: {len(results['skipped'])}")
    print(f"  ⏱️  Elapsed: {hours}h {minutes}m {seconds}s")
    if total - len(results['skipped']) > 0:
        avg_ms = (elapsed * 1000) / max(len(results['success']) + len(results['failed']), 1)
        print(f"  📊 Avg per file: {avg_ms:.0f} ms")
    if results['failed']:
        n_show = min(10, len(results['failed']))
        print(f"\nFirst {n_show} failed files:")
        for name, err in results['failed'][:n_show]:
            print(f"  - {name}: {err}")
        if len(results['failed']) > n_show:
            print(f"  ... and {len(results['failed']) - n_show} more (see {report_path})")
    print(f"  📄 Full report: {report_path}")
    print(f"{'='*60}")

    return results


def main(args):
    quiet = getattr(args, 'quiet', False)

    # ========== Batch mode ==========
    if args.batch_dir:
        batch_process(
            input_dir=args.batch_dir,
            output_dir=args.output_dir,
            method=args.method,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            save_aedat4=args.save_aedat4,
            save_txt=args.save_txt,
            threshold=args.threshold,
            skip_existing=not args.overwrite,
            num_workers=args.workers,
            sim_backend=args.sim_backend,
        )
        return

    # ========== Single file mode (original behavior) ==========
    if args.aedat4:
        raw_frames, timestamps_us, events = load_from_aedat4(args.aedat4)
        if events is not None:
            if not quiet:
                print(f"Read {len(events)} events from AEDAT4 file")
            if args.save_txt:
                output_txt = f"events_from_aedat4.txt"
                np.savetxt(output_txt, events, fmt="%d")
            if args.save_aedat4:
                output_aedat4 = f"events_from_aedat4.aedat4"
                save_to_aedat4(events, filename=output_aedat4, quiet=quiet)
            return
    elif args.video:
        raw_frames, timestamps_us = load_from_video(args.video, quiet=quiet)
        input_resolution = (raw_frames.shape[2], raw_frames.shape[1])
        is_rgb_input = _is_rgb_video(args.video)
    else:
        raw_frames, timestamps_us = load_from_file(args.raw_frames, args.metadata,
                                                 args.frame_width, args.frame_height,
                                                 args.is_rgb, quiet=quiet)
        input_resolution = (args.frame_width, args.frame_height)
        is_rgb_input = args.is_rgb

    if not quiet:
        print(f"Read {len(timestamps_us)} frames")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"events_from_{'video' if args.video else 'file'}_{args.method}"

    n_events = process_single(
        raw_frames, timestamps_us, input_resolution,
        method=args.method, output_path=output_path,
        save_aedat4=args.save_aedat4, save_txt=args.save_txt,
        threshold=args.threshold, is_rgb=is_rgb_input, quiet=quiet,
        sim_backend=args.sim_backend
    )

    if not quiet:
        print("Generation finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Event generation from frames (single file or batch mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode:
  python generate_event.py --video input.mp4 --save_aedat4 --method naive
  python generate_event.py --raw_frames raw.dat --metadata meta.dat --save_aedat4 --frame_width 692 --frame_height 520

  # Batch mode (auto-discovers .dat groups and video files):
  python generate_event.py --batch_dir /path/to/data --output_dir /path/to/output --save_aedat4
  python generate_event.py --batch_dir /path/to/data --output_dir /path/to/output --save_aedat4 --workers 20
  python generate_event.py --batch_dir /path/to/data --output_dir /path/to/output --save_aedat4 --workers 0  # auto
        """
    )
    # Event generation parameters
    parser.add_argument('--method', type=str, choices=['naive', 'dvs'], default='naive', help='Event generation method')
    parser.add_argument('--threshold', type=int, default=20, help='Threshold for naive method (default: 20)')
    parser.add_argument(
        '--sim_backend', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu', 'numpy'],
        help="Simulator backend for DVS mode (default: auto)"
    )

    # Single file inputs
    parser.add_argument('--video', type=str, help='Path to input video (mp4, mkv)')
    parser.add_argument('--aedat4', type=str, help='Path to input AEDAT4 file')
    parser.add_argument('--raw_frames', type=str, default='raw_frames.npy', help='Path to raw_frames file (.npy, .dat)')
    parser.add_argument('--metadata', type=str, default='metadata.dat', help='Path to metadata file (.npy, .dat, .csv, .txt)')
    parser.add_argument('--frame_width', type=int, default=692, help='Width of each frame (for .dat files)')
    parser.add_argument('--frame_height', type=int, default=520, help='Height of each frame (for .dat files)')
    parser.add_argument('--is_rgb', action='store_true', help='Input is RGB data (4 channels)')

    # Output options
    parser.add_argument('--output', type=str, default=None, help='Output file path (without extension, for single mode)')
    parser.add_argument('--save_aedat4', action='store_true', help='Save output as AEDAT4')
    parser.add_argument('--save_txt', action='store_true', help='Save output as TXT')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress all output (used internally by subprocess workers)')

    # Batch mode
    parser.add_argument('--batch_dir', type=str, default=None,
                       help='Input directory for batch mode (auto-discovers .dat groups and video files)')
    parser.add_argument('--output_dir', type=str, default='batch_output',
                       help='Output directory for batch results (default: batch_output)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files in batch mode (default: skip)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1=sequential, 0=auto use all CPUs-2)')

    args = parser.parse_args()
    main(args)
