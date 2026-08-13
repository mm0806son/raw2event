#!/usr/bin/env python3
"""Associate each real event with the luminance of its bracketing frames.

Writes ``events_with_luminance_{source}.pt``, a float32 ``(N, 7)`` tensor with
columns ``[timestamp_us, x, y, polarity, prev_lum, next_lum, frame_dt_us]``.
Prerequisite for ``k_estimate``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import src.process_data.file_read as file_read  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI_IMAGE_WIDTH = 346
PI_IMAGE_HEIGHT = 260
BATCH_SIZE = 1_000_000


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Associate DV events with frame luminance for K calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--data_dir",
        help="Single calibration data directory (contains aedat4, raw/rgb .dat, metadata .dat)",
    )
    grp.add_argument(
        "--root_dir",
        help="Root directory containing multiple calibration subfolders",
    )
    p.add_argument(
        "--source",
        choices=["raw", "rgb", "both"],
        default="raw",
        help="Which luminance source(s) to compute",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <data_dir>/frames_analysis_full/)",
    )
    p.add_argument(
        "--file_suffix",
        default=None,
        help="Common token in filenames for find_matching_files (e.g. '20250509')",
    )
    p.add_argument(
        "--no_gpu", action="store_true", help="Force CPU even if CUDA available"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_calibration_files(data_dir: str, suffix: str | None) -> dict[str, str]:
    """Find calibration files by suffix or by file-type pattern matching.

    Supports both current naming (``dv_output_*.aedat4``) and legacy naming
    (``time_filtered_dv_events*.aedat4``).  Raises if a file type has multiple
    candidates (use ``--file_suffix`` to disambiguate).
    """
    if suffix is not None:
        return file_read.find_matching_files(data_dir, suffix)

    import re

    # Each type maps to a list of patterns tried in priority order.
    type_patterns: dict[str, list[re.Pattern]] = {
        "dv": [
            re.compile(r"dv_output_.*\.aedat4$"),
            re.compile(r"time_filtered_dv_events.*\.aedat4$"),
        ],
        "metadata": [re.compile(r"metadata_.*\.dat$")],
        "raw_frames": [re.compile(r"raw_frames_.*\.dat$")],
        "rgb_frames": [re.compile(r"rgb_frames_.*\.dat$")],
    }
    files: dict[str, str] = {}
    listing = sorted(os.listdir(data_dir))

    for ftype, pats in type_patterns.items():
        candidates: list[str] = []
        for pat in pats:
            candidates = [f for f in listing if pat.search(f)]
            if candidates:
                break
        if len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple candidates for '{ftype}' in {data_dir}: "
                f"{candidates}. Use --file_suffix to disambiguate."
            )
        if candidates:
            files[ftype] = os.path.join(data_dir, candidates[0])

    missing = [k for k in type_patterns if k not in files]
    if missing:
        raise FileNotFoundError(
            f"Cannot auto-detect calibration files in {data_dir}. "
            f"Missing: {', '.join(missing)}. Use --file_suffix."
        )
    return files


# ---------------------------------------------------------------------------
# Core: event-luminance association
# ---------------------------------------------------------------------------


def _prepare_gray_frames(
    frames: np.ndarray | torch.Tensor,
    is_rgb: bool,
) -> np.ndarray:
    """Convert frame stack to a contiguous float64 luminance array (N, H, W).

    RAW Bayer mosaics are debayered to BT.601 luminance Y via
    ``file_read.bayer_mosaic_to_luminance`` so that the calibration target
    matches the monochromatic DV sensor response. RGB frames (in R, G, B
    channel order after the fix in ``read_rgb_frames``) use the same BT.601
    coefficients: Y = 0.299*R + 0.587*G + 0.114*B.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    frames = np.asarray(frames)

    if is_rgb and frames.ndim == 4 and frames.shape[3] >= 3:
        gray = (
            0.299 * frames[:, :, :, 0].astype(np.float64)
            + 0.587 * frames[:, :, :, 1].astype(np.float64)
            + 0.114 * frames[:, :, :, 2].astype(np.float64)
        )
    elif frames.ndim == 4 and frames.shape[3] == 1:
        gray = file_read.bayer_mosaic_to_luminance(frames[:, :, :, 0]).astype(np.float64)
    else:
        gray = file_read.bayer_mosaic_to_luminance(frames).astype(np.float64)

    return np.ascontiguousarray(gray)  # (N, H, W) float64


def _lookup_pixel_luminance(
    gray_frames: np.ndarray,
    frame_indices: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """Vectorized pixel luminance lookup for a batch of (frame_idx, x, y).

    Parameters
    ----------
    gray_frames : (N, H, W) float64 grayscale frames
    frame_indices, x_coords, y_coords : 1-D int arrays, same length

    Returns
    -------
    1-D float64 array of luminance values.
    """
    return gray_frames[frame_indices, y_coords, x_coords]


def compute_event_luminance(
    events: np.ndarray,
    frames: np.ndarray | torch.Tensor,
    timestamps: np.ndarray,
    is_rgb: bool,
    batch_size: int = BATCH_SIZE,
) -> torch.Tensor | None:
    """Associate each event with prev/next frame luminance.

    Parameters
    ----------
    events : (M, 4) float64 -- [t_us, x, y, polarity]
    frames : (N, H, W) or (N, H, W, C) -- Pi frames
    timestamps : (N,) float64 -- Pi frame timestamps in microseconds
    is_rgb : whether frames are RGB (need grayscale conversion)

    Returns
    -------
    Tensor of shape (K, 7): [t, x, y, p, prev_lum, next_lum, dt_frames]
    where K <= M (events outside frame coverage are dropped).
    """
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    n_frames = len(ts_arr)
    if n_frames < 2:
        print("  WARNING: Need at least 2 frames for bracketing.")
        return None

    # Pre-convert all frames to grayscale float64 once
    gray_frames = _prepare_gray_frames(frames, is_rgb)
    h, w = gray_frames.shape[1], gray_frames.shape[2]

    results: list[np.ndarray] = []
    n_events = events.shape[0]
    source_label = "RGB" if is_rgb else "RAW"

    for start in tqdm(
        range(0, n_events, batch_size),
        desc=f"Luminance ({source_label})",
        unit="batch",
    ):
        end = min(start + batch_size, n_events)
        batch = events[start:end]

        ev_t = batch[:, 0]
        ev_x = np.round(batch[:, 1]).astype(np.intp)
        ev_y = np.round(batch[:, 2]).astype(np.intp)

        # Find bracketing frame indices via searchsorted
        next_idx = np.searchsorted(ts_arr, ev_t)

        # Build validity mask: event must be strictly between first and
        # last frame timestamps, and pixel coords must be in bounds.
        valid = (
            (next_idx >= 1)
            & (next_idx < n_frames)
            & (ev_x >= 0)
            & (ev_x < w)
            & (ev_y >= 0)
            & (ev_y < h)
        )
        if not valid.any():
            continue

        # Restrict to valid events
        b_valid = batch[valid]
        nx = next_idx[valid]
        px = nx - 1
        x_v = ev_x[valid]
        y_v = ev_y[valid]

        # Frame time difference
        dt = ts_arr[nx] - ts_arr[px]

        # Vectorized luminance lookups
        prev_lum = _lookup_pixel_luminance(gray_frames, px, x_v, y_v)
        next_lum = _lookup_pixel_luminance(gray_frames, nx, x_v, y_v)

        out = np.column_stack(
            [
                b_valid[:, :4],  # t, x, y, p
                prev_lum,
                next_lum,
                dt,
            ]
        )
        results.append(out)

    if not results:
        return None

    all_data = np.concatenate(results, axis=0)
    return torch.from_numpy(all_data).double()


# ---------------------------------------------------------------------------
# Single-folder processing
# ---------------------------------------------------------------------------


def process_folder(
    data_dir: str,
    source: str,
    output_dir: str | None,
    file_suffix: str | None,
    device: torch.device,
) -> None:
    """Run event-luminance preprocessing for one calibration folder."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {data_dir}")
    print(f"{'=' * 60}")
    t0 = time.time()

    # Discover files
    files = find_calibration_files(data_dir, file_suffix)

    # Output directory
    if output_dir is None:
        out_dir = os.path.join(data_dir, "frames_analysis_full")
    else:
        out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load DV events
    print("Loading DV events ...")
    dv_events = file_read.load_events(files["dv"])
    if dv_events is None or dv_events.shape[0] == 0:
        print("ERROR: No events found in AEDAT4 file.")
        return
    events_np = dv_events.numpy().astype(np.float64)
    print(f"  {events_np.shape[0]} events loaded.")

    # Load timestamps -- use RealTime (Unix microseconds) so they share
    # the same epoch as DV event timestamps.  SensorTimestamp is a Pi-internal
    # clock that starts near zero and cannot be compared to DV times.
    sensor_ts, real_ts = file_read.read_metadata(files["metadata"])
    if real_ts is None or len(real_ts) == 0:
        print("ERROR: Cannot read metadata timestamps.")
        return
    ts_np = np.asarray(real_ts, dtype=np.float64)
    print(f"  {len(ts_np)} frame timestamps loaded.")

    # Determine which sources to process
    sources = ["raw", "rgb"] if source == "both" else [source]

    for src in sources:
        print(f"\n--- Source: {src} ---")
        # Load frames
        if src == "raw":
            frames = file_read.read_raw_frames(
                files["raw_frames"], PI_IMAGE_HEIGHT, PI_IMAGE_WIDTH
            )
            is_rgb = False
        else:
            frames = file_read.read_rgb_frames(
                files["rgb_frames"], PI_IMAGE_HEIGHT, PI_IMAGE_WIDTH
            )
            is_rgb = True

        if frames is None:
            print(f"  ERROR: Cannot read {src} frames.")
            continue
        print(f"  {len(frames)} frames loaded, shape={frames[0].shape}")

        # Compute luminance
        result = compute_event_luminance(events_np, frames, ts_np, is_rgb)

        if result is None or result.shape[0] == 0:
            print(f"  WARNING: No valid events for {src}.")
            continue

        # Save
        out_path = os.path.join(out_dir, f"events_with_luminance_{src}.pt")
        torch.save(result.cpu(), out_path)
        print(f"  Saved {result.shape[0]} events -> {out_path}")
        print(
            "  Columns: [timestamp_us, x, y, polarity, prev_lum, next_lum, "
            "frame_time_diff_us]"
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    device = torch.device("cpu")
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU.")

    if args.data_dir:
        # Single folder mode
        process_folder(
            args.data_dir, args.source, args.output_dir, args.file_suffix, device
        )
    else:
        # Batch mode: iterate subfolders
        root = args.root_dir
        subfolders = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        if not subfolders:
            print(f"No subfolders found under {root}.")
            return
        print(f"Found {len(subfolders)} subfolders.")
        for sf in subfolders:
            sf_path = os.path.join(root, sf)
            # In batch mode, --output_dir is treated as root output dir;
            # each subfolder gets its own sub-directory to avoid overwrite.
            if args.output_dir is not None:
                sf_out = os.path.join(args.output_dir, sf)
            else:
                sf_out = None  # default: <data_dir>/frames_analysis_full/
            try:
                process_folder(
                    sf_path, args.source, sf_out, args.file_suffix, device
                )
            except Exception as e:
                print(f"FAILED: {sf}: {e}")


if __name__ == "__main__":
    main()
