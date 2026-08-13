"""Helpers shared by v2e baseline scripts.

Responsibilities:
 - Load Pi RAW (10-bit Bayer) or RGB MKV frames into uint8 grayscale at 346x260.
 - Write a temp grayscale video (mp4 / image folder) suitable for v2e CLI.
 - Build & invoke the v2e CLI subprocess with a chosen threshold set + protocol.
 - Read v2e .h5 output -> events tensor [N, 4] in (t_us, x, y, p) format.
 - Apply existing AprilTag-driven event_filter pipeline + unified80 rescale.
 - Save NPZ in the same schema as Raw2Event's *_filtered_*.npz.

The wrapper scripts (run_v2e_batch, threshold_sweep) call into this module.

Pipeline insertion point (verified Stage 0.7):
    process_single_batch.py:201-215  -> events generated (here we substitute
                                        v2e events for dvs_generate.generate_events_tensor)
    process_single_batch.py:298-312  -> AprilTag crop + transform via event_filter
                                        (we reuse this verbatim)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# h5py is only needed by read_v2e_h5_events (the v2e output reader). Lazy-load
# so that raw_dvsv_default_k_gen on the base image (no h5py) can import this
# module to reuse apply_apriltag_crop / save_filtered_npz / stream helpers.
try:
    import h5py
except ImportError:
    h5py = None

# Allow `from src.process_data...` imports when run from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.process_data import file_read  # noqa: E402

# ----------------------------------------------------------------------------
# Threshold + variant config
# ----------------------------------------------------------------------------

THRESHOLDS_JSON = Path(__file__).parent / "v2e_thresholds.json"

# Canonical variant matrix (mirrors Stage 0.9 of the planning doc).
# fields: input_modality in {"rgb", "rawY"}, protocol in {"native50", "slomo"},
# threshold_set in {"default", "tuned"}.
VARIANT_SPEC = {
    "V05": {"input_modality": "rgb",  "protocol": "native50", "threshold_set": "default"},
    "V06": {"input_modality": "rgb",  "protocol": "native50", "threshold_set": "tuned"},
    "V07": {"input_modality": "rawY", "protocol": "native50", "threshold_set": "default"},
    "V08": {"input_modality": "rawY", "protocol": "native50", "threshold_set": "tuned"},
    "V09": {"input_modality": "rgb",  "protocol": "slomo",    "threshold_set": "default"},
    "V10": {"input_modality": "rgb",  "protocol": "slomo",    "threshold_set": "tuned"},
    "V11": {"input_modality": "rawY", "protocol": "slomo",    "threshold_set": "default"},
    "V12": {"input_modality": "rawY", "protocol": "slomo",    "threshold_set": "tuned"},
}

TARGET_W, TARGET_H = 346, 260  # DAVIS346 native + project standard


def load_thresholds() -> dict:
    with open(THRESHOLDS_JSON) as f:
        return json.load(f)


def get_threshold_dict(threshold_set: str, input_modality: str) -> dict:
    """Resolve a threshold set name to a concrete v2e param dict.

    'tuned_<modality>' falls back to 'default' when its values are still null
    (Stage 2 not yet executed). The wrapper scripts log this fallback.
    """
    cfg = load_thresholds()
    if threshold_set == "default":
        return dict(cfg["default"])
    if threshold_set == "tuned":
        key = f"tuned_{input_modality}"
        tuned = cfg.get(key, {})
        if tuned.get("pos_thres") is None or tuned.get("neg_thres") is None:
            print(
                f"[v2e_helpers] WARNING: tuned thresholds for {input_modality} "
                f"not yet set in {THRESHOLDS_JSON}; falling back to default."
            )
            return dict(cfg["default"])
        # Fill missing fields from default
        merged = dict(cfg["default"])
        merged.update({k: v for k, v in tuned.items() if v is not None})
        return merged
    raise ValueError(f"Unknown threshold_set={threshold_set}")


# ----------------------------------------------------------------------------
# Frame loading & temp-video creation for v2e
# ----------------------------------------------------------------------------


@dataclass
class PiFrameStream:
    """Grayscale uint8 frames at 346x260 + per-frame timestamps in microseconds."""

    frames: np.ndarray   # (N, 260, 346) uint8
    timestamps_us: np.ndarray  # (N,) int64 microseconds, monotonically increasing
    nominal_fps: float


def _make_stream_rgb(rgb_frames_uint8: np.ndarray, timestamps_us: np.ndarray) -> PiFrameStream:
    """Pi RGB grayscale frames at native res (520x692) -> uint8 grayscale at TARGET resolution."""
    n = rgb_frames_uint8.shape[0]
    out = np.empty((n, TARGET_H, TARGET_W), dtype=np.uint8)
    for i in range(n):
        out[i] = cv2.resize(rgb_frames_uint8[i], (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    nominal_fps = _nominal_fps(timestamps_us)
    return PiFrameStream(frames=out, timestamps_us=timestamps_us, nominal_fps=nominal_fps)


def _make_stream_rawY(raw_uint16_bayer: np.ndarray, timestamps_us: np.ndarray) -> PiFrameStream:
    """Pi RAW Bayer 10-bit (uint16) -> Bayer→Y luminance -> resize TARGET -> uint8.

    Mirrors the production Raw2Event path (Scheme A): bayer_mosaic_to_luminance
    BEFORE downsample, then INTER_AREA resize, then quantize to uint8 by
    rescaling 10-bit (~0..1023) to 0..255.
    """
    luminance = file_read.bayer_mosaic_to_luminance(raw_uint16_bayer)  # uint16 luminance
    n = luminance.shape[0]
    out = np.empty((n, TARGET_H, TARGET_W), dtype=np.uint8)
    scale = 255.0 / 1023.0
    for i in range(n):
        ds = cv2.resize(luminance[i], (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        out[i] = np.clip(ds * scale, 0, 255).astype(np.uint8)
    nominal_fps = _nominal_fps(timestamps_us)
    return PiFrameStream(frames=out, timestamps_us=timestamps_us, nominal_fps=nominal_fps)


def _nominal_fps(timestamps_us: np.ndarray) -> float:
    diffs = np.diff(timestamps_us.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 50.0  # fallback to Pi's nominal 50 fps
    return float(np.clip(1e6 / np.median(diffs), 1.0, 1000.0))


def write_v2e_temp_input(stream: PiFrameStream, out_dir: str | os.PathLike,
                          *, mode: str = "png_folder") -> Path:
    """Write a temp v2e input. PNG folder is the safer default (avoids
    FFV1/codec mismatches between cv2 versions). v2e ingests image folders via
    its `ImageFolderReader` when --input is a directory.

    mode="png_folder": writes <out_dir>/frame_NNNNNN.png and returns out_dir
    mode="ffv1_avi":  legacy fallback; writes <out_dir>.avi and returns that
    """
    out_dir = Path(out_dir)
    if mode == "png_folder":
        out_dir.mkdir(parents=True, exist_ok=True)
        n = stream.frames.shape[0]
        for i in range(n):
            cv2.imwrite(str(out_dir / f"frame_{i:06d}.png"), stream.frames[i])
        return out_dir
    if mode == "ffv1_avi":
        out_path = out_dir.with_suffix(".avi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = stream.frames.shape[1:]
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        writer = cv2.VideoWriter(str(out_path), fourcc, stream.nominal_fps, (w, h), isColor=False)
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"Y800")
            writer = cv2.VideoWriter(str(out_path), fourcc, stream.nominal_fps, (w, h), isColor=False)
        if not writer.isOpened():
            raise RuntimeError(f"cv2.VideoWriter failed to open {out_path}; try mode='png_folder'")
        for f in stream.frames:
            writer.write(f)
        writer.release()
        return out_path
    raise ValueError(f"Unknown mode={mode}; expected 'png_folder' or 'ffv1_avi'")


# Back-compat alias (older code may import this name)
write_v2e_temp_video = write_v2e_temp_input


# ----------------------------------------------------------------------------
# v2e CLI wrapper
# ----------------------------------------------------------------------------


def _autodetect_v2e_paths() -> tuple[str, str]:
    """Resolve V2E python + CLI script paths.

    Precedence:
      1. V2E_VENV_PYTHON / V2E_SCRIPT env vars (explicit override).
      2. **Inside a Docker image with v2e system-installed**: v2e
         is installed system-wide so `python` (sys.executable) imports v2ecore
         and `v2e.py` is on PATH at /opt/v2e/v2e.py. This is the production
         path inside the image.
      3. Local dev fallback: ~/.virtualenvs/v2e/{bin/python, bin/v2e.py}.
      4. Submodule layout: ~/.virtualenvs/v2e/bin/python + REPO_ROOT/external/v2e/v2e.py.
    Fails fast if none resolve, with a message naming the env-var pair.
    """
    py_env = os.environ.get("V2E_VENV_PYTHON")
    script_env = os.environ.get("V2E_SCRIPT")
    if py_env and script_env:
        return py_env, script_env

    # In-container: v2e is system-installed; use sys.executable + v2e.py from PATH.
    try:
        import v2ecore  # noqa: F401
        v2e_on_path = shutil.which("v2e.py")
        if v2e_on_path:
            return sys.executable, v2e_on_path
    except ImportError:
        pass

    # Local dev fallbacks.
    candidates = [
        (Path.home() / ".virtualenvs/v2e/bin/python", Path.home() / ".virtualenvs/v2e/bin/v2e.py"),
        (Path.home() / ".virtualenvs/v2e/bin/python", REPO_ROOT / "external/v2e/v2e.py"),
    ]
    for py, script in candidates:
        if py.exists() and script.exists():
            return str(py), str(script)

    raise RuntimeError(
        "Cannot locate v2e Python + CLI. Either run inside a Docker image "
        "mm0806son/raw2event:v2e-* (where v2ecore is system-installed and "
        "v2e.py is on PATH), or set V2E_VENV_PYTHON and V2E_SCRIPT env vars."
    )


def build_v2e_cmd(
    input_video: Path,
    output_folder: Path,
    threshold_dict: dict,
    protocol: str,
    *,
    input_frame_rate: float,
    slomo_model: Optional[str] = None,
    output_h5: str = "v2e_events.h5",
    extra: Optional[list[str]] = None,
) -> list[str]:
    """Compose v2e.py CLI invocation.

    Always invoke through the v2e venv's python so the wrapper scripts can
    run from the DVS-Voltmeter venv (which doesn't have v2e installed).

    protocol = "native50": --disable_slomo, single accumulation per source frame
    protocol = "slomo": SuperSloMo enabled, requires --slomo_model
    """
    py, script = _autodetect_v2e_paths()
    cmd = [
        py, script,
        "-i", str(input_video),
        "--output_folder", str(output_folder),
        "--overwrite",
        "--no_preview",
        "--dvs346",
        "--input_frame_rate", str(input_frame_rate),
        "--pos_thres", str(threshold_dict["pos_thres"]),
        "--neg_thres", str(threshold_dict["neg_thres"]),
        "--sigma_thres", str(threshold_dict["sigma_thres"]),
        "--cutoff_hz", str(threshold_dict["cutoff_hz"]),
        "--leak_rate_hz", str(threshold_dict["leak_rate_hz"]),
        "--shot_noise_rate_hz", str(threshold_dict["shot_noise_rate_hz"]),
        "--refractory_period", str(threshold_dict.get("refractory_period", 0.0)),
        "--dvs_h5", output_h5,
        "--dvs_aedat2", "None",
        "--dvs_aedat4", "None",
        "--dvs_text", "None",
        "--dvs_vid", "None",  # skip preview AVI
        "--skip_video_output",
    ]
    if protocol == "native50":
        cmd.append("--disable_slomo")
    elif protocol == "slomo":
        if not slomo_model:
            raise ValueError("slomo protocol requires slomo_model path")
        cmd.extend(["--slomo_model", str(slomo_model)])
        cmd.extend(["--auto_timestamp_resolution", "true"])
    else:
        raise ValueError(f"Unknown protocol={protocol}")
    if extra:
        cmd.extend(extra)
    return cmd


def run_v2e(cmd: list[str], log_path: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """Run v2e CLI as subprocess; return exit code. Logs full stdout/stderr if path given."""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as logf:
            return subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=proc_env).returncode
    return subprocess.run(cmd, env=proc_env).returncode


def read_v2e_h5_events(h5_path: str | os.PathLike) -> np.ndarray:
    """Load v2e .h5 events as [N, 4] tensor with cols (t_us, x, y, p in {0,1}).

    v2e stores events under dataset key 'events' with cols (t_us int, x, y, p),
    where p=1 for ON and p=0 for OFF (verified from v2ecore source). Newer
    versions may use float timestamps; we coerce to int microseconds.
    """
    if h5py is None:
        raise ImportError("h5py is required for read_v2e_h5_events; install it or run on the v2e Docker image.")
    with h5py.File(h5_path, "r") as f:
        # Prefer 'events' (DDD20-style); fall back to common alternates
        key = "events"
        if key not in f:
            for alt in ("dvs_events", "data", "ev"):
                if alt in f:
                    key = alt
                    break
            else:
                raise KeyError(f"No events dataset in {h5_path}; have {list(f.keys())}")
        ev = f[key][...]
    if ev.ndim != 2 or ev.shape[1] != 4:
        raise ValueError(f"Unexpected v2e events shape {ev.shape} in {h5_path}")
    # Coerce timestamp column to int microseconds (v2e usually writes int us already)
    out = np.empty(ev.shape, dtype=np.int64)
    out[:, 0] = ev[:, 0].astype(np.int64)
    out[:, 1] = ev[:, 1].astype(np.int64)
    out[:, 2] = ev[:, 2].astype(np.int64)
    # Polarity: the project-wide NPZ convention is {0,1} (src/simulator.py
    # emits {0,1} on both backends). v2e already writes {0,1}, so the (p > 0)
    # normalization below is defensive. Readers tolerate {-1,+1} too, but newly
    # generated NPZs must use {0,1}.
    p_v2e = ev[:, 3].astype(np.int64)
    out[:, 3] = (p_v2e > 0).astype(np.int64)
    return out


# ----------------------------------------------------------------------------
# AprilTag crop + unified80 rescale (reuse production code)
# ----------------------------------------------------------------------------


def apply_apriltag_crop(
    events_tensor: np.ndarray,
    pi_frames_for_tag: np.ndarray,
    pi_frame_timestamps_us: np.ndarray,
    *,
    n_workers: int = 8,
    is_luminance: bool = True,
    is_raw: bool = False,
) -> tuple[np.ndarray, int]:
    """Apply the AprilTag detection + rotate/scale transform used by
    train_class/process_single_batch.py:241-316 to a v2e events tensor.

    Returns (filtered_events, box_size). Mirrors production: per-batch
    ThreadPoolExecutor + filter by `info[2] is not None` (timestamp), then
    polygon bbox -> round_up_to_10 box_size, then event_filter transform.

    Thread safety: pupil_apriltags' C ``detect()`` is not safe to call
    concurrently from multiple Python threads when numba or OpenMP is also
    loaded — it aborts with `malloc(): mismatching next->prev_size`. The
    AprilTag executor is therefore pinned to a single worker. The downstream
    event_filter loop keeps the caller's n_workers since it only does
    numpy/torch and is safe in parallel.
    """
    import concurrent.futures
    import torch  # heavy; defer
    from src.process_data import event_filter, tag_detector

    n = pi_frames_for_tag.shape[0]
    apriltag_workers = 1   # see THREAD-SAFETY NOTE above; do not raise.
    event_filter_workers = max(1, min(n_workers, n))
    n_workers = apriltag_workers
    frame_batches = tag_detector.split_batches(pi_frames_for_tag, n_workers)
    ts_batches = tag_detector.split_batches(pi_frame_timestamps_us, n_workers)
    margin_ratios = [0.03] * n_workers
    tag_ref_widths = [287] * n_workers
    barbara_ref_sizes = [861] * n_workers
    barbara_gaps = [82] * n_workers
    # is_raw signals high-bit-depth (uint16) frames; tag_detector applies
    # an explicit stretch from 10/16-bit to uint8 instead of the lossy /256
    # fallback. Required for the rawY path where _frame_array_to_tensor +
    # bayer_mosaic_to_luminance produce uint16 luminance.
    is_raws = [is_raw] * n_workers
    is_luminances = [is_luminance] * n_workers

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        all_results = list(executor.map(
            tag_detector.process_batch,
            frame_batches, ts_batches,
            margin_ratios, tag_ref_widths, barbara_ref_sizes, barbara_gaps,
            is_raws, is_luminances,
        ))

    # Match production: keep entries with timestamp (info[2]) not None, then sort.
    crops_info = [item for batch in all_results for item in batch if item[2] is not None]
    crops_info.sort(key=lambda x: x[2])
    valid = [info[0] for info in crops_info if info[0] is not None]
    if not valid:
        return np.empty((0, 4), dtype=np.int64), 0
    box_max = max(info["polygon"].ptp(axis=0).max() for info in valid)
    box_size = int(np.ceil(box_max / 10.0) * 10)

    ev_tensor = torch.as_tensor(events_tensor) if not isinstance(events_tensor, torch.Tensor) else events_tensor
    filtered = event_filter.filter_events_parallel(
        events_tensor=ev_tensor, crops_info=crops_info,
        target_size=box_size, transform=True,
        batch_size=100000, n_workers=event_filter_workers,
    )
    if isinstance(filtered, torch.Tensor):
        filtered = filtered.numpy()
    return filtered, box_size


def save_filtered_npz(events: np.ndarray, out_path: str | os.PathLike) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, events=events)


def is_valid_filtered_npz(path: Path | str, *, require_nonempty: bool = False) -> bool:
    """Used by --skip_existing to avoid re-running on partial / corrupt outputs.

    Validates the NPZ has the 'events' key with shape (N, 4). When
    require_nonempty is True also asserts N > 0; usually False because legitimate
    runs may produce 0 events on degenerate prefixes.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as f:
            if "events" not in f:
                return False
            ev = f["events"]
            if ev.ndim != 2 or ev.shape[1] != 4:
                return False
            if require_nonempty and ev.shape[0] == 0:
                return False
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Convenience: end-to-end one prefix
# ----------------------------------------------------------------------------


def process_one_prefix_v2e_subprocess(
    prefix: str,
    input_dir: str | os.PathLike,
    output_npz_path: str | os.PathLike,
    *,
    input_modality: str,
    protocol: str,
    threshold_dict: dict,
    slomo_model: Optional[str] = None,
    work_dir: Optional[str | os.PathLike] = None,
    log_dir: Optional[str | os.PathLike] = None,
    keep_intermediate: bool = False,
    n_workers: int = 8,
    timeout_sec: int = 900,
) -> dict:
    """Run process_one_prefix_v2e in an isolated subprocess so that a
    pupil_apriltags SIGABRT doesn't kill the whole batch.

    Returns the same stats dict as process_one_prefix_v2e on success.
    Raises RuntimeError on non-zero subprocess exit (parent stays alive).
    """
    import subprocess
    import tempfile

    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix=f"v2eparent_{prefix}_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    threshold_json = work_dir / "_thresholds.json"
    with open(threshold_json, "w") as f:
        json.dump(threshold_dict, f)
    result_json = work_dir / "_result.json"

    cmd = [
        sys.executable, "-m", "tools.v2e_baseline._one_prefix_subproc",
        "--prefix", prefix,
        "--input_dir", str(input_dir),
        "--output_npz_path", str(output_npz_path),
        "--input_modality", input_modality,
        "--protocol", protocol,
        "--threshold_json", str(threshold_json),
        "--n_workers", str(n_workers),
        "--result_json", str(result_json),
    ]
    if slomo_model:
        cmd += ["--slomo_model", str(slomo_model)]
    if log_dir:
        cmd += ["--log_dir", str(log_dir)]
    if keep_intermediate:
        cmd.append("--keep_intermediate")
    # Pass a per-prefix work_dir so v2e temp lives under work_root if caller set one
    cmd += ["--work_dir", str(work_dir / "inner")]

    env = os.environ.copy()
    # Pass thread-limit env vars (in case caller forgot to export them)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMBA_NUM_THREADS", "1")
    env.setdefault("TBB_NUM_THREADS", "1")
    # Run from REPO_ROOT so `python -m tools.v2e_baseline...` resolves
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout_sec,
    )
    if proc.returncode != 0:
        # Drop stdout/stderr tails into the exception so the caller logs them.
        tail_out = (proc.stdout or "").splitlines()[-20:]
        tail_err = (proc.stderr or "").splitlines()[-20:]
        raise RuntimeError(
            f"v2e subprocess for {prefix} exited rc={proc.returncode}\n"
            f"--- stdout tail ---\n" + "\n".join(tail_out) + "\n"
            f"--- stderr tail ---\n" + "\n".join(tail_err)
        )
    if not result_json.exists():
        raise RuntimeError(f"v2e subprocess for {prefix} produced no result_json")
    with open(result_json) as f:
        return json.load(f)


def process_one_prefix_v2e(
    prefix: str,
    input_dir: str | os.PathLike,
    output_npz_path: str | os.PathLike,
    *,
    input_modality: str,           # "rgb" | "rawY"
    protocol: str,                 # "native50" | "slomo"
    threshold_dict: dict,
    slomo_model: Optional[str] = None,
    work_dir: Optional[str | os.PathLike] = None,
    log_dir: Optional[str | os.PathLike] = None,
    keep_intermediate: bool = False,
    n_workers: int = 8,
) -> dict:
    """End-to-end: load Pi MKV -> temp grayscale video -> v2e -> AprilTag crop -> NPZ.

    Returns a dict with stats: n_events_v2e, n_events_after_crop, box_size, time_seconds.
    """
    import time
    import glob

    t0 = time.time()
    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix=f"v2e_{prefix}_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # File naming convention (mirrors train_class/process_single_batch.py):
    #   rgb_frames_<base_id>*_rgb.mkv
    #   raw_frames_<base_id>*_raw_10bit.mkv
    #   metadata_<base_id>*.dat
    # where base_id = prefix.rsplit("_", 1)[0] (drops the trailing time field).
    base_id = prefix.rsplit("_", 1)[0]
    input_dir = Path(input_dir)
    rgb_mkv_glob = sorted(input_dir.glob(f"rgb_frames_{base_id}*_rgb.mkv"))
    raw_mkv_glob = sorted(input_dir.glob(f"raw_frames_{base_id}*_raw_10bit.mkv"))
    metadata_glob = sorted(input_dir.glob(f"metadata_{base_id}*.dat"))
    if not metadata_glob:
        raise FileNotFoundError(f"metadata_*.dat for {prefix} not found in {input_dir}")
    pi_timestamps, _real_ts = file_read.read_metadata(str(metadata_glob[0]))
    pi_timestamps_np = pi_timestamps.numpy() if hasattr(pi_timestamps, "numpy") else np.asarray(pi_timestamps)

    if input_modality == "rgb":
        if not rgb_mkv_glob:
            raise FileNotFoundError(f"rgb mkv for {prefix}")
        # Frame count may differ from metadata; truncate per production convention.
        from generate_event import load_from_video as _lfv
        rgb_frames, _ = _lfv(str(rgb_mkv_glob[0]), quiet=True)
        n = min(rgb_frames.shape[0], pi_timestamps_np.shape[0])
        ts_us = pi_timestamps_np[:n].astype(np.int64)
        stream = _make_stream_rgb(rgb_frames[:n], ts_us)
        is_luminance = False
    elif input_modality == "rawY":
        if not raw_mkv_glob:
            raise FileNotFoundError(f"raw_10bit mkv for {prefix}")
        from generate_event import load_from_video as _lfv
        raw_frames, _ = _lfv(str(raw_mkv_glob[0]), quiet=True)
        n = min(raw_frames.shape[0], pi_timestamps_np.shape[0])
        ts_us = pi_timestamps_np[:n].astype(np.int64)
        stream = _make_stream_rawY(raw_frames[:n], ts_us)
        is_luminance = True
    else:
        raise ValueError(f"Unknown input_modality={input_modality}")

    temp_video = write_v2e_temp_input(
        stream, work_dir / f"{prefix}_{input_modality}_frames", mode="png_folder",
    )

    v2e_out_dir = work_dir / "v2e_out"
    cmd = build_v2e_cmd(
        input_video=temp_video,
        output_folder=v2e_out_dir,
        threshold_dict=threshold_dict,
        protocol=protocol,
        input_frame_rate=stream.nominal_fps,
        slomo_model=slomo_model,
        output_h5="v2e_events.h5",
    )
    log_path = (Path(log_dir) / f"{prefix}_{input_modality}_{protocol}.log") if log_dir else None
    rc = run_v2e(cmd, log_path=log_path)
    if rc != 0:
        raise RuntimeError(f"v2e failed for {prefix} (rc={rc}); see {log_path}")

    h5_files = list(v2e_out_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"v2e produced no .h5 in {v2e_out_dir}")
    events_v2e = read_v2e_h5_events(h5_files[0])
    n_events_v2e = events_v2e.shape[0]

    # CRITICAL: v2e emits 0-based timestamps relative
    # to its first input frame. apply_apriltag_crop interpolates between
    # crops_info entries that carry Pi sensor timestamps. Shift v2e events into
    # Pi timestamp domain by adding stream.timestamps_us[0]. Drop any events
    # that fall outside the Pi frame timestamp range to avoid extrapolation.
    if events_v2e.shape[0] > 0:
        t0_pi = int(stream.timestamps_us[0])
        events_v2e = events_v2e.copy()
        events_v2e[:, 0] = events_v2e[:, 0] + t0_pi
        t_lo, t_hi = stream.timestamps_us[0], stream.timestamps_us[-1]
        keep = (events_v2e[:, 0] >= t_lo) & (events_v2e[:, 0] <= t_hi)
        events_v2e = events_v2e[keep]

    # AprilTag crop on the SAME grayscale frames we fed to v2e (so crop polygon
    # matches v2e's coordinate system).
    filtered, box_size = apply_apriltag_crop(
        events_v2e, stream.frames, stream.timestamps_us,
        n_workers=n_workers, is_luminance=is_luminance,
    )
    save_filtered_npz(filtered, output_npz_path)

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "prefix": prefix,
        "input_modality": input_modality,
        "protocol": protocol,
        "n_events_v2e": int(n_events_v2e),
        "n_events_after_crop": int(filtered.shape[0]),
        "box_size": int(box_size),
        "time_seconds": float(time.time() - t0),
    }
