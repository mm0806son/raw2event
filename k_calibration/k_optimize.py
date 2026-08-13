#!/usr/bin/env python3
"""Optional Optuna refinement of the full K vector.

Minimizes a Sinkhorn-EMD objective plus an event-count penalty between simulated
and real events, optionally warm-started from the three-step regression. The
calibration scene is a uniform brightness ramp with no motion, so all events are
used full-frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# Ensure project root is on sys.path so ``src.*`` imports work when running
# this script directly (e.g. ``python k_calibration/k_optimize.py``).
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import optuna  # noqa: E402
from geomloss import SamplesLoss  # noqa: E402

import src.process_data.dvs_generate as dvs_generate  # noqa: E402
import src.process_data.file_read as file_read  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI_IMAGE_WIDTH = 346
PI_IMAGE_HEIGHT = 260

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Weight for the event-count penalty term in the composite loss.
# loss = EMD + COUNT_PENALTY_WEIGHT * |log(N_gen / N_dv)|
#      + SPATIAL_PENALTY_WEIGHT * TV(per-pixel count dist)
#      + TEMPORAL_PENALTY_WEIGHT * TV(per-pixel Δt hist dist)
COUNT_PENALTY_WEIGHT = 0.1
# Default-off so legacy runs keep reproducing. When enabled the loss adds
# total-variation distance between per-pixel normalized event-count
# distributions of sim vs DV, which distinguishes "fire on every pixel"
# (bias-drift degenerate solution) from "fire on edges" (signal-driven).
SPATIAL_PENALTY_WEIGHT = 0.0
# Default-off temporal counterpart to the spatial TV penalty. Aligns pooled
# per-pixel consecutive-Δt histograms so frame-sync plateaus are penalized.
TEMPORAL_PENALTY_WEIGHT = 0.0
# polarity-aware penalty weight. When > 0 the loss adds
# |pos_ratio_gen - pos_ratio_dv| so the calibration objective is no longer
# polarity-blind. Default-off so legacy 6D K runs reproduce.
POLARITY_PENALTY_WEIGHT = 0.0
READONLY_SQLITE_HINT = (
    "Check file permissions, parent directory permissions, free disk space, "
    "and whether the target path is mounted read-only."
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna K1-K6 optimization via Sinkhorn-EMD (full-frame)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- data ---
    p.add_argument(
        "--data_dir",
        required=True,
        help="Calibration data directory (contains aedat4, raw/rgb .dat, metadata .dat)",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory for output JSON, Optuna DB, and logs",
    )
    p.add_argument(
        "--source",
        choices=["raw", "rgb"],
        default="raw",
        help="Which Pi frames to use for event generation",
    )
    p.add_argument("--pair", default="Raw2DVS346", help="Device-pair name")
    # --- warm start ---
    p.add_argument(
        "--stage1_json",
        default=None,
        help="Path to stage1_params.json from k_estimate.py (warm-start k1,k2,k4,k5)",
    )
    p.add_argument(
        "--warm_factor",
        type=float,
        default=5.0,
        help="Multiplicative factor for warm-start search range (value/factor .. value*factor)",
    )
    # --- Optuna ---
    p.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    p.add_argument(
        "--timeout", type=int, default=None, help="Optuna timeout in seconds"
    )
    p.add_argument(
        "--study_name",
        default=None,
        help="Optuna study name (default: auto from pair+source)",
    )
    # --- EMD ---
    p.add_argument(
        "--emd_blur", type=float, default=0.01, help="Sinkhorn blur parameter"
    )
    p.add_argument(
        "--emd_scaling", type=float, default=0.9, help="Sinkhorn scaling parameter"
    )
    p.add_argument(
        "--max_pts",
        type=int,
        default=2_000_000,
        help="Max event points for EMD (random subsample if exceeded)",
    )
    p.add_argument(
        "--count_penalty",
        type=float,
        default=COUNT_PENALTY_WEIGHT,
        help="Weight for |log(N_gen/N_dv)| count penalty in the loss",
    )
    p.add_argument(
        "--spatial_penalty_weight",
        type=float,
        default=SPATIAL_PENALTY_WEIGHT,
        help=(
            "Weight for TV-distance between per-pixel normalized event-count "
            "distributions (sim vs DV). 0 disables. Distinguishes 'fire on "
            "every pixel' (bias-drift degeneracy) from 'fire on edges'."
        ),
    )
    p.add_argument(
        "--temporal_penalty_weight",
        type=float,
        default=TEMPORAL_PENALTY_WEIGHT,
        help=(
            "Weight for TV-distance between pooled per-pixel consecutive-Δt "
            "histograms (sim vs DV). 0 disables. Temporal dual of the "
            "spatial TV penalty for suppressing frame-sync artifacts."
        ),
    )
    p.add_argument(
        "--pol_penalty_weight",
        type=float,
        default=POLARITY_PENALTY_WEIGHT,
        help=(
            "R1: weight for |pos_ratio_gen - pos_ratio_dv| polarity penalty. "
            "0 disables (default; preserves legacy polarity-blind loss). "
            "Activates 8D K [k1..k6, k_on, k_off] sampling when > 0."
        ),
    )
    p.add_argument(
        "--calibrate_polarity",
        action="store_true",
        help=(
            "R1: search 8D K [k1..k6, k_on, k_off] instead of 6D. Forces "
            "k_on/k_off to be sampled in [0.1, 10] (log scale). Implies "
            "--pol_penalty_weight should be > 0 to break ratio degeneracy."
        ),
    )
    # --- misc ---
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--file_suffix",
        default=None,
        help="Common token in filenames for find_matching_files (e.g. '20250509')",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_events_fixed(
    ev: torch.Tensor,
    t_min: float,
    t_max: float,
    x_max: float = PI_IMAGE_WIDTH,
    y_max: float = PI_IMAGE_HEIGHT,
) -> torch.Tensor:
    """Normalize (t, x, y) to [0, 1] using shared, fixed bounds.

    Both DV and generated events must use the same bounds so that spatial
    and temporal scales are comparable across point clouds.
    """
    ev = ev.clone().float()
    # t
    t_range = t_max - t_min
    if t_range > 0:
        ev[:, 0] = (ev[:, 0] - t_min) / t_range
    else:
        ev[:, 0] = 0.0
    # x -> [0, x_max]
    if x_max > 0:
        ev[:, 1] = ev[:, 1] / x_max
    # y -> [0, y_max]
    if y_max > 0:
        ev[:, 2] = ev[:, 2] / y_max
    return ev


def subsample_deterministic(pts: torch.Tensor, max_n: int, seed: int) -> torch.Tensor:
    """Uniformly subsample to at most *max_n* points, deterministically."""
    if pts.shape[0] <= max_n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], size=max_n, replace=False)
    return pts[torch.from_numpy(idx)]


def per_pixel_probs(
    events: torch.Tensor,
    height: int = PI_IMAGE_HEIGHT,
    width: int = PI_IMAGE_WIDTH,
) -> torch.Tensor:
    """Return a ``(H*W,)`` tensor of per-pixel normalized event-count probabilities.

    Events columns are assumed to be ``[t, x, y, p]``. Coordinates out of
    bounds are clamped. When events are empty, returns a uniform distribution
    so that TV distance against a concentrated reference is well-defined.
    """
    n_pix = height * width
    if events.shape[0] == 0:
        return torch.full((n_pix,), 1.0 / n_pix, dtype=torch.float64)
    ev_cpu = events.detach().to("cpu")
    x = ev_cpu[:, 1].long().clamp_(0, width - 1)
    y = ev_cpu[:, 2].long().clamp_(0, height - 1)
    flat_idx = y * width + x
    counts = torch.bincount(flat_idx, minlength=n_pix).to(torch.float64)
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts


def spatial_tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance between two per-pixel probability vectors."""
    return float(0.5 * torch.abs(p - q).sum().item())


def per_pixel_probs_pol(
    events: torch.Tensor,
    height: int = PI_IMAGE_HEIGHT,
    width: int = PI_IMAGE_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R1: polarity-split per-pixel normalized event-count distributions.

    Returns ``(P_pos, P_neg)`` — two ``(H*W,)`` tensors. Polarity column is
    taken from ``events[:, 3]`` and split with ``> 0`` (matches the simulator
    convention where ON events have ``p == 1`` and OFF have ``p == 0``).
    Each branch is independently normalized; if a branch has no events it
    falls back to the uniform distribution so TV distance is well-defined.
    """
    n_pix = height * width
    if events.shape[0] == 0:
        uni = torch.full((n_pix,), 1.0 / n_pix, dtype=torch.float64)
        return uni, uni.clone()
    ev_cpu = events.detach().to("cpu")
    pol = ev_cpu[:, 3]
    pos_mask = pol > 0
    neg_mask = ~pos_mask

    def _branch(ev: torch.Tensor) -> torch.Tensor:
        if ev.shape[0] == 0:
            return torch.full((n_pix,), 1.0 / n_pix, dtype=torch.float64)
        x = ev[:, 1].long().clamp_(0, width - 1)
        y = ev[:, 2].long().clamp_(0, height - 1)
        flat_idx = y * width + x
        counts = torch.bincount(flat_idx, minlength=n_pix).to(torch.float64)
        total = counts.sum()
        if total > 0:
            counts = counts / total
        return counts

    return _branch(ev_cpu[pos_mask]), _branch(ev_cpu[neg_mask])


def pos_ratio(events: torch.Tensor) -> float:
    """R1: fraction of ON events. Returns 0.5 when ``events`` is empty."""
    if events.shape[0] == 0:
        return 0.5
    pol = events[:, 3] if isinstance(events, torch.Tensor) else events[:, 3]
    if isinstance(pol, torch.Tensor):
        return float((pol > 0).to(torch.float64).mean().item())
    return float((np.asarray(pol) > 0).mean())


def temporal_dt_probs(
    events: torch.Tensor | np.ndarray,
    *,
    min_events_per_pixel: int = 5,
    max_dt_us: float = 100_000.0,
    num_bins: int = 100,
) -> torch.Tensor:
    """Return a normalized pooled per-pixel consecutive-Δt histogram.

    Events columns are assumed to be ``[t, x, y, p]``. Per-pixel timestamps
    are sorted before differencing. Pixels with fewer than
    ``min_events_per_pixel`` events are discarded. Positive ``Δt`` values are
    binned uniformly over ``[0, max_dt_us]``; any ``Δt >= max_dt_us`` is
    clipped into the last bin.

    Returns a zero tensor of length ``num_bins`` when the input is empty or no
    pixel has enough events.
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}.")
    if max_dt_us <= 0:
        raise ValueError(f"max_dt_us must be positive, got {max_dt_us}.")

    if isinstance(events, torch.Tensor):
        ev = events.detach().to("cpu").numpy()
    else:
        ev = np.asarray(events)

    if ev.ndim != 2 or ev.shape[1] < 4:
        raise ValueError(f"Event array must have shape (N, >=4), got {ev.shape}.")
    if ev.shape[0] == 0:
        return torch.zeros(num_bins, dtype=torch.float64)

    timestamps = ev[:, 0].astype(np.float64, copy=False)
    x_coords = ev[:, 1].astype(np.int64, copy=False)
    y_coords = ev[:, 2].astype(np.int64, copy=False)

    x_coords = x_coords - x_coords.min()
    y_coords = y_coords - y_coords.min()
    width = int(x_coords.max()) + 1 if x_coords.size > 0 else 1
    flat_idx = y_coords * width + x_coords

    order = np.lexsort((timestamps, flat_idx))
    sorted_idx = flat_idx[order]
    sorted_ts = timestamps[order]

    _, start_idx, counts = np.unique(sorted_idx, return_index=True, return_counts=True)
    eligible = counts >= min_events_per_pixel
    if not np.any(eligible):
        return torch.zeros(num_bins, dtype=torch.float64)

    hist = np.zeros(num_bins, dtype=np.float64)
    bin_width = max_dt_us / num_bins

    for start, count in zip(start_idx[eligible], counts[eligible], strict=False):
        diffs = np.diff(sorted_ts[start : start + count])
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            continue
        bin_ids = np.floor(diffs / bin_width).astype(np.int64, copy=False)
        np.clip(bin_ids, 0, num_bins - 1, out=bin_ids)
        hist += np.bincount(bin_ids, minlength=num_bins)

    total = hist.sum()
    if total <= 0:
        return torch.zeros(num_bins, dtype=torch.float64)
    return torch.from_numpy(hist / total).to(torch.float64)


def temporal_tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """0.5 * sum(|p - q|), matching the spatial TV convention."""
    return float(0.5 * torch.abs(p - q).sum().item())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to reduce the chance of partial checkpoint files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


class LocalOptunaTrialRecorder:
    """Persist per-trial snapshots so long Optuna runs can be recovered locally."""

    CSV_FIELDNAMES = [
        "pair",
        "source",
        "study_name",
        "trial_number",
        "status",
        "trial_started_at",
        "trial_finished_at",
        "trial_seconds",
        "sinkhorn_seconds",
        "k1",
        "k2",
        "k3",
        "k4",
        "k5",
        "k6",
        # optional 8D K columns (NaN when 6D K is used).
        "k_on",
        "k_off",
        "n_generated_events",
        "n_reference_events",
        "count_ratio",
        "emd",
        "count_penalty",
        "spatial_penalty",
        "temporal_penalty",
        # polarity-aware loss diagnostics.
        "polarity_penalty",
        "gen_pos_ratio",
        "loss",
        "loss_delta_from_prev",
        "best_loss_so_far",
        "best_improvement",
        "error",
    ]

    def __init__(
        self, output_dir: Path, study_name: str, pair: str, source: str
    ) -> None:
        self.output_dir = output_dir
        self.study_name = study_name
        self.pair = pair
        self.source = source
        self.last_trial_path = output_dir / f"{study_name}_last_trial.json"
        self.best_trial_path = output_dir / f"{study_name}_best_local.json"
        self.trial_csv_path = output_dir / f"{study_name}_trials.csv"
        self.best_payload: dict[str, Any] | None = None
        self.previous_loss: float | None = None

        if self.last_trial_path.exists():
            with open(self.last_trial_path) as f:
                self.previous_loss = float(json.load(f)["loss"])
        if self.best_trial_path.exists():
            with open(self.best_trial_path) as f:
                self.best_payload = json.load(f)

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        self.trial_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (
            not self.trial_csv_path.exists() or self.trial_csv_path.stat().st_size == 0
        )
        with open(self.trial_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.CSV_FIELDNAMES, lineterminator="\n"
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def record_trial(
        self,
        *,
        trial_number: int,
        status: str,
        k_values: list[float],
        n_gen: int | None,
        n_dv: int,
        emd: float | None,
        count_pen: float | None,
        loss: float,
        trial_seconds: float,
        sinkhorn_seconds: float | None = None,
        error: str | None = None,
        trial_started_at: float | None = None,
        trial_finished_at: float | None = None,
        spatial_pen: float | None = None,
        temporal_pen: float | None = None,
        pol_pen: float | None = None,
        gen_pos_ratio: float | None = None,
    ) -> None:
        count_ratio = None
        if n_gen is not None and n_dv > 0:
            count_ratio = n_gen / n_dv

        loss_delta_from_prev = None
        if self.previous_loss is not None:
            loss_delta_from_prev = loss - self.previous_loss

        previous_best_loss = None
        if self.best_payload is not None:
            previous_best_loss = float(self.best_payload["loss"])

        best_loss_so_far = previous_best_loss
        best_improvement = None
        is_finite_loss = math.isfinite(loss)
        is_better = is_finite_loss and (
            previous_best_loss is None or loss < previous_best_loss
        )
        if is_better:
            best_loss_so_far = loss
            if previous_best_loss is not None:
                best_improvement = previous_best_loss - loss

        # K may be 6D (legacy) or 8D ([k1..k6, k_on, k_off]).
        if len(k_values) == 8:
            k_names_payload = ["k1", "k2", "k3", "k4", "k5", "k6", "k_on", "k_off"]
            k_on_csv = k_values[6]
            k_off_csv = k_values[7]
        else:
            k_names_payload = ["k1", "k2", "k3", "k4", "k5", "k6"]
            k_on_csv = None
            k_off_csv = None

        payload = {
            "pair": self.pair,
            "source": self.source,
            "study_name": self.study_name,
            "trial_number": trial_number,
            "status": status,
            "K": list(k_values),
            "K_names": k_names_payload,
            "n_generated_events": n_gen,
            "n_reference_events": n_dv,
            "count_ratio": count_ratio,
            "emd": emd,
            "count_penalty": count_pen,
            "spatial_penalty": spatial_pen,
            "temporal_penalty": temporal_pen,
            "polarity_penalty": pol_pen,
            "gen_pos_ratio": gen_pos_ratio,
            "loss": loss,
            "trial_started_at": trial_started_at,
            "trial_finished_at": trial_finished_at,
            "trial_seconds": trial_seconds,
            "sinkhorn_seconds": sinkhorn_seconds,
            "error": error,
            "timestamp": trial_finished_at
            if trial_finished_at is not None
            else time.time(),
        }
        _write_json_atomic(self.last_trial_path, payload)

        csv_row = {
            "pair": self.pair,
            "source": self.source,
            "study_name": self.study_name,
            "trial_number": trial_number,
            "status": status,
            "trial_started_at": trial_started_at,
            "trial_finished_at": trial_finished_at,
            "trial_seconds": trial_seconds,
            "sinkhorn_seconds": sinkhorn_seconds,
            "k1": k_values[0],
            "k2": k_values[1],
            "k3": k_values[2],
            "k4": k_values[3],
            "k5": k_values[4],
            "k6": k_values[5],
            "k_on": k_on_csv,
            "k_off": k_off_csv,
            "n_generated_events": n_gen,
            "n_reference_events": n_dv,
            "count_ratio": count_ratio,
            "emd": emd,
            "count_penalty": count_pen,
            "spatial_penalty": spatial_pen,
            "temporal_penalty": temporal_pen,
            "polarity_penalty": pol_pen,
            "gen_pos_ratio": gen_pos_ratio,
            "loss": loss,
            "loss_delta_from_prev": loss_delta_from_prev,
            "best_loss_so_far": best_loss_so_far,
            "best_improvement": best_improvement,
            "error": error,
        }
        self._append_csv_row(csv_row)
        self.previous_loss = loss

        if is_better:
            self.best_payload = payload
            _write_json_atomic(self.best_trial_path, payload)


def _ensure_optuna_storage_writable(db_path: Path) -> None:
    """Fail fast if the Optuna SQLite storage cannot be written."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __raw2event_storage_probe__ "
            "(id INTEGER PRIMARY KEY, touched_at REAL)"
        )
        connection.execute(
            "INSERT INTO __raw2event_storage_probe__ (touched_at) VALUES (?)",
            (time.time(),),
        )
        connection.execute("DROP TABLE __raw2event_storage_probe__")
        connection.commit()
    except sqlite3.OperationalError as exc:
        raise RuntimeError(
            f"Optuna SQLite storage is not writable: {db_path}. {READONLY_SQLITE_HINT}"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    """Return ``exc`` plus any chained causes/contexts without infinite loops."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _is_sqlite_readonly_error(exc: BaseException) -> bool:
    """Detect readonly SQLite failures even when Optuna wraps them oddly."""
    readonly_markers = (
        "readonly database",
        "attempt to write a readonly database",
        "read-only database",
        "read-only file system",
    )
    for current in _iter_exception_chain(exc):
        message = str(current).lower()
        if any(marker in message for marker in readonly_markers):
            return True
    return False


def _build_result_payload(
    *,
    args: argparse.Namespace,
    k_values: list[float],
    best_value: float,
    n_trials: int,
    ranges: dict[str, tuple[float, float, bool]],
    stage1: dict[str, Any] | None,
    recovered_from_local_checkpoint: bool = False,
    best_trial_number: int | None = None,
) -> dict[str, Any]:
    """Construct the JSON payload compatible with ``load_K_from_file``.

    R1: payload K may be 6D or 8D. ``version`` field reflects the dimension
    (1 for 6D legacy; 2 for 8D polarity-aware).
    """
    if len(k_values) == 8:
        k_names = ["k1", "k2", "k3", "k4", "k5", "k6", "k_on", "k_off"]
        version = 2
    else:
        k_names = ["k1", "k2", "k3", "k4", "k5", "k6"]
        version = 1
    result = {
        "pair": args.pair,
        "source": args.source,
        "version": version,
        "K": k_values,
        "K_names": k_names,
        "stage2_best_value": best_value,
        "stage2_n_trials": n_trials,
        "stage2_search_ranges": {k: list(v[:2]) for k, v in ranges.items()},
        "stage2_loss_weights": {
            "count_penalty": args.count_penalty,
            "spatial_penalty_weight": args.spatial_penalty_weight,
            "temporal_penalty_weight": args.temporal_penalty_weight,
            "pol_penalty_weight": getattr(args, "pol_penalty_weight", 0.0),
        },
        "calibration_date": str(date.today()),
    }
    if best_trial_number is not None:
        result["stage2_best_trial"] = best_trial_number
    if recovered_from_local_checkpoint:
        result["stage2_recovered_from_local_checkpoint"] = True
    if stage1 is not None:
        result["stage1_params"] = stage1.get("stage1_params", {})
    return result


def _save_recovered_results(
    recorder: LocalOptunaTrialRecorder,
    args: argparse.Namespace,
    ranges: dict[str, tuple[float, float, bool]],
    stage1: dict[str, Any] | None,
    output_dir: Path,
) -> Path:
    """Persist the best locally checkpointed trial when Optuna storage breaks."""
    if recorder.best_payload is None:
        raise RuntimeError("No local trial checkpoint is available for recovery.")

    payload = _build_result_payload(
        args=args,
        k_values=list(recorder.best_payload["K"]),
        best_value=float(recorder.best_payload["loss"]),
        n_trials=int(recorder.best_payload["trial_number"]) + 1,
        ranges=ranges,
        stage1=stage1,
        recovered_from_local_checkpoint=True,
        best_trial_number=int(recorder.best_payload["trial_number"]),
    )
    out_path = output_dir / f"{args.pair}_K_recovered.json"
    _write_json_atomic(out_path, payload)
    return out_path


def _optimize_with_storage_guard(
    *,
    study: Any,
    objective: Callable[[optuna.Trial], float],
    args: argparse.Namespace,
    db_path: Path,
    recovery_writer: Callable[[], Path] | None = None,
) -> None:
    """Run Optuna and rewrite opaque readonly-storage failures into actionable errors."""
    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    except Exception as exc:
        if not _is_sqlite_readonly_error(exc):
            raise

        recovered_path: Path | None = None
        if recovery_writer is not None:
            recovered_path = recovery_writer()

        message = (
            f"Optuna SQLite storage became unwritable during optimization "
            f"(readonly database): {db_path}. "
            f"{READONLY_SQLITE_HINT}"
        )
        if recovered_path is not None:
            message += f" Recovered best local trial to {recovered_path}."
        raise RuntimeError(message) from exc


def _find_calibration_files(data_dir: str, suffix: str | None) -> dict[str, str]:
    """Find calibration files in *data_dir*.

    If *suffix* is given (e.g. '20250509'), it is passed to
    ``find_matching_files`` which matches any file containing that token.
    If *suffix* is None, auto-detect by type-pattern matching.  Supports
    both current (``dv_output_*.aedat4``) and legacy
    (``time_filtered_dv_events*.aedat4``) DV naming.  Raises if a type
    has multiple candidates.
    """
    if suffix is not None:
        return file_read.find_matching_files(data_dir, suffix)

    import re

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


def _looks_like_stage1_analysis_dir(path: Path) -> bool:
    """Heuristically detect Stage 1 analysis directories from saved `.pt` outputs."""
    return any(path.rglob("events_with_luminance_*.pt"))


def _looks_like_raw_calibration_dir(path: Path) -> bool:
    """Detect whether a directory already contains the Stage 2 raw inputs."""
    try:
        _find_calibration_files(str(path), None)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _resolve_optimize_data_dir(data_dir: str) -> str:
    """Resolve Stage 1 analysis directories to their matching raw calibration directory."""
    candidate = Path(data_dir).expanduser().resolve()

    if _looks_like_raw_calibration_dir(candidate):
        return str(candidate)

    if _looks_like_stage1_analysis_dir(candidate):
        raw_dir = candidate.parent / "data" / candidate.name
        if _looks_like_raw_calibration_dir(raw_dir):
            return str(raw_dir)
        raise FileNotFoundError(
            "Stage 1 analysis directory was provided to k_optimize, but the matching "
            f"raw calibration directory was not found. Expected something like: {raw_dir}"
        )

    return str(candidate)


# ---------------------------------------------------------------------------
# Search ranges
# ---------------------------------------------------------------------------

# Parameters that can be negative (based on physical model / Stage 1 outputs).
_SIGNED_PARAMS = {"k2", "k4", "k5"}

# Default cold-start ranges.  Signed params use linear scale with negative
# lower bound; positive-only params use log scale.
_COLD_RANGES: dict[str, tuple[float, float, bool]] = {
    #             (lo,     hi,      log_scale)
    "k1": (1e-1, 1e2, True),
    "k2": (-1e2, 1e2, False),
    "k3": (1e-8, 1e-2, True),
    "k4": (-1e-4, 1e-4, False),
    "k5": (-1e-5, 1e-5, False),
    "k6": (1e-8, 1e-2, True),
}

# cold-start ranges for the polarity-aware threshold params.
# Search log-scale around 1.0 covering 0.5..2.0 (Stoffregen ECCV 2020 reports
# DAVIS Cp / Cn typically within ~2× of each other). Wider ranges (e.g.
# [0.1, 10]) push event_generation_torch into runaway recursion when k_on and
# k_off are both << 1, exhausting host memory before a trial can complete.
_POL_COLD_RANGES: dict[str, tuple[float, float, bool]] = {
    "k_on": (0.5, 2.0, True),
    "k_off": (0.5, 2.0, True),
}


def build_search_ranges(
    stage1: dict | None,
    warm_factor: float,
    *,
    calibrate_polarity: bool = False,
) -> dict[str, tuple[float, float, bool]]:
    """Return (lo, hi, log_scale) search ranges for K parameters.

    If *stage1* is provided, k1/k2/k4/k5 are centered on the regression
    estimates; k3 and k6 always use wide default ranges (no Stage 1 info).

    R1: when ``calibrate_polarity`` is True, also include search ranges for
    ``k_on`` and ``k_off`` (log-scale around 1.0, ±1 decade).
    """
    if stage1 is None:
        ranges = dict(_COLD_RANGES)
    else:
        params = stage1.get("stage1_params", {})
        ranges = dict(_COLD_RANGES)
        for key in ("k1", "k2", "k4", "k5"):
            val = params.get(key)
            if val is None or val == 0:
                continue
            abs_val = abs(val)
            spread = abs_val * warm_factor
            if key in _SIGNED_PARAMS:
                # Linear, centered on the signed value
                ranges[key] = (val - spread, val + spread, False)
            else:
                # Log-scale, positive only
                lo = abs_val / warm_factor
                hi = abs_val * warm_factor
                ranges[key] = (lo, hi, True)

    if calibrate_polarity:
        # k_on / k_off are not part of stage1 regression; always cold-start.
        ranges.update(_POL_COLD_RANGES)
    return ranges


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_calibration_data(
    data_dir: str, source: str, file_suffix: str | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load Pi frames, Pi timestamps, and DV events for a calibration scene.

    Returns
    -------
    frames : (N, H, W) or (N, H, W, 4)
    pi_timestamps : (N,) microseconds
    dv_events : (M, 4) [t, x, y, p]
    """
    resolved_data_dir = _resolve_optimize_data_dir(data_dir)
    files = _find_calibration_files(resolved_data_dir, file_suffix)

    # Pi frames
    if source == "raw":
        frames = file_read.read_raw_frames(
            files["raw_frames"], PI_IMAGE_HEIGHT, PI_IMAGE_WIDTH
        )
    else:
        frames = file_read.read_rgb_frames(
            files["rgb_frames"], PI_IMAGE_HEIGHT, PI_IMAGE_WIDTH
        )

    pi_timestamps, real_timestamps = file_read.read_metadata(files["metadata"])

    # DV events
    dv_events = file_read.load_events(files["dv"])

    # Time alignment: shift DV timestamps to Pi clock
    time_offset = file_read.calculate_time_offset(pi_timestamps, real_timestamps)
    dv_events[:, 0] = dv_events[:, 0] - time_offset

    return frames, pi_timestamps, dv_events


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def make_objective(
    frames: torch.Tensor,
    pi_timestamps: torch.Tensor,
    dv_events: torch.Tensor,
    is_rgb: bool,
    sinkhorn_fn: SamplesLoss,
    ranges: dict[str, tuple[float, float, bool]],
    max_pts: int,
    base_seed: int,
    count_penalty_weight: float,
    spatial_penalty_weight: float = 0.0,
    temporal_penalty_weight: float = 0.0,
    pol_penalty_weight: float = 0.0,
    calibrate_polarity: bool = False,
    trial_recorder: LocalOptunaTrialRecorder | None = None,
):
    """Return a closure suitable for ``study.optimize``."""

    # Pre-compute DV reference points (full-frame, time-window-aligned).
    t_min = float(pi_timestamps[0].item())
    t_max = float(pi_timestamps[-1].item())
    dv_mask = (dv_events[:, 0] >= t_min) & (dv_events[:, 0] <= t_max)
    dv_trimmed = dv_events[dv_mask]
    n_dv = dv_trimmed.shape[0]

    if n_dv == 0:
        raise RuntimeError(
            "No DV events fall within Pi timestamp range after alignment. "
            "Check time offset / data integrity."
        )

    # Normalize with shared, fixed bounds.
    # when polarity-aware loss is active, retain the polarity
    # column so Sinkhorn-EMD treats opposite-polarity matches as separated by
    # one cost unit (polarity ∈ {0, 1}). Legacy path (no polarity weight)
    # drops the column for byte-identical reproduction of pre-R1 EMD values.
    _retain_pol = pol_penalty_weight > 0 or calibrate_polarity
    if _retain_pol:
        dv_pts_full = normalize_events_fixed(dv_trimmed, t_min, t_max)[:, :4]
    else:
        dv_pts_full = normalize_events_fixed(dv_trimmed, t_min, t_max)[:, :3]
    dv_pts = subsample_deterministic(dv_pts_full, max_pts, base_seed).to(DEVICE)
    w_dv = torch.full((dv_pts.shape[0],), 1.0 / dv_pts.shape[0], device=DEVICE)

    # Per-pixel reference distribution (full DV in the aligned window). Kept on
    # CPU because it is a tiny vector and compared against CPU bincount below.
    # when polarity penalty is on, use per-polarity spatial probs.
    dv_spatial_probs = None
    dv_spatial_probs_pol: tuple[torch.Tensor, torch.Tensor] | None = None
    if spatial_penalty_weight > 0:
        if _retain_pol:
            dv_spatial_probs_pol = per_pixel_probs_pol(dv_trimmed)
        else:
            dv_spatial_probs = per_pixel_probs(dv_trimmed)
    dv_temporal_probs = (
        temporal_dt_probs(dv_trimmed) if temporal_penalty_weight > 0 else None
    )
    # precompute DV pos_ratio for the polarity penalty term.
    dv_pos_ratio: float | None = (
        pos_ratio(dv_trimmed) if pol_penalty_weight > 0 else None
    )

    print(f"DV reference: {n_dv} events (subsampled to {dv_pts.shape[0]})")
    if spatial_penalty_weight > 0:
        if dv_spatial_probs_pol is not None:
            empty_pos = float((dv_spatial_probs_pol[0] == 0).float().mean().item())
            empty_neg = float((dv_spatial_probs_pol[1] == 0).float().mean().item())
            print(
                f"  Spatial penalty ON / per-polarity (weight={spatial_penalty_weight:.4g}); "
                f"DV empty_pixel_frac POS={empty_pos:.4f} NEG={empty_neg:.4f}"
            )
        elif dv_spatial_probs is not None:
            empty_frac = float((dv_spatial_probs == 0).float().mean().item())
            print(
                f"  Spatial penalty ON (weight={spatial_penalty_weight:.4g}); "
                f"DV empty_pixel_frac={empty_frac:.4f}"
            )
    if temporal_penalty_weight > 0 and dv_temporal_probs is not None:
        print(
            f"  Temporal penalty ON (weight={temporal_penalty_weight:.4g}); "
            f"DV eligible_dt_mass={float(dv_temporal_probs.sum().item()):.4f}"
        )
    if pol_penalty_weight > 0 and dv_pos_ratio is not None:
        print(
            f"  Polarity penalty ON (weight={pol_penalty_weight:.4g}); "
            f"DV pos_ratio={dv_pos_ratio:.4f}"
        )
    if calibrate_polarity:
        print("  R1: 8D K sampling enabled (k_on, k_off jointly searched).")

    def objective(trial: optuna.Trial) -> float:
        trial_started_at = time.time()
        # Suggest K parameters with correct scale per parameter.
        # when calibrate_polarity, also sample (k_on, k_off) so the K
        # vector becomes 8D [k1..k6, k_on, k_off].
        k_param_names = ["k1", "k2", "k3", "k4", "k5", "k6"]
        if calibrate_polarity:
            k_param_names.extend(["k_on", "k_off"])
        k_params = []
        for k_name in k_param_names:
            lo, hi, log_scale = ranges[k_name]
            if log_scale:
                k_params.append(trial.suggest_float(k_name, lo, hi, log=True))
            else:
                k_params.append(trial.suggest_float(k_name, lo, hi))

        # Bad K proposals can drive the simulator into runaway recursion.
        # Treat these as invalid trials so Optuna can continue searching.
        try:
            gen_events = dvs_generate.generate_events_tensor(
                pi_timestamps, frames, is_rgb=is_rgb, k_values=k_params
            )
        except RecursionError as exc:
            trial_finished_at = time.time()
            print(
                f"  [trial {trial.number}] Event generation recursion error: {exc}. "
                "Treating this proposal as invalid."
            )
            if trial_recorder is not None:
                trial_recorder.record_trial(
                    trial_number=trial.number,
                    status="invalid_recursion_error",
                    k_values=k_params,
                    n_gen=None,
                    n_dv=n_dv,
                    emd=None,
                    count_pen=None,
                    loss=float("inf"),
                    trial_seconds=trial_finished_at - trial_started_at,
                    sinkhorn_seconds=None,
                    error=str(exc),
                    trial_started_at=trial_started_at,
                    trial_finished_at=trial_finished_at,
                    temporal_pen=None,
                )
            return float("inf")

        if gen_events is None or gen_events.shape[0] == 0:
            trial_finished_at = time.time()
            if trial_recorder is not None:
                trial_recorder.record_trial(
                    trial_number=trial.number,
                    status="invalid_empty_generated_events",
                    k_values=k_params,
                    n_gen=0,
                    n_dv=n_dv,
                    emd=None,
                    count_pen=None,
                    loss=float("inf"),
                    trial_seconds=trial_finished_at - trial_started_at,
                    sinkhorn_seconds=None,
                    error="Generated event tensor is empty before alignment.",
                    trial_started_at=trial_started_at,
                    trial_finished_at=trial_finished_at,
                    temporal_pen=None,
                )
            return float("inf")

        # Trim to same time window
        gen_events = gen_events[
            (gen_events[:, 0] >= t_min) & (gen_events[:, 0] <= t_max)
        ]
        n_gen = gen_events.shape[0]

        if n_gen == 0:
            trial_finished_at = time.time()
            if trial_recorder is not None:
                trial_recorder.record_trial(
                    trial_number=trial.number,
                    status="invalid_empty_aligned_events",
                    k_values=k_params,
                    n_gen=0,
                    n_dv=n_dv,
                    emd=None,
                    count_pen=None,
                    loss=float("inf"),
                    trial_seconds=trial_finished_at - trial_started_at,
                    sinkhorn_seconds=None,
                    error="Generated events were empty after time-window alignment.",
                    trial_started_at=trial_started_at,
                    trial_finished_at=trial_finished_at,
                    temporal_pen=None,
                )
            return float("inf")

        # Normalize with the SAME fixed bounds as DV.
        # keep the polarity column when polarity-aware loss is on so
        # Sinkhorn-EMD penalizes cross-polarity matches by ~1 cost unit.
        if _retain_pol:
            gen_pts_full = normalize_events_fixed(gen_events, t_min, t_max)[:, :4]
        else:
            gen_pts_full = normalize_events_fixed(gen_events, t_min, t_max)[:, :3]
        # Deterministic subsample keyed on trial number
        gen_pts = subsample_deterministic(
            gen_pts_full, max_pts, base_seed + trial.number + 1
        ).to(DEVICE)
        w_gen = torch.full((gen_pts.shape[0],), 1.0 / gen_pts.shape[0], device=DEVICE)

        # Sinkhorn-EMD
        t0 = time.time()
        try:
            emd = sinkhorn_fn(w_gen, gen_pts, w_dv, dv_pts).item()
        except RuntimeError as exc:
            trial_finished_at = time.time()
            sinkhorn_seconds = trial_finished_at - t0
            print(f"  [trial {trial.number}] Sinkhorn error: {exc}")
            if trial_recorder is not None:
                trial_recorder.record_trial(
                    trial_number=trial.number,
                    status="invalid_sinkhorn_error",
                    k_values=k_params,
                    n_gen=n_gen,
                    n_dv=n_dv,
                    emd=None,
                    count_pen=None,
                    loss=float("inf"),
                    trial_seconds=trial_finished_at - trial_started_at,
                    sinkhorn_seconds=sinkhorn_seconds,
                    error=str(exc),
                    trial_started_at=trial_started_at,
                    trial_finished_at=trial_finished_at,
                    temporal_pen=None,
                )
            return float("inf")
        sinkhorn_seconds = time.time() - t0

        # Event count penalty: |log(N_gen / N_dv)| penalizes density mismatch
        count_ratio = n_gen / n_dv if n_dv > 0 else 1.0
        count_pen = abs(math.log(max(count_ratio, 1e-6)))

        # Spatial sparsity penalty: total-variation between per-pixel normalized
        # event-count distributions. Needed because Sinkhorn-EMD + count penalty
        # alone cannot distinguish "fire on every pixel" (bias-drift degenerate
        # solution) from "fire on edges" on uniform calibration scenes.
        # when polarity-aware, average TV distance over per-polarity
        # distributions instead of polarity-collapsed.
        spatial_pen: float | None = None
        if spatial_penalty_weight > 0:
            if dv_spatial_probs_pol is not None:
                gen_pos_probs, gen_neg_probs = per_pixel_probs_pol(gen_events)
                tv_pos = spatial_tv_distance(gen_pos_probs, dv_spatial_probs_pol[0])
                tv_neg = spatial_tv_distance(gen_neg_probs, dv_spatial_probs_pol[1])
                spatial_pen = 0.5 * (tv_pos + tv_neg)
            elif dv_spatial_probs is not None:
                gen_spatial_probs = per_pixel_probs(gen_events)
                spatial_pen = spatial_tv_distance(gen_spatial_probs, dv_spatial_probs)
        temporal_pen: float | None = None
        if temporal_penalty_weight > 0 and dv_temporal_probs is not None:
            gen_temporal_probs = temporal_dt_probs(gen_events)
            temporal_pen = temporal_tv_distance(gen_temporal_probs, dv_temporal_probs)
        # polarity ratio penalty.
        pol_pen: float | None = None
        gen_pos_ratio: float | None = None
        if pol_penalty_weight > 0 and dv_pos_ratio is not None:
            gen_pos_ratio = pos_ratio(gen_events)
            pol_pen = abs(gen_pos_ratio - dv_pos_ratio)

        loss = emd + count_penalty_weight * count_pen
        if spatial_pen is not None:
            loss = loss + spatial_penalty_weight * spatial_pen
        if temporal_pen is not None:
            loss = loss + temporal_penalty_weight * temporal_pen
        if pol_pen is not None:
            loss = loss + pol_penalty_weight * pol_pen
        trial_finished_at = time.time()
        trial_seconds = trial_finished_at - trial_started_at

        trial.set_user_attr("emd", float(emd))
        trial.set_user_attr("count_penalty", float(count_pen))
        if spatial_pen is not None:
            trial.set_user_attr("spatial_penalty", float(spatial_pen))
        if temporal_pen is not None:
            trial.set_user_attr("temporal_penalty", float(temporal_pen))
        if pol_pen is not None:
            trial.set_user_attr("polarity_penalty", float(pol_pen))
        if gen_pos_ratio is not None:
            trial.set_user_attr("gen_pos_ratio", float(gen_pos_ratio))
        trial.set_user_attr("n_generated_events", int(n_gen))
        trial.set_user_attr("loss", float(loss))

        spatial_pen_str = (
            f" spatial_pen={spatial_pen:.4f}" if spatial_pen is not None else ""
        )
        temporal_pen_str = (
            f" temporal_pen={temporal_pen:.4f}" if temporal_pen is not None else ""
        )
        pol_pen_str = (
            f" pol_pen={pol_pen:.4f}(r={gen_pos_ratio:.3f})"
            if pol_pen is not None and gen_pos_ratio is not None
            else ""
        )
        print(
            f"  [trial {trial.number}] "
            f"K=[{', '.join(f'{v:.4g}' for v in k_params)}] "
            f"gen={n_gen} EMD={emd:.6f} count_pen={count_pen:.4f}"
            f"{spatial_pen_str}{temporal_pen_str}{pol_pen_str} "
            f"loss={loss:.6f} ({trial_seconds:.1f}s)"
        )
        if trial_recorder is not None:
            trial_recorder.record_trial(
                trial_number=trial.number,
                status="completed",
                k_values=k_params,
                n_gen=n_gen,
                n_dv=n_dv,
                emd=emd,
                count_pen=count_pen,
                spatial_pen=spatial_pen,
                temporal_pen=temporal_pen,
                pol_pen=pol_pen,
                gen_pos_ratio=gen_pos_ratio,
                loss=loss,
                trial_seconds=trial_seconds,
                sinkhorn_seconds=sinkhorn_seconds,
                error=None,
                trial_started_at=trial_started_at,
                trial_finished_at=trial_finished_at,
            )
        return loss

    return objective


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(
    study: optuna.Study,
    args: argparse.Namespace,
    ranges: dict[str, tuple[float, float, bool]],
    stage1: dict | None,
    output_dir: Path,
) -> Path:
    """Write the final ``{pair}_K.json`` compatible with ``load_K_from_file``."""
    best = study.best_trial
    base_keys = ("k1", "k2", "k3", "k4", "k5", "k6")
    k_values = [best.params[k] for k in base_keys]
    # include polarity-aware threshold params when present in the trial.
    if "k_on" in best.params and "k_off" in best.params:
        k_values.append(best.params["k_on"])
        k_values.append(best.params["k_off"])
    result = _build_result_payload(
        args=args,
        k_values=k_values,
        best_value=float(best.value),
        n_trials=len(study.trials),
        ranges=ranges,
        stage1=stage1,
        best_trial_number=best.number,
    )

    out_path = output_dir / f"{args.pair}_K.json"
    _write_json_atomic(out_path, result)
    return out_path


def _config_fingerprint(args: argparse.Namespace) -> str:
    """Short hash of key config params to detect study DB mismatches."""
    blob = json.dumps(
        {
            "data_dir": os.path.abspath(args.data_dir),
            "source": args.source,
            "emd_blur": args.emd_blur,
            "emd_scaling": args.emd_scaling,
            "max_pts": args.max_pts,
            "count_penalty": args.count_penalty,
            "spatial_penalty_weight": args.spatial_penalty_weight,
            "temporal_penalty_weight": args.temporal_penalty_weight,
            # include polarity-aware loss / search dimension so a study
            # started with a different setting is flagged on resume.
            "pol_penalty_weight": getattr(args, "pol_penalty_weight", 0.0),
            "calibrate_polarity": bool(getattr(args, "calibrate_polarity", False)),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device info
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available. Sinkhorn-EMD will be slow on CPU.")

    # Load data
    print(f"Loading calibration data from {args.data_dir} (source={args.source}) ...")
    frames, pi_timestamps, dv_events = load_calibration_data(
        args.data_dir, args.source, args.file_suffix
    )
    print(
        f"  Pi frames: {frames.shape}, timestamps: {pi_timestamps.shape}, "
        f"DV events: {dv_events.shape}"
    )

    is_rgb = args.source == "rgb"

    # Warm start
    stage1: dict | None = None
    if args.stage1_json is not None:
        with open(args.stage1_json) as f:
            stage1 = json.load(f)
        s1p = stage1.get("stage1_params", {})
        print(f"Warm-start from Stage 1: {s1p}")

    ranges = build_search_ranges(
        stage1, args.warm_factor, calibrate_polarity=args.calibrate_polarity
    )
    print("Search ranges:")
    for k, (lo, hi, log) in ranges.items():
        scale = "log" if log else "linear"
        print(f"  {k}: [{lo:.4g}, {hi:.4g}] ({scale})")

    # Sinkhorn loss
    sinkhorn_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.emd_blur,
        scaling=args.emd_scaling,
        backend="online",
    )

    # Build objective
    study_name = args.study_name or f"k_optimize_{args.pair}_{args.source}"
    db_path = output_dir / f"{study_name}.db"
    _ensure_optuna_storage_writable(db_path)
    trial_recorder = LocalOptunaTrialRecorder(
        output_dir, study_name, args.pair, args.source
    )
    print(f"Trial CSV log: {trial_recorder.trial_csv_path}")

    objective = make_objective(
        frames,
        pi_timestamps,
        dv_events,
        is_rgb,
        sinkhorn_fn,
        ranges,
        args.max_pts,
        args.seed,
        args.count_penalty,
        spatial_penalty_weight=args.spatial_penalty_weight,
        temporal_penalty_weight=args.temporal_penalty_weight,
        pol_penalty_weight=args.pol_penalty_weight,
        calibrate_polarity=args.calibrate_polarity,
        trial_recorder=trial_recorder,
    )

    # Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    # Validate config consistency when resuming
    fp = _config_fingerprint(args)
    if study.user_attrs.get("config_fingerprint"):
        old_fp = study.user_attrs["config_fingerprint"]
        if old_fp != fp:
            print(
                f"WARNING: Config fingerprint mismatch (stored={old_fp}, "
                f"current={fp}). Results may be inconsistent."
            )
    study.set_user_attr("config_fingerprint", fp)

    # Enqueue Stage 1 values as the first trial (true warm-start)
    if stage1 is not None and len(study.trials) == 0:
        s1p = stage1.get("stage1_params", {})
        enqueue_params: dict[str, float] = {}
        for k_name in ("k1", "k2", "k4", "k5"):
            if k_name in s1p:
                enqueue_params[k_name] = s1p[k_name]
        # k3 and k6: use geometric mean of their range as initial guess
        for k_name in ("k3", "k6"):
            lo, hi, _ = ranges[k_name]
            enqueue_params[k_name] = math.sqrt(lo * hi)
        # when calibrating polarity, seed k_on=k_off=1.0 (legacy-equivalent
        # threshold) so the first trial is the byte-identical 6D baseline.
        if args.calibrate_polarity:
            enqueue_params["k_on"] = 1.0
            enqueue_params["k_off"] = 1.0
        study.enqueue_trial(enqueue_params)
        print(f"Enqueued Stage 1 trial: {enqueue_params}")

    print(f"\nStarting Optuna optimization: {args.n_trials} trials ...")
    _optimize_with_storage_guard(
        study=study,
        objective=objective,
        args=args,
        db_path=db_path,
        recovery_writer=lambda: _save_recovered_results(
            trial_recorder, args, ranges, stage1, output_dir
        ),
    )

    # Results
    best = study.best_trial
    k_names = ["k1", "k2", "k3", "k4", "k5", "k6"]
    if args.calibrate_polarity:
        k_names = k_names + ["k_on", "k_off"]
    print("\n=== Best Trial ===")
    print(f"  Trial {best.number}, loss = {best.value:.6f}")
    for k in k_names:
        print(f"  {k} = {best.params[k]:.6g}")

    out_path = save_results(study, args, ranges, stage1, output_dir)
    print(f"\nSaved: {out_path}")

    # Verify load_K_from_file compatibility
    from src.config import load_K_from_file

    loaded_pair = load_K_from_file(str(out_path))
    print(f"Verified: load_K_from_file('{out_path}') -> pair='{loaded_pair}'")


if __name__ == "__main__":
    main()
