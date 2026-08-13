"""Compare native DAVIS event streams with their unified80 descendants.

The audit quantifies what is and is not preserved by the production
AprilTag-driven spatial filtering and coordinate normalization. It compares
event retention, duration, polarity balance, and normalized temporal activity.
Spatial image fidelity is intentionally not inferred from these statistics;
that requires a separate frame/alignment audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import signal

SOURCE_RE = re.compile(r"^dv_output_(?P<prefix>.+)\.aedat4$")
SPECTRAL_FREQUENCIES_HZ = (50.0, 60.0, 120.0, 180.0)


def normalized_histogram(timestamps: np.ndarray, bins: int) -> np.ndarray:
    """Histogram event activity over normalized recording phase."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("at least two timestamps are required")
    span = float(timestamps.max() - timestamps.min())
    if span <= 0:
        raise ValueError("timestamps have non-positive duration")
    phase = (timestamps - timestamps.min()) / span
    counts, _ = np.histogram(phase, bins=bins, range=(0.0, 1.0))
    total = int(counts.sum())
    if total == 0:
        raise ValueError("temporal histogram is empty")
    return counts.astype(np.float64) / total


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in bits."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl_divergence(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def histogram_correlation(p: np.ndarray, q: np.ndarray) -> float:
    """Pearson correlation between two temporal activity histograms."""
    p_centered = np.asarray(p, dtype=np.float64) - np.mean(p)
    q_centered = np.asarray(q, dtype=np.float64) - np.mean(q)
    denominator = float(np.linalg.norm(p_centered) * np.linalg.norm(q_centered))
    return float(np.dot(p_centered, q_centered) / denominator) if denominator else 0.0


def spectral_line_contrast(
    timestamps_us: np.ndarray,
    target_hz: float,
    *,
    bin_us: int = 500,
    local_inner_hz: float = 1.0,
    local_outer_hz: float = 5.0,
) -> float:
    """Measure an event-rate line against its local spectral background."""
    timestamps = np.asarray(timestamps_us, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("at least two timestamps are required")
    relative = timestamps - timestamps.min()
    if relative.max() <= 0:
        raise ValueError("timestamps have non-positive duration")
    bin_ids = np.floor(relative / bin_us).astype(np.int64)
    counts = np.bincount(bin_ids, minlength=int(bin_ids.max()) + 1).astype(np.float64)
    stabilized = np.sqrt(counts + 3.0 / 8.0)
    frequencies, power = signal.periodogram(
        stabilized,
        fs=1_000_000.0 / bin_us,
        window="hann",
        detrend="linear",
        scaling="density",
    )
    power = np.maximum(power, np.finfo(np.float64).tiny)
    log_power = np.log10(power)
    target_index = int(np.argmin(np.abs(frequencies - target_hz)))
    distance = np.abs(frequencies - target_hz)
    background = log_power[(distance >= local_inner_hz) & (distance <= local_outer_hz)]
    if background.size == 0:
        return float("nan")
    return 10.0 * float(log_power[target_index] - np.median(background))


def spatial_distribution_stats(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, float]:
    """Return resolution-normalized occupancy and concentration descriptors."""
    flat = np.asarray(y, dtype=np.int64) * width + np.asarray(x, dtype=np.int64)
    if flat.size == 0:
        raise ValueError("event stream is empty")
    if flat.min() < 0 or flat.max() >= width * height:
        raise ValueError("event coordinates fall outside declared resolution")
    counts = np.bincount(flat, minlength=width * height).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / np.log(width * height)
    top_count = max(1, int(np.ceil(0.01 * width * height)))
    top_fraction = float(np.sort(counts)[-top_count:].sum() / counts.sum())
    return {
        "occupied_pixel_fraction": float(np.mean(counts > 0)),
        "normalized_spatial_entropy": normalized_entropy,
        "top_1pct_pixel_event_fraction": top_fraction,
    }


def compare_event_arrays(
    native_events: np.ndarray,
    unified_events: np.ndarray,
    *,
    temporal_bins: int = 200,
    native_resolution: tuple[int, int] = (346, 260),
    unified_resolution: tuple[int, int] = (80, 80),
) -> dict[str, float | int]:
    """Compare native structured events with unified ``[t,x,y,p]`` events."""
    if native_events.dtype.names is None:
        raise ValueError("native_events must be a structured array")
    required = {"timestamp", "x", "y", "polarity"}
    if not required.issubset(native_events.dtype.names):
        raise ValueError(
            f"native_events is missing fields: {sorted(required - set(native_events.dtype.names))}"
        )
    unified_events = np.asarray(unified_events)
    if unified_events.ndim != 2 or unified_events.shape[1] != 4:
        raise ValueError("unified_events must have shape (N, 4)")
    if len(native_events) < 2 or len(unified_events) < 2:
        raise ValueError("both streams need at least two events")

    native_t = native_events["timestamp"].astype(np.float64)
    unified_t = unified_events[:, 0].astype(np.float64)
    native_hist = normalized_histogram(native_t, temporal_bins)
    unified_hist = normalized_histogram(unified_t, temporal_bins)
    native_duration = float(native_t.max() - native_t.min()) / 1_000_000.0
    unified_duration = float(unified_t.max() - unified_t.min()) / 1_000_000.0
    native_p = native_events["polarity"].astype(np.float64)
    unified_p = unified_events[:, 3].astype(np.float64)

    native_w, native_h = native_resolution
    unified_w, unified_h = unified_resolution
    native_spatial = spatial_distribution_stats(
        native_events["x"],
        native_events["y"],
        width=native_w,
        height=native_h,
    )
    unified_spatial = spatial_distribution_stats(
        unified_events[:, 1],
        unified_events[:, 2],
        width=unified_w,
        height=unified_h,
    )

    spectral_metrics = {}
    for frequency in SPECTRAL_FREQUENCIES_HZ:
        key = f"line_{frequency:g}_hz_contrast_db"
        spectral_metrics[f"native_{key}"] = spectral_line_contrast(native_t, frequency)
        spectral_metrics[f"unified_{key}"] = spectral_line_contrast(
            unified_t, frequency
        )

    return {
        "native_events": int(len(native_events)),
        "unified_events": int(len(unified_events)),
        "event_retention_fraction": float(len(unified_events) / len(native_events)),
        "native_duration_s": native_duration,
        "unified_duration_s": unified_duration,
        "duration_ratio": unified_duration / native_duration,
        "native_positive_fraction": float(np.mean(native_p > 0)),
        "unified_positive_fraction": float(np.mean(unified_p > 0)),
        "positive_fraction_abs_delta": float(
            abs(np.mean(native_p > 0) - np.mean(unified_p > 0))
        ),
        "temporal_histogram_correlation": histogram_correlation(
            native_hist, unified_hist
        ),
        "temporal_js_divergence_bits": jensen_shannon_divergence(
            native_hist, unified_hist
        ),
        "temporal_cdf_max_distance": float(
            np.max(np.abs(np.cumsum(native_hist) - np.cumsum(unified_hist)))
        ),
        **{f"native_{key}": value for key, value in native_spatial.items()},
        **{f"unified_{key}": value for key, value in unified_spatial.items()},
        **spectral_metrics,
    }


def _load_native_events(path: Path) -> np.ndarray:
    import dv_processing as dv

    recording = dv.io.MonoCameraRecording(str(path))
    if not recording.isEventStreamAvailable():
        raise ValueError("AEDAT4 has no event stream")
    chunks = []
    while True:
        batch = recording.getNextEventBatch()
        if batch is None:
            break
        array = batch.numpy()
        if len(array):
            chunks.append(array.copy())
    if not chunks:
        raise ValueError("AEDAT4 event stream is empty")
    return np.concatenate(chunks)


def _prefix_from_source(path: Path) -> str:
    match = SOURCE_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected AEDAT4 filename: {path.name}")
    return match.group("prefix")


def discover_pairs(
    native_dir: Path, unified_dir: Path, limit: int | None = None
) -> list[tuple[Path, Path, str]]:
    """Return source/final pairs in deterministic prefix order."""
    pairs = []
    for native_path in native_dir.glob("dv_output_*.aedat4"):
        prefix = _prefix_from_source(native_path)
        unified_path = unified_dir / f"{prefix}_filtered_dv.npz"
        if unified_path.exists():
            pairs.append((native_path, unified_path, prefix))
    pairs.sort(key=lambda item: item[2])
    return pairs if limit is None else pairs[:limit]


def _analyze_pair(args: tuple[str, str, str, int]) -> dict:
    native_path_raw, unified_path_raw, prefix, temporal_bins = args
    native_path = Path(native_path_raw)
    unified_path = Path(unified_path_raw)
    try:
        native_events = _load_native_events(native_path)
        with np.load(unified_path, allow_pickle=False) as payload:
            unified_events = payload["events"]
        metrics = compare_event_arrays(
            native_events, unified_events, temporal_bins=temporal_bins
        )
        return {"prefix": prefix, "status": "ok", "error": "", **metrics}
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        return {
            "prefix": prefix,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _metric_summary(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray(
        [row[key] for row in rows if row["status"] == "ok"], dtype=np.float64
    )
    return {
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _write_report(summary: dict, path: Path) -> None:
    metrics = summary["metrics"]
    lines = [
        "# Native DAVIS → unified80 retention audit",
        "",
        f"- Matched source/final pairs: {summary['n_pairs']}",
        f"- Successfully analyzed: {summary['n_ok']}",
        f"- Errors: {summary['n_errors']}",
        "",
        "| Metric | Median | IQR | Range |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("event_retention_fraction", "Event retention"),
        ("duration_ratio", "Duration ratio"),
        ("positive_fraction_abs_delta", "Absolute positive-fraction change"),
        ("temporal_histogram_correlation", "Temporal histogram correlation"),
        ("temporal_js_divergence_bits", "Temporal JS divergence (bits)"),
        ("temporal_cdf_max_distance", "Temporal CDF max distance"),
        ("native_normalized_spatial_entropy", "Native normalized spatial entropy"),
        ("unified_normalized_spatial_entropy", "unified80 normalized spatial entropy"),
        ("native_line_60_hz_contrast_db", "Native 60 Hz line contrast (dB)"),
        ("unified_line_60_hz_contrast_db", "unified80 60 Hz line contrast (dB)"),
        ("native_line_120_hz_contrast_db", "Native 120 Hz line contrast (dB)"),
        ("unified_line_120_hz_contrast_db", "unified80 120 Hz line contrast (dB)"),
    ):
        item = metrics[key]
        lines.append(
            f"| {label} | {item['median']:.4f} | [{item['q25']:.4f}, {item['q75']:.4f}] | "
            f"[{item['min']:.4f}, {item['max']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Retention and temporal/polarity agreement quantify how the selected event stream changes.",
            "- Native and unified spatial entropy are resolution-normalized descriptors, not a proof of",
            "  spatial alignment or edge preservation: the native stream includes the full sensor while",
            "  unified80 selects and transforms the display region.",
            "- A separate matched-frame audit is required for geometric residuals and focus/blur.",
            "",
        ]
    )
    if summary["errors"]:
        lines.extend(["## Errors", ""])
        for error in summary["errors"]:
            lines.append(f"- `{error['prefix']}`: {error['error']}")
        lines.append("")
    path.write_text("\n".join(lines))


def run_audit(
    native_dir: Path,
    unified_dir: Path,
    output_dir: Path,
    *,
    temporal_bins: int = 200,
    workers: int = 1,
    limit: int | None = None,
) -> dict:
    """Run the matched native/final audit and write CSV, JSON, and Markdown."""
    pairs = discover_pairs(native_dir, unified_dir, limit=limit)
    if not pairs:
        raise FileNotFoundError("no matched native AEDAT4 / unified80 DV pairs found")
    work = [
        (str(native), str(unified), prefix, temporal_bins)
        for native, unified, prefix in pairs
    ]
    if workers == 1:
        rows = [_analyze_pair(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_analyze_pair, work, chunksize=2))
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("every matched pair failed analysis")

    metric_keys = [
        "event_retention_fraction",
        "duration_ratio",
        "positive_fraction_abs_delta",
        "temporal_histogram_correlation",
        "temporal_js_divergence_bits",
        "temporal_cdf_max_distance",
        "native_normalized_spatial_entropy",
        "unified_normalized_spatial_entropy",
        "native_line_50_hz_contrast_db",
        "unified_line_50_hz_contrast_db",
        "native_line_60_hz_contrast_db",
        "unified_line_60_hz_contrast_db",
        "native_line_120_hz_contrast_db",
        "unified_line_120_hz_contrast_db",
        "native_line_180_hz_contrast_db",
        "unified_line_180_hz_contrast_db",
    ]
    errors = [
        {"prefix": row["prefix"], "error": row["error"]}
        for row in rows
        if row["status"] != "ok"
    ]
    summary = {
        "native_dir": str(native_dir),
        "unified_dir": str(unified_dir),
        "n_pairs": len(pairs),
        "n_ok": len(ok_rows),
        "n_errors": len(errors),
        "temporal_bins": temporal_bins,
        "metrics": {key: _metric_summary(rows, key) for key in metric_keys},
        "errors": errors,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with (output_dir / "per_prefix_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary, output_dir / "report.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--unified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temporal-bins", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        args.native_dir,
        args.unified_dir,
        args.output_dir,
        temporal_bins=args.temporal_bins,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
