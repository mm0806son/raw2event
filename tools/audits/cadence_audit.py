"""Audit Pi hardware-timestamp cadence across Raw2Event metadata sidecars."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np

METADATA_DTYPE = np.dtype(
    [
        ("SensorTimestamp", "float64"),
        ("RealTime", "S30"),
    ]
)


def analyze_metadata(path: Path) -> dict[str, float | int | str]:
    """Measure cadence from the monotonic Pi hardware timestamps."""
    metadata = np.memmap(path, dtype=METADATA_DTYPE, mode="r")
    timestamps = np.asarray(metadata["SensorTimestamp"], dtype=np.float64)
    if len(timestamps) < 3:
        raise ValueError(f"{path}: expected at least three metadata records")
    intervals = np.diff(timestamps)
    if not np.all(np.isfinite(intervals)) or np.any(intervals <= 0):
        raise ValueError(f"{path}: SensorTimestamp is not strictly increasing")
    median_interval = float(np.median(intervals))
    median_absolute_deviation = float(np.median(np.abs(intervals - median_interval)))
    return {
        "file": path.name,
        "n_frames": int(len(timestamps)),
        "duration_s": float((timestamps[-1] - timestamps[0]) / 1_000_000.0),
        "median_interval_us": median_interval,
        "median_fps": 1_000_000.0 / median_interval,
        "interval_mad_us": median_absolute_deviation,
        "interval_q01_us": float(np.quantile(intervals, 0.01)),
        "interval_q99_us": float(np.quantile(intervals, 0.99)),
    }


def _analyze_path(path_raw: str) -> dict:
    path = Path(path_raw)
    try:
        return {"status": "ok", "error": "", **analyze_metadata(path)}
    except (OSError, ValueError) as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "file": path.name,
        }


def _describe(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_audit(
    metadata_dir: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    limit: int | None = None,
) -> dict:
    paths = sorted(metadata_dir.glob("metadata_*.dat"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no metadata_*.dat files found under {metadata_dir}")
    if workers == 1:
        rows = [_analyze_path(str(path)) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_analyze_path, map(str, paths), chunksize=16))
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise RuntimeError("every metadata sidecar failed analysis")
    fps = np.asarray([row["median_fps"] for row in ok_rows], dtype=np.float64)
    intervals = np.asarray(
        [row["median_interval_us"] for row in ok_rows], dtype=np.float64
    )
    summary = {
        "metadata_dir": str(metadata_dir),
        "n_files": len(paths),
        "n_ok": len(ok_rows),
        "n_errors": len(paths) - len(ok_rows),
        "median_fps": _describe(fps),
        "median_interval_us": _describe(intervals),
        "fraction_within_0_5_hz_of_60": float(np.mean(np.abs(fps - 60.0) <= 0.5)),
        "fraction_within_0_5_hz_of_50": float(np.mean(np.abs(fps - 50.0) <= 0.5)),
        "errors": [
            {"file": row["file"], "error": row["error"]}
            for row in rows
            if row["status"] != "ok"
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with (output_dir / "per_file_cadence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fps_summary = summary["median_fps"]
    interval_summary = summary["median_interval_us"]
    report = "\n".join(
        [
            "# Pi metadata cadence audit",
            "",
            f"- Files analyzed: {summary['n_ok']} / {summary['n_files']}",
            (
                f"- Median per-file frame rate: {fps_summary['median']:.4f} Hz "
                f"(1--99%: {fps_summary['q01']:.4f}--{fps_summary['q99']:.4f})"
            ),
            (
                f"- Median per-file hardware interval: {interval_summary['median']:.2f} us "
                f"(1--99%: {interval_summary['q01']:.2f}--{interval_summary['q99']:.2f})"
            ),
            f"- Within 0.5 Hz of 60 Hz: {100 * summary['fraction_within_0_5_hz_of_60']:.2f}%",
            f"- Within 0.5 Hz of 50 Hz: {100 * summary['fraction_within_0_5_hz_of_50']:.2f}%",
            "",
            "The audit uses the Pi SensorTimestamp field rather than the MKV container frame-rate tag.",
            "",
        ]
    )
    (output_dir / "report.md").write_text(report)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        args.metadata_dir,
        args.output_dir,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
