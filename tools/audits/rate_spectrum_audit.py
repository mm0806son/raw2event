"""Audit stable temporal spectral lines in Raw2Event event streams.

The LCD acquisition protocol may add refresh-rate or PWM artifacts that are
absent from frame-driven simulations. This read-only diagnostic measures event-rate spectra for matched real-DV,
raw-simulated, and RGB-simulated prefixes. It deliberately reports spectral
lines rather than assigning their physical cause: attribution to the display
requires either display metadata or a separate acquisition control.

Input files use the training NPZ convention:
``events`` has shape ``(N, 4)`` with columns ``[t_us, x, y, p]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

DEFAULT_CANDIDATE_FREQUENCIES = (
    50.0,
    60.0,
    75.0,
    90.0,
    100.0,
    120.0,
    144.0,
    180.0,
    200.0,
    240.0,
    288.0,
    300.0,
    360.0,
    480.0,
)
MODALITIES = ("dv", "raw", "rgb")
FILE_RE = re.compile(r"^(?P<prefix>.+)_filtered_(?P<modality>dv|raw|rgb)\.npz$")
DATE_RE = re.compile(r"_(?P<date>20\d{6})_\d{6}$")


@dataclass(frozen=True)
class SpectrumConfig:
    """Frequency-analysis settings shared across every prefix."""

    bin_us: int = 500
    min_freq_hz: float = 20.0
    max_freq_hz: float = 500.0
    freq_step_hz: float = 0.25
    local_inner_hz: float = 1.0
    local_outer_hz: float = 5.0

    @property
    def sampling_hz(self) -> float:
        return 1_000_000.0 / self.bin_us

    def validate(self) -> None:
        if self.bin_us <= 0:
            raise ValueError("bin_us must be positive")
        if self.min_freq_hz <= 0 or self.max_freq_hz <= self.min_freq_hz:
            raise ValueError("expected 0 < min_freq_hz < max_freq_hz")
        if self.max_freq_hz >= self.sampling_hz / 2:
            raise ValueError(
                f"max_freq_hz={self.max_freq_hz} must be below Nyquist "
                f"({self.sampling_hz / 2:.1f} Hz)"
            )
        if not 0 < self.local_inner_hz < self.local_outer_hz:
            raise ValueError("expected 0 < local_inner_hz < local_outer_hz")


def frequency_grid(config: SpectrumConfig) -> np.ndarray:
    """Return the common frequency grid used to aggregate variable durations."""
    config.validate()
    stop = config.max_freq_hz + config.freq_step_hz * 0.5
    return np.arange(config.min_freq_hz, stop, config.freq_step_hz, dtype=np.float64)


def spectrum_from_counts(
    counts: np.ndarray,
    sampling_hz: float,
    grid_hz: np.ndarray,
) -> np.ndarray:
    """Return log10 periodogram power interpolated onto ``grid_hz``.

    Square-root stabilization reduces the dependence of spectral variance on
    total event count. A Hann window and linear detrending limit leakage from
    the slow circular camera trajectory into the 20--500 Hz audit band.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size < 16:
        raise ValueError(
            "counts must be a one-dimensional series with at least 16 bins"
        )
    stabilized = np.sqrt(counts + 3.0 / 8.0)
    freqs, power = signal.periodogram(
        stabilized,
        fs=sampling_hz,
        window="hann",
        detrend="linear",
        scaling="density",
    )
    power = np.maximum(power, np.finfo(np.float64).tiny)
    return np.interp(grid_hz, freqs, np.log10(power), left=np.nan, right=np.nan)


def spectrum_from_timestamps(
    timestamps_us: np.ndarray,
    config: SpectrumConfig,
    grid_hz: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Convert event timestamps to a spectral estimate and duration."""
    timestamps = np.asarray(timestamps_us)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("at least two event timestamps are required")
    relative = timestamps.astype(np.float64) - float(timestamps.min())
    duration_us = float(relative.max())
    if not math.isfinite(duration_us) or duration_us <= 0:
        raise ValueError("event timestamps have non-positive duration")
    bin_ids = np.floor(relative / config.bin_us).astype(np.int64)
    counts = np.bincount(bin_ids, minlength=int(bin_ids.max()) + 1)
    log_power = spectrum_from_counts(counts, config.sampling_hz, grid_hz)
    return log_power, duration_us / 1_000_000.0


def local_line_contrast_db(
    grid_hz: np.ndarray,
    log10_power: np.ndarray,
    target_hz: float,
    *,
    inner_hz: float = 1.0,
    outer_hz: float = 5.0,
) -> float:
    """Measure a spectral line relative to its local annular background."""
    grid = np.asarray(grid_hz, dtype=np.float64)
    log_power = np.asarray(log10_power, dtype=np.float64)
    if grid.shape != log_power.shape:
        raise ValueError("grid_hz and log10_power must have the same shape")
    target_idx = int(np.argmin(np.abs(grid - target_hz)))
    distance = np.abs(grid - target_hz)
    background = log_power[(distance >= inner_hz) & (distance <= outer_hz)]
    background = background[np.isfinite(background)]
    if background.size == 0 or not np.isfinite(log_power[target_idx]):
        return float("nan")
    return 10.0 * float(log_power[target_idx] - np.median(background))


def _parse_prefix(path: Path, modality: str) -> str:
    match = FILE_RE.match(path.name)
    if match is None or match.group("modality") != modality:
        raise ValueError(f"unexpected {modality} filename: {path.name}")
    return match.group("prefix")


def _prefix_metadata(prefix: str) -> tuple[str, str]:
    parts = prefix.split("_")
    label = parts[1] if len(parts) > 1 else "unknown"
    match = DATE_RE.search(prefix)
    session = match.group("date")[:6] if match else "unknown"
    return label, session


def discover_prefixes(
    data_dir: Path,
    modalities: Sequence[str],
    limit: int | None = None,
) -> list[str]:
    """Return prefixes for which every requested modality exists."""
    modality_sets: list[set[str]] = []
    for modality in modalities:
        paths = data_dir.glob(f"*_filtered_{modality}.npz")
        modality_sets.append({_parse_prefix(path, modality) for path in paths})
    if not modality_sets:
        return []
    prefixes = sorted(set.intersection(*modality_sets))
    return prefixes if limit is None else prefixes[:limit]


def _analyze_one(
    args: tuple[str, str, str, SpectrumConfig, np.ndarray, tuple[float, ...]],
) -> dict:
    data_dir_raw, prefix, modality, config, grid_hz, candidates = args
    path = Path(data_dir_raw) / f"{prefix}_filtered_{modality}.npz"
    with np.load(path, allow_pickle=False) as payload:
        if "events" not in payload:
            raise ValueError(f"{path}: missing events array")
        events = payload["events"]
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{path}: events must have shape (N, 4), got {events.shape}")
    label, session = _prefix_metadata(prefix)
    try:
        log_power, duration_s = spectrum_from_timestamps(events[:, 0], config, grid_hz)
        analysis_status = "ok"
        analysis_error = ""
    except ValueError as exc:
        log_power = np.full_like(grid_hz, np.nan)
        duration_s = float("nan")
        analysis_status = "invalid_stream"
        analysis_error = str(exc)
    contrasts = {
        f"line_{frequency:g}_hz_db": local_line_contrast_db(
            grid_hz,
            log_power,
            frequency,
            inner_hz=config.local_inner_hz,
            outer_hz=config.local_outer_hz,
        )
        for frequency in candidates
        if config.min_freq_hz <= frequency <= config.max_freq_hz
    }
    return {
        "prefix": prefix,
        "label": label,
        "session": session,
        "modality": modality,
        "n_events": int(len(events)),
        "duration_s": duration_s,
        "analysis_status": analysis_status,
        "analysis_error": analysis_error,
        "log10_power": log_power,
        **contrasts,
    }


def _local_contrast_curve(
    grid_hz: np.ndarray, log_power: np.ndarray, config: SpectrumConfig
):
    return np.asarray(
        [
            local_line_contrast_db(
                grid_hz,
                log_power,
                frequency,
                inner_hz=config.local_inner_hz,
                outer_hz=config.local_outer_hz,
            )
            for frequency in grid_hz
        ],
        dtype=np.float64,
    )


def _top_aggregate_lines(
    grid_hz: np.ndarray,
    median_log_power: np.ndarray,
    config: SpectrumConfig,
    n_lines: int = 12,
) -> list[dict]:
    contrast = _local_contrast_curve(grid_hz, median_log_power, config)
    finite = np.nan_to_num(contrast, nan=-np.inf)
    min_distance_bins = max(1, int(round(4.0 / config.freq_step_hz)))
    peaks, properties = signal.find_peaks(
        finite,
        distance=min_distance_bins,
        prominence=0.5,
    )
    prominences = properties.get("prominences", np.zeros(len(peaks)))
    ranked = sorted(
        zip(peaks, prominences),
        key=lambda item: (finite[item[0]], item[1]),
        reverse=True,
    )
    return [
        {
            "frequency_hz": float(grid_hz[index]),
            "local_contrast_db": float(contrast[index]),
            "prominence_db": float(prominence),
        }
        for index, prominence in ranked[:n_lines]
    ]


def _candidate_summary(rows: list[dict], candidates: Sequence[float]) -> list[dict]:
    summary: list[dict] = []
    for modality in MODALITIES:
        modality_rows = [row for row in rows if row["modality"] == modality]
        if not modality_rows:
            continue
        for frequency in candidates:
            key = f"line_{frequency:g}_hz_db"
            values = np.asarray(
                [row[key] for row in modality_rows if key in row], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            summary.append(
                {
                    "modality": modality,
                    "frequency_hz": frequency,
                    "n_prefixes": int(values.size),
                    "median_contrast_db": float(np.median(values)),
                    "q25_contrast_db": float(np.quantile(values, 0.25)),
                    "q75_contrast_db": float(np.quantile(values, 0.75)),
                    "prevalence_gt_6db": float(np.mean(values > 6.0)),
                    "prevalence_gt_10db": float(np.mean(values > 10.0)),
                }
            )
    return summary


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [key for key in rows[0] if key != "log10_power"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_plot(
    grid_hz: np.ndarray,
    aggregates: dict[str, dict],
    config: SpectrumConfig,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = {"dv": "#222222", "raw": "#0072B2", "rgb": "#D55E00"}
    for modality in MODALITIES:
        if modality not in aggregates:
            continue
        median_power = aggregates[modality]["median_log10_power"]
        centered_db = 10.0 * (median_power - np.nanmedian(median_power))
        axes[0].plot(
            grid_hz, centered_db, label=modality, color=colors[modality], lw=1.1
        )
        contrast = _local_contrast_curve(grid_hz, median_power, config)
        axes[1].plot(grid_hz, contrast, label=modality, color=colors[modality], lw=1.1)
    axes[0].set_ylabel("Median log power (dB, band-centered)")
    axes[1].set_ylabel("Local line contrast (dB)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend()
    axes[1].axhline(6.0, color="#777777", ls="--", lw=0.8, label="6 dB")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _format_markdown(
    prefixes: Sequence[str],
    config: SpectrumConfig,
    aggregates: dict[str, dict],
    candidate_summary: list[dict],
    anomalies: list[dict],
) -> str:
    lines = [
        "# Monitor/display temporal artifact audit",
        "",
        "This audit detects stable event-rate spectral lines. A line is evidence of a periodic",
        "component, but is not by itself proof that the LCD caused it.",
        "",
        "## Protocol",
        "",
        f"- Matched prefixes: {len(prefixes)}",
        f"- Event-rate bin: {config.bin_us} us ({config.sampling_hz:.1f} Hz sampling)",
        f"- Audit band: {config.min_freq_hz:g}--{config.max_freq_hz:g} Hz",
        "- Spectrum: square-root-stabilized event counts, linear detrending, Hann periodogram",
        (
            f"- Line contrast: target power relative to the median local background "
            f"({config.local_inner_hz:g}--{config.local_outer_hz:g} Hz annulus)"
        ),
        f"- Invalid/empty modality streams excluded from spectra: {len(anomalies)}",
        "",
        "## Strongest aggregate spectral lines",
        "",
        "| Modality | Frequency (Hz) | Local contrast (dB) | Peak prominence (dB) |",
        "|---|---:|---:|---:|",
    ]
    for modality in MODALITIES:
        for row in aggregates.get(modality, {}).get("top_lines", [])[:10]:
            lines.append(
                f"| {modality} | {row['frequency_hz']:.2f} | "
                f"{row['local_contrast_db']:.2f} | {row['prominence_db']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Candidate display/frame frequencies",
            "",
            "| Modality | Frequency (Hz) | Median dB | IQR dB | >6 dB | >10 dB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_summary:
        lines.append(
            f"| {row['modality']} | {row['frequency_hz']:.0f} | "
            f"{row['median_contrast_db']:.2f} | "
            f"[{row['q25_contrast_db']:.2f}, {row['q75_contrast_db']:.2f}] | "
            f"{100 * row['prevalence_gt_6db']:.1f}% | "
            f"{100 * row['prevalence_gt_10db']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Invalid or empty streams",
            "",
            "| Prefix | Modality | Events | Reason |",
            "|---|---|---:|---|",
        ]
    )
    if anomalies:
        for row in anomalies:
            lines.append(
                f"| {row['prefix']} | {row['modality']} | {row['n_events']} | "
                f"{row['analysis_error']} |"
            )
    else:
        lines.append("| — | — | — | None |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A stable DV-only line supports a high-frequency real-acquisition component that the",
            "  frame-driven simulators do not reproduce.",
            "- A line shared by DV, raw, and RGB may instead arise from common motion or processing.",
            "- Display attribution requires display refresh/PWM metadata or a separate control capture.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    data_dir: Path,
    output_dir: Path,
    *,
    modalities: Sequence[str] = MODALITIES,
    config: SpectrumConfig | None = None,
    candidates: Sequence[float] = DEFAULT_CANDIDATE_FREQUENCIES,
    workers: int = 1,
    limit: int | None = None,
) -> dict:
    """Run the directory-level audit and write reproducible artifacts."""
    config = config or SpectrumConfig()
    config.validate()
    invalid_modalities = sorted(set(modalities) - set(MODALITIES))
    if invalid_modalities:
        raise ValueError(f"unsupported modalities: {invalid_modalities}")
    prefixes = discover_prefixes(data_dir, modalities, limit=limit)
    if not prefixes:
        raise FileNotFoundError(f"no matched event NPZ files found under {data_dir}")
    grid_hz = frequency_grid(config)
    work_items = [
        (str(data_dir), prefix, modality, config, grid_hz, tuple(candidates))
        for prefix in prefixes
        for modality in modalities
    ]
    if workers == 1:
        rows = [_analyze_one(item) for item in work_items]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_analyze_one, work_items, chunksize=8))

    aggregates: dict[str, dict] = {}
    for modality in modalities:
        modality_rows = [
            row
            for row in rows
            if row["modality"] == modality and row["analysis_status"] == "ok"
        ]
        if not modality_rows:
            continue
        spectra = np.stack(
            [row["log10_power"] for row in modality_rows],
            axis=0,
        )
        median_power = np.nanmedian(spectra, axis=0)
        aggregates[modality] = {
            "n_prefixes": int(spectra.shape[0]),
            "median_log10_power": median_power,
            "q25_log10_power": np.nanquantile(spectra, 0.25, axis=0),
            "q75_log10_power": np.nanquantile(spectra, 0.75, axis=0),
            "top_lines": _top_aggregate_lines(grid_hz, median_power, config),
        }
    candidate_summary = _candidate_summary(rows, candidates)
    anomalies = [
        {
            "prefix": row["prefix"],
            "modality": row["modality"],
            "n_events": row["n_events"],
            "analysis_error": row["analysis_error"],
        }
        for row in rows
        if row["analysis_status"] != "ok"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "per_prefix_spectral_lines.csv")
    np.savez_compressed(
        output_dir / "aggregate_spectra.npz",
        frequency_hz=grid_hz,
        **{
            f"{modality}_{stat}": values
            for modality, aggregate in aggregates.items()
            for stat, values in aggregate.items()
            if isinstance(values, np.ndarray)
        },
    )
    serializable_aggregates = {
        modality: {
            "n_prefixes": aggregate["n_prefixes"],
            "top_lines": aggregate["top_lines"],
        }
        for modality, aggregate in aggregates.items()
    }
    summary = {
        "data_dir": str(data_dir),
        "n_matched_prefixes": len(prefixes),
        "modalities": list(modalities),
        "config": config.__dict__,
        "aggregates": serializable_aggregates,
        "candidate_summary": candidate_summary,
        "anomalies": anomalies,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(
        _format_markdown(prefixes, config, aggregates, candidate_summary, anomalies)
    )
    _write_plot(grid_hz, aggregates, config, output_dir / "aggregate_spectra.png")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES)
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bin-us", type=int, default=500)
    parser.add_argument("--min-freq-hz", type=float, default=20.0)
    parser.add_argument("--max-freq-hz", type=float, default=500.0)
    parser.add_argument("--freq-step-hz", type=float, default=0.25)
    parser.add_argument(
        "--candidate-frequencies",
        type=float,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_FREQUENCIES),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SpectrumConfig(
        bin_us=args.bin_us,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        freq_step_hz=args.freq_step_hz,
    )
    summary = run_audit(
        args.data_dir,
        args.output_dir,
        modalities=args.modalities,
        config=config,
        candidates=args.candidate_frequencies,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
