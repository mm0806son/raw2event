"""Recompute the upstream temporal ranking with the frame cadence removed.

The frame-driven simulations carry a strong line at the capture cadence that the
real recordings do not, so the temporal metrics could be penalising a
discretisation artifact of the acquisition rather than simulator behavior. This
audit notches the cadence fundamental and its harmonics out of the event-rate
series, excises the corresponding comb from the interval histogram, and reports
the Spearman correlation between the original and filtered rankings.
``count_ratio`` is carried through as an invariant control; spatial metrics are
out of scope because a temporal notch cannot move them.

``events`` has shape ``(N, 4)`` with columns ``[t_us, x, y, p]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audits.rate_spectrum_audit import (  # noqa: E402
    SpectrumConfig,
    frequency_grid,
    local_line_contrast_db,
    spectrum_from_counts,
)

# Pi SensorTimestamp cadence measured over 5899 capture files (median 60.0601 Hz,
# 1--99% range 60.0601--60.0601 Hz).
CADENCE_HZ = 60.0601
CADENCE_PERIOD_US = 1_000_000.0 / CADENCE_HZ


@dataclass(frozen=True)
class NotchConfig:
    """Cadence-suppression settings shared by every stream and variant."""

    fundamental_hz: float = CADENCE_HZ
    half_width_hz: float = 2.0
    max_freq_hz: float = 500.0
    bin_us: int = 500
    # Per-pixel interval histogram, matching tools/v2e_baseline/threshold_sweep.py:88-111.
    dt_max_us: int = 100_000
    dt_bins: int = 100
    dt_min_group: int = 6
    # Half-width of the excised interval band around each multiple of the frame
    # period; the default equals one interval-histogram bin (1000 us).
    comb_tol_us: float = 1000.0
    # Lag range (in 500 us bins) of the event-rate autocorrelation comparison.
    acf_max_lag: int = 100
    # Background annulus for the line-suppression diagnostic. The spectral audit
    # used 1--5 Hz, which straddles the +/-2 Hz excised band: after the notch the
    # background estimate itself collapses, so the measured contrast overstates
    # the residual line by ~6 dB. Both arms are therefore measured against an
    # annulus that lies entirely outside the excised band.
    line_inner_hz: float = 3.0
    line_outer_hz: float = 8.0

    @property
    def sampling_hz(self) -> float:
        return 1_000_000.0 / self.bin_us

    def validate(self) -> None:
        if self.bin_us <= 0:
            raise ValueError("bin_us must be positive")
        if self.fundamental_hz <= 0:
            raise ValueError("fundamental_hz must be positive")
        if self.half_width_hz <= 0:
            raise ValueError("half_width_hz must be positive")
        if self.half_width_hz >= self.fundamental_hz / 2:
            raise ValueError(
                "half_width_hz must stay below half the fundamental so that "
                "adjacent harmonics do not merge into one another"
            )
        if self.max_freq_hz <= self.fundamental_hz:
            raise ValueError("max_freq_hz must exceed fundamental_hz")
        if self.max_freq_hz >= self.sampling_hz / 2:
            raise ValueError(
                f"max_freq_hz={self.max_freq_hz} must stay below Nyquist "
                f"({self.sampling_hz / 2:.1f} Hz)"
            )
        if self.dt_max_us <= 0 or self.dt_bins <= 0:
            raise ValueError("dt_max_us and dt_bins must be positive")
        if self.comb_tol_us <= 0:
            raise ValueError("comb_tol_us must be positive")
        if self.acf_max_lag < 1:
            raise ValueError("acf_max_lag must be at least 1")
        if not self.half_width_hz < self.line_inner_hz < self.line_outer_hz:
            raise ValueError(
                "expected half_width_hz < line_inner_hz < line_outer_hz so that the "
                "background annulus lies outside the excised band"
            )

    def harmonics_hz(self) -> tuple[float, ...]:
        """Return the cadence fundamental and every harmonic inside the band."""
        self.validate()
        order = 1
        harmonics: list[float] = []
        while order * self.fundamental_hz <= self.max_freq_hz:
            harmonics.append(order * self.fundamental_hz)
            order += 1
        return tuple(harmonics)


@dataclass(frozen=True)
class MetricSpec:
    """One scalar fidelity metric and how it is ranked across variants."""

    name: str
    ideal: float
    aggregate: str  # "median" or "mean"
    domain: str  # "rate", "delta_t" or "count"
    description: str

    def aggregate_values(self, values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0 or not np.isfinite(array).any():
            return float("nan")
        if self.aggregate == "median":
            return float(np.nanmedian(array))
        if self.aggregate == "mean":
            return float(np.nanmean(array))
        raise ValueError(f"unsupported aggregate: {self.aggregate}")


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="rate_logpsd_l1_db",
        ideal=0.0,
        aggregate="median",
        domain="rate",
        description="mean |sim - ref| of the 20-500 Hz event-rate log-power spectra",
    ),
    MetricSpec(
        name="rate_cv_ratio",
        ideal=1.0,
        aggregate="median",
        domain="rate",
        description="sim/ref ratio of the event-rate coefficient of variation",
    ),
    MetricSpec(
        name="rate_acf_l1",
        ideal=0.0,
        aggregate="median",
        domain="rate",
        description="mean |sim - ref| of the event-rate autocorrelation over lags 1..L",
    ),
    MetricSpec(
        name="tv_delta_t",
        ideal=0.0,
        aggregate="mean",
        domain="delta_t",
        description="total-variation distance of per-pixel interval histograms",
    ),
    MetricSpec(
        name="count_ratio",
        ideal=1.0,
        aggregate="median",
        domain="count",
        description="sim/ref total event-count ratio (notch-invariant control)",
    ),
)

# Metric domains that the cadence suppression can move at all. ``count`` is kept
# in the table as an explicit invariance check.
NOTCH_SENSITIVE_DOMAINS = frozenset({"rate", "delta_t"})

ARMS = ("baseline", "notched")

# ``rawY`` is one on-disk spelling of the RAW branch for V03-V12; some layouts
# stage the same files under ``raw`` instead, so both names are accepted.
VALID_SUFFIXES = frozenset({"raw", "rawY", "rgb", "dv"})


@dataclass
class StreamCache:
    """Per-stream quantities that both arms reuse."""

    n_events: int
    counts: np.ndarray
    counts_notched: np.ndarray
    log_power: np.ndarray
    log_power_notched: np.ndarray
    dt_hist: np.ndarray
    negative_bin_fraction: float
    dc_relative_error: float
    line_db: dict[str, float] = field(default_factory=dict)


def event_rate_series(timestamps_us: np.ndarray, bin_us: int) -> np.ndarray:
    """Bin event timestamps into a uniformly sampled event-rate series."""
    timestamps = np.asarray(timestamps_us)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("at least two event timestamps are required")
    relative = timestamps.astype(np.float64) - float(timestamps.min())
    span_us = float(relative.max())
    if not np.isfinite(span_us) or span_us <= 0:
        raise ValueError("event timestamps have non-positive duration")
    bin_ids = np.floor(relative / bin_us).astype(np.int64)
    return np.bincount(bin_ids, minlength=int(bin_ids.max()) + 1).astype(np.float64)


def notch_series(series: np.ndarray, config: NotchConfig) -> np.ndarray:
    """Zero the cadence harmonics of ``series`` in the discrete Fourier domain.

    A zero-phase spectral excision is preferred over a cascade of
    ``scipy.signal.iirnotch`` sections for three reasons. First, the excised
    bands are defined on exactly the frequency axis that the spectral audit used
    to *measure* the lines, so "what is removed" and "what was reported" cannot
    drift apart. Second, the records are short (~2-5 s at a 500 us bin, i.e.
    4k-10k samples) and eight narrow IIR sections would spend a non-negligible
    part of that record on start-up transients. Third, excision needs a single
    half-width parameter instead of a per-harmonic quality factor.

    The DC bin is never inside an excised band (the fundamental is 60 Hz with a
    2 Hz half-width), so the total event count is preserved by construction.
    """
    config.validate()
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("series must be one-dimensional with at least 4 samples")
    spectrum = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(values.size, d=1.0 / config.sampling_hz)
    mask = np.zeros(freqs.shape, dtype=bool)
    for harmonic in config.harmonics_hz():
        mask |= np.abs(freqs - harmonic) <= config.half_width_hz
    mask[0] = False  # never touch DC: it carries the total event count
    spectrum[mask] = 0.0
    return np.fft.irfft(spectrum, n=values.size)


ANSCOMBE_OFFSET = 3.0 / 8.0


def stabilize(counts: np.ndarray) -> np.ndarray:
    """Anscombe variance stabilisation, matching ``spectrum_from_counts``.

    ``tools/audits/rate_spectrum_audit.py`` applies ``sqrt(c + 3/8)``
    before the periodogram, so the 60 Hz lines the audit reported live in *this*
    domain, not in the raw counts.
    """
    return np.sqrt(
        np.clip(np.asarray(counts, dtype=np.float64), 0.0, None) + ANSCOMBE_OFFSET
    )


def spectrum_from_stabilized(
    stabilized: np.ndarray, sampling_hz: float, grid_hz: np.ndarray
) -> np.ndarray:
    """Periodogram of an already-stabilised series, via the audit's estimator.

    The notch must act in the stabilised domain (see ``stabilize``): notching the
    raw counts and letting ``spectrum_from_counts`` square-root afterwards leaves
    most of the line standing, because the square root is a non-linearity that
    re-generates energy at the excised frequencies. On the local smoke set,
    in-domain notching drops the 60.06 Hz target bin by ~17 dB -- the full height
    of the line, i.e. down to the local background floor -- whereas linear-domain
    notching manages ~6 dB and on one variant raised the line instead.

    ``spectrum_from_counts`` hard-codes its own ``sqrt(c + 3/8)`` step, so the
    argument is pre-inverted (``c = s**2 - 3/8``) to make that step the identity.
    This reuses the audit's exact window, detrending, scaling and interpolation
    instead of duplicating them, which is what keeps the baseline arm bit-identical
    to the published spectral audit.
    """
    values = np.clip(np.asarray(stabilized, dtype=np.float64), 0.0, None)
    return spectrum_from_counts(values**2 - ANSCOMBE_OFFSET, sampling_hz, grid_hz)


def per_pixel_delta_t_histogram(events: np.ndarray, config: NotchConfig) -> np.ndarray:
    """Pooled per-pixel inter-event-interval histogram.

    Numerically mirrors the ``per_pixel_dt_tv`` closure inside ``k_health_one``
    (``tools/v2e_baseline/threshold_sweep.py:88-111``) so that the baseline arm
    reproduces the unfiltered ``tv_delta_t``; it is lifted to module level here
    because the comb-excised arm needs the histogram itself, not just the
    total-variation scalar.
    """
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape (N, 4)")
    if events.shape[0] < 2:
        return np.zeros(config.dt_bins, dtype=np.float64)
    events = events[np.argsort(events[:, 0])]
    keys = events[:, 1].astype(np.int64) * 10000 + events[:, 2].astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sorted_events = events[order]
    sorted_keys = keys[order]
    cuts = np.flatnonzero(np.diff(sorted_keys)) + 1
    intervals: list[np.ndarray] = []
    for group in np.split(sorted_events[:, 0], cuts):
        if group.size < config.dt_min_group:
            continue
        deltas = np.diff(group).astype(np.int64)
        deltas = deltas[(deltas > 0) & (deltas < config.dt_max_us)]
        if deltas.size:
            intervals.append(deltas)
    if not intervals:
        return np.zeros(config.dt_bins, dtype=np.float64)
    pooled = np.concatenate(intervals)
    histogram, _ = np.histogram(
        pooled, bins=config.dt_bins, range=(0, config.dt_max_us)
    )
    total = histogram.sum()
    return histogram / total if total > 0 else histogram.astype(np.float64)


def cadence_comb_mask(config: NotchConfig) -> np.ndarray:
    """Interval-histogram bins that fall on a multiple of the frame period."""
    config.validate()
    edges = np.linspace(0.0, float(config.dt_max_us), config.dt_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    period_us = 1_000_000.0 / config.fundamental_hz
    mask = np.zeros(config.dt_bins, dtype=bool)
    order = 1
    while order * period_us <= config.dt_max_us + config.comb_tol_us:
        mask |= np.abs(centers - order * period_us) <= config.comb_tol_us
        order += 1
    return mask


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    """Total-variation distance between two histograms over a shared support."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("histograms must share the same shape")
    if left.sum() <= 0 or right.sum() <= 0:
        return float("nan")
    return 0.5 * float(np.abs(left / left.sum() - right / right.sum()).sum())


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of ``series`` for lags ``1..max_lag``."""
    values = np.asarray(series, dtype=np.float64)
    centered = values - values.mean()
    denominator = float(np.dot(centered, centered))
    lags = np.arange(1, max_lag + 1)
    if denominator <= 0 or centered.size <= 1:
        return np.full(lags.shape, np.nan)
    usable = lags[lags < centered.size]
    result = np.full(lags.shape, np.nan)
    for index, lag in enumerate(usable):
        result[index] = float(np.dot(centered[:-lag], centered[lag:])) / denominator
    return result


def coefficient_of_variation(series: np.ndarray) -> float:
    values = np.asarray(series, dtype=np.float64)
    mean = float(values.mean())
    if not np.isfinite(mean) or mean <= 0:
        return float("nan")
    return float(values.std()) / mean


def build_stream_cache(
    events: np.ndarray,
    config: NotchConfig,
    spectrum_config: SpectrumConfig,
    grid_hz: np.ndarray,
    report_lines_hz: Sequence[float],
) -> StreamCache:
    """Precompute both arms of every per-stream quantity."""
    counts = event_rate_series(events[:, 0], config.bin_us)
    # Linear domain: preserves the total event count exactly, so the coefficient
    # of variation and the autocorrelation stay interpretable as rate statistics.
    counts_notched = notch_series(counts, config)
    # Stabilised domain: the domain in which the spectral audit measured the
    # lines, hence the domain the spectral metrics must be notched in.
    stabilized_notched = notch_series(stabilize(counts), config)
    negative_fraction = float(np.mean(counts_notched < 0.0))
    total = float(counts.sum())
    dc_error = (
        abs(float(counts_notched.sum()) - total) / total if total > 0 else float("nan")
    )
    log_power = spectrum_from_counts(counts, config.sampling_hz, grid_hz)
    log_power_notched = spectrum_from_stabilized(
        stabilized_notched, config.sampling_hz, grid_hz
    )
    line_db: dict[str, float] = {}
    for frequency in report_lines_hz:
        for arm, power in (("baseline", log_power), ("notched", log_power_notched)):
            line_db[f"line_{frequency:g}_hz_db_{arm}"] = local_line_contrast_db(
                grid_hz,
                power,
                frequency,
                inner_hz=config.line_inner_hz,
                outer_hz=config.line_outer_hz,
            )
    return StreamCache(
        n_events=int(events.shape[0]),
        counts=counts,
        counts_notched=counts_notched,
        log_power=log_power,
        log_power_notched=log_power_notched,
        dt_hist=per_pixel_delta_t_histogram(events, config),
        negative_bin_fraction=negative_fraction,
        dc_relative_error=dc_error,
        line_db=line_db,
    )


def paired_metrics(
    sim: StreamCache,
    reference: StreamCache,
    config: NotchConfig,
    arm: str,
) -> dict[str, float]:
    """Compute every scalar metric for one (sim, reference) pair and one arm."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    notched = arm == "notched"
    sim_counts = sim.counts_notched if notched else sim.counts
    ref_counts = reference.counts_notched if notched else reference.counts
    sim_power = sim.log_power_notched if notched else sim.log_power
    ref_power = reference.log_power_notched if notched else reference.log_power

    difference = 10.0 * np.abs(sim_power - ref_power)
    finite = difference[np.isfinite(difference)]
    rate_logpsd_l1_db = float(finite.mean()) if finite.size else float("nan")

    sim_cv = coefficient_of_variation(sim_counts)
    ref_cv = coefficient_of_variation(ref_counts)
    rate_cv_ratio = (
        sim_cv / ref_cv if np.isfinite(ref_cv) and ref_cv > 0 else float("nan")
    )

    acf_difference = np.abs(
        autocorrelation(sim_counts, config.acf_max_lag)
        - autocorrelation(ref_counts, config.acf_max_lag)
    )
    finite_acf = acf_difference[np.isfinite(acf_difference)]
    rate_acf_l1 = float(finite_acf.mean()) if finite_acf.size else float("nan")

    sim_hist = sim.dt_hist
    ref_hist = reference.dt_hist
    if notched:
        keep = ~cadence_comb_mask(config)
        sim_hist = sim_hist[keep]
        ref_hist = ref_hist[keep]

    return {
        "rate_logpsd_l1_db": rate_logpsd_l1_db,
        "rate_cv_ratio": rate_cv_ratio,
        "rate_acf_l1": rate_acf_l1,
        "tv_delta_t": total_variation(sim_hist, ref_hist),
        # DC is preserved by the notch, so this stays identical across arms and
        # acts as an executable invariance check rather than a result.
        "count_ratio": float(sim.n_events / max(reference.n_events, 1)),
    }


def load_events(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "events" not in payload:
            raise ValueError(f"{path}: missing events array")
        events = payload["events"]
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{path}: events must have shape (N, 4), got {events.shape}")
    return events


def _analyze_variant(
    args: tuple[str, str, str, str, str, list[str], NotchConfig, tuple[float, ...]],
) -> dict:
    (
        variant,
        directory_raw,
        suffix,
        reference_dir_raw,
        reference_suffix,
        prefixes,
        config,
        report_lines,
    ) = args
    directory = Path(directory_raw)
    reference_dir = Path(reference_dir_raw)
    spectrum_config = SpectrumConfig(
        bin_us=config.bin_us, max_freq_hz=config.max_freq_hz
    )
    grid_hz = frequency_grid(spectrum_config)

    rows: list[dict] = []
    n_missing = 0
    n_invalid = 0
    for prefix in prefixes:
        sim_path = directory / f"{prefix}_filtered_{suffix}.npz"
        reference_path = reference_dir / f"{prefix}_filtered_{reference_suffix}.npz"
        if not sim_path.exists() or not reference_path.exists():
            n_missing += 1
            continue
        try:
            sim_events = load_events(sim_path)
            reference_events = load_events(reference_path)
            if sim_events.shape[0] < 2 or reference_events.shape[0] < 2:
                raise ValueError("at least two events are required per stream")
            sim_cache = build_stream_cache(
                sim_events, config, spectrum_config, grid_hz, report_lines
            )
            reference_cache = build_stream_cache(
                reference_events, config, spectrum_config, grid_hz, report_lines
            )
        except ValueError:
            n_invalid += 1
            continue
        row: dict[str, float | str] = {"variant": variant, "prefix": prefix}
        for arm in ARMS:
            for key, value in paired_metrics(
                sim_cache, reference_cache, config, arm
            ).items():
                row[f"{key}__{arm}"] = value
        row["sim_negative_bin_fraction"] = sim_cache.negative_bin_fraction
        row["ref_negative_bin_fraction"] = reference_cache.negative_bin_fraction
        row["sim_dc_relative_error"] = sim_cache.dc_relative_error
        row["ref_dc_relative_error"] = reference_cache.dc_relative_error
        for key, value in sim_cache.line_db.items():
            row[f"sim_{key}"] = value
        for key, value in reference_cache.line_db.items():
            row[f"ref_{key}"] = value
        rows.append(row)

    aggregates: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for spec in METRIC_SPECS:
        for arm in ARMS:
            aggregates[arm][spec.name] = spec.aggregate_values(
                [float(row[f"{spec.name}__{arm}"]) for row in rows]
            )
    line_summary: dict[str, float] = {}
    for frequency in report_lines:
        for role in ("sim", "ref"):
            for arm in ARMS:
                key = f"{role}_line_{frequency:g}_hz_db_{arm}"
                values = np.asarray(
                    [float(row[key]) for row in rows if key in row], dtype=np.float64
                )
                values = values[np.isfinite(values)]
                line_summary[key] = (
                    float(np.median(values)) if values.size else float("nan")
                )
    return {
        "variant": variant,
        "directory": str(directory),
        "suffix": suffix,
        "n_prefixes_used": len(rows),
        "n_missing": n_missing,
        "n_invalid": n_invalid,
        "aggregates": aggregates,
        "line_summary": line_summary,
        "median_negative_bin_fraction": float(
            np.median([row["sim_negative_bin_fraction"] for row in rows])
        )
        if rows
        else float("nan"),
        # The excision never touches the DC bin, so the notched series must carry
        # the same total event count. A non-negligible value here is a bug, not a
        # finding (see the interpretation boundary in the report).
        "max_dc_relative_error": float(
            max(
                max(float(row["sim_dc_relative_error"]) for row in rows),
                max(float(row["ref_dc_relative_error"]) for row in rows),
            )
        )
        if rows
        else float("nan"),
        "rows": rows,
    }


def ranking_stability(results: Sequence[dict]) -> list[dict]:
    """Spearman correlation between the baseline and cadence-suppressed rankings."""
    rows: list[dict] = []
    for spec in METRIC_SPECS:
        baseline = np.asarray(
            [
                abs(result["aggregates"]["baseline"][spec.name] - spec.ideal)
                for result in results
            ],
            dtype=np.float64,
        )
        notched = np.asarray(
            [
                abs(result["aggregates"]["notched"][spec.name] - spec.ideal)
                for result in results
            ],
            dtype=np.float64,
        )
        finite = np.isfinite(baseline) & np.isfinite(notched)
        if finite.sum() < 3:
            correlation = float("nan")
        else:
            correlation = float(spearmanr(baseline[finite], notched[finite]).statistic)
        baseline_order = [
            results[index]["variant"] for index in np.argsort(baseline, kind="stable")
        ]
        notched_order = [
            results[index]["variant"] for index in np.argsort(notched, kind="stable")
        ]
        rows.append(
            {
                "metric": spec.name,
                "domain": spec.domain,
                "ideal": spec.ideal,
                "aggregate": spec.aggregate,
                "n_variants": int(finite.sum()),
                "spearman_baseline_vs_notched": correlation,
                "baseline_ranking": baseline_order,
                "notched_ranking": notched_order,
                "top1_changed": bool(baseline_order[:1] != notched_order[:1]),
                "notch_sensitive": spec.domain in NOTCH_SENSITIVE_DOMAINS,
            }
        )
    return rows


def parse_variant(specification: str) -> tuple[str, str, str]:
    """Parse ``ID=directory:suffix`` without splitting colons inside the path."""
    if "=" not in specification or ":" not in specification:
        raise ValueError(f"invalid variant specification: {specification}")
    variant, remainder = specification.split("=", 1)
    directory, suffix = remainder.rsplit(":", 1)
    # ``rawY`` is the historical dev-machine spelling of the RAW branch for
    # V03-V12; some layouts carry ``raw`` instead for the same files.
    if suffix not in VALID_SUFFIXES:
        raise ValueError(
            f"variant suffix must be one of {sorted(VALID_SUFFIXES)}: {specification}"
        )
    return variant, directory, suffix


def _write_per_prefix_csv(results: Sequence[dict], path: Path) -> None:
    rows = [row for result in results for row in result["rows"]]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_report(summary: dict) -> str:
    config = summary["notch_config"]
    harmonics = summary["harmonics_hz"]
    lines = [
        "# Cadence-notch robustness of the upstream simulator ranking",
        "",
        "The 60.0601 Hz Pi capture cadence and its harmonics are removed from the",
        "comparison, and the temporal upstream metrics are recomputed. A ranking that",
        "survives the removal is not an artifact of the 60 fps acquisition grid.",
        "",
        "## Protocol",
        "",
        f"- Variants: {len(summary['variants'])}",
        f"- Reference stream: {summary['reference_dir']} (suffix `{summary['reference_suffix']}`)",
        f"- Prefix roster: {summary['n_prefixes']}",
        f"- Event-rate bin: {config['bin_us']} us ({1e6 / config['bin_us']:.1f} Hz sampling)",
        (
            f"- Notch: zero-phase DFT excision of {config['fundamental_hz']:.4f} Hz "
            f"+/- {config['half_width_hz']:g} Hz and its "
            f"{len(harmonics) - 1} harmonics up to {config['max_freq_hz']:g} Hz"
        ),
        f"- Excised harmonics (Hz): {', '.join(f'{value:.2f}' for value in harmonics)}",
        (
            f"- Excised fraction of the 0--{1e6 / config['bin_us'] / 2:.0f} Hz axis: "
            f"{100 * summary['excised_band_fraction']:.2f}%"
        ),
        (
            f"- Interval-histogram comb: multiples of {summary['cadence_period_us']:.1f} us "
            f"+/- {config['comb_tol_us']:g} us are dropped and the histogram renormalised "
            f"({summary['comb_excised_bins']}/{config['dt_bins']} bins)"
        ),
        "",
        "## Metric scope",
        "",
        "| Metric | Domain | Ideal | Aggregate | Cadence-sensitive | Description |",
        "|---|---|---:|---|---|---|",
    ]
    for spec in METRIC_SPECS:
        sensitive = "yes" if spec.domain in NOTCH_SENSITIVE_DOMAINS else "no (control)"
        lines.append(
            f"| `{spec.name}` | {spec.domain} | {spec.ideal:g} | {spec.aggregate} | "
            f"{sensitive} | {spec.description} |"
        )
    lines.extend(
        [
            "",
            "Spatial metrics (`per_pixel_count_emd`, `spatial_entropy_ratio`,",
            "`active_pixel_ratio`) are out of scope: a temporal notch leaves event",
            "coordinates untouched, so their ranking is unchanged by construction. Their",
            "own robustness is covered by the resolution-sensitivity audit.",
            "",
            "## Notch effectiveness",
            "",
            "Median local line contrast at the cadence fundamental, before and after the notch.",
            (
                f"Both arms use a {config['line_inner_hz']:g}--{config['line_outer_hz']:g} Hz "
                "background annulus rather than the spectral audit's 1--5 Hz: the latter "
                "straddles the excised band, so after the notch it depresses the background "
                "estimate along with the line and overstates the residual by ~6 dB."
            ),
            "",
            "| Variant | Sim baseline (dB) | Sim notched (dB) | Suppression (dB) | Max DC rel. error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    fundamental = summary["notch_config"]["fundamental_hz"]
    baseline_key = f"sim_line_{fundamental:g}_hz_db_baseline"
    notched_key = f"sim_line_{fundamental:g}_hz_db_notched"
    for result in summary["variants"]:
        before = result["line_summary"].get(baseline_key, float("nan"))
        after = result["line_summary"].get(notched_key, float("nan"))
        lines.append(
            f"| {result['variant']} | {before:.2f} | {after:.2f} | {before - after:.2f} | "
            f"{result['max_dc_relative_error']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Rank stability under cadence suppression",
            "",
            "| Metric | Cadence-sensitive | n | Spearman rho | Top-1 changed |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in summary["ranking_stability"]:
        sensitive = "yes" if row["notch_sensitive"] else "no (control)"
        lines.append(
            f"| `{row['metric']}` | {sensitive} | {row['n_variants']} | "
            f"{row['spearman_baseline_vs_notched']:.3f} | "
            f"{'yes' if row['top1_changed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Per-variant aggregates",
            "",
            "| Variant | n | Metric | Baseline | Notched |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for result in summary["variants"]:
        for spec in METRIC_SPECS:
            lines.append(
                f"| {result['variant']} | {result['n_prefixes_used']} | `{spec.name}` | "
                f"{result['aggregates']['baseline'][spec.name]:.4f} | "
                f"{result['aggregates']['notched'][spec.name]:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- A high Spearman rho shows the reported ordering does not depend on the",
            "  60.0601 Hz acquisition cadence; it does not claim the cadence is absent",
            "  from the simulated streams, which the spectral audit already quantified.",
            "- The notch removes a fixed comb, not everything the frame grid induces:",
            "  aperiodic frame-quantisation effects (for example the hard floor on",
            "  simulated inter-event intervals) survive it.",
            "- `count_ratio` must be bit-identical across arms; a difference indicates a",
            "  bug in the DC-preserving excision rather than a finding.",
            "- Ranking is by |aggregate - ideal|; ties are broken by variant order and",
            "  can make the top-1 flag noisy when two variants are numerically level.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    reference_dir: Path,
    prefix_list: Path,
    variant_specs: Sequence[str],
    output_dir: Path,
    *,
    reference_suffix: str = "dv",
    config: NotchConfig | None = None,
    report_lines_hz: Sequence[float] | None = None,
    workers: int = 1,
    limit: int | None = None,
) -> dict:
    """Run the variant-level audit and write reproducible artifacts."""
    config = config or NotchConfig()
    config.validate()
    # Follow the configured fundamental so a --fundamental-hz override cannot
    # desynchronise the reported line frequencies from the excised ones.
    if report_lines_hz is None:
        report_lines_hz = (config.fundamental_hz, 2.0 * config.fundamental_hz)
    prefixes = [
        line.strip()
        for line in prefix_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if limit is not None:
        prefixes = prefixes[:limit]
    if not prefixes:
        raise ValueError(f"no usable prefixes in {prefix_list}")
    variants = [parse_variant(specification) for specification in variant_specs]
    work = [
        (
            variant,
            directory,
            suffix,
            str(reference_dir),
            reference_suffix,
            prefixes,
            config,
            tuple(report_lines_hz),
        )
        for variant, directory, suffix in variants
    ]
    if workers == 1:
        results = [_analyze_variant(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_analyze_variant, work))
    results.sort(key=lambda result: result["variant"])
    usable = [result for result in results if result["n_prefixes_used"] > 0]
    if not usable:
        raise FileNotFoundError(
            "no variant produced a single usable (sim, reference) prefix pair"
        )

    harmonics = config.harmonics_hz()
    excised_fraction = (
        2.0 * config.half_width_hz * len(harmonics) / (config.sampling_hz / 2.0)
    )
    summary = {
        "reference_dir": str(reference_dir),
        "reference_suffix": reference_suffix,
        "prefix_list": str(prefix_list),
        "n_prefixes": len(prefixes),
        "notch_config": asdict(config),
        "harmonics_hz": list(harmonics),
        "excised_band_fraction": float(excised_fraction),
        "cadence_period_us": 1_000_000.0 / config.fundamental_hz,
        "comb_excised_bins": int(cadence_comb_mask(config).sum()),
        "metrics": [asdict(spec) for spec in METRIC_SPECS],
        "variants": [
            {key: value for key, value in result.items() if key != "rows"}
            for result in usable
        ],
        "ranking_stability": ranking_stability(usable),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(_format_report(summary) + "\n")
    _write_per_prefix_csv(usable, output_dir / "per_prefix_metrics.csv")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-suffix", default="dv", choices=["dv", "raw", "rgb"]
    )
    parser.add_argument("--prefix-list", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bin-us", type=int, default=500)
    parser.add_argument("--fundamental-hz", type=float, default=CADENCE_HZ)
    parser.add_argument("--half-width-hz", type=float, default=2.0)
    parser.add_argument("--max-freq-hz", type=float, default=500.0)
    parser.add_argument("--comb-tol-us", type=float, default=1000.0)
    parser.add_argument("--acf-max-lag", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = NotchConfig(
        fundamental_hz=args.fundamental_hz,
        half_width_hz=args.half_width_hz,
        max_freq_hz=args.max_freq_hz,
        bin_us=args.bin_us,
        comb_tol_us=args.comb_tol_us,
        acf_max_lag=args.acf_max_lag,
    )
    summary = run_audit(
        args.reference_dir,
        args.prefix_list,
        args.variant,
        args.output_dir,
        reference_suffix=args.reference_suffix,
        config=config,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(summary["ranking_stability"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
