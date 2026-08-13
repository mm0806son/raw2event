"""Audit whether upstream spatial-fidelity rankings depend on unified80 resolution."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr


def rescale_coordinates(
    events: np.ndarray, resolution: int, source_resolution: int = 80
) -> np.ndarray:
    """Copy events and map integer coordinates to a lower square grid."""
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape (N, 4)")
    if not 1 <= resolution <= source_resolution:
        raise ValueError("resolution must be between 1 and source_resolution")
    output = events.copy()
    if len(output):
        output[:, 1] = np.minimum(
            resolution - 1,
            np.floor(output[:, 1].astype(np.float64) * resolution / source_resolution),
        ).astype(np.int64)
        output[:, 2] = np.minimum(
            resolution - 1,
            np.floor(output[:, 2].astype(np.float64) * resolution / source_resolution),
        ).astype(np.int64)
    return output


def _count_map(events: np.ndarray, resolution: int) -> np.ndarray:
    counts = np.zeros((resolution, resolution), dtype=np.int64)
    if len(events) == 0:
        return counts
    x = events[:, 1].astype(np.int64)
    y = events[:, 2].astype(np.int64)
    valid = (x >= 0) & (x < resolution) & (y >= 0) & (y < resolution)
    np.add.at(counts, (y[valid], x[valid]), 1)
    return counts


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0].astype(np.float64)
    if probabilities.size == 0:
        return float("nan")
    probabilities /= probabilities.sum()
    return -float(np.sum(probabilities * np.log(probabilities)))


def spatial_metrics(
    sim_events: np.ndarray, dv_events: np.ndarray, resolution: int
) -> dict[str, float]:
    """Reproduce the benchmark's spatial descriptors on a chosen grid."""
    sim_counts = _count_map(sim_events, resolution)
    dv_counts = _count_map(dv_events, resolution)
    if sim_counts.sum() == 0 or dv_counts.sum() == 0:
        return {
            "per_pixel_count_emd": float("nan"),
            "spatial_entropy_ratio": float("nan"),
            "active_pixel_ratio": float("nan"),
        }
    sim_distribution = sim_counts.ravel().astype(np.float64)
    dv_distribution = dv_counts.ravel().astype(np.float64)
    sim_distribution /= sim_distribution.sum()
    dv_distribution /= dv_distribution.sum()
    per_pixel_count_emd = float(
        np.abs(np.cumsum(sim_distribution) - np.cumsum(dv_distribution)).mean()
    )
    sim_entropy = _entropy(sim_counts)
    dv_entropy = _entropy(dv_counts)
    return {
        "per_pixel_count_emd": per_pixel_count_emd,
        "spatial_entropy_ratio": sim_entropy / dv_entropy,
        "active_pixel_ratio": float(
            np.count_nonzero(sim_counts) / np.count_nonzero(dv_counts)
        ),
    }


def parse_variant(specification: str) -> tuple[str, str, str]:
    """Parse ``ID=directory:suffix`` without splitting colons in the path."""
    if "=" not in specification or ":" not in specification:
        raise ValueError(f"invalid variant specification: {specification}")
    variant, remainder = specification.split("=", 1)
    directory, suffix = remainder.rsplit(":", 1)
    if suffix not in {"raw", "rgb"}:
        raise ValueError(f"variant suffix must be raw or rgb: {specification}")
    return variant, directory, suffix


def _analyze_variant(
    args: tuple[str, str, str, str, list[str], tuple[int, ...]],
) -> dict:
    variant, directory_raw, suffix, dv_dir_raw, prefixes, resolutions = args
    directory = Path(directory_raw)
    dv_dir = Path(dv_dir_raw)
    rows = {resolution: [] for resolution in resolutions}
    n_missing = 0
    for prefix in prefixes:
        sim_path = directory / f"{prefix}_filtered_{suffix}.npz"
        dv_path = dv_dir / f"{prefix}_filtered_dv.npz"
        if not sim_path.exists() or not dv_path.exists():
            n_missing += 1
            continue
        with np.load(sim_path, allow_pickle=False) as payload:
            sim_events_80 = payload["events"]
        with np.load(dv_path, allow_pickle=False) as payload:
            dv_events_80 = payload["events"]
        if len(sim_events_80) == 0 or len(dv_events_80) == 0:
            continue
        for resolution in resolutions:
            sim_events = rescale_coordinates(sim_events_80, resolution)
            dv_events = rescale_coordinates(dv_events_80, resolution)
            rows[resolution].append(spatial_metrics(sim_events, dv_events, resolution))
    aggregates = {}
    for resolution, resolution_rows in rows.items():
        aggregates[str(resolution)] = {
            "n_prefixes": len(resolution_rows),
            "per_pixel_count_emd_median": float(
                np.nanmedian([row["per_pixel_count_emd"] for row in resolution_rows])
            ),
            "spatial_entropy_ratio_mean": float(
                np.nanmean([row["spatial_entropy_ratio"] for row in resolution_rows])
            ),
            "active_pixel_ratio_mean": float(
                np.nanmean([row["active_pixel_ratio"] for row in resolution_rows])
            ),
        }
    return {
        "variant": variant,
        "directory": str(directory),
        "suffix": suffix,
        "n_missing": n_missing,
        "aggregates": aggregates,
    }


def _ranking_stability(results: list[dict], resolutions: Sequence[int]) -> list[dict]:
    reference = str(max(resolutions))
    rows = []
    for metric, ideal in (
        ("per_pixel_count_emd_median", 0.0),
        ("spatial_entropy_ratio_mean", 1.0),
        ("active_pixel_ratio_mean", 1.0),
    ):
        reference_values = np.asarray(
            [abs(result["aggregates"][reference][metric] - ideal) for result in results]
        )
        for resolution in sorted(resolutions, reverse=True):
            values = np.asarray(
                [
                    abs(result["aggregates"][str(resolution)][metric] - ideal)
                    for result in results
                ]
            )
            correlation = spearmanr(reference_values, values).statistic
            rows.append(
                {
                    "metric": metric,
                    "resolution": resolution,
                    "spearman_vs_highest_resolution": float(correlation),
                }
            )
    return rows


def _write_report(summary: dict, path: Path) -> None:
    lines = [
        "# unified80 spatial-metric resolution sensitivity",
        "",
        f"- Variants: {len(summary['variants'])}",
        f"- Prefix roster: {summary['n_prefixes']}",
        f"- Resolutions: {', '.join(map(str, summary['resolutions']))}",
        "",
        "## Rank stability against the 80×80 result",
        "",
        "| Metric | Resolution | Spearman ρ |",
        "|---|---:|---:|",
    ]
    for row in summary["ranking_stability"]:
        lines.append(
            f"| {row['metric']} | {row['resolution']} | "
            f"{row['spearman_vs_highest_resolution']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Stable rankings under further downsampling show robustness within the already",
            "  normalized representation; they cannot recover or validate native-resolution edges.",
            "- A rank reversal indicates that the reported spatial conclusion is grid-sensitive.",
            "- Count and polarity metrics are omitted because coordinate downsampling does not change them.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run_audit(
    dv_dir: Path,
    prefix_list: Path,
    variant_specs: Sequence[str],
    output_dir: Path,
    *,
    resolutions: Sequence[int] = (80, 64, 40, 20),
    workers: int = 1,
) -> dict:
    prefixes = [
        line.strip()
        for line in prefix_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    variants = [parse_variant(specification) for specification in variant_specs]
    work = [
        (variant, directory, suffix, str(dv_dir), prefixes, tuple(resolutions))
        for variant, directory, suffix in variants
    ]
    if workers == 1:
        results = [_analyze_variant(item) for item in work]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_analyze_variant, work))
    results.sort(key=lambda result: result["variant"])
    summary = {
        "dv_dir": str(dv_dir),
        "prefix_list": str(prefix_list),
        "n_prefixes": len(prefixes),
        "resolutions": list(resolutions),
        "variants": results,
        "ranking_stability": _ranking_stability(results, resolutions),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(summary, output_dir / "report.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dv-dir", type=Path, required=True)
    parser.add_argument("--prefix-list", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[80, 64, 40, 20])
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_audit(
        args.dv_dir,
        args.prefix_list,
        args.variant,
        args.output_dir,
        resolutions=args.resolutions,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
