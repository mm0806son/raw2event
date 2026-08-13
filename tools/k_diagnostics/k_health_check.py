"""Multi-dimensional K-health check — post-hoc analysis of pre-computed event NPZs.

This script does NOT re-run the simulator. It reads the DV + Stage1 {raw, rgb}
+ Stage2 polarity-aware {raw, rgb} NPZs from
``output/multi_sample_k_compare_20260422/{sample}/events/`` and reports the
the core upstream fidelity dimensions:

    1. count_ratio           = N_sim / N_dv
    2. tv_delta_t            = TV(per-pixel Δt histogram)
    3. spatial_entropy_ratio = H(per-pixel count dist) / H_dv
    4. active_pixel_ratio    = (active pixel frac)_sim / (active pixel frac)_dv
    5. polarity_delta        = |pos_frac_sim - pos_frac_dv|

Pass/fail is judged against the ``THRESHOLDS`` table; one report row is
emitted per (sample, K-set, modality) tuple. ``tv_delta_t`` is read from the
upstream ``results.json`` rather than recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "output" / "multi_sample_k_compare_20260422"
RESULTS_JSON = BASE / "results.json"

SAMPLES = [
    "42868_horse_1_8698_20251228_234242",
    "10000_automobile_5_1087_20251224_105416",
    "12001_bird_1_1003_20251224_172825",
]
K_SETS = ("stage1", "stage2_pa")
MODS = ("raw", "rgb")
H_IMG, W_IMG = 260, 346

# Pass/fail thresholds (min, max). A metric passes when value is in [min, max].
THRESHOLDS = {
    "count_ratio":           (0.5, 2.0),
    "tv_delta_t":            (0.0, 0.25),
    "spatial_entropy_ratio": (0.8, 1.2),
    "active_pixel_ratio":    (0.8, 1.2),
    "polarity_delta":        (0.0, 0.10),
}


def _load_events(npz_path: Path) -> np.ndarray:
    return np.load(npz_path)["events"]


def _count_map(ev: np.ndarray) -> np.ndarray:
    cmap = np.zeros((H_IMG, W_IMG), dtype=np.int64)
    if ev.shape[0] == 0:
        return cmap
    ys = ev[:, 2].astype(np.int64)
    xs = ev[:, 1].astype(np.int64)
    in_bounds = (ys >= 0) & (ys < H_IMG) & (xs >= 0) & (xs < W_IMG)
    np.add.at(cmap, (ys[in_bounds], xs[in_bounds]), 1)
    return cmap


def _spatial_entropy(count_map: np.ndarray) -> float:
    flat = count_map.ravel().astype(np.float64)
    s = flat.sum()
    if s == 0:
        return 0.0
    p = flat / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _active_pixel_fraction(count_map: np.ndarray) -> float:
    return float((count_map > 0).sum()) / float(count_map.size)


def _pos_fraction(ev: np.ndarray) -> float:
    if ev.shape[0] == 0:
        return 0.0
    return float((ev[:, 3] == 1).sum()) / float(ev.shape[0])


def _passes(metric: str, value: float) -> bool:
    lo, hi = THRESHOLDS[metric]
    return lo <= value <= hi


def _load_tv_from_results() -> dict[str, dict[str, dict[str, float]]]:
    """Extract TV(Δt) per (sample, K set, mod) from existing results.json."""
    if not RESULTS_JSON.exists():
        return {}
    data = json.loads(RESULTS_JSON.read_text())
    tv: dict[str, dict[str, dict[str, float]]] = {}
    for sample, rec in data.get("samples", {}).items():
        tv[sample] = {}
        for ks in K_SETS:
            tv[sample][ks] = {}
            for mod in MODS:
                val = rec.get(ks, {}).get(mod, {}).get("tv_vs_dv")
                tv[sample][ks][mod] = float(val) if val is not None else float("nan")
    return tv


def evaluate_sample(sample: str, tv_map: dict) -> dict[str, Any]:
    events_dir = BASE / sample / "events"
    dv = _load_events(events_dir / f"{sample}_dv.npz")
    dv_cmap = _count_map(dv)
    dv_entropy = _spatial_entropy(dv_cmap)
    dv_active = _active_pixel_fraction(dv_cmap)
    dv_pos = _pos_fraction(dv)

    record: dict[str, Any] = {
        "dv": {
            "n_events": int(dv.shape[0]),
            "spatial_entropy": dv_entropy,
            "active_pixel_frac": dv_active,
            "pos_fraction": dv_pos,
        }
    }

    for ks in K_SETS:
        record[ks] = {}
        for mod in MODS:
            ev = _load_events(events_dir / f"{sample}_{ks}_{mod}.npz")
            cmap = _count_map(ev)
            entropy = _spatial_entropy(cmap)
            active = _active_pixel_fraction(cmap)
            pos = _pos_fraction(ev)
            tv_dt = tv_map.get(sample, {}).get(ks, {}).get(mod, float("nan"))

            metrics = {
                "n_events": int(ev.shape[0]),
                "count_ratio":           float(ev.shape[0] / max(dv.shape[0], 1)),
                "tv_delta_t":            float(tv_dt),
                "spatial_entropy":       entropy,
                "spatial_entropy_ratio": entropy / dv_entropy if dv_entropy > 0 else float("nan"),
                "active_pixel_frac":     active,
                "active_pixel_ratio":    active / dv_active if dv_active > 0 else float("nan"),
                "pos_fraction":          pos,
                "polarity_delta":        abs(pos - dv_pos),
            }
            metrics["pass"] = {m: _passes(m, metrics[m]) for m in THRESHOLDS}
            metrics["pass_count"] = int(sum(metrics["pass"].values()))
            metrics["pass_total"] = len(THRESHOLDS)
            record[ks][mod] = metrics

    return record


def _format_row(sample: str, ks: str, mod: str, m: dict) -> str:
    def fmt(v: float, passed: bool, width: int = 7, prec: int = 3) -> str:
        mark = "✓" if passed else "✗"
        return f"{v:>{width}.{prec}f}{mark}"

    return (
        f"{sample[:28]:28} {ks:>9} {mod.upper():>3}  "
        f"{fmt(m['count_ratio'],          m['pass']['count_ratio'])}  "
        f"{fmt(m['tv_delta_t'],           m['pass']['tv_delta_t'])}  "
        f"{fmt(m['spatial_entropy_ratio'], m['pass']['spatial_entropy_ratio'])}  "
        f"{fmt(m['active_pixel_ratio'],   m['pass']['active_pixel_ratio'])}  "
        f"{fmt(m['polarity_delta'],       m['pass']['polarity_delta'])}  "
        f"{m['pass_count']}/{m['pass_total']}"
    )


def main() -> None:
    tv_map = _load_tv_from_results()
    overall: dict[str, Any] = {
        "thresholds": {m: list(THRESHOLDS[m]) for m in THRESHOLDS},
        "samples": {},
    }

    header = (
        f"{'sample':28} {'K set':>9} {'mod':>3}  "
        f"{'cnt_rat':>8}  {'TV(Δt)':>8}  {'H_rat':>8}  {'act_rat':>8}  {'|Δp|':>8}  pass"
    )
    print(header)
    print("-" * len(header))

    for sample in SAMPLES:
        rec = evaluate_sample(sample, tv_map)
        overall["samples"][sample] = rec
        for ks in K_SETS:
            for mod in MODS:
                m = rec[ks][mod]
                print(_format_row(sample, ks, mod, m))
        print()

    out_path = BASE / "k_health_check.json"
    out_path.write_text(json.dumps(overall, indent=2))
    print(f"→ full report: {out_path}")

    # Aggregate cross-sample summary (mean per K set × mod)
    print("\n" + "=" * 80)
    print("CROSS-SAMPLE MEAN (3 samples)")
    print("=" * 80)
    print(f"{'K set':>9} {'mod':>3}  {'cnt_rat':>8}  {'TV(Δt)':>8}  {'H_rat':>8}  {'act_rat':>8}  {'|Δp|':>8}  avg_pass")
    print("-" * 80)
    for ks in K_SETS:
        for mod in MODS:
            agg: dict[str, list[float]] = {
                k: [] for k in ("count_ratio", "tv_delta_t", "spatial_entropy_ratio",
                                "active_pixel_ratio", "polarity_delta", "pass_count")
            }
            for sample in SAMPLES:
                m = overall["samples"][sample][ks][mod]
                for k in agg:
                    agg[k].append(m[k])
            means = {k: float(np.mean(v)) for k, v in agg.items()}
            print(
                f"{ks:>9} {mod.upper():>3}  "
                f"{means['count_ratio']:>8.3f}  "
                f"{means['tv_delta_t']:>8.3f}  "
                f"{means['spatial_entropy_ratio']:>8.3f}  "
                f"{means['active_pixel_ratio']:>8.3f}  "
                f"{means['polarity_delta']:>8.3f}  "
                f"{means['pass_count']:>5.2f}/{len(THRESHOLDS)}"
            )


if __name__ == "__main__":
    main()
