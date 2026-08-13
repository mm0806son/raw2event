"""Stage 2: 1-D pos_thres / neg_thres sweep on a small calibration prefix set.

For each (input_modality, pos_thres == neg_thres) ∈ {0.10, 0.15, 0.20, 0.25, 0.30}:
  1. Generate v2e events on each calibration prefix
  2. Apply AprilTag crop (same as production)
  3. Run 5-D K health check vs the matching DV NPZ
Aggregate: pick the threshold with max pass count; write the winners back into
v2e_thresholds.json under tuned_rgb / tuned_rawY.

Cost: 5 prefix * 2 modality * 5 thresholds = 50 v2e runs (~7-12 GPU-hours).

Usage (run on GPU):
  python tools/v2e_baseline/threshold_sweep.py \
    --input_dir ./data \
    --dv_npz_dir ./data/cifar10_npz_346x260 \
    --calib_prefix_list tools/v2e_baseline/calib_5_prefixes.txt \
    --output_root ./output/threshold_sweep \
    --gpu_id 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from tools.v2e_baseline.v2e_helpers import (  # noqa: E402
    load_thresholds, process_one_prefix_v2e_subprocess, THRESHOLDS_JSON,
)

SWEEP_VALUES = [0.10, 0.15, 0.20, 0.25, 0.30]
INPUT_MODALITIES = ["rgb", "rawY"]


def parse_calib(path: Path) -> list[str]:
    with open(path) as f:
        out = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return out


def k_health_one(sim_npz: Path, dv_npz: Path, h_img: int = 260, w_img: int = 346) -> dict:
    """Cut-down 5-D K health check matching tools/k_diagnostics/k_health_check.py
    THRESHOLDS but operating on just one (sim, dv) NPZ pair."""
    THRESHOLDS = {
        "count_ratio":           (0.5, 2.0),
        "tv_delta_t":            (0.0, 0.25),
        "spatial_entropy_ratio": (0.8, 1.2),
        "active_pixel_ratio":    (0.8, 1.2),
        "polarity_delta":        (0.0, 0.10),
    }

    def load(p: Path) -> np.ndarray:
        return np.load(p)["events"]

    def count_map(ev: np.ndarray) -> np.ndarray:
        cmap = np.zeros((h_img, w_img), dtype=np.int64)
        if ev.shape[0] == 0:
            return cmap
        ys = ev[:, 2].astype(np.int64); xs = ev[:, 1].astype(np.int64)
        ok = (ys >= 0) & (ys < h_img) & (xs >= 0) & (xs < w_img)
        np.add.at(cmap, (ys[ok], xs[ok]), 1)
        return cmap

    def spatial_entropy(cm: np.ndarray) -> float:
        flat = cm.ravel().astype(np.float64); s = flat.sum()
        if s == 0:
            return 0.0
        p = flat / s; p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def active_pixel_frac(cm: np.ndarray) -> float:
        return float((cm > 0).sum()) / float(cm.size)

    def pos_frac(ev: np.ndarray) -> float:
        if ev.shape[0] == 0:
            return 0.0
        return float((ev[:, 3] > 0).sum()) / float(ev.shape[0])

    def per_pixel_dt_tv(ev: np.ndarray, max_dt_us: int = 100000, num_bins: int = 100) -> np.ndarray:
        # Pool consecutive Δt across pixels
        if ev.shape[0] < 2:
            return np.zeros(num_bins)
        ev = ev[np.argsort(ev[:, 0])]
        # Group by (x, y)
        keys = ev[:, 1].astype(np.int64) * 10000 + ev[:, 2].astype(np.int64)
        order = np.argsort(keys, kind="stable")
        ev_s = ev[order]; keys_s = keys[order]
        dt_all = []
        # group boundaries
        cuts = np.flatnonzero(np.diff(keys_s)) + 1
        groups = np.split(ev_s[:, 0], cuts)
        for grp in groups:
            if grp.size < 6:
                continue
            d = np.diff(grp).astype(np.int64)
            d = d[(d > 0) & (d < max_dt_us)]
            if d.size:
                dt_all.append(d)
        if not dt_all:
            return np.zeros(num_bins)
        dt = np.concatenate(dt_all)
        hist, _ = np.histogram(dt, bins=num_bins, range=(0, max_dt_us))
        s = hist.sum()
        return hist / s if s > 0 else hist.astype(np.float64)

    def tv(p: np.ndarray, q: np.ndarray) -> float:
        return 0.5 * float(np.abs(p - q).sum())

    sim_ev = load(sim_npz); dv_ev = load(dv_npz)
    sim_cm = count_map(sim_ev); dv_cm = count_map(dv_ev)
    metrics = {
        "n_sim": int(sim_ev.shape[0]),
        "n_dv": int(dv_ev.shape[0]),
        "count_ratio": float(sim_ev.shape[0] / max(dv_ev.shape[0], 1)),
        "spatial_entropy_ratio": (spatial_entropy(sim_cm) / spatial_entropy(dv_cm))
            if spatial_entropy(dv_cm) > 0 else float("nan"),
        "active_pixel_ratio": (active_pixel_frac(sim_cm) / active_pixel_frac(dv_cm))
            if active_pixel_frac(dv_cm) > 0 else float("nan"),
        "polarity_delta": abs(pos_frac(sim_ev) - pos_frac(dv_ev)),
        "tv_delta_t": tv(per_pixel_dt_tv(sim_ev), per_pixel_dt_tv(dv_ev)),
    }
    metrics["pass"] = {k: bool(THRESHOLDS[k][0] <= metrics[k] <= THRESHOLDS[k][1]) for k in THRESHOLDS}
    metrics["pass_count"] = int(sum(metrics["pass"].values()))
    metrics["pass_total"] = len(THRESHOLDS)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--dv_npz_dir", required=True,
                    help="Directory containing the matching {prefix}_filtered_dv.npz files")
    ap.add_argument("--calib_prefix_list", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--gpu_id", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--write_back", action="store_true",
                    help="If set, write winning thresholds into v2e_thresholds.json (otherwise dry-run)")
    args = ap.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prefixes = parse_calib(Path(args.calib_prefix_list))
    print(f"[threshold_sweep] {len(prefixes)} calib prefixes; {len(SWEEP_VALUES)} thresholds; "
          f"{len(INPUT_MODALITIES)} modalities -> {len(prefixes)*len(SWEEP_VALUES)*len(INPUT_MODALITIES)} runs")

    cfg = load_thresholds()
    all_results = []
    for modality in INPUT_MODALITIES:
        for thr in SWEEP_VALUES:
            tag = f"{modality}_pos{thr:.2f}_neg{thr:.2f}"
            out_dir = output_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            threshold_dict = dict(cfg["default"])
            threshold_dict["pos_thres"] = thr
            threshold_dict["neg_thres"] = thr
            for prefix in prefixes:
                out_npz = out_dir / f"{prefix}_filtered_{('rawY' if modality == 'rawY' else 'rgb')}.npz"
                t0 = time.time()
                try:
                    if not out_npz.exists() or out_npz.stat().st_size == 0:
                        process_one_prefix_v2e_subprocess(
                            prefix=prefix,
                            input_dir=args.input_dir,
                            output_npz_path=out_npz,
                            input_modality=modality,
                            protocol="native50",
                            threshold_dict=threshold_dict,
                            n_workers=args.num_workers,
                        )
                    dv_npz = Path(args.dv_npz_dir) / f"{prefix}_filtered_dv.npz"
                    if not dv_npz.exists():
                        print(f"WARNING: missing DV NPZ {dv_npz}; skipping K-health for {prefix}", file=sys.stderr)
                        continue
                    health = k_health_one(out_npz, dv_npz)
                    rec = {
                        "modality": modality, "pos_thres": thr, "neg_thres": thr,
                        "prefix": prefix, "elapsed_s": time.time() - t0,
                        **health,
                    }
                    all_results.append(rec)
                    print(f"[{tag}] {prefix} pass={health['pass_count']}/{health['pass_total']} "
                          f"cnt_r={health['count_ratio']:.2f} |Δp|={health['polarity_delta']:.3f}", flush=True)
                except Exception as exc:
                    print(f"[{tag}] {prefix} FAILED: {exc}", file=sys.stderr, flush=True)

    # Aggregate per (modality, threshold) => mean pass_count
    agg = {}
    for r in all_results:
        k = (r["modality"], r["pos_thres"])
        agg.setdefault(k, []).append(r["pass_count"])
    summary = []
    for (modality, thr), passes in sorted(agg.items()):
        summary.append({
            "modality": modality, "pos_thres": thr, "neg_thres": thr,
            "n_prefix_eval": len(passes),
            "mean_pass_count": float(np.mean(passes)),
            "max_pass_count": int(np.max(passes)),
        })
    summary_path = output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"runs": all_results, "agg": summary}, f, indent=2)
    print(f"[threshold_sweep] summary -> {summary_path}")

    # Pick winner per modality (highest mean_pass_count, ties broken by lower threshold)
    winners = {}
    for modality in INPUT_MODALITIES:
        candidates = [s for s in summary if s["modality"] == modality]
        if not candidates:
            continue
        winner = max(candidates, key=lambda s: (s["mean_pass_count"], -s["pos_thres"]))
        winners[modality] = winner
        print(f"[threshold_sweep] WINNER {modality}: pos=neg={winner['pos_thres']:.2f} "
              f"mean_pass={winner['mean_pass_count']:.2f}")

    if args.write_back and winners:
        for modality, winner in winners.items():
            key = f"tuned_{modality}"
            cfg.setdefault(key, {})
            cfg[key]["pos_thres"] = winner["pos_thres"]
            cfg[key]["neg_thres"] = winner["neg_thres"]
            cfg[key]["sigma_thres"] = cfg["default"]["sigma_thres"]
            cfg[key]["cutoff_hz"] = cfg["default"]["cutoff_hz"]
            cfg[key]["leak_rate_hz"] = cfg["default"]["leak_rate_hz"]
            cfg[key]["shot_noise_rate_hz"] = cfg["default"]["shot_noise_rate_hz"]
            cfg[key]["refractory_period"] = cfg["default"].get("refractory_period", 0.0)
            cfg[key]["source"] = (
                f"Stage 2 1-D sweep on {len(parse_calib(Path(args.calib_prefix_list)))} prefixes "
                f"(mean_pass={winner['mean_pass_count']:.2f})"
            )
        with open(THRESHOLDS_JSON, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[threshold_sweep] wrote winners -> {THRESHOLDS_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
