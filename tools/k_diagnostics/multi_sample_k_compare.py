"""Stage 1 vs Stage 2 K, across 3 samples — count + Δt distribution comparison.

For each of 3 cifar10-xdvs samples, generate RAW-sim and RGB-sim events under
two K sets (Stage 1 physical regression and current Stage 2 + P-A K from
``K_MAP``) in the production config_A path (346x260, Bayer→Y→resize for RAW,
INTER_AREA for RGB), and render overlay Δt histograms against real DV.

K-propagation safety: after each ``generate_events_tensor`` call we assert that
``cfg.SENSOR.K`` equals the K we passed, which is the same tensor the just-built
``EventSim`` copied into ``self.k1..k6`` (see ``src/simulator.py:39``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_event import load_from_video  # noqa: E402
from src.config import cfg, K_MAP  # noqa: E402
from src.process_data import dvs_generate, file_read, interval_fit  # noqa: E402
from train_class.process_single_batch import (  # noqa: E402
    _downsample_frames,
    _raw_bayer_to_y_downsampled,
)

DATASET = Path("./data")
SAMPLES = [
    "42868_horse_1_8698_20251228_234242",
    "10000_automobile_5_1087_20251224_105416",
    "12001_bird_1_1003_20251224_172825",
]

STAGE1_RAW_K = [
    1.6612259044483269,
    -35.55831455106693,
    1e-4,
    2.079202656350928e-06,
    -2.0353994379776813e-09,
    1e-5,
]
STAGE1_RGB_K = [
    7.612017342884227,
    204.46017142991028,
    1e-4,
    7.354632959846225e-06,
    -2.5565783709920732e-08,
    1e-5,
]
K_SETS = {
    "stage1": {"raw": STAGE1_RAW_K, "rgb": STAGE1_RGB_K},
    "stage2_pa": {"raw": list(K_MAP["Raw2DVS346"]), "rgb": list(K_MAP["RGB2DVS346"])},
}

OUT_DIR = ROOT / "output" / "multi_sample_k_compare_20260422"
TARGET_W, TARGET_H = 346, 260
BAD_PIXEL = (108, 108)

INTERVAL_MIN_EVENTS_PER_PIXEL = 5
INTERVAL_MAX_DT_US = 100_000
INTERVAL_BINS = 100


def _files_for(prefix: str) -> dict[str, Path]:
    return {
        "dv": DATASET / f"dv_output_{prefix}.aedat4",
        "raw_mkv": DATASET / f"raw_frames_{prefix}_raw_10bit.mkv",
        "rgb_mkv": DATASET / f"rgb_frames_{prefix}_rgb.mkv",
        "metadata": DATASET / f"metadata_{prefix}.dat",
    }


def _load_dv_events(aedat_path: Path) -> np.ndarray:
    events = file_read.load_events(str(aedat_path))
    if events is None:
        raise RuntimeError(f"No events in {aedat_path}")
    events = events.numpy()
    mask = ~((events[:, 1] == BAD_PIXEL[0]) & (events[:, 2] == BAD_PIXEL[1]))
    return events[mask]


def _assert_k_propagated(expected: list[float]) -> None:
    actual = [float(x) for x in cfg.SENSOR.K]
    if len(actual) != 6 or any(a != b for a, b in zip(actual, expected)):
        raise RuntimeError(
            f"K propagation failure:\n  expected={expected}\n  cfg.SENSOR.K={actual}"
        )


def _generate(pi_ts_tensor, frames_np, *, is_rgb: bool, raw_is_luminance: bool,
              k_values: list[float], backend: str) -> np.ndarray:
    if is_rgb:
        frames_tensor = torch.from_numpy(frames_np)
    else:
        frames_tensor = torch.from_numpy(frames_np.astype(np.int32, copy=False))
    print(f"  [pre ] cfg.SENSOR.K = {list(cfg.SENSOR.K)}")
    events = dvs_generate.generate_events_tensor(
        pi_ts_tensor,
        frames_tensor,
        is_rgb=is_rgb,
        raw_is_luminance=raw_is_luminance,
        k_values=k_values,
        sim_backend=backend,
    )
    print(f"  [post] cfg.SENSOR.K = {list(cfg.SENSOR.K)}")
    _assert_k_propagated(k_values)
    return events.cpu().numpy()


def _delta_t_histogram(events: np.ndarray) -> dict[str, Any]:
    intervals = interval_fit.compute_event_intervals(
        events,
        mode="per-pixel",
        min_events_per_pixel=INTERVAL_MIN_EVENTS_PER_PIXEL,
    )
    clipped = intervals[(intervals > 0) & (intervals <= INTERVAL_MAX_DT_US)]
    if clipped.size == 0:
        return {
            "interval_count": 0,
            "bin_edges_us": None,
            "hist_density": None,
            "mean_us": None,
            "median_us": None,
        }
    hist, edges = np.histogram(
        clipped, bins=INTERVAL_BINS, range=(0, INTERVAL_MAX_DT_US), density=True
    )
    return {
        "interval_count": int(clipped.size),
        "bin_edges_us": edges.tolist(),
        "hist_density": hist.tolist(),
        "mean_us": float(clipped.mean()),
        "median_us": float(np.median(clipped)),
    }


def _tv_vs_dv(dens: list[float] | None, dv_dens: list[float] | None,
              edges: list[float] | None) -> float | None:
    if dens is None or dv_dens is None or edges is None:
        return None
    a = np.asarray(dens); b = np.asarray(dv_dens)
    w = np.diff(np.asarray(edges))
    pa = a * w; pb = b * w
    pa = pa / pa.sum() if pa.sum() > 0 else pa
    pb = pb / pb.sum() if pb.sum() > 0 else pb
    return float(0.5 * np.abs(pa - pb).sum())


def _plot_sample_overlay(sample_result: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    curves = [
        ("DV",         sample_result["dv"],                "#111111", "-"),
        ("RAW stage1", sample_result["stage1"]["raw"],     "#1f77b4", "-"),
        ("RGB stage1", sample_result["stage1"]["rgb"],     "#d62728", "-"),
        ("RAW stage2", sample_result["stage2_pa"]["raw"],  "#1f77b4", "--"),
        ("RGB stage2", sample_result["stage2_pa"]["rgb"],  "#d62728", "--"),
    ]
    for label, node, color, ls in curves:
        h = node.get("histogram") or node
        edges = h.get("bin_edges_us")
        dens = h.get("hist_density")
        if edges is None or dens is None:
            continue
        centers = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:])) / 1000.0
        n = node.get("n_events", node.get("interval_count"))
        tv = node.get("tv_vs_dv")
        tag = f"{label}  N={n:,}"
        if tv is not None:
            tag += f"  TV={tv:.3f}"
        axes[0].plot(centers, dens, color=color, linestyle=ls, lw=1.4, label=tag)
        axes[1].semilogy(centers, dens, color=color, linestyle=ls, lw=1.4, label=tag)
    for ax in axes:
        ax.set_xlabel("Δt (ms)")
        ax.set_ylabel("per-pixel density")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_title(f"{sample_result['prefix']} — linear")
    axes[1].set_title(f"{sample_result['prefix']} — log-y")
    axes[1].set_ylim(bottom=1e-8)
    fig.suptitle("Δt histogram, per-pixel, min_events=5, max_dt=100ms — Stage1 (solid) vs Stage2+P-A (dashed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] backend={backend}  K_SETS={list(K_SETS)}")

    overall: dict[str, Any] = {
        "dataset_dir": str(DATASET),
        "config": "A (346x260, Bayer→Y→resize for RAW, INTER_AREA for RGB)",
        "bad_pixel": list(BAD_PIXEL),
        "interval": {
            "mode": "per-pixel",
            "min_events_per_pixel": INTERVAL_MIN_EVENTS_PER_PIXEL,
            "max_dt_us": INTERVAL_MAX_DT_US,
            "bins": INTERVAL_BINS,
        },
        "k_sets": K_SETS,
        "backend": backend,
        "samples": {},
    }

    for prefix in SAMPLES:
        print("\n" + "=" * 78)
        print(f"[sample] {prefix}")
        print("=" * 78)
        sample_dir = OUT_DIR / prefix
        events_dir = sample_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        files = _files_for(prefix)
        for key, fp in files.items():
            if not fp.exists():
                raise FileNotFoundError(f"Missing file for {prefix}: {key}={fp}")

        # DV
        print(f"[dv] loading {files['dv'].name}")
        dv_events = _load_dv_events(files["dv"])
        dv_npz = events_dir / f"{prefix}_dv.npz"
        np.savez_compressed(dv_npz, events=dv_events)
        dv_hist = _delta_t_histogram(dv_events)
        dv_node = {
            "path": str(dv_npz),
            "n_events": int(dv_events.shape[0]),
            "histogram": dv_hist,
            "tv_vs_dv": 0.0,
        }

        # Frames
        print("[frames] loading metadata + RAW/RGB mkv")
        pi_ts, _ = file_read.read_metadata(str(files["metadata"]))
        raw_np, _ = load_from_video(str(files["raw_mkv"]), quiet=True)
        rgb_np, _ = load_from_video(str(files["rgb_mkv"]), quiet=True)
        L = min(len(pi_ts), len(raw_np), len(rgb_np))
        pi_ts, raw_np, rgb_np = pi_ts[:L], raw_np[:L], rgb_np[:L]
        print(f"[frames] trimmed to {L}; raw={raw_np.shape} rgb={rgb_np.shape}")

        raw_y_ds = _raw_bayer_to_y_downsampled(raw_np, TARGET_W, TARGET_H)
        rgb_ds = _downsample_frames(rgb_np, TARGET_W, TARGET_H)

        sample_record: dict[str, Any] = {"prefix": prefix, "dv": dv_node}

        for tag, ks in K_SETS.items():
            print(f"\n  [K set: {tag}] raw={ks['raw']}")
            print(f"                rgb={ks['rgb']}")
            node: dict[str, Any] = {}

            # RAW
            t0 = time.time()
            raw_ev = _generate(
                pi_ts, raw_y_ds, is_rgb=False, raw_is_luminance=True,
                k_values=ks["raw"], backend=backend,
            )
            dt_raw = time.time() - t0
            raw_npz = events_dir / f"{prefix}_{tag}_raw.npz"
            np.savez_compressed(raw_npz, events=raw_ev)
            raw_hist = _delta_t_histogram(raw_ev)
            node["raw"] = {
                "path": str(raw_npz),
                "K": list(ks["raw"]),
                "n_events": int(raw_ev.shape[0]),
                "count_ratio_vs_dv": float(raw_ev.shape[0] / dv_events.shape[0]),
                "sim_seconds": round(dt_raw, 2),
                "histogram": raw_hist,
                "tv_vs_dv": _tv_vs_dv(
                    raw_hist["hist_density"], dv_hist["hist_density"], dv_hist["bin_edges_us"]
                ),
            }
            print(f"    RAW N={node['raw']['n_events']:,}  ratio={node['raw']['count_ratio_vs_dv']:.3f}x  TV={node['raw']['tv_vs_dv']:.3f}  ({dt_raw:.1f}s)")

            # RGB
            t0 = time.time()
            rgb_ev = _generate(
                pi_ts, rgb_ds, is_rgb=True, raw_is_luminance=False,
                k_values=ks["rgb"], backend=backend,
            )
            dt_rgb = time.time() - t0
            rgb_npz = events_dir / f"{prefix}_{tag}_rgb.npz"
            np.savez_compressed(rgb_npz, events=rgb_ev)
            rgb_hist = _delta_t_histogram(rgb_ev)
            node["rgb"] = {
                "path": str(rgb_npz),
                "K": list(ks["rgb"]),
                "n_events": int(rgb_ev.shape[0]),
                "count_ratio_vs_dv": float(rgb_ev.shape[0] / dv_events.shape[0]),
                "sim_seconds": round(dt_rgb, 2),
                "histogram": rgb_hist,
                "tv_vs_dv": _tv_vs_dv(
                    rgb_hist["hist_density"], dv_hist["hist_density"], dv_hist["bin_edges_us"]
                ),
            }
            print(f"    RGB N={node['rgb']['n_events']:,}  ratio={node['rgb']['count_ratio_vs_dv']:.3f}x  TV={node['rgb']['tv_vs_dv']:.3f}  ({dt_rgb:.1f}s)")

            sample_record[tag] = node

        overlay_path = sample_dir / "overlay.png"
        _plot_sample_overlay(sample_record, overlay_path)
        print(f"  → overlay: {overlay_path}")
        (sample_dir / "summary.json").write_text(json.dumps(sample_record, indent=2))
        overall["samples"][prefix] = sample_record

    (OUT_DIR / "results.json").write_text(json.dumps(overall, indent=2))
    print(f"\n[done] results → {OUT_DIR}")

    # Summary table to stdout
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    hdr = ("sample", "K set", "mod", "N", "ratio", "TV(vs DV)")
    print("{:40} {:>10} {:>4} {:>10} {:>8} {:>10}".format(*hdr))
    for prefix, rec in overall["samples"].items():
        n_dv = rec["dv"]["n_events"]
        print("{:40} {:>10} {:>4} {:>10,} {:>8} {:>10}".format(
            prefix[:40], "DV", "-", n_dv, "1.000x", "0.000"))
        for tag in ("stage1", "stage2_pa"):
            for mod in ("raw", "rgb"):
                n = rec[tag][mod]
                print("{:40} {:>10} {:>4} {:>10,} {:>7.3f}x {:>10.3f}".format(
                    prefix[:40], tag, mod.upper(),
                    n["n_events"], n["count_ratio_vs_dv"], n["tv_vs_dv"]))
        print()


if __name__ == "__main__":
    main()
