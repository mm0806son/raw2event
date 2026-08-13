"""Global k4 polarity probe — 1D scan over Stage 1 RAW K's k4.

Hypothesis (Ding 2025 IVT): fixing Stage 1 k1/k2/k3/k5/k6 and varying only k4 = α,
there exists a unique α* making polarity_delta ≈ 0. Analytical prediction under
static-background approximation μ ≈ k4 + k5·L̄ with L̄ ≈ 237 and Stage 1
k5 = -2.04e-9 gives α* = -k5·L̄ ≈ +4.83e-7 (vs Stage 1 baseline k4 ≈ +2.08e-6).

This scan verifies the zero location empirically and produces a "Stage 1.5 K"
candidate that should pass 4/5 health check dims (up from Stage 1's 3/5).

Re-uses infrastructure from multi_sample_k_compare.py:
    _files_for, _load_dv_events, _generate, _delta_t_histogram, _tv_vs_dv
so K propagation assert and preprocessing are identical to that experiment.
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
from src.process_data import file_read  # noqa: E402
from train_class.process_single_batch import _raw_bayer_to_y_downsampled  # noqa: E402

from tools.k_diagnostics.multi_sample_k_compare import (  # noqa: E402
    SAMPLES,
    TARGET_H,
    TARGET_W,
    _delta_t_histogram,
    _files_for,
    _generate,
    _load_dv_events,
    _tv_vs_dv,
)

# Stage 1 RAW K (from output/k_estimate_raw/stage1_params.json + DVS346 priors)
STAGE1_K = [
    1.6612259044483269,
    -35.55831455106693,
    1e-4,
    2.079202656350928e-06,  # k4 — overridden per α below
    -2.0353994379776813e-09,
    1e-5,
]
STAGE1_K4_BASELINE = STAGE1_K[3]
K5 = STAGE1_K[4]

# Predicted zero: α* = -k5·L̄ with L̄ ≈ 237 (uniform calibration scene mean Y)
LBAR_APPROX = 237.0
ALPHA_STAR_PRED = -K5 * LBAR_APPROX  # ≈ +4.83e-7

ALPHAS = [
    -5e-7,               # IVT bracketing: μ < 0, expect OFF dominant
    0.0,                 # μ = k5·L̄ ≈ -4.83e-7, still OFF dominant
    ALPHA_STAR_PRED,     # predicted zero
    1e-6,                # μ > 0, mildly ON dominant
    STAGE1_K4_BASELINE,  # Stage 1 baseline anchor (should reproduce 89% ON)
]

OUT_DIR = ROOT / "output" / "k4_probe_20260422"
COMPARE_DIR = ROOT / "output" / "multi_sample_k_compare_20260422"

H_IMG, W_IMG = 260, 346


def _k_for_alpha(alpha: float) -> list[float]:
    k = list(STAGE1_K)
    k[3] = float(alpha)
    return k


def _metrics_from_events(ev: np.ndarray, dv: np.ndarray) -> dict[str, float]:
    """count_ratio + polarity_delta + spatial_entropy_ratio + active_pixel_ratio."""
    def cmap(e: np.ndarray) -> np.ndarray:
        m = np.zeros((H_IMG, W_IMG), dtype=np.int64)
        if e.shape[0] == 0:
            return m
        ys = e[:, 2].astype(np.int64)
        xs = e[:, 1].astype(np.int64)
        ok = (ys >= 0) & (ys < H_IMG) & (xs >= 0) & (xs < W_IMG)
        np.add.at(m, (ys[ok], xs[ok]), 1)
        return m

    def entropy(m: np.ndarray) -> float:
        flat = m.ravel().astype(np.float64)
        s = flat.sum()
        if s == 0:
            return 0.0
        p = flat / s
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def pos_frac(e: np.ndarray) -> float:
        if e.shape[0] == 0:
            return 0.0
        return float((e[:, 3] == 1).sum()) / float(e.shape[0])

    def active_frac(m: np.ndarray) -> float:
        return float((m > 0).sum()) / float(m.size)

    sim_cmap = cmap(ev)
    dv_cmap = cmap(dv)
    sim_h = entropy(sim_cmap)
    dv_h = entropy(dv_cmap)
    sim_active = active_frac(sim_cmap)
    dv_active = active_frac(dv_cmap)
    sim_pos = pos_frac(ev)
    dv_pos = pos_frac(dv)

    return {
        "n_events": int(ev.shape[0]),
        "count_ratio": float(ev.shape[0] / max(dv.shape[0], 1)),
        "pos_fraction": sim_pos,
        "polarity_delta": abs(sim_pos - dv_pos),
        "active_pixel_frac": sim_active,
        "active_pixel_ratio": sim_active / dv_active if dv_active > 0 else float("nan"),
        "spatial_entropy": sim_h,
        "spatial_entropy_ratio": sim_h / dv_h if dv_h > 0 else float("nan"),
    }


def _plot_curves(overall: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    alphas = np.asarray(ALPHAS)

    # Left: polarity_delta vs α, per sample + mean
    per_sample: list[list[float]] = []
    for sample in SAMPLES:
        vals = [overall["samples"][sample][f"{a:.3e}"]["polarity_delta"] for a in ALPHAS]
        per_sample.append(vals)
        axes[0].plot(alphas, vals, "o-", alpha=0.5, label=sample[:20])
    mean = np.mean(per_sample, axis=0)
    axes[0].plot(alphas, mean, "k-", lw=2.2, label="mean")
    axes[0].axhline(0.10, color="red", ls="--", alpha=0.5, label="threshold 0.10")
    axes[0].axvline(ALPHA_STAR_PRED, color="green", ls=":", alpha=0.7,
                    label=f"predicted α*={ALPHA_STAR_PRED:.2e}")
    axes[0].axvline(STAGE1_K4_BASELINE, color="gray", ls=":", alpha=0.5,
                    label=f"Stage 1 baseline={STAGE1_K4_BASELINE:.2e}")
    axes[0].set_xlabel("α (k4 value)")
    axes[0].set_ylabel("|pos_frac_sim − pos_frac_dv|")
    axes[0].set_title("polarity_delta vs α")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)

    # Right: pos_fraction vs α (shows IVT crossing)
    for sample in SAMPLES:
        vals = [overall["samples"][sample][f"{a:.3e}"]["pos_fraction"] for a in ALPHAS]
        axes[1].plot(alphas, vals, "o-", alpha=0.5, label=f"{sample[:20]} (sim)")
    dv_pos_mean = float(np.mean([overall["samples"][s]["dv_pos_fraction"] for s in SAMPLES]))
    axes[1].axhline(dv_pos_mean, color="black", ls="-", lw=2,
                    label=f"DV mean pos_frac={dv_pos_mean:.3f}")
    axes[1].axvline(ALPHA_STAR_PRED, color="green", ls=":", alpha=0.7)
    axes[1].set_xlabel("α (k4 value)")
    axes[1].set_ylabel("pos_fraction (ON / total)")
    axes[1].set_title("pos_fraction vs α — IVT crossing")
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Global k4 polarity probe — Stage 1 RAW K, k4 = α, "
        f"predicted α* = -k5·L̄ = {ALPHA_STAR_PRED:.3e}", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] backend={backend}")
    print(f"[setup] α values = {ALPHAS}")
    print(f"[setup] predicted α* = {ALPHA_STAR_PRED:.3e} (from -k5·L̄, L̄={LBAR_APPROX})")
    print(f"[setup] stage1 baseline k4 = {STAGE1_K4_BASELINE:.3e}")

    overall: dict[str, Any] = {
        "config": "Config A (346x260, Bayer→Y→resize for RAW)",
        "stage1_k_base": STAGE1_K,
        "alphas": ALPHAS,
        "predicted_alpha_star": ALPHA_STAR_PRED,
        "lbar_approx": LBAR_APPROX,
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

        # DV — reuse existing NPZ from multi_sample_k_compare if present (avoids re-AEDAT)
        dv_npz_existing = COMPARE_DIR / prefix / "events" / f"{prefix}_dv.npz"
        if dv_npz_existing.exists():
            print(f"[dv] reuse {dv_npz_existing}")
            dv_events = np.load(dv_npz_existing)["events"]
        else:
            print(f"[dv] loading {files['dv'].name}")
            dv_events = _load_dv_events(files["dv"])

        dv_pos = float((dv_events[:, 3] == 1).sum()) / float(dv_events.shape[0])
        print(f"[dv] N={dv_events.shape[0]:,}  pos_fraction={dv_pos:.4f}")

        # Frames (RAW only — RGB out of scope for this probe)
        print("[frames] loading metadata + RAW mkv")
        pi_ts, _ = file_read.read_metadata(str(files["metadata"]))
        raw_np, _ = load_from_video(str(files["raw_mkv"]), quiet=True)
        L = min(len(pi_ts), len(raw_np))
        pi_ts, raw_np = pi_ts[:L], raw_np[:L]
        raw_y_ds = _raw_bayer_to_y_downsampled(raw_np, TARGET_W, TARGET_H)
        print(f"[frames] trimmed L={L}, raw_y_ds={raw_y_ds.shape}")

        sample_record: dict[str, Any] = {
            "prefix": prefix,
            "dv_n_events": int(dv_events.shape[0]),
            "dv_pos_fraction": dv_pos,
        }

        for alpha in ALPHAS:
            key = f"{alpha:.3e}"
            k_vec = _k_for_alpha(alpha)
            print(f"\n  [α={key}] K = {k_vec}")
            t0 = time.time()
            ev = _generate(
                pi_ts, raw_y_ds, is_rgb=False, raw_is_luminance=True,
                k_values=k_vec, backend=backend,
            )
            dt = time.time() - t0

            npz_path = events_dir / f"{prefix}_alpha_{key}.npz"
            np.savez_compressed(npz_path, events=ev)

            hist = _delta_t_histogram(ev)
            dv_hist = overall.get("_dv_hist_cache", {}).get(prefix)
            if dv_hist is None:
                dv_hist = _delta_t_histogram(dv_events)
                overall.setdefault("_dv_hist_cache", {})[prefix] = dv_hist

            tv = _tv_vs_dv(hist["hist_density"], dv_hist["hist_density"], dv_hist["bin_edges_us"])

            metrics = _metrics_from_events(ev, dv_events)
            metrics["alpha"] = alpha
            metrics["K"] = k_vec
            metrics["path"] = str(npz_path)
            metrics["sim_seconds"] = round(dt, 2)
            metrics["tv_delta_t"] = float(tv) if tv is not None else float("nan")
            sample_record[key] = metrics

            mu_approx = alpha + K5 * LBAR_APPROX
            print(f"    N={metrics['n_events']:,}  ratio={metrics['count_ratio']:.3f}×  "
                  f"pos_frac={metrics['pos_fraction']:.4f}  |Δp|={metrics['polarity_delta']:.4f}  "
                  f"TV(Δt)={metrics['tv_delta_t']:.3f}  "
                  f"μ_static≈{mu_approx:+.3e}  ({dt:.1f}s)")

        overall["samples"][prefix] = sample_record
        (sample_dir / "summary.json").write_text(
            json.dumps({k: v for k, v in sample_record.items() if not k.startswith("_")}, indent=2)
        )

    # Strip internal caches before writing results.json
    overall.pop("_dv_hist_cache", None)
    (OUT_DIR / "results.json").write_text(json.dumps(overall, indent=2))

    # Plot
    plot_path = OUT_DIR / "polarity_vs_alpha.png"
    _plot_curves(overall, plot_path)
    print(f"\n[done] results → {OUT_DIR / 'results.json'}")
    print(f"[done] plot    → {plot_path}")

    # Summary table + pick best α
    print("\n" + "=" * 78)
    print("CROSS-SAMPLE MEAN (3 samples)")
    print("=" * 78)
    hdr = f"{'α':>12}  {'μ_static':>11}  {'|Δp|_mean':>10}  {'ratio_mean':>10}  {'TV(Δt)_mean':>11}  {'act_rat_mean':>12}"
    print(hdr)
    print("-" * len(hdr))

    best_alpha = None
    best_polarity = float("inf")
    for alpha in ALPHAS:
        key = f"{alpha:.3e}"
        pd = np.mean([overall["samples"][s][key]["polarity_delta"] for s in SAMPLES])
        cr = np.mean([overall["samples"][s][key]["count_ratio"] for s in SAMPLES])
        tv = np.mean([overall["samples"][s][key]["tv_delta_t"] for s in SAMPLES])
        ar = np.mean([overall["samples"][s][key]["active_pixel_ratio"] for s in SAMPLES])
        mu_static = alpha + K5 * LBAR_APPROX
        print(f"{alpha:>12.3e}  {mu_static:>+11.3e}  {pd:>10.4f}  {cr:>9.3f}x  {tv:>11.4f}  {ar:>12.3f}")
        if pd < best_polarity:
            best_polarity = pd
            best_alpha = alpha

    print(f"\n[best α by min polarity_delta] α = {best_alpha:.3e}  →  mean |Δp| = {best_polarity:.4f}")
    if best_polarity <= 0.10:
        print(f"[verdict] ✓ PASS polarity_delta threshold at α = {best_alpha:.3e}")
        print(f"          → \"Stage 1.5 K\" candidate: {_k_for_alpha(best_alpha)}")
    else:
        print(f"[verdict] ✗ FAIL polarity_delta threshold (best |Δp|={best_polarity:.4f} > 0.10)")
        print("          → static-background approximation may be too coarse; consider 2D (k4, k5) scan")


if __name__ == "__main__":
    main()
