"""k4 polarity probe — refinement pass.

Initial scan (global_k4_polarity_probe.py) showed |Δp| minimum between
α = 4.83e-7 and α = 1e-6 (best so far: α = 1e-6, mean |Δp| = 0.108, misses
0.10 threshold by 0.008). Linear interpolation across 3 samples predicts
the true zero-crossing at α ≈ 7.6e-7. This pass probes 3 α in that band
to locate the empirical sweet spot and verify 5-dim pass count transition.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tools.k_diagnostics.global_k4_polarity_probe import (  # noqa: E402
    ALPHA_STAR_PRED,
    COMPARE_DIR,
    K5,
    LBAR_APPROX,
    OUT_DIR as INITIAL_OUT_DIR,
    ROOT,
    STAGE1_K4_BASELINE,
    _k_for_alpha,
    _metrics_from_events,
)
from tools.k_diagnostics.multi_sample_k_compare import (  # noqa: E402
    SAMPLES,
    TARGET_H,
    TARGET_W,
    _delta_t_histogram,
    _files_for,
    _generate,
    _tv_vs_dv,
)
from generate_event import load_from_video  # noqa: E402
from src.process_data import file_read  # noqa: E402
from train_class.process_single_batch import _raw_bayer_to_y_downsampled  # noqa: E402

REFINE_ALPHAS = [6e-7, 7.6e-7, 9e-7]
OUT_DIR = ROOT / "output" / "k4_probe_20260422_refine"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] backend={backend}  refine α = {REFINE_ALPHAS}")

    overall: dict[str, Any] = {
        "config": "Config A, RAW, refinement around α ≈ 7.6e-7",
        "refine_alphas": REFINE_ALPHAS,
        "samples": {},
    }

    for prefix in SAMPLES:
        print(f"\n{'='*78}\n[sample] {prefix}\n{'='*78}")
        sample_dir = OUT_DIR / prefix
        events_dir = sample_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        files = _files_for(prefix)

        # Reuse DV NPZ from initial compare
        dv_npz = COMPARE_DIR / prefix / "events" / f"{prefix}_dv.npz"
        dv_events = np.load(dv_npz)["events"]
        dv_pos = float((dv_events[:, 3] == 1).sum()) / float(dv_events.shape[0])
        dv_hist = _delta_t_histogram(dv_events)
        print(f"[dv] N={dv_events.shape[0]:,}  pos_fraction={dv_pos:.4f}")

        pi_ts, _ = file_read.read_metadata(str(files["metadata"]))
        raw_np, _ = load_from_video(str(files["raw_mkv"]), quiet=True)
        L = min(len(pi_ts), len(raw_np))
        pi_ts, raw_np = pi_ts[:L], raw_np[:L]
        raw_y_ds = _raw_bayer_to_y_downsampled(raw_np, TARGET_W, TARGET_H)

        sample_record: dict[str, Any] = {
            "prefix": prefix,
            "dv_n_events": int(dv_events.shape[0]),
            "dv_pos_fraction": dv_pos,
        }
        for alpha in REFINE_ALPHAS:
            key = f"{alpha:.3e}"
            k_vec = _k_for_alpha(alpha)
            print(f"\n  [α={key}]")
            t0 = time.time()
            ev = _generate(
                pi_ts, raw_y_ds, is_rgb=False, raw_is_luminance=True,
                k_values=k_vec, backend=backend,
            )
            dt = time.time() - t0
            npz_path = events_dir / f"{prefix}_alpha_{key}.npz"
            np.savez_compressed(npz_path, events=ev)
            hist = _delta_t_histogram(ev)
            tv = _tv_vs_dv(hist["hist_density"], dv_hist["hist_density"], dv_hist["bin_edges_us"])
            metrics = _metrics_from_events(ev, dv_events)
            metrics.update({
                "alpha": alpha,
                "K": k_vec,
                "path": str(npz_path),
                "sim_seconds": round(dt, 2),
                "tv_delta_t": float(tv) if tv is not None else float("nan"),
            })
            sample_record[key] = metrics
            mu = alpha + K5 * LBAR_APPROX
            print(f"    N={metrics['n_events']:,}  ratio={metrics['count_ratio']:.3f}×  "
                  f"pos_frac={metrics['pos_fraction']:.4f}  |Δp|={metrics['polarity_delta']:.4f}  "
                  f"TV={metrics['tv_delta_t']:.3f}  act_rat={metrics['active_pixel_ratio']:.3f}  "
                  f"μ≈{mu:+.3e}  ({dt:.1f}s)")
        overall["samples"][prefix] = sample_record
        (sample_dir / "summary.json").write_text(json.dumps(sample_record, indent=2))

    (OUT_DIR / "results.json").write_text(json.dumps(overall, indent=2))

    # Combined plot: initial 5 α + refine 3 α, per sample + mean
    initial = json.loads((INITIAL_OUT_DIR / "results.json").read_text())
    all_alphas = sorted(set(initial["alphas"]) | set(REFINE_ALPHAS))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    per_sample_pos: dict[str, list[float]] = {s: [] for s in SAMPLES}
    per_sample_pd:  dict[str, list[float]] = {s: [] for s in SAMPLES}
    per_sample_ar:  dict[str, list[float]] = {s: [] for s in SAMPLES}
    per_sample_cr:  dict[str, list[float]] = {s: [] for s in SAMPLES}

    def _lookup(sample: str, alpha: float, field: str) -> float:
        key = f"{alpha:.3e}"
        if key in initial["samples"][sample]:
            return initial["samples"][sample][key][field]
        return overall["samples"][sample][key][field]

    for s in SAMPLES:
        for a in all_alphas:
            per_sample_pos[s].append(_lookup(s, a, "pos_fraction"))
            per_sample_pd[s].append(_lookup(s, a, "polarity_delta"))
            per_sample_ar[s].append(_lookup(s, a, "active_pixel_ratio"))
            per_sample_cr[s].append(_lookup(s, a, "count_ratio"))

    xs = np.asarray(all_alphas)
    # pos_fraction with DV line
    for s in SAMPLES:
        axes[0].plot(xs, per_sample_pos[s], "o-", alpha=0.6, label=s[:16])
    dv_mean = float(np.mean([initial["samples"][s]["dv_pos_fraction"] for s in SAMPLES]))
    axes[0].axhline(dv_mean, color="black", lw=2, label=f"DV mean={dv_mean:.3f}")
    axes[0].axvline(ALPHA_STAR_PRED, color="green", ls=":", label=f"pred α*={ALPHA_STAR_PRED:.2e}")
    axes[0].axvline(STAGE1_K4_BASELINE, color="gray", ls=":", label=f"Stage 1={STAGE1_K4_BASELINE:.2e}")
    axes[0].set_xlabel("α (k4 value)")
    axes[0].set_ylabel("pos_fraction")
    axes[0].set_title("pos_fraction vs α (IVT crossing)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # polarity_delta
    for s in SAMPLES:
        axes[1].plot(xs, per_sample_pd[s], "o-", alpha=0.6, label=s[:16])
    axes[1].plot(xs, np.mean([per_sample_pd[s] for s in SAMPLES], axis=0), "k-", lw=2, label="mean")
    axes[1].axhline(0.10, color="red", ls="--", alpha=0.5, label="threshold 0.10")
    axes[1].set_xlabel("α")
    axes[1].set_ylabel("|Δp|")
    axes[1].set_title("polarity_delta vs α")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # active_ratio & count_ratio (secondary)
    axes[2].plot(xs, np.mean([per_sample_ar[s] for s in SAMPLES], axis=0), "o-", color="C0", label="active_ratio (mean)")
    axes[2].plot(xs, np.mean([per_sample_cr[s] for s in SAMPLES], axis=0), "s-", color="C1", label="count_ratio (mean)")
    axes[2].axhspan(0.8, 1.2, color="C0", alpha=0.1, label="active_rat ok")
    axes[2].axhspan(0.5, 2.0, color="C1", alpha=0.05)
    axes[2].set_xlabel("α")
    axes[2].set_ylabel("ratio")
    axes[2].set_title("active_ratio & count_ratio vs α")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Global k4 probe — combined initial + refinement scan", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "combined_scan.png", dpi=140)
    plt.close(fig)

    # Final summary: mean across all α
    print("\n" + "=" * 78)
    print("COMBINED CROSS-SAMPLE MEAN (initial + refine)")
    print("=" * 78)
    THRESH = {
        "count_ratio":           (0.5, 2.0),
        "tv_delta_t":            (0.0, 0.25),
        "spatial_entropy_ratio": (0.8, 1.2),
        "active_pixel_ratio":    (0.8, 1.2),
        "polarity_delta":        (0.0, 0.10),
    }

    def _pass(k, v): return THRESH[k][0] <= v <= THRESH[k][1]

    print(f"{'α':>12}  {'μ':>11}  {'|Δp|':>6}  {'count':>6}  {'TV':>6}  {'H_rat':>6}  {'act_rat':>7}  pass")
    print("-" * 80)
    rows = []
    for a in all_alphas:
        vals = {}
        for field in ("count_ratio", "tv_delta_t", "spatial_entropy_ratio",
                      "active_pixel_ratio", "polarity_delta"):
            vals[field] = float(np.mean([_lookup(s, a, field) for s in SAMPLES]))
        pc = sum(_pass(k, v) for k, v in vals.items())
        mu = a + K5 * LBAR_APPROX
        rows.append((a, mu, vals, pc))
        m = lambda k: "✓" if _pass(k, vals[k]) else "✗"
        print(
            f"{a:>12.3e}  {mu:>+11.3e}  "
            f"{vals['polarity_delta']:>4.3f}{m('polarity_delta')}  "
            f"{vals['count_ratio']:>4.2f}x{m('count_ratio')}  "
            f"{vals['tv_delta_t']:>4.3f}{m('tv_delta_t')}  "
            f"{vals['spatial_entropy_ratio']:>4.3f}{m('spatial_entropy_ratio')}  "
            f"{vals['active_pixel_ratio']:>5.3f}{m('active_pixel_ratio')}  {pc}/5"
        )

    best_a, best_mu, best_v, best_pc = max(rows, key=lambda r: (r[3], -r[2]["polarity_delta"]))
    print(f"\n[best α by pass count + min |Δp|] α = {best_a:.3e}  →  pass {best_pc}/5  |Δp|={best_v['polarity_delta']:.4f}")
    print(f"\"Stage 1.5 K\" candidate (RAW): {_k_for_alpha(best_a)}")


if __name__ == "__main__":
    main()
