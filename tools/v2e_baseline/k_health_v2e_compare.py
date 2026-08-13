"""Aggregate the upstream distance metrics across simulator variants.

Reads a variant-to-NPZ-directory config and reports, per variant and recording:
count ratio, per-pixel interval-histogram total variation, spatial entropy ratio,
active-pixel ratio, polarity deviation, Sinkhorn interval EMD, per-pixel count
EMD, and optionally a Gromov-Wasserstein cost over the (x, y, t) geometry.

Aggregation is nan-safe mean and median over the clean-recording subset. The GWD
column drops recordings below ``min_points``, so its denominator can be smaller
than the others'. Writes JSON plus a Markdown rendering of the aggregate table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

# ``k_health_one`` is imported lazily inside ``main()``: it pulls in cv2 via
# v2e_helpers, which the lightweight metric functions below (and their unit
# tests) do not need.

# Optional Sinkhorn (POT package) - skip if unavailable
try:
    import ot  # noqa: F401
    HAVE_POT = True
except ImportError:
    HAVE_POT = False


def per_pixel_count_emd(sim_ev: np.ndarray, dv_ev: np.ndarray, h: int = 260, w: int = 346) -> float:
    """1-Wasserstein on per-pixel count distributions (sorted-CDF method)."""
    def cmap(ev):
        c = np.zeros((h, w), dtype=np.int64)
        if ev.shape[0] == 0:
            return c
        ys = ev[:, 2].astype(np.int64); xs = ev[:, 1].astype(np.int64)
        ok = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
        np.add.at(c, (ys[ok], xs[ok]), 1)
        return c
    a = cmap(sim_ev).ravel().astype(np.float64); b = cmap(dv_ev).ravel().astype(np.float64)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    a /= a.sum(); b /= b.sum()
    # 1-Wasserstein on integer indices: use sorted CDF identity
    return float(np.abs(np.cumsum(a) - np.cumsum(b)).mean())


def sinkhorn_dt_emd(sim_ev: np.ndarray, dv_ev: np.ndarray, n_bins: int = 100, max_dt_us: int = 100000,
                    reg: float = 1e-2) -> float:
    """Sinkhorn EMD on per-pixel Δt histograms (best-effort; returns NaN if POT missing)."""
    if not HAVE_POT:
        return float("nan")
    # Reuse local Δt extractor inline
    def per_pixel_dt(ev):
        if ev.shape[0] < 2:
            return np.zeros(n_bins)
        ev = ev[np.argsort(ev[:, 0])]
        keys = ev[:, 1].astype(np.int64) * 10000 + ev[:, 2].astype(np.int64)
        order = np.argsort(keys, kind="stable")
        ev_s = ev[order]; keys_s = keys[order]
        dt_all = []
        cuts = np.flatnonzero(np.diff(keys_s)) + 1
        for grp in np.split(ev_s[:, 0], cuts):
            if grp.size < 6:
                continue
            d = np.diff(grp).astype(np.int64)
            d = d[(d > 0) & (d < max_dt_us)]
            if d.size:
                dt_all.append(d)
        if not dt_all:
            return np.zeros(n_bins)
        dt = np.concatenate(dt_all)
        h, _ = np.histogram(dt, bins=n_bins, range=(0, max_dt_us))
        s = h.sum()
        return (h / s) if s > 0 else h.astype(np.float64)
    a = per_pixel_dt(sim_ev); b = per_pixel_dt(dv_ev)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    M = np.abs(np.arange(n_bins)[:, None] - np.arange(n_bins)[None, :]).astype(np.float64)
    M /= M.max() if M.max() > 0 else 1.0
    import ot
    return float(ot.sinkhorn2(a, b, M, reg))


def _spatiotemporal_cloud(ev: np.ndarray, n_sample: int, seed: int) -> np.ndarray | None:
    """Extract a per-axis z-scored (x, y, t) point cloud, subsampled to n_sample.

    Returns None when there are too few finite events for a meaningful structure.
    The cloud is canonically ordered (lexicographic by x, y, t) before
    subsampling so the metric is a pure function of the point *set* — it does
    not depend on the row order in which events were serialised — and uses a
    fixed seed so it is reproducible across runs.
    """
    if ev.shape[0] == 0:
        return None
    # Columns are [t, x, y, p]; GWD operates on the spatiotemporal geometry.
    feats = ev[:, [1, 2, 0]].astype(np.float64)
    # Drop non-finite rows rather than letting NaN/inf propagate into the GW
    # solver (fail-soft per-prefix: a degenerate prefix yields NaN, not a crash).
    feats = feats[np.isfinite(feats).all(axis=1)]
    if feats.shape[0] == 0:
        return None
    # Canonical order → subsampling (and hence GWD) is shuffle-invariant.
    order = np.lexsort((feats[:, 2], feats[:, 1], feats[:, 0]))
    feats = feats[order]
    n = feats.shape[0]
    if n > n_sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=n_sample, replace=False))
        feats = feats[idx]
    # Per-axis z-score so the microsecond t-axis does not dominate the
    # intra-cloud distance structure (GWD is not invariant to per-axis scale).
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    return (feats - mean) / std


def _normalised_distance_matrix(cloud: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances within a cloud, normalised by the max."""
    diff = cloud[:, None, :] - cloud[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    dmax = dist.max()
    if dmax > 0:
        dist /= dmax
    return dist


def gwd_one(sim_ev: np.ndarray, dv_ev: np.ndarray, n_sample: int = 512,
            seed: int = 0, min_points: int = 8, epsilon: float | None = None) -> float:
    """Gromov-Wasserstein cost between sim and real event point clouds.

    Measures the joint spatiotemporal structural fidelity of the simulated
    events relative to the real DAVIS346 reference. Per-axis z-score and
    max-normalised intra-cloud distances make it invariant to per-axis and
    global scale; uniform masses make it invariant to the *total event count*
    (that absolute-density axis is already covered by ``count_ratio``). Note
    this does NOT remove local density-distribution differences. Lower is
    closer to real.

    Returns the POT ``square_loss`` Gromov-Wasserstein cost (the
    ``<L(C1,C2) \\otimes T, T>`` term), not the ``0.5*sqrt(.)`` GW distance;
    ordering between variants is unaffected by that convention.

    ``epsilon=None`` uses exact GW (deterministic, exactly 0 for identical
    clouds); a positive ``epsilon`` switches to the faster entropic solver.
    Returns NaN when POT is missing or either cloud has < ``min_points`` finite
    events (matching the NaN convention of the EMD metrics).
    """
    if n_sample < min_points:
        raise ValueError(f"n_sample ({n_sample}) must be >= min_points ({min_points})")
    if epsilon is not None and epsilon <= 0:
        raise ValueError(f"epsilon must be None or > 0, got {epsilon}")
    if not HAVE_POT:
        return float("nan")
    if sim_ev.shape[0] < min_points or dv_ev.shape[0] < min_points:
        return float("nan")
    c_sim = _spatiotemporal_cloud(sim_ev, n_sample, seed)
    c_dv = _spatiotemporal_cloud(dv_ev, n_sample, seed)
    if c_sim is None or c_dv is None or c_sim.shape[0] < min_points or c_dv.shape[0] < min_points:
        return float("nan")
    cmat_sim = _normalised_distance_matrix(c_sim)
    cmat_dv = _normalised_distance_matrix(c_dv)
    p = np.full(c_sim.shape[0], 1.0 / c_sim.shape[0])
    q = np.full(c_dv.shape[0], 1.0 / c_dv.shape[0])
    import ot
    if epsilon is None:
        return float(ot.gromov.gromov_wasserstein2(cmat_sim, cmat_dv, p, q, loss_fun="square_loss"))
    return float(ot.gromov.entropic_gromov_wasserstein2(
        cmat_sim, cmat_dv, p, q, loss_fun="square_loss", epsilon=epsilon))


def render_md_table(summary: list[dict]) -> str:
    """Render the simulator-fidelity Markdown table from a list of per-variant
    aggregate dicts (as produced in ``main``). Shared by the standalone tool
    and the fan-out merge step so the two cannot drift.
    """
    lines = [
        "# Simulator fidelity table (5-D K health + EMD + GWD distances)",
        "",
        "Per-variant aggregates over the canonical test set, with "
        "n_dv == 0 prefixes excluded (DV capture failures, not simulator "
        "properties).  See JSON `raw` block for un-aggregated per-prefix "
        "metrics.  `cnt_med` is more robust than `cnt_mean` against the "
        "long-tailed event-density distribution.  `GWD med` is the median "
        "Gromov-Wasserstein square-loss cost of the (x, y, t) event geometry "
        "(structure-only, total-count-invariant; lower is closer to real).",
        "",
        "| Variant | Label | n_used | skip | cnt_med | cnt_mean | TV(Δt) | H_rat | act_rat | \\|Δp\\| | pass | px-EMD med | Δt-EMD med | GWD med |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        if s.get("n_prefix_used", 0) == 0:
            lines.append(
                f"| {s['variant']} | {s['label']} | 0 | {s.get('n_skipped', 0)} | "
                "— | — | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {s['variant']} | {s['label']} | {s['n_prefix_used']} | {s['n_skipped']} | "
            f"{s['count_ratio_median']:.2f} | {s['count_ratio_mean']:.2f} | "
            f"{s['tv_delta_t_mean']:.2f} | "
            f"{s['spatial_entropy_ratio_mean']:.2f} | {s['active_pixel_ratio_mean']:.2f} | "
            f"{s['polarity_delta_mean']:.3f} | {s['pass_count_mean']:.1f}/5 | "
            f"{s['per_pixel_count_emd_median']:.4f} | {s['sinkhorn_dt_emd_median']:.4f} | "
            f"{s['gwd_median']:.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="JSON mapping variant_id -> {npz_dir, npz_suffix, label}")
    ap.add_argument("--dv_npz_dir", required=True)
    ap.add_argument("--prefix_list", required=True,
                    help="Canonical test prefix list (600 prefixes)")
    ap.add_argument("--output", required=True,
                    help="Output JSON path; .md sibling will also be written")
    ap.add_argument("--sinkhorn_reg", type=float, default=1e-2)
    ap.add_argument("--gwd_n_sample", type=int, default=512,
                    help="Points subsampled per cloud for GWD (lower = faster).")
    ap.add_argument("--gwd_epsilon", type=float, default=None,
                    help="If set (>0), use entropic GW with this regularisation "
                         "instead of exact GW (faster, approximate).")
    args = ap.parse_args()

    # Imported here (not at module top) to keep the cv2 dependency out of the
    # import path for the standalone metric helpers and their unit tests.
    from tools.v2e_baseline.threshold_sweep import k_health_one

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.prefix_list) as f:
        prefixes = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    print(f"[k_health_v2e_compare] {len(cfg)} variants × {len(prefixes)} prefixes")

    dv_dir = Path(args.dv_npz_dir)
    results = {}
    for variant, spec in cfg.items():
        sim_dir = Path(spec["npz_dir"])
        npz_suffix = spec.get("npz_suffix", "rgb")
        label = spec.get("label", variant)
        rows = []
        for pf in prefixes:
            sim = sim_dir / f"{pf}_filtered_{npz_suffix}.npz"
            dv = dv_dir / f"{pf}_filtered_dv.npz"
            if not sim.exists() or not dv.exists():
                continue
            sim_ev = np.load(sim)["events"]
            dv_ev = np.load(dv)["events"]
            health = k_health_one(sim, dv)
            health["per_pixel_count_emd"] = per_pixel_count_emd(sim_ev, dv_ev)
            health["sinkhorn_dt_emd"] = sinkhorn_dt_emd(sim_ev, dv_ev, reg=args.sinkhorn_reg)
            health["gwd"] = gwd_one(sim_ev, dv_ev, n_sample=args.gwd_n_sample,
                                    epsilon=args.gwd_epsilon)
            rows.append({"prefix": pf, **health})
        results[variant] = {"label": label, "n_prefix": len(rows), "rows": rows}
        print(f"[{variant}] {label}: n={len(rows)}")

    # Aggregate
    # Skip prefixes whose ground-truth DV NPZ has 0 events: they are degenerate
    # (DV capture failure, not a property of the simulator), and they corrupt
    # the aggregate via:
    #   - count_ratio = sim_count / max(0,1) = sim_count  (e.g. 190100x outlier)
    #   - spatial_entropy_ratio = sim_H / 0 → NaN
    #   - active_pixel_ratio = sim_act / 0 → NaN
    #   - tv_delta_t = tv(p_sim, zeros) ≈ 0.5  (artificially elevated)
    # Symmetrically also skip n_sim == 0. Counts are reported in `n_skipped`.
    # Per-prefix data is preserved untouched in the JSON `raw` block for full
    # transparency / reanalysis.
    summary = []
    for variant, blk in results.items():
        rows = blk["rows"]
        if not rows:
            continue
        clean = [r for r in rows if r["n_dv"] > 0 and r["n_sim"] > 0]
        n_skipped = len(rows) - len(clean)
        if not clean:
            agg = {
                "variant": variant, "label": blk["label"],
                "n_prefix_total": blk["n_prefix"], "n_prefix_used": 0, "n_skipped": n_skipped,
            }
            summary.append(agg)
            continue
        cnt = np.array([r["count_ratio"]            for r in clean], dtype=float)
        tvd = np.array([r["tv_delta_t"]             for r in clean], dtype=float)
        her = np.array([r["spatial_entropy_ratio"]  for r in clean], dtype=float)
        apr = np.array([r["active_pixel_ratio"]     for r in clean], dtype=float)
        pol = np.array([r["polarity_delta"]         for r in clean], dtype=float)
        psc = np.array([r["pass_count"]             for r in clean], dtype=float)
        pem = np.array([r["per_pixel_count_emd"]    for r in clean], dtype=float)
        sem = np.array([r["sinkhorn_dt_emd"]        for r in clean], dtype=float)
        gwd = np.array([r["gwd"]                    for r in clean], dtype=float)
        agg = {
            "variant": variant, "label": blk["label"],
            "n_prefix_total": blk["n_prefix"], "n_prefix_used": len(clean),
            "n_skipped": n_skipped,
            # GWD additionally drops < min_points-event prefixes (NaN), so its
            # effective denominator can be smaller than n_prefix_used.
            "n_gwd_used": int(np.isfinite(gwd).sum()),
            # Mean (nan-safe for entropy/active which can still NaN if sim_H==0)
            "count_ratio_mean":           float(np.nanmean(cnt)),
            "tv_delta_t_mean":            float(np.nanmean(tvd)),
            "spatial_entropy_ratio_mean": float(np.nanmean(her)),
            "active_pixel_ratio_mean":    float(np.nanmean(apr)),
            "polarity_delta_mean":        float(np.nanmean(pol)),
            "pass_count_mean":            float(np.nanmean(psc)),
            "per_pixel_count_emd_mean":   float(np.nanmean(pem)),
            "sinkhorn_dt_emd_mean":       float(np.nanmean(sem)),
            "gwd_mean":                   float(np.nanmean(gwd)),
            # Median (robust to outliers — recommended for count_ratio in particular,
            # where the mean is sensitive to a few high-event-density prefixes)
            "count_ratio_median":           float(np.nanmedian(cnt)),
            "spatial_entropy_ratio_median": float(np.nanmedian(her)),
            "active_pixel_ratio_median":    float(np.nanmedian(apr)),
            "per_pixel_count_emd_median":   float(np.nanmedian(pem)),
            "sinkhorn_dt_emd_median":       float(np.nanmedian(sem)),
            "gwd_median":                   float(np.nanmedian(gwd)),
        }
        summary.append(agg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": {v: r["rows"] for v, r in results.items()}}, f, indent=2)

    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write(render_md_table(summary) + "\n")
    print(f"[k_health_v2e_compare] wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
