"""Paired bootstrap CI for cross-modal retrieval metrics.

Consumes a manifest of per-seed per-query metric NPZ files written by
``eval_retrieval`` and reports R@1, R@5, R@10 and mAP deltas with percentile
confidence intervals. All NPZ files within one comparison must share the same
query order, otherwise the pairing is invalid. See ``--help`` for the manifest
schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

K_LIST = (1, 5, 10)
METRIC_NAMES = ("R@1", "R@5", "R@10", "mAP")
GAIN_THRESHOLD_PP = 5.0  # same as classification protocol


def load_per_query(npz_path: Path) -> dict:
    z = np.load(npz_path)
    out = {
        "query_prefix": np.asarray(z["query_prefix"]),
        "query_label": z["query_label"].astype(np.int64),
        "r_at_k": z["r_at_k"].astype(np.float32),  # (N, 3)
        "ap": z["ap"].astype(np.float32),          # (N,)
    }
    sidecar = npz_path.with_suffix(".meta.json")
    out["meta"] = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return out


def stack_per_seed(per_query_npz_paths: list[Path]) -> dict:
    """Stack per-seed per-query metric vectors into (S, N) arrays. All seeds
    must share the same prefix order; we assert that to catch silent pairing
    bugs that would invalidate the bootstrap CI."""
    seeds = [load_per_query(p) for p in per_query_npz_paths]
    ref = seeds[0]["query_prefix"]
    for i, s in enumerate(seeds[1:], start=1):
        assert np.array_equal(s["query_prefix"], ref), (
            f"seed[{i}] prefix order disagrees with seed[0]; paired bootstrap requires "
            f"identical query order across seeds"
        )
    return {
        "R@1":  np.stack([s["r_at_k"][:, 0] for s in seeds], axis=0),
        "R@5":  np.stack([s["r_at_k"][:, 1] for s in seeds], axis=0),
        "R@10": np.stack([s["r_at_k"][:, 2] for s in seeds], axis=0),
        "mAP":  np.stack([s["ap"] for s in seeds], axis=0),
        "n_seeds": len(seeds),
        "query_prefix": ref,
        "metas": [s["meta"] for s in seeds],
    }


def paired_bootstrap_ci(
    delta_stack: np.ndarray, n_bootstrap: int = 1000, rng_seed: int = 0
) -> dict:
    """Mirror cross_modal_eval_with_ci.paired_bootstrap_ci.

    delta_stack : (S, N) per-seed-pair per-query (left - right) values.
    Returns mean (in pp), 95% percentile CI (in pp), n_seeds, n_test.
    """
    rng = np.random.default_rng(rng_seed)
    s, n = delta_stack.shape
    point_mean = float(delta_stack.mean())
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[b] = float(delta_stack[:, idx].mean())
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "n_seeds": s,
        "n_test": n,
        "mean_pp": 100.0 * point_mean,
        "ci_low_pp": 100.0 * float(lo),
        "ci_high_pp": 100.0 * float(hi),
        "ci_crosses_zero": bool(lo < 0 < hi),
    }


def absolute_bootstrap_ci(
    metric_stack: np.ndarray, n_bootstrap: int = 1000, rng_seed: int = 0
) -> dict:
    """Bootstrap CI for an absolute (non-paired) per-seed per-query metric.
    Used for self-retrieval upper-bound rows where there is no 'right' to
    subtract. Same resampling: draw N indices, average over (S, idx)."""
    rng = np.random.default_rng(rng_seed)
    s, n = metric_stack.shape
    point_mean = float(metric_stack.mean())
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[b] = float(metric_stack[:, idx].mean())
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "n_seeds": s,
        "n_test": n,
        "mean_pp": 100.0 * point_mean,
        "ci_low_pp": 100.0 * float(lo),
        "ci_high_pp": 100.0 * float(hi),
    }


def verdict_from_r1(mean_pp: float, ci_low_pp: float, ci_high_pp: float,
                    threshold: float = GAIN_THRESHOLD_PP) -> str:
    """R@1-driven direction verdict by whether the paired-bootstrap CI crosses
    zero; the effect size is reported separately (mean + CI). The ``threshold``
    arg is retained for signature back-compat but is UNUSED — no arbitrary pp
    threshold and no PASS/FAIL collapse (report first, judge significance-in-
    context second;)."""
    del threshold  # deprecated: verdicts no longer collapse on a pp threshold
    if ci_low_pp > 0:
        return "significant positive (95% CI > 0)"
    if ci_high_pp < 0:
        return "significant negative (95% CI < 0)"
    return "no significant direction (95% CI crosses zero)"


def fmt_ci(mean_pp: float, lo_pp: float, hi_pp: float) -> str:
    sign = "+" if mean_pp >= 0 else ""
    sign_lo = "+" if lo_pp >= 0 else ""
    sign_hi = "+" if hi_pp >= 0 else ""
    return f"{sign}{mean_pp:.2f} [{sign_lo}{lo_pp:.2f}, {sign_hi}{hi_pp:.2f}]"


def fmt_abs(mean_pp: float, lo_pp: float, hi_pp: float) -> str:
    return f"{mean_pp:.2f} [{lo_pp:.2f}, {hi_pp:.2f}]"


def write_markdown(rows: list[dict], self_rows: list[dict], out_md: Path,
                   n_bootstrap: int) -> None:
    """Wide format: one row per comparison, four metric Δ columns + verdict."""
    lines: list[str] = []
    lines.append("# Cross-modal retrieval — paired bootstrap 95% CI on canonical 559 prefixes")
    lines.append("")
    lines.append(
        f"Query = real Davis346 DV (canonical 559 test prefixes) · "
        f"Gallery = simulator variant · Encoder = each variant's own ckpt · "
        f"Embedding = ``model.head`` input (B, 256), L2-normalized · "
        f"Bootstrap B={n_bootstrap}, percentile [2.5%, 97.5%]"
    )
    lines.append("")
    lines.append("Reporting rule (R@1-driven):")
    lines.append("- 95% CI entirely > 0 → significant positive effect")
    lines.append("- 95% CI entirely < 0 → significant negative effect")
    lines.append("- 95% CI crosses zero → no significant direction")
    lines.append("")
    lines.append("## Paired comparisons (8 rows)")
    lines.append("")
    lines.append(
        "| Pair | Left R@1 | Right R@1 | Δ R@1 (pp) [95% CI] | Δ R@5 (pp) [95% CI] | "
        "Δ R@10 (pp) [95% CI] | Δ mAP (pp) [95% CI] | n seeds | Verdict |"
    )
    lines.append("|---|---:|---:|---|---|---|---|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['left_means']['R@1']*100:.2f} | {r['right_means']['R@1']*100:.2f} | "
            f"{fmt_ci(r['ci']['R@1']['mean_pp'], r['ci']['R@1']['ci_low_pp'], r['ci']['R@1']['ci_high_pp'])} | "
            f"{fmt_ci(r['ci']['R@5']['mean_pp'], r['ci']['R@5']['ci_low_pp'], r['ci']['R@5']['ci_high_pp'])} | "
            f"{fmt_ci(r['ci']['R@10']['mean_pp'], r['ci']['R@10']['ci_low_pp'], r['ci']['R@10']['ci_high_pp'])} | "
            f"{fmt_ci(r['ci']['mAP']['mean_pp'], r['ci']['mAP']['ci_low_pp'], r['ci']['mAP']['ci_high_pp'])} | "
            f"{r['n_seeds']} | {r['verdict']} |"
        )
    if self_rows:
        lines.append("")
        lines.append("## Self-retrieval upper bound (query=gallery=same variant; "
                     "same-prefix masked)")
        lines.append("")
        lines.append(
            "| Variant | R@1 (pp) [95% CI] | R@5 (pp) [95% CI] | R@10 (pp) [95% CI] | "
            "mAP (pp) [95% CI] | n seeds |"
        )
        lines.append("|---|---|---|---|---|---:|")
        for r in self_rows:
            lines.append(
                f"| {r['name']} | "
                f"{fmt_abs(r['ci']['R@1']['mean_pp'], r['ci']['R@1']['ci_low_pp'], r['ci']['R@1']['ci_high_pp'])} | "
                f"{fmt_abs(r['ci']['R@5']['mean_pp'], r['ci']['R@5']['ci_low_pp'], r['ci']['R@5']['ci_high_pp'])} | "
                f"{fmt_abs(r['ci']['R@10']['mean_pp'], r['ci']['R@10']['ci_low_pp'], r['ci']['R@10']['ci_high_pp'])} | "
                f"{fmt_abs(r['ci']['mAP']['mean_pp'], r['ci']['mAP']['ci_low_pp'], r['ci']['mAP']['ci_high_pp'])} | "
                f"{r['n_seeds']} |"
            )
    out_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output_json", type=Path, required=True)
    ap.add_argument("--output_md", type=Path, required=True)
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for cmp in manifest.get("comparisons", []):
        L = stack_per_seed([Path(p) for p in cmp["left"]["per_query_npz"]])
        R = stack_per_seed([Path(p) for p in cmp["right"]["per_query_npz"]])
        assert np.array_equal(L["query_prefix"], R["query_prefix"]), (
            f"{cmp['name']}: left/right query prefixes disagree — invalid pairing"
        )
        assert L["n_seeds"] == R["n_seeds"], (
            f"{cmp['name']}: seed count mismatch L={L['n_seeds']} R={R['n_seeds']}"
        )
        ci_per_metric = {}
        for m in METRIC_NAMES:
            delta = L[m] - R[m]  # (S, N), per-seed per-query
            ci_per_metric[m] = paired_bootstrap_ci(delta, args.n_bootstrap, args.rng_seed)
        verdict = verdict_from_r1(
            ci_per_metric["R@1"]["mean_pp"],
            ci_per_metric["R@1"]["ci_low_pp"],
            ci_per_metric["R@1"]["ci_high_pp"],
        )
        rows.append({
            "name": cmp["name"],
            "left_label":  f"{cmp['left'].get('variant')}-{cmp['left'].get('modality')}",
            "right_label": f"{cmp['right'].get('variant')}-{cmp['right'].get('modality')}",
            "left_means":  {m: float(L[m].mean()) for m in METRIC_NAMES},
            "right_means": {m: float(R[m].mean()) for m in METRIC_NAMES},
            "ci": ci_per_metric,
            "n_seeds": L["n_seeds"],
            "verdict": verdict,
        })
        print(
            f"[ci] {cmp['name']}: ΔR@1={ci_per_metric['R@1']['mean_pp']:+.2f} "
            f"[{ci_per_metric['R@1']['ci_low_pp']:+.2f}, {ci_per_metric['R@1']['ci_high_pp']:+.2f}]  "
            f"verdict={verdict}",
            flush=True,
        )

    self_rows: list[dict] = []
    for sb in manifest.get("self_baselines", []):
        S = stack_per_seed([Path(p) for p in sb["per_query_npz"]])
        ci_per_metric = {m: absolute_bootstrap_ci(S[m], args.n_bootstrap, args.rng_seed)
                         for m in METRIC_NAMES}
        self_rows.append({
            "name": sb["name"],
            "variant": sb.get("variant"),
            "ci": ci_per_metric,
            "n_seeds": S["n_seeds"],
        })
        print(
            f"[ci] self {sb['name']}: R@1={ci_per_metric['R@1']['mean_pp']:.2f} "
            f"[{ci_per_metric['R@1']['ci_low_pp']:.2f}, {ci_per_metric['R@1']['ci_high_pp']:.2f}]",
            flush=True,
        )

    out_payload = {
        "n_bootstrap": args.n_bootstrap,
        "rng_seed": args.rng_seed,
        "gain_threshold_pp": GAIN_THRESHOLD_PP,
        "comparisons": rows,
        "self_baselines": self_rows,
    }
    args.output_json.write_text(json.dumps(out_payload, indent=2))
    print(f"[ci] wrote JSON → {args.output_json}", flush=True)

    write_markdown(rows, self_rows, args.output_md, args.n_bootstrap)
    print(f"[ci] wrote markdown → {args.output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
