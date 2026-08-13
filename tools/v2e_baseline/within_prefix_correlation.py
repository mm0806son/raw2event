"""Per-variant Spearman rho between upstream distance and transfer correctness.

Correlates a per-recording upstream metric against per-recording 0/1 transfer
correctness averaged over training seeds, with a bootstrap CI over recordings.
Spearman rather than Pearson because correctness is a four-level ordinal and the
distance is heavy-tailed; the bootstrap resamples recordings rather than seeds
because the sampling noise of interest is which recordings were drawn.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


VARIANT_LABELS = {
    "V01": "Raw2Event-RAW (Stage 1.5 K)",
    "V02": "Raw2Event-RGB (Stage 1.5 K)",
    "V03": "DVS-V default K (raw)",
    "V04": "DVS-V default K (rgb)",
    "V05": "v2e native50 default (rgb, 0.15)",
    "V06": "v2e native50 tuned (rgb, 0.25)",
    "V07": "v2e native50 default (raw, 0.15)",
    "V08": "v2e native50 tuned (raw, 0.30)",
    "V09": "v2e slomo default (rgb, 0.15)",
    "V10": "v2e slomo tuned (rgb, 0.25)",
    "V11": "v2e slomo default (raw, 0.15)",
    "V12": "v2e slomo tuned (raw, 0.30)",
}


def load_emd_per_prefix(results_dir: Path, variant: str, metric: str) -> dict[str, float]:
    """Load result_V<NN>.json and return ``{prefix: metric_value}``.

    Drops rows with ``n_dv == 0`` (3 fail-mode prefixes consistent across all
    12 variants).
    """
    rec = json.loads((results_dir / f"result_{variant}.json").read_text())
    block = rec["raw"][variant]
    out: dict[str, float] = {}
    for r in block:
        if r.get("n_dv", 0) == 0:
            continue
        v = r.get(metric)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        out[r["prefix"]] = float(v)
    return out


def load_correctness(json_path: Path) -> dict:
    return json.loads(json_path.read_text())


def average_correctness_per_prefix(correctness: dict, variant: str) -> dict[str, float]:
    """Return ``{prefix: mean(0/1) over seeds}`` for one variant.

    Combines the 600-element ``per_run[<key>].correct`` arrays across all
    seeds matching the variant ID, then maps positionally to ``prefixes``.
    Each per-prefix value lies in {0, 1/k, 2/k, ..., 1} for k seeds.
    """
    prefixes: list[str] = correctness["prefixes"]
    per_run: dict = correctness["per_run"]
    matched_keys = sorted(k for k in per_run if k.split(".")[1] == variant)
    if not matched_keys:
        raise KeyError(f"no per_run entries for variant {variant}")
    arrs = []
    for k in matched_keys:
        arr = np.asarray(per_run[k]["correct"], dtype=np.float32)
        if arr.shape[0] != len(prefixes):
            raise ValueError(
                f"{k}: correct len={arr.shape[0]} != prefixes len={len(prefixes)}"
            )
        arrs.append(arr)
    avg = np.stack(arrs, axis=0).mean(axis=0)
    return {p: float(v) for p, v in zip(prefixes, avg)}


def join_for_variant(
    emd_map: dict[str, float],
    correctness_map: dict[str, float],
    canonical: set[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Intersect EMD universe with correctness test prefixes and canonical
    list. Returns (prefixes, emd, correctness) aligned by prefix.

    Sorted output for determinism (Spearman is order-invariant but reproducibility
    of any optional CSV export hinges on a stable order).
    """
    keys = sorted(set(emd_map.keys()) & set(correctness_map.keys()) & canonical)
    emd = np.array([emd_map[p] for p in keys], dtype=np.float64)
    cor = np.array([correctness_map[p] for p in keys], dtype=np.float64)
    return keys, emd, cor


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, n_bootstrap: int, rng_seed: int
) -> tuple[float, float, int]:
    """Per-prefix bootstrap 95% CI for Spearman ρ.

    Returns (ci_low, ci_high, n_valid_resamples). Resamples that produce a
    constant-rank vector (and hence undefined Spearman) are dropped from the
    percentile computation rather than imputed — matches paired-bootstrap-CI
    convention in cross_modal_eval_with_ci.py:70-96.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), 0
    rho_samples = np.empty(n_bootstrap, dtype=np.float64)
    rho_samples[:] = np.nan
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        with warnings.catch_warnings():
            # spearmanr emits a RuntimeWarning when an input is constant;
            # already filtered above, but suppress just in case.
            warnings.simplefilter("ignore")
            rho_samples[b] = spearmanr(xb, yb).statistic
    valid = rho_samples[np.isfinite(rho_samples)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return float(lo), float(hi), int(valid.size)


def correlation_for_variant(
    variant: str,
    emd_map: dict[str, float],
    correctness_map: dict[str, float],
    canonical: set[str],
    n_bootstrap: int,
    rng_seed: int,
) -> dict:
    keys, emd, cor = join_for_variant(emd_map, correctness_map, canonical)
    n_used = len(keys)
    out = {
        "variant": variant,
        "label": VARIANT_LABELS.get(variant, ""),
        "n_used": n_used,
        "n_emd": len(emd_map),
        "n_correctness": len(correctness_map),
        "n_canonical": len(canonical),
    }
    if n_used < 4:
        out.update({
            "rho": float("nan"), "p_value": float("nan"),
            "rho_ci_low": float("nan"), "rho_ci_high": float("nan"),
            "n_bootstrap_valid": 0,
            "note": f"n_used={n_used} < 4 — cannot compute Spearman",
        })
        return out
    sp = spearmanr(emd, cor)
    ci_lo, ci_hi, n_valid = bootstrap_spearman_ci(emd, cor, n_bootstrap, rng_seed)
    out.update({
        "rho": float(sp.statistic),
        "p_value": float(sp.pvalue),
        "rho_ci_low": ci_lo,
        "rho_ci_high": ci_hi,
        "n_bootstrap_valid": n_valid,
    })
    return out


def summarize_across_variants(per_variant: list[dict]) -> dict:
    """Median + IQR + min/max of per-variant ρ for the summary abstract claim."""
    rhos = np.array([r["rho"] for r in per_variant if np.isfinite(r["rho"])],
                    dtype=np.float64)
    if rhos.size == 0:
        return {"n_variants_used": 0}
    q1, med, q3 = np.percentile(rhos, [25, 50, 75])
    return {
        "n_variants_used": int(rhos.size),
        "rho_median": float(med),
        "rho_iqr_low": float(q1),
        "rho_iqr_high": float(q3),
        "rho_min": float(rhos.min()),
        "rho_max": float(rhos.max()),
    }


def write_markdown(out_md: Path, per_variant: list[dict], summary: dict,
                   emd_metric: str, n_bootstrap: int) -> None:
    lines = [
        "# Within-recording correlation — K-health vs cross-modal transfer",
        "",
        f"EMD metric: `{emd_metric}` · bootstrap B={n_bootstrap} on prefix axis · "
        f"correctness avg over training seeds.",
        "",
        "| Variant | Label | n_used | ρ (Spearman) | 95% CI | p |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in per_variant:
        if np.isfinite(r["rho"]):
            ci = f"[{r['rho_ci_low']:+.3f}, {r['rho_ci_high']:+.3f}]"
            rho_s = f"{r['rho']:+.3f}"
            p_s = f"{r['p_value']:.2e}"
        else:
            ci = "—"
            rho_s = "NaN"
            p_s = "—"
        lines.append(
            f"| {r['variant']} | {r['label']} | {r['n_used']} | {rho_s} | {ci} | {p_s} |"
        )
    lines += ["", "## Across-variants summary"]
    if summary.get("n_variants_used", 0) > 0:
        lines.append(
            f"- n_variants_used = {summary['n_variants_used']}"
        )
        lines.append(
            f"- ρ median = {summary['rho_median']:+.3f}"
        )
        lines.append(
            f"- ρ IQR    = [{summary['rho_iqr_low']:+.3f}, {summary['rho_iqr_high']:+.3f}]"
        )
        lines.append(
            f"- ρ range  = [{summary['rho_min']:+.3f}, {summary['rho_max']:+.3f}]"
        )
    else:
        lines.append("- (no usable variants)")
    out_md.write_text("\n".join(lines) + "\n")


def parse_csv_list(s: str | None) -> list[str]:
    return [tok.strip() for tok in (s or "").split(",") if tok.strip()]


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emd_results_dir", required=True,
                    help="Dir containing result_V<NN>.json (k_health_v2e_compare output).")
    ap.add_argument("--correctness_json", required=True,
                    help="per_prefix_correctness.json from dump_per_prefix_correctness.py.")
    ap.add_argument("--canonical_prefix_list", required=True,
                    help="canonical_test_v2e_runnable.txt (559 lines).")
    ap.add_argument("--variants", required=True,
                    help="Comma-separated variant IDs (e.g. V01,V02,...,V08).")
    ap.add_argument("--emd_metric", default="per_pixel_count_emd",
                    choices=["per_pixel_count_emd", "sinkhorn_dt_emd",
                             "count_ratio", "polarity_delta"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    variants = parse_csv_list(args.variants)
    if not variants:
        raise SystemExit("--variants empty after parsing")

    emd_dir = Path(args.emd_results_dir)
    canonical = {
        ln.strip() for ln in Path(args.canonical_prefix_list).read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }
    correctness = load_correctness(Path(args.correctness_json))

    per_variant: list[dict] = []
    for v in variants:
        emd_map = load_emd_per_prefix(emd_dir, v, args.emd_metric)
        try:
            cor_map = average_correctness_per_prefix(correctness, v)
        except KeyError as e:
            print(f"  {v}: SKIPPED — {e}", file=sys.stderr)
            continue
        result = correlation_for_variant(
            v, emd_map, cor_map, canonical, args.bootstrap, args.rng_seed
        )
        per_variant.append(result)
        if np.isfinite(result["rho"]):
            print(f"  {v}: ρ={result['rho']:+.3f} "
                  f"CI=[{result['rho_ci_low']:+.3f}, {result['rho_ci_high']:+.3f}] "
                  f"n_used={result['n_used']}")
        else:
            print(f"  {v}: NaN — {result.get('note', '')}")

    summary = summarize_across_variants(per_variant)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "emd_metric": args.emd_metric,
            "bootstrap": args.bootstrap,
            "rng_seed": args.rng_seed,
            "variants": variants,
            "emd_results_dir": str(emd_dir),
            "correctness_json": str(args.correctness_json),
            "canonical_prefix_list": str(args.canonical_prefix_list),
            "n_canonical": len(canonical),
        },
        "per_variant": per_variant,
        "across_variants": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    write_markdown(out_dir / "results.md", per_variant, summary,
                   args.emd_metric, args.bootstrap)
    print(f"[within_prefix_correlation] wrote {out_dir/'results.json'} "
          f"and {out_dir/'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
