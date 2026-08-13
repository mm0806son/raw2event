"""Correlate a per-variant upstream metric (e.g. EQS) against downstream transfer.

This is where the paper's claim lands: across the ~12 simulator variants, does a
*higher* EQS similarity actually predict *better* sim-to-real downstream transfer
(as EQS's authors claim for DSEC detection)?  With only ~12 variants this is an
**exploratory** analysis: a non-significant p-value is NOT evidence of "no
correlation".  We therefore report several complementary statistics — Spearman
rho, Kendall tau-b, Pearson r (each with its analytic p), a permutation p-value,
and a bootstrap CI on Spearman rho — and emphasise rank *stability* over any
single p-value.

Pure NumPy + SciPy; no torch, no project-data dependency, so it is unit-testable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _align(
    upstream: dict[str, float], downstream: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intersect on variant id, drop pairs with non-finite values, keep order."""
    variants = [v for v in upstream if v in downstream]
    xs, ys, kept = [], [], []
    for v in variants:
        x, y = float(upstream[v]), float(downstream[v])
        if np.isfinite(x) and np.isfinite(y):
            xs.append(x)
            ys.append(y)
            kept.append(v)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), kept


def rank_correlation(
    upstream: dict[str, float],
    downstream: dict[str, float],
    n_perm: int = 10000,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Multi-statistic correlation between two per-variant metric maps.

    Returns Spearman/Kendall/Pearson point estimates with analytic p-values, a
    permutation p-value for Spearman (label-shuffle, two-sided), and a bootstrap
    percentile CI for Spearman rho.  Degenerate inputs (n < 3 or a constant
    column) yield NaNs rather than raising.
    """
    x, y, kept = _align(upstream, downstream)
    n = x.size
    out: dict = {"n": n, "variants": kept}
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        out.update(
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            kendall_tau=float("nan"),
            kendall_p=float("nan"),
            pearson_r=float("nan"),
            pearson_p=float("nan"),
            permutation_p=float("nan"),
            spearman_ci=[float("nan"), float("nan")],
            note="degenerate (n<3 or constant input)",
        )
        return out

    sp = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    pr = stats.pearsonr(x, y)
    rho = float(sp.statistic)

    rng = np.random.default_rng(seed)
    perm_rho = np.empty(n_perm)
    for i in range(n_perm):
        perm_rho[i] = stats.spearmanr(x, rng.permutation(y)).statistic
    perm_p = float((np.sum(np.abs(perm_rho) >= abs(rho)) + 1) / (n_perm + 1))

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, by = x[idx], y[idx]
        boot[i] = (
            stats.spearmanr(bx, by).statistic if np.ptp(bx) and np.ptp(by) else np.nan
        )
    boot = boot[np.isfinite(boot)]
    lo, hi = (
        (
            float(np.quantile(boot, (1 - ci) / 2)),
            float(np.quantile(boot, 1 - (1 - ci) / 2)),
        )
        if boot.size
        else (float("nan"), float("nan"))
    )

    out.update(
        spearman_rho=rho,
        spearman_p=float(sp.pvalue),
        kendall_tau=float(kt.statistic),
        kendall_p=float(kt.pvalue),
        pearson_r=float(pr.statistic),
        pearson_p=float(pr.pvalue),
        permutation_p=perm_p,
        spearman_ci=[lo, hi],
    )
    return out


def render_md(results: list[dict]) -> str:
    """Render a correlation table; each row is one (upstream, downstream) pair."""
    lines = [
        "# EQS vs downstream transfer — rank-correlation (EXPLORATORY, n~12)",
        "",
        "A non-significant p is NOT evidence of no correlation at this n; read "
        "rho/tau with the bootstrap CI and treat as exploratory.  `permutation_p` "
        "is a two-sided label-shuffle test on Spearman rho.",
        "",
        "| Upstream | Downstream | n | Spearman rho | 95% CI | perm p | Kendall tau | Pearson r |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in results:
        ci = r.get("spearman_ci", [float("nan"), float("nan")])
        lines.append(
            f"| {r['upstream_name']} | {r['downstream_name']} | {r['n']} | "
            f"{r['spearman_rho']:.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{r['permutation_p']:.3f} | {r['kendall_tau']:.3f} | {r['pearson_r']:.3f} |"
        )
    return "\n".join(lines)


def _load_metric_map(path: str, key: str) -> dict[str, float]:
    """Load {variant: value} from a JSON list of dicts, each with 'variant'+key."""
    with open(path) as f:
        data = json.load(f)
    rows = data["summary"] if isinstance(data, dict) and "summary" in data else data
    return {row["variant"]: row[key] for row in rows if key in row}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--upstream_json", required=True, help="JSON with per-variant upstream summary"
    )
    ap.add_argument("--upstream_key", default="eqs_similarity_mean")
    ap.add_argument(
        "--downstream_json",
        required=True,
        help="JSON with per-variant downstream summary",
    )
    ap.add_argument(
        "--downstream_key", required=True, help="e.g. transfer_acc or retrieval_r1"
    )
    ap.add_argument(
        "--output", required=True, help="Output JSON path; .md sibling also written"
    )
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    up = _load_metric_map(args.upstream_json, args.upstream_key)
    down = _load_metric_map(args.downstream_json, args.downstream_key)
    res = rank_correlation(
        up, down, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed
    )
    res["upstream_name"] = args.upstream_key
    res["downstream_name"] = args.downstream_key

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    with open(out_path.with_suffix(".md"), "w") as f:
        f.write(render_md([res]) + "\n")
    print(
        f"[eqs_downstream_correlation] wrote {out_path} and {out_path.with_suffix('.md')}"
    )
    print(
        f"  Spearman rho={res['spearman_rho']:.3f} perm_p={res['permutation_p']:.3f} n={res['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
