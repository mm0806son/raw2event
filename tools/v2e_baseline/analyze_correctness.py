"""Offline analysis of a dumped per-sample correctness array.

Emits two Markdown reports from the output of ``dump_eval_correctness``: a
degeneracy audit (per-variant prediction entropy, top-1 share, distinct classes,
cross-seed identity) that separates a genuine constant-class collapse from a
checkpoint-reuse bug, and a comparison report contrasting the fixed-seed and
seed x recording bootstraps with a max-T simultaneous family.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tools.v2e_baseline.cross_modal_eval_with_ci import paired_bootstrap_ci, verdict as fverdict
from tools.v2e_baseline.hierarchical_ci import (
    hierarchical_paired_bootstrap_ci,
    seed_level_paired_t,
    simultaneous_vs_reference,
    verdict,
)

VARIANT_MODALITY = {
    "DV": "dv", "V01": "raw", "V02": "rgb", "V03": "raw", "V04": "rgb",
    "V05": "rgb", "V06": "rgb", "V07": "raw", "V08": "raw",
    "V09": "rgb", "V10": "rgb", "V11": "raw", "V12": "raw",
}
ORDER = ["DV", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08",
         "V09", "V10", "V11", "V12"]
PAIRS = [
    ("V01", "V07", "R2E-RAW vs v2e-RAW-default"),
    ("V01", "V08", "R2E-RAW vs v2e-RAW-tuned (★)"),
    ("V01", "V03", "R2E-RAW vs DVS-V-defK-raw"),
    ("V02", "V05", "R2E-RGB vs v2e-RGB-default"),
    ("V02", "V06", "R2E-RGB vs v2e-RGB-tuned"),
    ("V02", "V04", "R2E-RGB vs DVS-V-defK-rgb"),
]


def _load(npz_path: Path):
    z = np.load(npz_path)
    correct: dict = collections.defaultdict(dict)
    preds: dict = collections.defaultdict(dict)
    for k in z.files:
        if k.endswith("__correct"):
            run = k[: -len("__correct")]
            _, var, seed = run.split(".")
            correct[var][seed] = z[k]
        elif k.endswith("__preds"):
            run = k[: -len("__preds")]
            _, var, seed = run.split(".")
            preds[var][seed] = z[k]
    return correct, preds


def _tag(v: str) -> str:
    return "PASS" if "PASS" in v else ("WITHDRAW" if "WITHDRAW" in v else "INCONCLUSIVE")


def audit_md(correct, preds, family: str, n_classes: int = 10) -> str:
    lines = [f"## Collapse / V02 audit — {family}", "",
             "`top1share` = mean fraction of test predictions in the single most "
             "predicted class; `entropy` = mean Shannon entropy of the predicted-"
             "class histogram (max = ln 10 = 2.303); collapse ⇒ high top1share, "
             "low entropy. `corr_ident` / `pred_ident` = are the per-seed "
             "correctness / prediction vectors element-wise identical across seeds?",
             "",
             "| V | mod | acc% | top1share | entropy | #uniq | corr_ident | pred_ident |",
             "|---|---|---:|---:|---:|---:|:--:|:--:|"]
    for var in ORDER:
        if var not in correct:
            continue
        seeds = sorted(correct[var])
        cs = [correct[var][s] for s in seeds]
        ps = [preds[var][s] for s in seeds]
        acc = np.mean([c.mean() for c in cs]) * 100
        shares, ents, uniqs = [], [], []
        for p in ps:
            h = np.bincount(p, minlength=n_classes)
            pr = h / h.sum()
            shares.append(pr.max())
            ents.append(-(pr[pr > 0] * np.log(pr[pr > 0])).sum())
            uniqs.append(len(np.unique(p)))
        corr_ident = all(np.array_equal(cs[0], c) for c in cs[1:])
        pred_ident = all(np.array_equal(ps[0], p) for p in ps[1:])
        lines.append(
            f"| {var} | {VARIANT_MODALITY[var]} | {acc:.2f} | {np.mean(shares):.2f} "
            f"| {np.mean(ents):.3f} | {np.mean(uniqs):.1f} | "
            f"{'Y' if corr_ident else 'N'} | {'Y' if pred_ident else 'N'} |")
    return "\n".join(lines)


def hier_md(correct, family: str, n_bootstrap: int, rng_seed: int) -> str:
    lines = [f"## Seed×prefix uncertainty — {family}", "",
             "Three statistics on the same per-seed deltas, increasingly honest "
             "about training-seed variance: (1) **fixed-seed** bootstrap resamples "
             "only prefixes (3 seeds held fixed — too narrow); (2) **crossed "
             "bootstrap** resamples seeds and a shared prefix vector; (3) "
             "**seed-level paired-t** over the 3 per-seed deltas (primary; most "
             "conservative). `seedΔ` = the three per-seed deltas (pp). With only "
             "3 seeds the paired-t is wide by design.",
             "",
             "| Comparison | Δpp | fixed 95% CI | crossed-boot 95% CI | paired-t 95% CI | seedΔ | paired-t |",
             "|---|---:|---|---|---|---|:--:|"]
    for left, right, desc in PAIRS:
        if left not in correct or right not in correct:
            continue
        seeds = sorted(set(correct[left]) & set(correct[right]))
        deltas = [(correct[left][s].astype(np.int8) - correct[right][s].astype(np.int8))
                  for s in seeds]
        fx = paired_bootstrap_ci(deltas, n_bootstrap=n_bootstrap, rng_seed=rng_seed)
        h = hierarchical_paired_bootstrap_ci(correct[left], correct[right],
                                             n_bootstrap=n_bootstrap, rng_seed=rng_seed)
        t = seed_level_paired_t(correct[left], correct[right])
        tv = _tag(verdict(t["mean_pp"], t["ci_low_pp"], t["ci_high_pp"]))
        sd = ", ".join(f"{x:+.1f}" for x in h["seed_deltas_pp"])
        lines.append(
            f"| {desc} | {h['mean_pp']:.2f} "
            f"| [{fx['ci_low_pp']:.2f}, {fx['ci_high_pp']:.2f}] "
            f"| [{h['ci_low_pp']:.2f}, {h['ci_high_pp']:.2f}] "
            f"| [{t['ci_low_pp']:.2f}, {t['ci_high_pp']:.2f}] | {sd} | {tv} |")

    others = {v: correct[v] for v in ORDER
              if v not in ("DV", "V01") and v in correct}
    s = simultaneous_vs_reference(correct["V01"], others,
                                  n_bootstrap=n_bootstrap, rng_seed=rng_seed)
    pm = sum(c["marginal_verdict"] == "PASS" for c in s["per_comparison"].values())
    ps = sum(c["simultaneous_verdict"] == "PASS" for c in s["per_comparison"].values())
    n = len(s["per_comparison"])
    lines += ["",
              f"### Simultaneous V01 vs {n} other simulators (max-T, FWER-controlled)",
              "",
              f"max-T critical value q = {s['maxT_q']:.2f}  ·  "
              f"**PASS marginal = {pm}/{n}  →  PASS simultaneous = {ps}/{n}**", "",
              "| vs | Δpp | marginal 95% CI | simultaneous 95% CI | sim verdict |",
              "|---|---:|---|---|:--:|"]
    for v, c in s["per_comparison"].items():
        lines.append(
            f"| {v} | {c['mean_pp']:.2f} | [{c['ci_low_pp']:.2f}, {c['ci_high_pp']:.2f}] "
            f"| [{c['sim_ci_low_pp']:.2f}, {c['sim_ci_high_pp']:.2f}] | "
            f"{_tag(c['simultaneous_verdict'])} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n_bootstrap", type=int, default=5000)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads(args.npz.with_suffix(args.npz.suffix + ".meta.json").read_text())
    family = meta["model_family"]
    correct, preds = _load(args.npz)

    md = "\n\n".join([
        f"# Correctness analysis — {family} (test on real DV, N={meta['n_test']})",
        audit_md(correct, preds, family),
        hier_md(correct, family, args.n_bootstrap, args.rng_seed),
    ])
    args.out.write_text(md + "\n")
    print(md)
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
