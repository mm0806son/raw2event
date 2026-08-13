"""Cross-modal evaluation with a paired bootstrap 95% CI.

Loads the checkpoints named in a manifest, runs each over the canonical real-event
test set, and reports the paired delta between two variants as a seed mean with a
percentile CI resampled over test recordings.

The bootstrap resamples the test set only and therefore ignores training-seed
variance; for comparisons across simulators use ``hierarchical_ci``, which
resamples seeds as well. See ``--help`` for the manifest schema.
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


def load_test_indices(split_source_run: Path) -> list[int]:
    info_path = split_source_run / "split_info.json"
    info = json.loads(info_path.read_text())
    return list(info["test_indices"])


def run_inference(ckpt_path: Path, test_data_dir: Path, test_indices: list[int],
                  device: str = "cuda:0", batch_size: int = 32,
                  test_modality: str = "dv", model_family: str = "qkformer") -> dict:
    """Run a QKF or MNet checkpoint on the canonical test set; return per-sample
    correctness vector + predictions.

    Defers to train_class.evaluate_cross_modality.run_inference_on_indices
    (the helper added in commit accompanying this file).
    """
    from train_class.evaluate_cross_modality import (  # type: ignore
        run_inference_on_indices,
    )
    return run_inference_on_indices(
        ckpt_path=str(ckpt_path),
        data_dir=str(test_data_dir),
        test_indices=test_indices,
        device=device,
        batch_size=batch_size,
        model_family=model_family,
        test_modality=test_modality,
    )


def paired_bootstrap_ci(deltas_per_seed: list[np.ndarray], n_bootstrap: int = 1000,
                        rng_seed: int = 0) -> dict:
    """Compute mean ± 95% CI of paired deltas using paired test-set bootstrap.

    deltas_per_seed: list of per-sample (correct_ours - correct_baseline) ∈ {-1,0,+1}
    arrays, one per training seed pair. We bootstrap *over the test set* and
    average across seeds within each bootstrap draw.
    """
    rng = np.random.default_rng(rng_seed)
    if not deltas_per_seed:
        return {"n_seeds": 0, "n_test": 0, "mean_pp": float("nan"),
                "ci_low_pp": float("nan"), "ci_high_pp": float("nan")}
    stack = np.stack(deltas_per_seed, axis=0)   # (S, N)
    s, n = stack.shape
    point_mean = float(stack.mean())
    samples = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[b] = stack[:, idx].mean()
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {
        "n_seeds": s, "n_test": n,
        "mean_pp": 100 * point_mean,
        "ci_low_pp": 100 * float(lo),
        "ci_high_pp": 100 * float(hi),
        "ci_crosses_zero": bool(lo < 0 < hi),
    }


def verdict(mean_pp: float, ci_low_pp: float, ci_high_pp: float) -> str:
    """Report direction by whether the paired-bootstrap CI crosses zero.

    The effect size is carried by mean_pp + the CI themselves and interpreted
    separately in context. No arbitrary pp threshold and no PASS/FAIL collapse.

    NOTE: this test-set-only bootstrap does not capture training-seed variance;
    for claims across simulators prefer the seed-level hierarchical / paired-t CI
    in tools/v2e_baseline/hierarchical_ci.py (fixed-seed CIs are anti-conservative).
    """
    if ci_low_pp > 0:
        return "significant positive (95% CI > 0)"
    if ci_high_pp < 0:
        return "significant negative (95% CI < 0)"
    return "no significant direction (95% CI crosses zero)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--gain_threshold_pp", type=float, default=5.0,
                    help="DEPRECATED / unused: retained for CLI back-compat only. "
                         "Verdicts no longer collapse on a pp threshold "
                         ".")
    args = ap.parse_args()

    with open(args.manifest) as f:
        m = json.load(f)
    test_indices = load_test_indices(Path(m["split_source_run"]))
    test_dir = Path(m["test_data_dir"])
    test_modality = m.get("test_modality", "dv")

    # Group runs by (model_family, variant, seed). Prefix convention:
    #   "<family>.V<id>.seed<k>" with kind/label in run dict.
    per_run = {}
    for key, spec in m["runs"].items():
        family, variant_id, seed_str = key.split(".")
        seed = int(seed_str.replace("seed", ""))
        result = run_inference(
            Path(spec["ckpt"]), test_dir, test_indices,
            device=args.device, batch_size=args.batch_size,
            test_modality=test_modality, model_family=family,
        )
        per_run[(family, variant_id, seed)] = {
            "label": spec["label"], "kind": spec.get("kind", ""),
            "correct": np.asarray(result["correct"], dtype=np.int8),
            "acc": float(np.asarray(result["correct"]).mean()),
        }
        print(f"  {key}: acc={per_run[(family, variant_id, seed)]['acc']:.4f}")

    # Pairing strategy:
    # If manifest provides explicit `comparisons` list (left/right variant IDs,
    # frozen ahead of time by the evaluation plan), iterate
    # each row, pair runs by matching seed across left/right variants, compute
    # paired bootstrap CI on test set.
    # Fallback: legacy kind-based pairing (kind="ours" vs all other kinds).
    explicit_cmp = m.get("comparisons")
    families = sorted({k[0] for k in per_run.keys()})
    comparisons = []

    if explicit_cmp:
        for cmp_spec in explicit_cmp:
            L, R, lbl = cmp_spec["left"], cmp_spec["right"], cmp_spec["label"]
            for family in families:
                deltas = []
                seeds_used = []
                for (f, v, s), data in per_run.items():
                    if f != family or v != L:
                        continue
                    right_key = (f, R, s)
                    if right_key not in per_run:
                        continue
                    deltas.append((data["correct"] - per_run[right_key]["correct"]).astype(np.int8))
                    seeds_used.append(s)
                if not deltas:
                    continue
                ci = paired_bootstrap_ci(deltas, n_bootstrap=args.n_bootstrap)
                comparisons.append({
                    "family": family, "label": lbl,
                    "left": L, "right": R,
                    "n_seed_pairs": len(deltas),
                    "seeds_used": sorted(seeds_used),
                    **ci,
                    "verdict": verdict(ci["mean_pp"], ci["ci_low_pp"], ci["ci_high_pp"]),
                })
    else:
        # Legacy kind-based pairing (kept for back-compat with old manifests).
        for family in families:
            ours_keys = [k for k in per_run if k[0] == family and per_run[k]["kind"] == "ours"]
            baseline_kinds = sorted({per_run[k]["kind"] for k in per_run
                                     if k[0] == family and per_run[k]["kind"] != "ours"})
            for bk in baseline_kinds:
                deltas = []
                for ok in ours_keys:
                    seed = ok[2]
                    bks = [k for k in per_run if k[0] == family and k[2] == seed and per_run[k]["kind"] == bk]
                    if not bks:
                        continue
                    deltas.append((per_run[ok]["correct"] - per_run[bks[0]]["correct"]).astype(np.int8))
                if not deltas:
                    continue
                ci = paired_bootstrap_ci(deltas, n_bootstrap=args.n_bootstrap)
                comparisons.append({
                    "family": family, "label": f"ours vs {bk}",
                    "left": "ours", "right": bk,
                    "n_seed_pairs": len(deltas),
                    **ci,
                    "verdict": verdict(ci["mean_pp"], ci["ci_low_pp"], ci["ci_high_pp"]),
                })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "per_run": {f"{f}.{v}.seed{s}": {"label": d["label"], "kind": d["kind"], "acc": d["acc"]}
                        for (f, v, s), d in per_run.items()},
            "comparisons": comparisons,
        }, f, indent=2)
    md_path = out_path.with_suffix(".md")
    lines = [
        "# Cross-modal eval — paired bootstrap 95% CI on real DV test set",
        f"Test modality: `{test_modality}`  ·  test indices from "
        f"`{Path(m['split_source_run']).name}`  ·  N test = "
        f"{len(test_indices)}  ·  bootstrap = {args.n_bootstrap}",
        "",
        "Reporting rule:",
        "- 95% CI entirely > 0 → significant positive effect",
        "- 95% CI entirely < 0 → significant negative effect",
        "- 95% CI crosses zero → no significant direction",
        "- effect size (mean Δ + CI) reported as-is, interpreted in context (e.g. vs "
        "the ~−76 pp sim→real gap); test-set-only CIs miss training-seed variance — "
        "use hierarchical_ci.py for cross-simulator claims",
        "",
    ]
    lines.append("| Family | Comparison | Left acc | Right acc | Mean Δ (pp) | 95% CI | n seeds | Verdict |")
    lines.append("|---|---|---:|---:|---:|---|---:|---|")
    for c in comparisons:
        # Aggregate left/right acc across seeds for context (mean across seeds).
        left_accs  = [d["acc"] for (f, v, _s), d in per_run.items() if f == c["family"] and v == c.get("left")]
        right_accs = [d["acc"] for (f, v, _s), d in per_run.items() if f == c["family"] and v == c.get("right")]
        left_mean  = (sum(left_accs)  / len(left_accs))  * 100 if left_accs  else float("nan")
        right_mean = (sum(right_accs) / len(right_accs)) * 100 if right_accs else float("nan")
        lines.append(
            f"| {c['family']} | {c.get('label', c.get('left','?')+' vs '+c.get('right','?'))} | "
            f"{left_mean:5.2f} | {right_mean:5.2f} | "
            f"{c['mean_pp']:+5.2f} | "
            f"[{c['ci_low_pp']:+5.2f}, {c['ci_high_pp']:+5.2f}] | "
            f"{c['n_seed_pairs']} | {c['verdict']} |"
        )
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[cross_modal_eval_with_ci] wrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
