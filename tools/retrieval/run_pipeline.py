"""End-to-end driver for the cross-modal retrieval evaluation.

Extracts penultimate embeddings for every (checkpoint, event-source) pair, runs
cosine retrieval per variant and seed, then builds the manifest for the bootstrap
CI. A thin orchestrator over three single-purpose entry points, each of which can
also be run alone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Comparison spec — frozen. Each entry is (left_variant, right_variant).
# Left / right are interpreted symmetrically; the table reads "Δ = left - right".
# ---------------------------------------------------------------------------
COMPARISONS: list[tuple[str, str]] = [
    ("V01", "V07"),  # Raw2Event-RAW  vs v2e-raw-default  (both raw simulators)
    ("V01", "V08"),  # Raw2Event-RAW  vs v2e-raw-tuned
    ("V02", "V05"),  # Raw2Event-RGB  vs v2e-rgb-default
    ("V02", "V06"),  # Raw2Event-RGB  vs v2e-rgb-tuned
    ("V01", "V03"),  # Raw2Event-RAW  vs DVS-V default-K (raw)
    ("V02", "V04"),  # Raw2Event-RGB  vs DVS-V default-K (rgb)
    ("V01", "DV"),   # Raw2Event-RAW  vs real-DV upper bound
    ("V02", "DV"),   # Raw2Event-RGB  vs real-DV upper bound
]

# Self-retrieval upper-bound rows — give cross-modal numbers a reference scale.
SELF_BASELINES: list[str] = ["V01", "V02", "DV"]

# Variant → modality (matches the QKFormer submit-script layout).
VARIANT_MODALITY: dict[str, str] = {
    "V01": "raw", "V02": "rgb", "V03": "raw", "V04": "rgb",
    "V05": "rgb", "V06": "rgb", "V07": "raw", "V08": "raw",
    "DV":  "dv",
}

SEEDS = (0, 1, 2)


def find_best_ckpt(output_root: Path, variant: str, seed: int) -> Path:
    """Locate the canonical best.pth for a (variant, seed).

    Layout (from the QKFormer submit script):
       {output_root}/qkf_{variant}_{modality}_s{seed}/QKFormer_{modality}_scratch_T{T}_{TS}/output_{modality}_scratch_best.pth
    Multiple training runs may have produced multiple timestamped subdirs;
    we pick the most recent one (sort by name → latest TS wins)."""
    modality = VARIANT_MODALITY[variant]
    run_dir = output_root / f"qkf_{variant}_{modality}_s{seed}"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"missing run dir: {run_dir}")
    candidates = sorted(run_dir.glob(f"QKFormer_{modality}_scratch_T*_*"))
    if not candidates:
        raise FileNotFoundError(f"no QKFormer_*_scratch_T*_* under {run_dir}")
    latest = candidates[-1]
    ckpt = latest / f"output_{modality}_scratch_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing best.pth: {ckpt}")
    return ckpt


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(f"[run] {' '.join(str(c) for c in cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(c) for c in cmd], check=True)


def stage_a_extract_all(
    output_root: Path,
    real_dv_dir: Path,
    sim_dirs: dict[str, Path],
    prefix_list: Path,
    embeddings_dir: Path,
    device: str,
    batch_size: int,
    limit: int,
    dry_run: bool,
) -> dict[tuple[str, int, str], Path]:
    """Drive extract_embeddings.py for every (variant, seed, source) we need.

    Returns a dict keyed by (variant, seed, source) → embedding NPZ path,
    where ``source`` ∈ {"real_dv", "own_sim"}."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, int, str], Path] = {}
    extract_cli = [sys.executable, "-m", "tools.retrieval.extract_embeddings"]

    sim_variants = [v for v in VARIANT_MODALITY if v != "DV"]
    for variant in sim_variants:
        for seed in SEEDS:
            ckpt = find_best_ckpt(output_root, variant, seed)
            mod = VARIANT_MODALITY[variant]

            # (a) real-DV encoded by this variant's ckpt → query.
            tag_q = f"{variant}_s{seed}__realDV"
            out_q = embeddings_dir / f"{tag_q}.npz"
            cmd = extract_cli + [
                "--ckpt", ckpt, "--data_dir", real_dv_dir, "--modality", "dv",
                "--prefix_list", prefix_list, "--variant", variant, "--seed", seed,
                "--output", out_q, "--device", device, "--batch_size", batch_size,
            ]
            if limit:
                cmd += ["--limit", limit]
            run(cmd, dry_run=dry_run)
            out[(variant, seed, "real_dv")] = out_q

            # (b) own-sim encoded by this variant's ckpt → gallery.
            sim_dir = sim_dirs[variant]
            tag_g = f"{variant}_s{seed}__ownSim"
            out_g = embeddings_dir / f"{tag_g}.npz"
            cmd = extract_cli + [
                "--ckpt", ckpt, "--data_dir", sim_dir, "--modality", mod,
                "--prefix_list", prefix_list, "--variant", variant, "--seed", seed,
                "--output", out_g, "--device", device, "--batch_size", batch_size,
            ]
            if limit:
                cmd += ["--limit", limit]
            run(cmd, dry_run=dry_run)
            out[(variant, seed, "own_sim")] = out_g

    # DV row: real-DV encoded by DV ckpt (one NPZ per seed; serves as both
    # query and gallery in the DV row → same-prefix mask drops self-hits).
    for seed in SEEDS:
        ckpt = find_best_ckpt(output_root, "DV", seed)
        tag = f"DV_s{seed}__realDV"
        out_p = embeddings_dir / f"{tag}.npz"
        cmd = extract_cli + [
            "--ckpt", ckpt, "--data_dir", real_dv_dir, "--modality", "dv",
            "--prefix_list", prefix_list, "--variant", "DV", "--seed", seed,
            "--output", out_p, "--device", device, "--batch_size", batch_size,
        ]
        if limit:
            cmd += ["--limit", limit]
        run(cmd, dry_run=dry_run)
        out[("DV", seed, "real_dv")] = out_p

    return out


def stage_b_eval_per_row(
    embeddings: dict[tuple[str, int, str], Path],
    metrics_dir: Path,
    dry_run: bool,
) -> dict[tuple[str, int], Path]:
    """Run eval_retrieval.py for each (variant, seed) row used in any
    comparison or self-baseline. Returns dict (variant, seed) → per_query NPZ."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    eval_cli = [sys.executable, "-m", "tools.retrieval.eval_retrieval"]
    out: dict[tuple[str, int], Path] = {}

    needed_variants: set[str] = set()
    for (l, r) in COMPARISONS:
        needed_variants.update([l, r])
    needed_variants.update(SELF_BASELINES)

    for variant in sorted(needed_variants):
        for seed in SEEDS:
            if variant == "DV":
                q = embeddings[(variant, seed, "real_dv")]
                g = embeddings[(variant, seed, "real_dv")]   # same NPZ; self mask handles it
            else:
                q = embeddings[(variant, seed, "real_dv")]
                g = embeddings[(variant, seed, "own_sim")]
            out_p = metrics_dir / f"row_{variant}_s{seed}.npz"
            run([*eval_cli, "--query", q, "--gallery", g, "--output", out_p],
                dry_run=dry_run)
            out[(variant, seed)] = out_p
    return out


def stage_c_write_manifest_and_ci(
    per_row: dict[tuple[str, int], Path],
    output_root: Path,
    n_bootstrap: int,
    dry_run: bool,
) -> tuple[Path, Path, Path]:
    manifest = {
        "comparisons": [
            {
                "name": f"{l}_vs_{r}",
                "left":  {"variant": l, "modality": VARIANT_MODALITY[l],
                          "per_query_npz": [str(per_row[(l, s)]) for s in SEEDS]},
                "right": {"variant": r, "modality": VARIANT_MODALITY[r],
                          "per_query_npz": [str(per_row[(r, s)]) for s in SEEDS]},
            }
            for (l, r) in COMPARISONS
        ],
        "self_baselines": [
            {"name": f"{v}_self", "variant": v, "modality": VARIANT_MODALITY[v],
             "per_query_npz": [str(per_row[(v, s)]) for s in SEEDS]}
            for v in SELF_BASELINES
        ],
    }
    manifest_path = output_root / "manifest.json"
    json_path = output_root / "retrieval_ci.json"
    md_path = output_root / "retrieval_ci.md"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[orchestrate] wrote manifest → {manifest_path}", flush=True)

    ci_cli = [sys.executable, "-m", "tools.retrieval.compute_retrieval_ci",
              "--manifest", manifest_path, "--output_json", json_path,
              "--output_md", md_path, "--n_bootstrap", n_bootstrap]
    run(ci_cli, dry_run=dry_run)
    return manifest_path, json_path, md_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_root", type=Path, required=True,
                    help="root containing qkf_{variant}_{modality}_s{seed}/ subdirs "
                         "(e.g. ./train_class/output)")
    ap.add_argument("--real_dv_dir", type=Path, required=True,
                    help="real Davis346 NPZ root (modality=dv)")
    ap.add_argument("--sim_dir_map", type=Path, required=True,
                    help="JSON dict { 'V01': '/abs/path/to/v01/sim/npz', ... }; "
                         "absolute paths because V01/V02 live under unified80/ "
                         "while V03..V08 live under v2e_compare/")
    ap.add_argument("--prefix_list", type=Path, required=True,
                    help="canonical prefix list (one prefix per line)")
    ap.add_argument("--output_root", type=Path, required=True,
                    help="dir for embeddings/, metrics/, retrieval_ci.{json,md}")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: cap each variant to N prefixes (0 = full 559)")
    ap.add_argument("--dry_run", action="store_true",
                    help="echo commands but do not execute (useful for dry checks)")
    args = ap.parse_args()

    sim_map_raw = json.loads(args.sim_dir_map.read_text())
    # Skip JSON "comment" keys (a leading underscore is the conventional
    # marker; the batch preflight uses the same filter).
    sim_dirs = {v: Path(sim_map_raw[v]) for v in sim_map_raw if not v.startswith("_")}
    missing = [v for v in VARIANT_MODALITY if v != "DV" and v not in sim_dirs]
    if missing:
        raise KeyError(f"sim_dir_map missing variants: {missing}")
    for v, p in sim_dirs.items():
        if not args.dry_run and not p.is_dir():
            raise NotADirectoryError(f"sim variant dir for {v!r} missing: {p}")

    embeddings_dir = args.output_root / "embeddings"
    metrics_dir = args.output_root / "metrics"

    print("=" * 72)
    print("[orchestrate] STAGE A — extract penultimate embeddings")
    print("=" * 72, flush=True)
    embeddings = stage_a_extract_all(
        args.ckpt_root, args.real_dv_dir, sim_dirs,
        args.prefix_list, embeddings_dir,
        args.device, args.batch_size, args.limit, args.dry_run,
    )

    print("=" * 72)
    print("[orchestrate] STAGE B — per-row cosine retrieval metrics")
    print("=" * 72, flush=True)
    per_row = stage_b_eval_per_row(embeddings, metrics_dir, args.dry_run)

    print("=" * 72)
    print("[orchestrate] STAGE C — paired bootstrap CI + markdown")
    print("=" * 72, flush=True)
    manifest_path, json_path, md_path = stage_c_write_manifest_and_ci(
        per_row, args.output_root, args.n_bootstrap, args.dry_run,
    )

    print()
    print(f"[orchestrate] DONE")
    print(f"  manifest: {manifest_path}")
    print(f"  json    : {json_path}")
    print(f"  markdown: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
