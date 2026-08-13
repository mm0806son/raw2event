"""Compute EQS for every simulator variant against matched real events.

Mirrors ``k_health_v2e_compare`` — same config, recording list, pairing and
nan-safe aggregation — but the per-pair metric is EQS. Needs a GPU and the RVT
backbone; those imports are deferred into ``main`` so ``--help`` works without
them. The real-event latent for each recording is extracted once and reused.

``--controls`` adds the validity battery: identity, a same-source temporal
split-half positive control, and polarity-flip / time-shuffle / coordinate-shuffle
lower bounds. If the positive controls do not clearly exceed the perturbation
bounds, the adapter is not discriminative on this data and the variant numbers
carry no information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _temporal_split(events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an event stream into two halves by timestamp (for the real-real UB)."""
    if events.shape[0] < 4:
        return events, events
    order = np.argsort(events[:, 0], kind="stable")
    ev = events[order]
    mid = ev.shape[0] // 2
    return ev[:mid], ev[mid:]


def _agg(rows: list[dict], key: str) -> dict:
    vals = np.array([r[key] for r in rows], dtype=float)
    return {
        f"{key}_mean": float(np.nanmean(vals)) if vals.size else float("nan"),
        f"{key}_median": float(np.nanmedian(vals)) if vals.size else float("nan"),
    }


def compute_eqs_table(
    extractor,
    cfg: dict,
    dv_dir: Path,
    prefixes: list[str],
    lfs_kw: dict,
    run_controls: bool,
):
    """Per-prefix EQS of every variant vs real DV, plus the optional controls.

    ``extractor`` is any callable ``events -> list[np.ndarray]`` (the RVT adapter
    in production, a stub in tests).  The real-DV latent is extracted ONCE per
    prefix and reused across all variants.  Fail-soft: a bad prefix/variant is
    recorded in ``skipped`` and skipped, so one corrupt NPZ never kills a
    559-prefix run; the latent-extraction import errors are NOT caught here so a
    genuine backbone/API failure still fails fast.

    Returns ``(per_variant, controls, skipped)``.
    """
    from tools.eqs.eqs_score import latent_feature_similarity
    from tools.eqs.perturbations import PERTURBATIONS

    per_variant: dict[str, list[dict]] = {v: [] for v in cfg}
    controls: dict[str, list[float]] = {
        "real_identity": [],
        "real_real_split": [],
        **{k: [] for k in PERTURBATIONS},
    }
    skipped: list[dict] = []

    for pf in prefixes:
        dv = dv_dir / f"{pf}_filtered_dv.npz"
        if not dv.exists():
            continue
        try:
            dv_ev = np.load(dv)["events"]
            if dv_ev.shape[0] == 0:
                continue
            dv_lat = extractor(dv_ev)  # real latent extracted ONCE; reused below
        except Exception as exc:  # noqa: BLE001 - fail-soft per prefix
            skipped.append({"prefix": pf, "variant": "<dv>", "error": repr(exc)})
            continue

        if run_controls:
            # real_identity is the true upper bound (must be ~1 if the adapter
            # works); real_real_split is a same-source positive control (two
            # temporal halves, re-binned independently, so < identity but still
            # well above the structure-destroying perturbation lower bounds).
            controls["real_identity"].append(
                latent_feature_similarity(dv_lat, dv_lat, **lfs_kw)[
                    "eqs_similarity_mean"
                ]
            )
            a, b = _temporal_split(dv_ev)
            controls["real_real_split"].append(
                latent_feature_similarity(extractor(a), extractor(b), **lfs_kw)[
                    "eqs_similarity_mean"
                ]
            )
            for name, fn in PERTURBATIONS.items():
                controls[name].append(
                    latent_feature_similarity(dv_lat, extractor(fn(dv_ev)), **lfs_kw)[
                        "eqs_similarity_mean"
                    ]
                )

        for variant, spec in cfg.items():
            sim = (
                Path(spec["npz_dir"])
                / f"{pf}_filtered_{spec.get('npz_suffix', 'rgb')}.npz"
            )
            if not sim.exists():
                continue
            try:
                sim_ev = np.load(sim)["events"]
                if sim_ev.shape[0] == 0:
                    continue
                res = latent_feature_similarity(dv_lat, extractor(sim_ev), **lfs_kw)
            except Exception as exc:  # noqa: BLE001 - fail-soft per (prefix, variant)
                skipped.append({"prefix": pf, "variant": variant, "error": repr(exc)})
                continue
            per_variant[variant].append({"prefix": pf, **res})

    return per_variant, controls, skipped


def summarise(per_variant: dict[str, list[dict]], cfg: dict) -> list[dict]:
    """Aggregate per-variant rows; ``n_eqs_used`` counts finite-EQS prefixes."""
    summary = []
    for variant, rows in per_variant.items():
        label = cfg[variant].get("label", variant)
        if not rows:
            summary.append(
                {"variant": variant, "label": label, "n_used": 0, "n_eqs_used": 0}
            )
            continue
        finite = int(np.isfinite([r["eqs_similarity_mean"] for r in rows]).sum())
        agg = {
            "variant": variant,
            "label": label,
            "n_used": len(rows),
            "n_eqs_used": finite,
        }
        for key in (
            "eqs_similarity_mean",
            "eqs_distance_mean",
            "zero_norm_patch_fraction",
        ):
            agg.update(_agg(rows, key))
        summary.append(agg)
    return summary


def summarise_controls(controls: dict[str, list[float]]) -> dict:
    """Control summary; ``n`` is the *finite* count, not the raw list length."""
    out = {}
    for k, v in controls.items():
        arr = np.array(v, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[k] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "n": int(finite.size),
        }
    return out


def render_md(summary: list[dict]) -> str:
    lines = [
        "# EQS (learned upstream realism) per variant vs real DV",
        "",
        "`eqs_sim` is mean cosine similarity of RVT stage-1/2/3 activations "
        "(higher = closer to real, matching the EQS authors' claim direction); "
        "`eqs_dist` = 1 - sim.  `znf` is the zero-norm patch fraction (sanity). "
        "Read alongside the control battery before trusting rankings.",
        "",
        "| Variant | Label | n_used | n_eqs | eqs_sim med | eqs_sim mean | eqs_dist med | znf mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        if s.get("n_used", 0) == 0:
            lines.append(f"| {s['variant']} | {s['label']} | 0 | 0 | — | — | — | — |")
            continue
        lines.append(
            f"| {s['variant']} | {s['label']} | {s['n_used']} | {s['n_eqs_used']} | "
            f"{s['eqs_similarity_mean_median']:.4f} | {s['eqs_similarity_mean_mean']:.4f} | "
            f"{s['eqs_distance_mean_median']:.4f} | {s['zero_norm_patch_fraction_mean']:.3f} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        required=True,
        help="JSON variant_id -> {npz_dir, npz_suffix, label}",
    )
    ap.add_argument("--dv_npz_dir", required=True)
    ap.add_argument("--prefix_list", required=True)
    ap.add_argument(
        "--output", required=True, help="Output JSON; .md sibling also written"
    )
    ap.add_argument(
        "--rvt_repo", required=True, help="Path to vendored RVT/ dir (from EQS repo)"
    )
    ap.add_argument(
        "--checkpoint", required=True, help="RVT-Gen1-small Lightning checkpoint"
    )
    ap.add_argument("--rvt_config", default="small")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--count_cutoff", type=int, default=10)
    ap.add_argument("--height", type=int, default=260)
    ap.add_argument("--width", type=int, default=346)
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument(
        "--zero_norm_policy", default="drop", choices=["drop", "eps", "ones"]
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--controls",
        action="store_true",
        help="Also run the real-real + perturbation sanity battery",
    )
    ap.add_argument(
        "--max_prefixes",
        type=int,
        default=0,
        help="Cap prefixes (0 = all; for smoke tests)",
    )
    args = ap.parse_args()

    # Deferred torch/RVT imports so --help works without torch (local sandbox).
    from tools.eqs.rvt_extractor import RVTStageExtractor, load_rvt_backbone

    lfs_kw = {"patch": args.patch, "zero_norm_policy": args.zero_norm_policy}

    with open(args.config) as f:
        cfg = json.load(f)
    with open(args.prefix_list) as f:
        prefixes = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if args.max_prefixes:
        prefixes = prefixes[: args.max_prefixes]
    print(f"[eqs_compare] {len(cfg)} variants x {len(prefixes)} prefixes")

    backbone = load_rvt_backbone(args.rvt_repo, args.checkpoint, args.rvt_config)
    extractor = RVTStageExtractor(
        backbone,
        bins=args.bins,
        height=args.height,
        width=args.width,
        count_cutoff=args.count_cutoff,
        device=args.device,
    )

    dv_dir = Path(args.dv_npz_dir)
    per_variant, controls, skipped = compute_eqs_table(
        extractor, cfg, dv_dir, prefixes, lfs_kw, args.controls
    )
    summary = summarise(per_variant, cfg)
    control_summary = summarise_controls(controls) if args.controls else None
    if skipped:
        print(f"[eqs_compare] fail-soft skipped {len(skipped)} (prefix, variant) pairs")

    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "rvt_repo": str(args.rvt_repo),
        "rvt_config": args.rvt_config,
        "bins": args.bins,
        "count_cutoff": args.count_cutoff,
        "height": args.height,
        "width": args.width,
        "patch": args.patch,
        "zero_norm_policy": args.zero_norm_policy,
        "stages": list(extractor.stages),
        "pad_multiple": extractor.pad_multiple,
        "n_prefixes": len(prefixes),
        "n_skipped": len(skipped),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "provenance": provenance,
                "controls": control_summary,
                "skipped": skipped,
                "raw": {v: r for v, r in per_variant.items()},
            },
            f,
            indent=2,
        )
    with open(out_path.with_suffix(".md"), "w") as f:
        f.write(render_md(summary) + "\n")
        if control_summary:
            f.write(
                "\n## Control battery (sanity: UB should exceed perturbation LBs)\n\n"
            )
            f.write("| Control | mean eqs_sim | n |\n|---|---:|---:|\n")
            for k, v in control_summary.items():
                f.write(f"| {k} | {v['mean']:.4f} | {v['n']} |\n")
    print(f"[eqs_compare] wrote {out_path} and {out_path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
