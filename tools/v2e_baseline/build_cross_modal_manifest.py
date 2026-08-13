"""Build the JSON manifest consumed by tools/v2e_baseline/cross_modal_eval_with_ci.py.

Scans a `train_class/output/` tree for QKFormer (or MNet) runs matching
the qkf_<VARIANT>_<modality>_s<seed>/ pattern, locates each run's
best.pth, and emits a manifest mapping run keys to ckpt paths.

Usage:
    python -m tools.v2e_baseline.build_cross_modal_manifest \\
        --output_root ./train_class/output \\
        --test_data_dir ./data/unified80 \\
        --split_source_run ./train_class/output/qkf_DV_dv_s0/<run_dir> \\
        --output cross_modal_manifest.json

`kind` is auto-assigned per variant (V01/V02 = ours, V03..V12 = baseline-<family>).
For paired bootstrap CI to fire, you need at least one `ours` run
and one `<other-kind>` run sharing the same model_family + seed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VARIANT_KIND = {
    "V01": "ours_raw",       # Raw2Event Stage 1.5 K, raw
    "V02": "ours_rgb",       # Raw2Event Stage 1.5 K, rgb
    "DV":  "real_dv",        # Davis346 ground truth
    "V03": "dvsv_default_raw",
    "V04": "dvsv_default_rgb",
    "V05": "v2e_native50_default_rgb",
    "V06": "v2e_native50_tuned_rgb",
    "V07": "v2e_native50_default_raw",
    "V08": "v2e_native50_tuned_raw",
    "V09": "v2e_slomo_default_rgb",
    "V10": "v2e_slomo_tuned_rgb",
    "V11": "v2e_slomo_default_raw",
    "V12": "v2e_slomo_tuned_raw",
}

VARIANT_LABEL = {
    "V01": "Raw2Event-RAW (Stage1.5 K)",
    "V02": "Raw2Event-RGB (Stage1.5 K)",
    "DV":  "Davis346 ground truth",
    "V03": "DVS-V default K (raw)",
    "V04": "DVS-V default K (rgb)",
    "V05": "v2e native50 default (rgb)",
    "V06": "v2e native50 tuned (rgb)",
    "V07": "v2e native50 default (raw)",
    "V08": "v2e native50 tuned (raw)",
    "V09": "v2e slomo default (rgb)",
    "V10": "v2e slomo tuned (rgb)",
    "V11": "v2e slomo default (raw)",
    "V12": "v2e slomo tuned (raw)",
}

# Each row is (left_variant, right_variant, label); the delta is left - right.
# A direction is significant iff the 95% CI does not cross zero.
DEFAULT_COMPARISONS = [
    # Raw2Event vs v2e
    ("V01", "V07", "Raw2Event-RAW vs v2e-RAW-default"),
    ("V01", "V08", "Raw2Event-RAW vs v2e-RAW-tuned"),
    ("V02", "V05", "Raw2Event-RGB vs v2e-RGB-default"),
    ("V02", "V06", "Raw2Event-RGB vs v2e-RGB-tuned"),
    # K-calibration ablation rows
    ("V01", "V03", "Raw2Event-RAW (Stage1.5 K) vs DVS-V default K (raw)"),
    ("V02", "V04", "Raw2Event-RGB (Stage1.5 K) vs DVS-V default K (rgb)"),
    # Sim-vs-real anchor (Raw2Event-trained vs Davis346-GT-trained on real DV test)
    ("V01", "DV",  "Raw2Event-RAW vs Davis346 GT (sim-vs-real, raw)"),
    ("V02", "DV",  "Raw2Event-RGB vs Davis346 GT (sim-vs-real, rgb)"),
]

# Match both QKFormer (`qkf_*`) and MobileNetV2 (`mnet_*`) run directories.
# Naming is shared between the QKFormer and MobileNetV2 submit layouts:
# same {variant, modality, seed} schema, only the family prefix differs.
# Group 1 = family prefix (qkf|mnet) so callers can filter by --model_family.
RUN_RE = re.compile(r"^(qkf|mnet)_(V\d+|DV)_(\w+)_s(\d+)$")

# Map run dir-name prefix → --model_family value, so the manifest
# builder never writes a qkformer ckpt into a mobilenetv2.* manifest key.
PREFIX_TO_FAMILY = {"qkf": "qkformer", "mnet": "mobilenetv2"}


# Extract the augmentation mode encoded in a MobileNetV2 run subdir name.
# The trainer writes ``MobileNetV2_<mod>_<mode_tag>_<rep_tag>[_aug-<aug>]_<ts>``
# where <ts> is ``YYYYMMDD_HHMMSS`` (train_mobileNetV2.py:main). A baseline
# (augmentation="none") subdir has no ``_aug-`` infix.
_AUG_SUBDIR_RE = re.compile(r"_aug-(.+)_\d{8}_\d{6}$")


def _subdir_augmentation(subdir_name: str) -> str:
    """Return the augmentation mode of a run subdir ('none' if not augmented)."""
    m = _AUG_SUBDIR_RE.search(subdir_name)
    return m.group(1) if m else "none"


def find_best_ckpt(
    run_dir: Path,
    representation: str | None = None,
    augmentation: str = "none",
) -> Path | None:
    """Return the most recent matching ``*_best.pth`` under ``run_dir``, or None.

    When ``representation`` is set (mobilenetv2 only), restrict the search
    to the matching per-run subdir prefix so a single ``mnet_*`` directory
    that holds both timestack and stacked_histogram runs disambiguates
    cleanly. Subdir naming convention is set by
    ``train_class/train_mobileNetV2.py:main`` (``rep_tag``).

    ``augmentation`` pins which training augmentation the picked checkpoint was
    trained with. Baseline and ``_aug-<mode>_`` runs coexist as sibling
    directories, so filtering on the exact mode rather than mtime keeps the
    selection independent of retrain order.
    """
    if representation == "stacked_histogram":
        subdir_glob = "MobileNetV2_*_STH_*/output_*_scratch_best.pth"
    elif representation == "timestack":
        subdir_glob = "MobileNetV2_*_TS*/output_*_scratch_best.pth"
    else:
        subdir_glob = "**/output_*_scratch_best.pth"
    candidates = sorted(run_dir.glob(subdir_glob),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [
        c for c in candidates
        if _subdir_augmentation(c.parent.name) == augmentation
    ]
    return candidates[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output_root", required=True,
                    help="Directory containing qkf_<V>_<mod>_s<seed>/ subdirs")
    ap.add_argument("--test_data_dir", required=True,
                    help="Directory with canonical test NPZ (must contain *_filtered_<test_modality>.npz)")
    ap.add_argument("--split_source_run", required=True,
                    help="A run dir whose split_info.json defines the canonical test indices")
    ap.add_argument("--test_modality", default="dv",
                    choices=["dv", "raw", "rgb"],
                    help="Which modality's NPZ to evaluate on (default: dv = real Davis346)")
    ap.add_argument("--model_family", default="qkformer",
                    choices=["qkformer", "mobilenetv2"])
    ap.add_argument("--representation", default=None,
                    choices=[None, "timestack", "stacked_histogram"],
                    help="(mobilenetv2 only) restrict ckpt search to the matching "
                         "per-run subdir prefix. None = pick latest of any "
                         "representation. Use to keep timestack and stacked_histogram "
                         "tables separate.")
    ap.add_argument("--augmentation", default="none",
                    help="Training augmentation the picked ckpt must have been "
                         "trained with (subdir '_aug-<mode>_' infix). 'none' "
                         "(default) = baseline ckpts only. Use 'eventdrop' etc. "
                         "to build a manifest over the augmented matrix. Keeps "
                         "baseline-eval and aug-eval reproducibly separated.")
    ap.add_argument("--ours_kind_override", default=None,
                    help="If set, mark V01 as kind=<this> (e.g. 'ours') so CI script "
                         "treats it as the reference. Default: per-variant kind tags above.")
    ap.add_argument("--output", required=True, help="Manifest JSON output path")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_dir():
        raise SystemExit(f"FATAL: output_root not found: {output_root}")

    runs = {}
    skipped = []
    for sub in sorted(output_root.iterdir()):
        if not sub.is_dir():
            continue
        m = RUN_RE.match(sub.name)
        if not m:
            continue
        # Hard filter on the dir-name family prefix: a `qkf_*` directory
        # is not a valid mobilenetv2 manifest source, and vice versa.
        # Without this guard a single output tree containing both qkformer
        # and mobilenet runs would silently bleed into the same manifest
        # under whichever `--model_family` is currently selected.
        prefix = m.group(1)
        if PREFIX_TO_FAMILY.get(prefix) != args.model_family:
            continue
        variant, _modality, seed = m.group(2), m.group(3), int(m.group(4))
        ckpt = find_best_ckpt(
            sub, representation=args.representation, augmentation=args.augmentation
        )
        if ckpt is None:
            skipped.append((sub.name, "no_best_pth"))
            continue
        kind = VARIANT_KIND.get(variant, "unknown")
        if args.ours_kind_override and variant == "V01":
            kind = args.ours_kind_override
        key = f"{args.model_family}.{variant}.seed{seed}"
        runs[key] = {
            "ckpt": str(ckpt.resolve()),
            "label": VARIANT_LABEL.get(variant, variant),
            "kind": kind,
            "variant": variant,
            "seed": seed,
        }

    # Filter DEFAULT_COMPARISONS to those where both sides have ≥1 run available.
    available_variants = {r["variant"] for r in runs.values()}
    comparisons = [
        {"left": L, "right": R, "label": lbl}
        for L, R, lbl in DEFAULT_COMPARISONS
        if L in available_variants and R in available_variants
    ]

    manifest = {
        "test_modality": args.test_modality,
        "test_data_dir": str(Path(args.test_data_dir).resolve()),
        "split_source_run": str(Path(args.split_source_run).resolve()),
        "model_family": args.model_family,
        "comparisons": comparisons,
        "runs": runs,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {out} with {len(runs)} runs across "
          f"{len(available_variants)} variants")
    print(f"[manifest] {len(comparisons)} paired comparisons enabled "
          f"(filtered from {len(DEFAULT_COMPARISONS)} declared rows):")
    for c in comparisons:
        print(f"    {c['left']:<4} vs {c['right']:<4}  — {c['label']}")
    if skipped:
        print(f"[manifest] skipped runs: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
