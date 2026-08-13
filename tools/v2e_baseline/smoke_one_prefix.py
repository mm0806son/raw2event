"""Smoke test: run one prefix end-to-end through the v2e pipeline.

Validates that v2e CLI is callable, MKV reading + Bayer→Y works, AprilTag crop
returns sane events, and the output NPZ has the right schema. No GPU training.

Usage (user runs on GPU-equipped box):
  python tools/v2e_baseline/smoke_one_prefix.py \
    --variant V05 \
    --prefix 42868_horse_1_8698_20251228_234242 \
    --input_dir ./data \
    --output_dir output/diagnostics_20260501_v2e_compare/smoke

Expected: a single NPZ at output_dir/{prefix}_filtered_<rgb|rawY>.npz with
~10K-1M events, plus a v2e CLI log under output_dir/_logs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v2e_baseline.v2e_helpers import (  # noqa: E402
    VARIANT_SPEC, get_threshold_dict, process_one_prefix_v2e,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list(VARIANT_SPEC.keys()))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--keep_intermediate", action="store_true")
    ap.add_argument("--slomo_model", default="./external/SuperSloMo39.ckpt",
                    help="Only used for --variant=V09..V12 (slomo protocol).")
    args = ap.parse_args()

    spec = VARIANT_SPEC[args.variant]
    thresholds = get_threshold_dict(spec["threshold_set"], spec["input_modality"])
    print(f"variant={args.variant} spec={spec} thresholds={thresholds}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_suffix = "rawY" if spec["input_modality"] == "rawY" else "rgb"
    out_npz = out_dir / f"{args.prefix}_filtered_{npz_suffix}.npz"
    log_dir = out_dir / "_logs"

    stats = process_one_prefix_v2e(
        prefix=args.prefix,
        input_dir=args.input_dir,
        output_npz_path=out_npz,
        input_modality=spec["input_modality"],
        protocol=spec["protocol"],
        threshold_dict=thresholds,
        slomo_model=args.slomo_model if spec["protocol"] == "slomo" else None,
        log_dir=log_dir,
        keep_intermediate=args.keep_intermediate,
        n_workers=args.num_workers,
    )
    print("\nSMOKE OK")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput NPZ: {out_npz}  (size = {out_npz.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
