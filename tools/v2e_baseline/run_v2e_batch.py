"""Batch v2e event generation for one variant (V05..V12).

Designed for manual splitting across cluster GPUs:
  --index_start / --index_end : closed/open prefix index window
  --gpu_id                    : sets CUDA_VISIBLE_DEVICES for the v2e subprocess
  --skip_existing             : resume; skip prefixes whose output NPZ already exists
  --keep_intermediate         : don't delete temp v2e working dirs (for debugging)
  --num_workers               : threads for AprilTag detection (per-prefix)

The user manually splits the prefix list across several jobs, e.g.:
  GPU 0:  --index_start=0     --index_end=1500
  GPU 1:  --index_start=1500  --index_end=3000
  GPU 2:  --index_start=3000  --index_end=4500
  GPU 3:  --index_start=4500  --index_end=5914

Output NPZ schema is identical to Raw2Event's *_filtered_*.npz:
  events: (N, 4) int64 / float, columns [t_us, x, y, p in {0, 1}]

After all four V05/V06/V07/V08 outputs land, run rescale_npz_to_unified.py
separately to produce the unified80 dataset for downstream training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v2e_baseline.v2e_helpers import (  # noqa: E402
    VARIANT_SPEC,
    get_threshold_dict,
    is_valid_filtered_npz,
    process_one_prefix_v2e_subprocess,
)


def parse_prefix_list(path: str | Path) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    # Sort by leading int for deterministic indexing (matches Raw2Event convention)
    def _key(p: str) -> tuple[int, str]:
        try:
            return int(p.split("_", 1)[0]), p
        except ValueError:
            return 10**9, p
    return sorted(out, key=_key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list(VARIANT_SPEC.keys()),
                    help="Variant ID; see VARIANT_SPEC for the twelve-variant matrix")
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing Pi RGB/RAW MKVs and metadata files (e.g. ./data)")
    ap.add_argument("--output_dir", required=True,
                    help="Output dir for *_filtered_<modality>.npz")
    ap.add_argument("--prefix_list", required=True,
                    help="Text file with one prefix per line")
    ap.add_argument("--index_start", type=int, default=0,
                    help="Inclusive start index into the (sorted) prefix list")
    ap.add_argument("--index_end", type=int, default=-1,
                    help="Exclusive end index; -1 means end of list")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip prefixes whose output NPZ already exists")
    ap.add_argument("--keep_intermediate", action="store_true",
                    help="Don't delete temp v2e dirs (debug)")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="Threads for AprilTag detection (per prefix)")
    ap.add_argument("--gpu_id", type=int, default=None,
                    help="CUDA device ID; sets CUDA_VISIBLE_DEVICES (None = inherit)")
    ap.add_argument("--slomo_model", default="./external/SuperSloMo39.ckpt",
                    help="Path to SuperSloMo .ckpt; only used for slomo protocol. "
                         "Default points to the local external/ copy.")
    ap.add_argument("--work_root", default=None,
                    help="Per-prefix temp work dir parent; default = system tmp")
    ap.add_argument("--log_dir", default=None,
                    help="Where to dump per-prefix v2e CLI logs (default = output_dir/_logs)")
    ap.add_argument("--manifest_jsonl", default=None,
                    help="Append per-prefix run stats here (default = output_dir/_manifest.jsonl)")
    args = ap.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    spec = VARIANT_SPEC[args.variant]
    input_modality = spec["input_modality"]
    protocol = spec["protocol"]
    threshold_set = spec["threshold_set"]
    thresholds = get_threshold_dict(threshold_set, input_modality)
    print(f"[run_v2e_batch] variant={args.variant} spec={spec}")
    print(f"[run_v2e_batch] thresholds={thresholds}")

    if protocol == "slomo":
        slomo_path = Path(args.slomo_model)
        if not slomo_path.exists():
            print(f"FATAL: --slomo_model not found at {slomo_path}; run download_slomo_ckpt.sh", file=sys.stderr)
            return 2
    else:
        slomo_path = None

    prefixes = parse_prefix_list(args.prefix_list)
    total = len(prefixes)
    end = total if args.index_end < 0 else min(args.index_end, total)
    if args.index_start < 0 or args.index_start >= end:
        print(f"FATAL: invalid index window [{args.index_start}, {end}) on list of {total}", file=sys.stderr)
        return 2
    window = prefixes[args.index_start:end]
    print(f"[run_v2e_batch] processing {len(window)} prefixes [{args.index_start}, {end}) of {total} total")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_jsonl) if args.manifest_jsonl else output_dir / "_manifest.jsonl"

    npz_suffix = "rawY" if input_modality == "rawY" else "rgb"

    t_start = time.time()
    n_done = n_skip = n_fail = 0
    for i, prefix in enumerate(window):
        out_npz = output_dir / f"{prefix}_filtered_{npz_suffix}.npz"
        if args.skip_existing and is_valid_filtered_npz(out_npz):
            n_skip += 1
            continue
        try:
            stats = process_one_prefix_v2e_subprocess(
                prefix=prefix,
                input_dir=args.input_dir,
                output_npz_path=out_npz,
                input_modality=input_modality,
                protocol=protocol,
                threshold_dict=thresholds,
                slomo_model=str(slomo_path) if slomo_path else None,
                work_dir=Path(args.work_root) / f"{args.variant}_{prefix}" if args.work_root else None,
                log_dir=log_dir,
                keep_intermediate=args.keep_intermediate,
                n_workers=args.num_workers,
            )
            stats["variant"] = args.variant
            stats["abs_index"] = args.index_start + i
            with open(manifest_path, "a") as mf:
                mf.write(json.dumps(stats) + "\n")
            n_done += 1
            elapsed = time.time() - t_start
            print(
                f"[{i+1}/{len(window)}] {prefix} ev={stats['n_events_v2e']:>9d} "
                f"-> crop={stats['n_events_after_crop']:>8d} box={stats['box_size']:3d} "
                f"t={stats['time_seconds']:.1f}s (cum {elapsed/60:.1f}m, "
                f"~{elapsed/(n_done):.1f}s/prefix)",
                flush=True,
            )
        except Exception as exc:
            n_fail += 1
            print(f"[{i+1}/{len(window)}] {prefix} FAILED: {exc}", file=sys.stderr, flush=True)
            with open(manifest_path, "a") as mf:
                mf.write(json.dumps({
                    "variant": args.variant, "prefix": prefix,
                    "abs_index": args.index_start + i,
                    "error": str(exc),
                }) + "\n")

    print(
        f"[run_v2e_batch] DONE variant={args.variant} window=[{args.index_start},{end}) "
        f"done={n_done} skip={n_skip} fail={n_fail} elapsed={(time.time()-t_start)/60:.1f}m"
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
