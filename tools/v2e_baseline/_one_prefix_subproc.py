"""Single-prefix v2e worker, used to isolate pupil_apriltags heap-corruption
SIGABRT crashes inside the training container. The parent process spawns
this script as a subprocess per prefix; if the subprocess SIGABRT-aborts,
the parent logs a failure and moves on to the next prefix instead of
losing the whole batch.

CLI:
  python -m tools.v2e_baseline._one_prefix_subproc \
    --prefix <prefix> \
    --input_dir ./data \
    --output_npz_path ./output/sim_branches/<variant>/{prefix}_filtered_<sfx>.npz \
    --input_modality {rgb|rawY} \
    --protocol {native50|slomo} \
    --threshold_json /tmp/thresholds.json \
    [--slomo_model ./external/SuperSloMo39.ckpt] \
    [--work_dir /tmp/v2e_work] [--log_dir ./logs/v2e] \
    [--keep_intermediate] [--n_workers 8] \
    [--result_json /tmp/result.json]   # writes process_one_prefix_v2e stats here

Exit codes: 0 on success, non-zero on any exception/abort.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v2e_baseline.v2e_helpers import process_one_prefix_v2e  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_npz_path", required=True)
    ap.add_argument("--input_modality", required=True, choices=["rgb", "rawY"])
    ap.add_argument("--protocol", required=True, choices=["native50", "slomo"])
    ap.add_argument("--threshold_json", required=True)
    ap.add_argument("--slomo_model", default=None)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--keep_intermediate", action="store_true")
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--result_json", default=None)
    args = ap.parse_args()

    with open(args.threshold_json) as f:
        threshold_dict = json.load(f)

    stats = process_one_prefix_v2e(
        prefix=args.prefix,
        input_dir=args.input_dir,
        output_npz_path=args.output_npz_path,
        input_modality=args.input_modality,
        protocol=args.protocol,
        threshold_dict=threshold_dict,
        slomo_model=args.slomo_model,
        work_dir=args.work_dir,
        log_dir=args.log_dir,
        keep_intermediate=args.keep_intermediate,
        n_workers=args.n_workers,
    )
    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(stats, f)
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
