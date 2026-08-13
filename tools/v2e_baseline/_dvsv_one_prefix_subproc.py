"""Single-prefix subprocess worker for V03/V04 (Raw + DVS-Voltmeter
default K). Mirror of _one_prefix_subproc.py but for the dvsv path.
Isolates pupil_apriltags SIGABRT crashes per prefix so the parent
batch driver survives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v2e_baseline.raw_dvsv_default_k_gen import process_one_prefix_dvsv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--variant", required=True, choices=["V03", "V04"])
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--sim_backend", default="auto", choices=["auto", "cuda", "cpu", "numpy"])
    ap.add_argument("--result_json", default=None)
    args = ap.parse_args()

    stats = process_one_prefix_dvsv(
        prefix=args.prefix,
        variant=args.variant,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        sim_backend=args.sim_backend,
    )
    if args.result_json:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(stats, f)
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
