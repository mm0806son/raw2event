"""Dump per-(family, variant, seed) per-sample correctness + predictions from a
cross-modal eval manifest, for offline hierarchical-CI and V02-audit analysis.

The cross-modal eval (``cross_modal_eval_with_ci.py``) recomputes per-sample
correctness via ``run_inference`` but only persists *aggregate* accuracy. Both
the hierarchical seed x prefix bootstrap and the V02 zero-variance audit need
the raw per-sample vectors, so this script re-runs the
same inference and saves everything to a single ``.npz`` (+ JSON sidecar).

Usage (on a GPU node):
    python -m tools.v2e_baseline.dump_eval_correctness \
        --manifest  <eval_output>/manifest.json \
        --output    <out>/correctness_<family>.npz \
        --test_data_dir ${DATA_DIR}/cifar10-xdvs_npz_ds346x260_unified80

Output ``.npz`` arrays:
    targets                      int16  (N,)   shared ground-truth labels
    test_indices                 int64  (N,)   canonical test split indices
    <family>.<V>.seed<k>__correct int8  (N,)   1 if prediction correct
    <family>.<V>.seed<k>__preds   int16 (N,)   predicted class
JSON sidecar (``<output>.meta.json``) maps each run key -> {label, kind, variant,
seed, acc, ckpt}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v2e_baseline.cross_modal_eval_with_ci import (
    load_test_indices,
    run_inference,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--test_data_dir", default=None,
                    help="Override manifest's test_data_dir (e.g. point at D36 unified80).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    runs: dict = manifest["runs"]
    model_family = manifest["model_family"]
    test_modality = manifest["test_modality"]
    test_data_dir = Path(args.test_data_dir or manifest["test_data_dir"])
    split_source_run = Path(manifest["split_source_run"])

    test_indices = load_test_indices(split_source_run)
    print(f"model_family={model_family}  test_modality={test_modality}  "
          f"N_test={len(test_indices)}  runs={len(runs)}")
    print(f"test_data_dir={test_data_dir}")

    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    targets_ref: np.ndarray | None = None

    for key, run in runs.items():
        res = run_inference(
            ckpt_path=Path(run["ckpt"]),
            test_data_dir=test_data_dir,
            test_indices=test_indices,
            device=args.device,
            batch_size=args.batch_size,
            test_modality=test_modality,
            model_family=model_family,
        )
        correct = np.asarray(res["correct"], dtype=np.int8)
        preds = np.asarray(res["preds"], dtype=np.int16)
        targets = np.asarray(res["targets"], dtype=np.int16)

        # All runs must share identical ground-truth ordering; assert to catch
        # split/index drift early (a silent mismatch would invalidate the audit).
        if targets_ref is None:
            targets_ref = targets
        elif not np.array_equal(targets_ref, targets):
            raise ValueError(f"targets mismatch for {key}: split ordering drifted")

        arrays[f"{key}__correct"] = correct
        arrays[f"{key}__preds"] = preds
        meta[key] = {
            "label": run.get("label"),
            "kind": run.get("kind"),
            "variant": run.get("variant"),
            "seed": run.get("seed"),
            "ckpt": run.get("ckpt"),
            "acc": float(correct.mean()),
        }
        print(f"  {key}: acc={correct.mean():.4f}")

    assert targets_ref is not None, "no runs in manifest"
    arrays["targets"] = targets_ref.astype(np.int16)
    arrays["test_indices"] = np.asarray(test_indices, dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    sidecar = args.output.with_suffix(args.output.suffix + ".meta.json")
    sidecar.write_text(json.dumps(
        {"model_family": model_family, "test_modality": test_modality,
         "n_test": len(test_indices), "runs": meta}, indent=2))
    print(f"[saved] {args.output}  ({len(arrays)} arrays)")
    print(f"[saved] {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
