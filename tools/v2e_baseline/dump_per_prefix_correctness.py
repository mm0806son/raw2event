"""Dump per-recording 0/1 cross-modal correctness for the correlation analysis.

Runs every checkpoint in a manifest over the canonical test split and persists
the per-sample correctness vector alongside the recording names. The name order
replicates ``EventNpzDataset``'s own file ordering so the dump is reproducible
across machines. Consumed by ``within_prefix_correlation``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_MODALITY_SUFFIX = "_filtered_{modality}.npz"


def list_dataset_prefixes(data_dir: str | os.PathLike, modality: str) -> list[str]:
    """Return the deterministic prefix list matching ``EventNpzDataset(data_dir, modality)``.

    ``EventNpzDataset`` (train_class/train_utils/dataset.py:153) sorts the glob
    by the leading integer of the basename. We replicate exactly that order so
    indices line up byte-for-byte with whatever the inference helper uses.
    """
    pattern = os.path.join(str(data_dir), f"*{_MODALITY_SUFFIX.format(modality=modality)}")
    files = sorted(glob.glob(pattern), key=lambda f: int(os.path.basename(f).split("_")[0]))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern}")
    suffix = _MODALITY_SUFFIX.format(modality=modality)
    return [os.path.basename(f)[: -len(suffix)] for f in files]


def load_test_indices(split_source_run: Path) -> list[int]:
    info = json.loads((split_source_run / "split_info.json").read_text())
    return list(info["test_indices"])


def select_run_keys(manifest_runs: dict, include_variants: Iterable[str] | None,
                    include_keys: Iterable[str] | None) -> list[str]:
    keys = list(manifest_runs.keys())
    if include_keys:
        wanted = set(include_keys)
        keys = [k for k in keys if k in wanted]
    if include_variants:
        v_set = set(include_variants)
        keys = [k for k in keys if k.split(".")[1] in v_set]
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="Cross-modal eval manifest (same format consumed by "
                         "cross_modal_eval_with_ci.py).")
    ap.add_argument("--output", required=True,
                    help="Output JSON path (per_prefix_correctness.json).")
    ap.add_argument("--data_dir_override", default=None,
                    help="Override manifest['test_data_dir'] (e.g. when running "
                         "on dev with a local mirror at a different path).")
    ap.add_argument("--split_source_override", default=None,
                    help="Override manifest['split_source_run'] (rarely needed).")
    ap.add_argument("--include_variants", default=None,
                    help="Comma-separated subset of variant IDs to dump "
                         "(e.g. V01,V02,...,V08). Default: all in manifest.")
    ap.add_argument("--include_keys", default=None,
                    help="Comma-separated subset of run keys "
                         "(e.g. qkformer.V01.seed0,qkformer.V02.seed1).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    test_modality = manifest.get("test_modality", "dv")
    test_data_dir = args.data_dir_override or manifest["test_data_dir"]
    split_source = Path(args.split_source_override or manifest["split_source_run"])

    test_indices = load_test_indices(split_source)
    prefix_universe = list_dataset_prefixes(test_data_dir, test_modality)
    if max(test_indices) >= len(prefix_universe):
        raise IndexError(
            f"test_indices max {max(test_indices)} >= dataset size {len(prefix_universe)}; "
            f"data_dir / modality mismatch?"
        )
    test_prefixes = [prefix_universe[i] for i in test_indices]

    include_variants = (args.include_variants.split(",")
                        if args.include_variants else None)
    include_keys = (args.include_keys.split(",")
                    if args.include_keys else None)
    run_keys = select_run_keys(manifest["runs"], include_variants, include_keys)
    if not run_keys:
        raise SystemExit("no runs selected after filtering — check --include_*")

    # Lazy import: torch is GPU-side, dev shell may not have it. Importing only
    # after argparse means --help works in any environment.
    from train_class.evaluate_cross_modality import (  # type: ignore
        run_inference_on_indices,
    )

    per_run: dict[str, dict] = {}
    for key in run_keys:
        spec = manifest["runs"][key]
        family = key.split(".")[0]
        result = run_inference_on_indices(
            ckpt_path=spec["ckpt"],
            data_dir=str(test_data_dir),
            test_indices=test_indices,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            model_family=family,
            test_modality=test_modality,
        )
        correct = [int(c) for c in result["correct"]]
        per_run[key] = {
            "label": spec.get("label", ""),
            "kind": spec.get("kind", ""),
            "correct": correct,
            "acc": float(sum(correct) / max(len(correct), 1)),
            "n": len(correct),
        }
        print(f"  {key}: acc={per_run[key]['acc']:.4f}  n={per_run[key]['n']}")

    payload = {
        "test_modality": test_modality,
        "split_source_run": str(split_source),
        "test_data_dir": str(test_data_dir),
        "n_test": len(test_indices),
        "test_indices": test_indices,
        "prefixes": test_prefixes,
        "per_run": per_run,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[dump_per_prefix_correctness] wrote {out_path} "
          f"({len(per_run)} runs × {len(test_indices)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
