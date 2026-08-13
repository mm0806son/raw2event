"""Rescale per-modality event coordinates onto a shared spatial box.

Each modality leaves ``process_single_batch`` in its own native coordinate box,
so the downstream bilinear resize would apply a different effective scale per
modality. This maps every stream onto one target box beforehand. Center
alignment is preserved because the events were placed with the tag center at
``box_size / 2``.
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import numpy as np
from tqdm import tqdm

MODALITIES = ("dv", "raw", "rgb")


def rescale_one_file(src_path: str, dst_path: str, target_size: int) -> dict:
    """Rescale one NPZ file. Returns stats dict for logging."""
    data = np.load(src_path)
    ev = data["events"]
    extra_keys = {k: data[k] for k in data.files if k != "events"}

    if len(ev) == 0:
        np.savez_compressed(dst_path, events=ev, **extra_keys)
        return {
            "src": src_path,
            "n_events": 0,
            "native_box": 0,
            "target": target_size,
        }

    x = ev[:, 1].astype(np.int64)
    y = ev[:, 2].astype(np.int64)
    native_box = int(max(x.max(), y.max())) + 1

    if native_box == target_size:
        # No-op rescale; just clip defensively.
        x_new = np.clip(x, 0, target_size - 1)
        y_new = np.clip(y, 0, target_size - 1)
    else:
        scale = target_size / native_box
        x_new = np.clip(np.floor(x * scale).astype(np.int64), 0, target_size - 1)
        y_new = np.clip(np.floor(y * scale).astype(np.int64), 0, target_size - 1)

    ev_new = ev.copy()
    ev_new[:, 1] = x_new
    ev_new[:, 2] = y_new

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.savez_compressed(dst_path, events=ev_new, **extra_keys)

    return {
        "src": src_path,
        "n_events": len(ev),
        "native_box": native_box,
        "target": target_size,
    }


def _worker(args):
    src, dst, target_size = args
    try:
        return rescale_one_file(src, dst, target_size)
    except Exception as exc:
        return {"src": src, "error": str(exc)}


def collect_files(src_dir: str, prefix_filter: str = None) -> list:
    """Return list of NPZ file paths under src_dir for all modalities."""
    files = []
    for mod in MODALITIES:
        if prefix_filter:
            pattern = os.path.join(src_dir, f"{prefix_filter}*_filtered_{mod}.npz")
        else:
            pattern = os.path.join(src_dir, f"*_filtered_{mod}.npz")
        files.extend(sorted(glob.glob(pattern)))
    return files


def smoke_test(src_dir: str, dst_dir: str, prefix: str, target_size: int):
    """Run rescale on a single prefix and print before/after stats."""
    print(f"=== Smoke test: prefix '{prefix}', target_size={target_size} ===")
    files = collect_files(src_dir, prefix_filter=prefix)
    if not files:
        print(f"ERROR: no files matching prefix '{prefix}' in {src_dir}")
        return 1

    for src in files:
        rel = os.path.basename(src)
        dst = os.path.join(dst_dir, rel)
        before = np.load(src)["events"]
        info = rescale_one_file(src, dst, target_size)
        after = np.load(dst)["events"]

        print(f"\n--- {rel} ---")
        print(f"  events: {info['n_events']}")
        print(f"  native_box (detected): {info['native_box']}")
        print(f"  target_size: {target_size}")
        if len(before) > 0:
            print(f"  before: x in [{before[:,1].min()}, {before[:,1].max()}], "
                  f"y in [{before[:,2].min()}, {before[:,2].max()}]")
            print(f"  after:  x in [{after[:,1].min()}, {after[:,1].max()}], "
                  f"y in [{after[:,2].min()}, {after[:,2].max()}]")
            # Polarity preserved?
            pol_before = np.bincount(before[:, 3].astype(np.int64), minlength=2)
            pol_after = np.bincount(after[:, 3].astype(np.int64), minlength=2)
            print(f"  polarity (neg, pos): before={tuple(pol_before)}, after={tuple(pol_after)}")
            # Timestamps unchanged?
            assert np.array_equal(before[:, 0], after[:, 0]), "timestamps mutated!"
            assert np.array_equal(before[:, 3], after[:, 3]), "polarity mutated!"
            print(f"  ✔ timestamps and polarity preserved")
    print("\n✔ Smoke test complete.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True,
                        help="Source NPZ directory (per-modality native box space)")
    parser.add_argument("--dst", required=True,
                        help="Destination directory for unified NPZs")
    parser.add_argument("--target_size", type=int, default=90,
                        help="Shared target box size (default 90, the DV mode)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default 8)")
    parser.add_argument("--smoke", type=str, default=None,
                        help="Run smoke test on a single prefix and exit")
    args = parser.parse_args()

    if args.smoke:
        return smoke_test(args.src, args.dst, args.smoke, args.target_size)

    files = collect_files(args.src)
    if not files:
        print(f"ERROR: no NPZ files found in {args.src}")
        return 1

    os.makedirs(args.dst, exist_ok=True)
    print(f"Source:   {args.src}")
    print(f"Dest:     {args.dst}")
    print(f"Target:   {args.target_size}")
    print(f"Files:    {len(files)} (all modalities)")
    print(f"Workers:  {args.workers}")

    tasks = []
    for src in files:
        rel = os.path.basename(src)
        dst = os.path.join(args.dst, rel)
        tasks.append((src, dst, args.target_size))

    t0 = time.time()
    native_box_counter = {}
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rescale"):
            res = fut.result()
            if "error" in res:
                errors.append(res)
                continue
            nb = res["native_box"]
            native_box_counter[nb] = native_box_counter.get(nb, 0) + 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Native box distribution: {sorted(native_box_counter.items())}")
    if errors:
        print(f"\nErrors: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
