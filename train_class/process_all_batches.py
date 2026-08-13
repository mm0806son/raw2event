import os
import sys
import glob
import argparse
import multiprocessing
import gc
import traceback
import re
from tqdm import tqdm
import torch

"""End-to-end batch processing pipeline.

Walk the input directory and process every Raw2Event CIFAR-10 capture batch:
  - Resumable: prefixes whose output already exists are skipped.
  - Strict matching: each batch must have the full RGB/RAW/DV/metadata set.
  - Memory-safe: ``maxtasksperchild=1`` forces per-prefix process recycling so
    GPU/CPU memory is released between samples.
  - Parallel: processes multiple prefixes concurrently across worker processes.

Example:
    python train_class/process_all_batches.py \\
        --input /path/to/data --output /path/to/output --workers 4
"""

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_class.process_single_batch import process_batch, parse_downsample_resolution


_PREFIX_INDEX_RE = re.compile(r"^(?P<index>\d+)_")


def extract_prefix_index(prefix: str) -> int:
    """Extract the leading integer index from a dataset prefix."""
    match = _PREFIX_INDEX_RE.match(prefix)
    if match is None:
        raise ValueError(
            f"Prefix '{prefix}' does not start with a leading integer index followed by '_'."
        )
    return int(match.group("index"))


def filter_prefixes_by_index_range(prefixes, index_start=None, index_end=None):
    """Keep prefixes whose leading integer index falls within the inclusive range."""
    if index_start is not None and index_end is not None and index_start > index_end:
        raise ValueError(
            f"index_start ({index_start}) must be less than or equal to index_end ({index_end})."
        )

    filtered = []
    for prefix in prefixes:
        prefix_index = extract_prefix_index(prefix)
        if index_start is not None and prefix_index < index_start:
            continue
        if index_end is not None and prefix_index > index_end:
            continue
        filtered.append(prefix)
    return filtered

def worker(args):
    """
    Worker function to process a single prefix.
    """
    prefix, input_dir, output_dir, sim_backend, downsample_resolution = args
    try:
        process_batch(prefix, input_dir, output_dir, sim_backend=sim_backend,
                      downsample_resolution=downsample_resolution)
        
        # Periodic GPU + RAM release.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return prefix, True, None
    except Exception as e:
        err_msg = traceback.format_exc()
        return prefix, False, err_msg

def main():
    parser = argparse.ArgumentParser(description="Process multiple batches of CIFAR10 event data.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing MKV and DAT files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for npz files")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of prefixes to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (Chunked Parallelism)")
    parser.add_argument(
        "--index_start",
        type=int,
        default=None,
        help="Inclusive lower bound for the leading integer index in each prefix",
    )
    parser.add_argument(
        "--index_end",
        type=int,
        default=None,
        help="Inclusive upper bound for the leading integer index in each prefix",
    )
    parser.add_argument(
        "--sim_backend",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "numpy"],
        help="Simulator backend for DVS event generation",
    )
    parser.add_argument(
        "--downsample",
        type=parse_downsample_resolution,
        # to match the calibration/DAVIS346 resolution.
        default="346x260",
        metavar="WxH",
        help="Downsample frames to WxH before event generation (e.g. 346x260). Pass 'none' to disable.",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning directory: {input_dir}")
    # Find all rgb mkv files to extract prefixes
    rgb_files = glob.glob(os.path.join(input_dir, "rgb_frames_*_rgb.mkv"))
    prefixes = []
    for f in rgb_files:
        filename = os.path.basename(f)
        # rgb_frames_PREFIX_rgb.mkv -> PREFIX
        prefix = filename.replace("rgb_frames_", "").replace("_rgb.mkv", "")
        prefixes.append(prefix)

    # Sort by leading integer index so "1 < 2 < 10 < 100", not
    # lexicographic "1 < 10 < 100 < 101 < ... < 2". Malformed prefixes sort
    # last and will raise at filter time.
    def _sort_key(prefix: str):
        m = _PREFIX_INDEX_RE.match(prefix)
        if m is None:
            return (1, 0, prefix)
        return (0, int(m.group("index")), prefix)

    prefixes.sort(key=_sort_key)
    
    # 1. Fuzzy File Matching with Fast O(1) Dictionary Lookup
    valid_prefixes = []
    bad_files_log = os.path.join(output_dir, "bad_files.log")
    print("Pre-caching directory contents for fast validation...")
    
    file_map = {}
    for f in os.listdir(input_dir):
        if f.startswith("rgb_frames_") and f.endswith("_rgb.mkv"):
            base = f.replace("rgb_frames_", "").replace("_rgb.mkv", "").rsplit('_', 1)[0]
            if base not in file_map: file_map[base] = {}
            file_map[base]['rgb'] = True
        elif f.startswith("raw_frames_") and f.endswith("_raw_10bit.mkv"):
            base = f.replace("raw_frames_", "").replace("_raw_10bit.mkv", "").rsplit('_', 1)[0]
            if base not in file_map: file_map[base] = {}
            file_map[base]['raw'] = True
        elif f.startswith("dv_output_") and f.endswith(".aedat4"):
            base = f.replace("dv_output_", "").replace(".aedat4", "").rsplit('_', 1)[0]
            if base not in file_map: file_map[base] = {}
            file_map[base]['dv'] = True
        elif f.startswith("metadata_") and f.endswith(".dat"):
            base = f.replace("metadata_", "").replace(".dat", "").rsplit('_', 1)[0]
            if base not in file_map: file_map[base] = {}
            file_map[base]['meta'] = True

    print("Validating file pairs for all prefixes...")
    with open(bad_files_log, "a") as f_log:
        for prefix in prefixes:
            base_id = prefix.rsplit('_', 1)[0]
            mapping = file_map.get(base_id, {})
            
            missing = []
            if 'rgb' not in mapping: missing.append("rgb_frames")
            if 'raw' not in mapping: missing.append("raw_frames")
            if 'dv' not in mapping: missing.append("dv_output")
            if 'meta' not in mapping: missing.append("metadata")
            
            if missing:
                err_str = f"[Missing Files] Prefix {prefix} (fuzzy base: {base_id}) is missing: {missing}"
                print(err_str)
                f_log.write(err_str + "\n")
                continue # Skip this prefix entirely
            
            valid_prefixes.append(prefix)

    filtered_prefixes = filter_prefixes_by_index_range(
        valid_prefixes,
        index_start=args.index_start,
        index_end=args.index_end,
    )

    # 2. Resumable processing: skip prefixes whose outputs already exist.
    to_process = []
    for prefix in filtered_prefixes:
        out_rgb = os.path.join(output_dir, f"{prefix}_filtered_rgb.npz")
        out_raw = os.path.join(output_dir, f"{prefix}_filtered_raw.npz")
        out_dv = os.path.join(output_dir, f"{prefix}_filtered_dv.npz")
        if os.path.exists(out_rgb) and os.path.exists(out_raw) and os.path.exists(out_dv):
            continue # Already processed, skipping
        to_process.append(prefix)

    # 3. Apply Limit if specified
    if args.limit is not None and args.limit > 0:
        to_process = to_process[:args.limit]

    print(f"Total found matching prefixes: {len(valid_prefixes)}")
    print(f"Total prefixes after index range filter: {len(filtered_prefixes)}")
    print(f"Total left to process (after resuming/limit): {len(to_process)}")

    if len(to_process) == 0:
        print("Nothing to process. All done.")
        return

    if args.downsample is not None:
        print(f"Frame downsampling enabled: {args.downsample[0]}x{args.downsample[1]}")

    # 4. Chunked Parallelism with Memory Release (maxtasksperchild=1)
    worker_args = [(prefix, input_dir, output_dir, args.sim_backend, args.downsample) for prefix in to_process]
    
    success_count = 0
    fail_count = 0
    
    # Pool definition: maxtasksperchild=1 ensures a process is killed and memory is fully cleared after EACH job.
    print(f"Launching pool with {args.workers} workers. Using maxtasksperchild=1 for safe VRAM/RAM recycling.")
    with multiprocessing.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        # imap_unordered yields results as soon as they are ready
        pbar = tqdm(pool.imap_unordered(worker, worker_args), total=len(worker_args), desc="Processing Batches")
        for prefix, success, err in pbar:
            if success:
                success_count += 1
            else:
                fail_count += 1
                tqdm.write(f"\n[ERROR] Task failed for {prefix}:\n{err}")
                
            pbar.set_postfix({'Success': success_count, 'Fail': fail_count})

    print(f"\nBatch processing complete! Success: {success_count}, Failed: {fail_count}")
    if fail_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    # Start method 'spawn' ensures child processes don't inherit a corrupted or bloated uncopyable RAM state
    multiprocessing.set_start_method('spawn', force=True)
    main()
