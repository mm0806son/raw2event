import argparse
import glob
import multiprocessing
import os
import queue as queue_mod
import re
import sys
import time
import traceback

from tqdm import tqdm


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from train_class.process_single_batch import parse_downsample_resolution, process_batch  # noqa: E402


_PREFIX_INDEX_RE = re.compile(r"^(?P<index>\d+)_")

DEFAULT_TIMEOUT_PER_PREFIX_SEC = 600


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


def _child_entry(result_q, prefix, input_dir, output_dir, sim_backend, downsample_resolution):
    """Run process_batch in a fresh spawned subprocess and report outcome via queue."""
    try:
        process_batch(
            prefix,
            input_dir,
            output_dir,
            sim_backend=sim_backend,
            downsample_resolution=downsample_resolution,
        )
        result_q.put((prefix, "ok", None))
    except Exception:
        result_q.put((prefix, "fail", traceback.format_exc()))


def _collect_finished(active, results, pbar, timeout_sec, poll_interval=1.0):
    """Poll active children and collect those that finished, crashed, or timed out.

    Mutates ``active`` and ``results`` in place. Returns the number of slots freed.
    """
    time.sleep(poll_interval)
    freed = 0
    for prefix in list(active.keys()):
        proc, queue, start_ts = active[prefix]
        elapsed = time.time() - start_ts
        finished = not proc.is_alive()

        if finished:
            proc.join()
            # queue.empty() has a known race with the child-side feeder thread:
            # proc.join() can return before the pickled result is visible on the
            # parent side, so we must actually attempt a bounded get().
            try:
                results.append(queue.get(timeout=2))
            except queue_mod.Empty:
                results.append(
                    (
                        prefix,
                        "crash",
                        f"child exited with code {proc.exitcode} without returning a result",
                    )
                )
            queue.close()
            queue.join_thread()
            del active[prefix]
            freed += 1
            pbar.update(1)
        elif timeout_sec is not None and elapsed > timeout_sec:
            proc.terminate()
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join()
            results.append(
                (
                    prefix,
                    "timeout",
                    f"exceeded per-prefix wall-clock limit of {timeout_sec}s",
                )
            )
            queue.close()
            queue.join_thread()
            del active[prefix]
            freed += 1
            pbar.update(1)
    return freed


def run_batch_driver(worker_args, workers, timeout_per_prefix_sec, child_target=None):
    """Run each prefix in its own spawned subprocess with a wall-clock timeout.

    This replaces multiprocessing.Pool.imap_unordered, which hangs forever when a
    child is killed by a signal (OOM, SIGSEGV, CUDA context death) because the
    try/except inside the worker can't catch kernel-delivered signals and the
    result queue never receives an entry.

    ``child_target`` is injectable for testing; production always uses
    ``_child_entry``.

    Returns a list of (prefix, status, err) where status is one of
    {"ok", "fail", "timeout", "crash"}.
    """
    if child_target is None:
        child_target = _child_entry
    if workers < 1:
        # A zero/negative workers value would make the driver loop spin forever
        # without ever launching a child, re-creating the original hang mode.
        raise ValueError(f"workers must be >= 1, got {workers}")

    ctx = multiprocessing.get_context("spawn")
    pending = list(worker_args)
    active = {}
    results = []

    pbar = tqdm(total=len(worker_args), desc="Processing Batches")
    try:
        while pending or active:
            while len(active) < workers and pending:
                args_tuple = pending.pop(0)
                prefix = args_tuple[0]
                queue = ctx.Queue()
                proc = ctx.Process(target=child_target, args=(queue, *args_tuple))
                proc.start()
                active[prefix] = (proc, queue, time.time())

            if active:
                # Only repaint the postfix when a slot actually frees,
                # otherwise non-TTY log collectors emit one progress line per
                # poll interval for no new information.
                freed = _collect_finished(
                    active, results, pbar, timeout_per_prefix_sec, poll_interval=5.0
                )
                if freed:
                    ok = sum(1 for _, status, _ in results if status == "ok")
                    bad = len(results) - ok
                    pbar.set_postfix({"Success": ok, "Fail": bad})
    finally:
        for prefix, (proc, leftover_q, _s) in list(active.items()):
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            results.append((prefix, "crash", "terminated during driver shutdown"))
            leftover_q.close()
            leftover_q.join_thread()
        pbar.close()

    return results


def collect_valid_prefixes(input_dir: str, bad_files_log: str) -> tuple[list[str], list[str]]:
    """Scan input_dir for valid (rgb/raw/dv/metadata) prefix quadruples."""
    print(f"Scanning directory: {input_dir}")
    rgb_files = glob.glob(os.path.join(input_dir, "rgb_frames_*_rgb.mkv"))
    prefixes = []
    for path in rgb_files:
        filename = os.path.basename(path)
        prefixes.append(filename.replace("rgb_frames_", "").replace("_rgb.mkv", ""))
    # Sort by the leading integer index so "1_..." < "2_..." < "10_..." < "100_..."
    # instead of the lexicographic order "1 < 10 < 100 < 101 < ... < 2".
    # Malformed prefixes (no leading <int>_) sort last to preserve visibility;
    # filter_prefixes_by_index_range will raise on them at filter time.
    def _sort_key(prefix: str):
        m = _PREFIX_INDEX_RE.match(prefix)
        if m is None:
            return (1, 0, prefix)
        return (0, int(m.group("index")), prefix)

    prefixes.sort(key=_sort_key)

    print("Pre-caching directory contents for fast validation...")
    file_map: dict[str, dict[str, bool]] = {}
    for filename in os.listdir(input_dir):
        if filename.startswith("rgb_frames_") and filename.endswith("_rgb.mkv"):
            base = filename.replace("rgb_frames_", "").replace("_rgb.mkv", "").rsplit("_", 1)[0]
            file_map.setdefault(base, {})["rgb"] = True
        elif filename.startswith("raw_frames_") and filename.endswith("_raw_10bit.mkv"):
            base = (
                filename.replace("raw_frames_", "")
                .replace("_raw_10bit.mkv", "")
                .rsplit("_", 1)[0]
            )
            file_map.setdefault(base, {})["raw"] = True
        elif filename.startswith("dv_output_") and filename.endswith(".aedat4"):
            base = filename.replace("dv_output_", "").replace(".aedat4", "").rsplit("_", 1)[0]
            file_map.setdefault(base, {})["dv"] = True
        elif filename.startswith("metadata_") and filename.endswith(".dat"):
            base = filename.replace("metadata_", "").replace(".dat", "").rsplit("_", 1)[0]
            file_map.setdefault(base, {})["meta"] = True

    valid_prefixes = []
    print("Validating file pairs for all prefixes...")
    with open(bad_files_log, "a", encoding="utf-8") as log_file:
        for prefix in prefixes:
            base_id = prefix.rsplit("_", 1)[0]
            mapping = file_map.get(base_id, {})

            missing = []
            if "rgb" not in mapping:
                missing.append("rgb_frames")
            if "raw" not in mapping:
                missing.append("raw_frames")
            if "dv" not in mapping:
                missing.append("dv_output")
            if "meta" not in mapping:
                missing.append("metadata")

            if missing:
                err_str = f"[Missing Files] Prefix {prefix} (fuzzy base: {base_id}) is missing: {missing}"
                print(err_str)
                log_file.write(err_str + "\n")
                continue

            valid_prefixes.append(prefix)

    return prefixes, valid_prefixes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process multiple batches of CIFAR10 event data using K defaults from src.config."
    )
    parser.add_argument("--input", type=str, required=True, help="Input directory containing MKV and DAT files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for npz files")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of prefixes to process")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent subprocesses")
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
    parser.add_argument(
        "--timeout_per_prefix_sec",
        type=int,
        default=DEFAULT_TIMEOUT_PER_PREFIX_SEC,
        help=(
            "Wall-clock timeout for a single prefix. Exceeding it marks the prefix "
            "as timeout and moves on, so a dead/hung worker never blocks the run. "
            "Pass 0 to disable."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.makedirs(args.output, exist_ok=True)

    bad_files_log = os.path.join(args.output, "bad_files.log")
    _, valid_prefixes = collect_valid_prefixes(args.input, bad_files_log)

    filtered_prefixes = filter_prefixes_by_index_range(
        valid_prefixes,
        index_start=args.index_start,
        index_end=args.index_end,
    )

    to_process = []
    for prefix in filtered_prefixes:
        out_rgb = os.path.join(args.output, f"{prefix}_filtered_rgb.npz")
        out_raw = os.path.join(args.output, f"{prefix}_filtered_raw.npz")
        out_dv = os.path.join(args.output, f"{prefix}_filtered_dv.npz")
        if os.path.exists(out_rgb) and os.path.exists(out_raw) and os.path.exists(out_dv):
            continue
        to_process.append(prefix)

    if args.limit is not None and args.limit > 0:
        to_process = to_process[: args.limit]

    print(f"Total found matching prefixes: {len(valid_prefixes)}")
    print(f"Total prefixes after index range filter: {len(filtered_prefixes)}")
    print(f"Total left to process (after resuming/limit): {len(to_process)}")

    if not to_process:
        print("Nothing to process. All done.")
        return

    if args.downsample is not None:
        print(f"Frame downsampling enabled: {args.downsample[0]}x{args.downsample[1]}")

    worker_args = [
        (prefix, args.input, args.output, args.sim_backend, args.downsample)
        for prefix in to_process
    ]

    timeout_sec = args.timeout_per_prefix_sec if args.timeout_per_prefix_sec > 0 else None
    print(
        f"Launching {args.workers}-way driver. "
        f"Per-prefix wall-clock timeout: {timeout_sec if timeout_sec is not None else 'disabled'}s"
    )

    results = run_batch_driver(
        worker_args,
        workers=args.workers,
        timeout_per_prefix_sec=timeout_sec,
    )

    success = [r for r in results if r[1] == "ok"]
    failures = [r for r in results if r[1] != "ok"]

    for prefix, status, err in failures:
        tqdm.write(f"\n[{status.upper()}] {prefix}\n{err}")

    print(
        f"\nBatch processing complete! Success: {len(success)}, "
        f"Failed: {len(failures)} (of which "
        f"{sum(1 for _, s, _ in failures if s == 'timeout')} timeout, "
        f"{sum(1 for _, s, _ in failures if s == 'crash')} crash, "
        f"{sum(1 for _, s, _ in failures if s == 'fail')} python exception)"
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
