"""V03 / V04 internal ablation: Raw / RGB through DVS-Voltmeter with the
*default* device K (K_MAP["DVS346"]), NOT the Raw2Event Stage 1.5 K.

Isolates the contribution of our pair-specific K calibration: if Raw2Event's
edge over v2e survives this baseline, then DVS-Voltmeter alone (without our
calibration) is doing the heavy lifting; if not, calibration is the source.

Reuses src/process_data/dvs_generate.generate_events_tensor with a substituted
K vector. Then routes events through the same AprilTag crop helper used for
v2e variants, so the downstream NPZ format is identical.

Usage (run on GPU):
  python tools/v2e_baseline/raw_dvsv_default_k_gen.py \
    --variant V03 \
    --input_dir ./data \
    --output_dir ./output/v2e_compare/V03_raw_dvsv_default_k \
    --prefix_list tools/v2e_baseline/canonical_test_559_prefixes.txt \
    --index_start 0 --index_end -1 \
    --gpu_id 0 --skip_existing
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

import numpy as np  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402

from src.config import K_MAP  # noqa: E402
from src.process_data import dvs_generate, file_read  # noqa: E402

from tools.v2e_baseline.run_v2e_batch import parse_prefix_list  # noqa: E402
from tools.v2e_baseline.v2e_helpers import (  # noqa: E402
    apply_apriltag_crop, is_valid_filtered_npz, save_filtered_npz,
)

TARGET_W, TARGET_H = 346, 260


def _downsample_frames_uint(frames_native: np.ndarray, target_w: int, target_h: int,
                            dtype) -> np.ndarray:
    n = frames_native.shape[0]
    out = np.empty((n, target_h, target_w), dtype=dtype)
    for i in range(n):
        out[i] = cv2.resize(frames_native[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
    return out


def _frames_to_tensor(frames_np: np.ndarray) -> torch.Tensor:
    """Mirror train_class/process_single_batch._frame_array_to_tensor:
    torch.from_numpy rejects uint16 on some builds; cast to int32 first."""
    if frames_np.dtype == np.uint16:
        frames_np = frames_np.astype(np.int32, copy=False)
    return torch.from_numpy(frames_np)


def process_one_prefix_dvsv(
    prefix: str,
    *,
    variant: str,
    input_dir: str,
    output_dir: str,
    num_workers: int,
    sim_backend: str,
) -> dict:
    """Per-prefix work for V03/V04 (Raw + DVS-Voltmeter default K).
    Pulled out of the batch loop so it can run in a child subprocess
    isolated from pupil_apriltags heap-corruption SIGABRTs.

    Returns stats dict; raises on any error.
    """
    spec = VARIANT_SPEC[variant]
    K_default = K_MAP[spec["k_label"]]
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    out_npz = output_dir_p / f"{prefix}_filtered_{spec['npz_suffix']}.npz"

    base_id = prefix.rsplit("_", 1)[0]
    input_dir_p = Path(input_dir)
    if spec["input_modality"] == "rgb":
        mkv_files = sorted(input_dir_p.glob(f"rgb_frames_{base_id}*_rgb.mkv"))
    else:
        mkv_files = sorted(input_dir_p.glob(f"raw_frames_{base_id}*_raw_10bit.mkv"))
    md_files = sorted(input_dir_p.glob(f"metadata_{base_id}*.dat"))
    if not mkv_files or not md_files:
        raise FileNotFoundError(f"missing inputs for {prefix}")

    from generate_event import load_from_video
    frames_native, _ = load_from_video(str(mkv_files[0]), quiet=True)
    pi_ts, _ = file_read.read_metadata(str(md_files[0]))
    n = min(frames_native.shape[0], int(pi_ts.shape[0]))

    if spec["input_modality"] == "rawY":
        luminance_native = file_read.bayer_mosaic_to_luminance(frames_native[:n])
        frames_for_sim = _downsample_frames_uint(luminance_native, TARGET_W, TARGET_H, np.uint16)
        is_rgb = False
        raw_is_lum = True
        is_luminance_for_tag = True
    else:
        frames_for_sim = _downsample_frames_uint(frames_native[:n], TARGET_W, TARGET_H, np.uint8)
        is_rgb = True
        raw_is_lum = False
        is_luminance_for_tag = False

    frames_tensor = _frames_to_tensor(frames_for_sim)
    ev_tensor = dvs_generate.generate_events_tensor(
        pi_ts[:n], frames_tensor, is_rgb=is_rgb, raw_is_luminance=raw_is_lum,
        k_values=K_default, sim_backend=sim_backend,
    )
    ev_np = ev_tensor.numpy() if isinstance(ev_tensor, torch.Tensor) else np.asarray(ev_tensor)

    ts_us_np = (pi_ts[:n].numpy() if isinstance(pi_ts, torch.Tensor) else np.asarray(pi_ts[:n])).astype(np.int64)
    filtered, box_size = apply_apriltag_crop(
        ev_np, frames_for_sim, ts_us_np,
        n_workers=num_workers,
        is_luminance=is_luminance_for_tag,
        is_raw=(spec["input_modality"] == "rawY"),
    )
    save_filtered_npz(filtered, out_npz)

    stats = {
        "variant": variant, "prefix": prefix,
        "n_events_sim": int(ev_np.shape[0]),
        "n_events_after_crop": int(filtered.shape[0]),
        "box_size": int(box_size),
    }
    if filtered.shape[0] == 0:
        stats["warning"] = "zero events after AprilTag crop — likely tag-detection failure"
    return stats

VARIANT_SPEC = {
    "V03": {"input_modality": "rawY", "k_label": "DVS346", "npz_suffix": "rawY"},
    "V04": {"input_modality": "rgb",  "k_label": "DVS346", "npz_suffix": "rgb"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=list(VARIANT_SPEC.keys()))
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix_list", required=True)
    ap.add_argument("--index_start", type=int, default=0)
    ap.add_argument("--index_end", type=int, default=-1)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--gpu_id", type=int, default=None)
    ap.add_argument("--sim_backend", default="auto", choices=["auto", "cuda", "cpu", "numpy"])
    args = ap.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    spec = VARIANT_SPEC[args.variant]
    K_default = K_MAP[spec["k_label"]]
    print(f"[raw_dvsv] variant={args.variant} input={spec['input_modality']} K={spec['k_label']}={K_default}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "_manifest.jsonl"

    prefixes = parse_prefix_list(args.prefix_list)
    end = len(prefixes) if args.index_end < 0 else min(args.index_end, len(prefixes))
    window = prefixes[args.index_start:end]
    print(f"[raw_dvsv] processing {len(window)} prefixes [{args.index_start}, {end})")

    import subprocess  # noqa: E402
    import tempfile  # noqa: E402

    n_done = n_skip = n_fail = 0
    t_start = time.time()
    for i, prefix in enumerate(window):
        out_npz = output_dir / f"{prefix}_filtered_{spec['npz_suffix']}.npz"
        if args.skip_existing and is_valid_filtered_npz(out_npz):
            n_skip += 1
            continue
        # Run per-prefix work in a child subprocess so a pupil_apriltags
        # SIGABRT (heap corruption observed in tag_detector.detect()) only
        # kills the child, not the whole batch driver. Same isolation
        # pattern as v2e_helpers.process_one_prefix_v2e_subprocess.
        with tempfile.TemporaryDirectory(prefix=f"dvsv_parent_{prefix}_") as parent_tmp:
            result_json = Path(parent_tmp) / "_result.json"
            cmd = [
                sys.executable, "-m", "tools.v2e_baseline._dvsv_one_prefix_subproc",
                "--prefix", prefix,
                "--variant", args.variant,
                "--input_dir", str(args.input_dir),
                "--output_dir", str(output_dir),
                "--num_workers", str(args.num_workers),
                "--sim_backend", args.sim_backend,
                "--result_json", str(result_json),
            ]
            env = os.environ.copy()
            # Force single-threaded native libs (override any caller value).
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                      "NUMBA_NUM_THREADS", "TBB_NUM_THREADS"):
                env[k] = "1"
            try:
                proc = subprocess.run(
                    cmd, cwd=str(Path(__file__).resolve().parents[2]),
                    env=env, capture_output=True, text=True, timeout=600,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"dvsv subprocess rc={proc.returncode}\n"
                        f"--- stdout tail ---\n"
                        + "\n".join((proc.stdout or "").splitlines()[-10:]) + "\n"
                        f"--- stderr tail ---\n"
                        + "\n".join((proc.stderr or "").splitlines()[-10:])
                    )
                if not result_json.exists():
                    raise RuntimeError("dvsv subprocess produced no result_json")
                with open(result_json) as f:
                    stats = json.load(f)
                stats["abs_index"] = args.index_start + i
                stats["elapsed_s"] = time.time() - t_start
                if stats.get("n_events_after_crop", 0) == 0:
                    print(f"[{i+1}/{len(window)}] {prefix} WARNING: zero cropped events", flush=True)
                with open(manifest_path, "a") as mf:
                    mf.write(json.dumps(stats) + "\n")
                n_done += 1
                print(f"[{i+1}/{len(window)}] {prefix} ev={stats['n_events_sim']:>9d} -> "
                      f"crop={stats['n_events_after_crop']:>8d} box={stats['box_size']}", flush=True)
            except Exception as exc:
                n_fail += 1
                print(f"[{i+1}/{len(window)}] {prefix} FAILED: {exc}", file=sys.stderr, flush=True)
                with open(manifest_path, "a") as mf:
                    mf.write(json.dumps({
                        "variant": args.variant, "prefix": prefix,
                        "abs_index": args.index_start + i,
                        "error": str(exc),
                    }) + "\n")

    print(f"[raw_dvsv] DONE done={n_done} skip={n_skip} fail={n_fail} elapsed={(time.time()-t_start)/60:.1f}m")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
