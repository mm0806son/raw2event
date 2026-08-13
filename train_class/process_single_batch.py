import argparse
import os
import sys
import glob
import math
import numpy as np
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

"""Single-prefix data processing script.

For a given prefix, process its RGB MKV, RAW MKV, AEDAT4, and metadata files:
  - Simulate DVS events.
  - Auto-detect AprilTag markers and recover the inter-sensor time offset.
  - Filter and align the event stream, then save it as a `.npz` file.

Example:
    python train_class/process_single_batch.py \\
        --input /path/to/data --prefix 18900_cat_1... --output ./output/
"""

# Add the project root to sys.path to import internal modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import src.process_data.file_read as file_read
import src.process_data.dvs_generate as dvs_generate
import src.process_data.tag_detector as tag_detector
import src.process_data.event_filter as event_filter
from src.config import get_K
from generate_event import load_from_video

def round_up_to_10(x):
    return int(math.ceil(x / 10.0) * 10)


def _frame_array_to_tensor(frames_np, name):
    """Convert video frames to a torch tensor with dtype compatible across torch builds."""
    if frames_np is None:
        raise ValueError(f"{name} video frames are empty or unavailable")

    if frames_np.dtype == np.uint16:
        # Some torch builds reject uint16 NumPy arrays in from_numpy.
        frames_np = frames_np.astype(np.int32, copy=False)

    return torch.from_numpy(frames_np)

def parse_downsample_resolution(value):
    """Parse a 'WxH' string into (width, height) with validation. 'none' disables downsampling."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "off", ""}:
        return None
    try:
        w_str, h_str = value.split("x")
        w, h = int(w_str), int(h_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Invalid downsample format '{value}'. Expected WxH, e.g. 346x260"
        )
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(
            f"Downsample dimensions must be positive, got {w}x{h}"
        )
    return (w, h)


def _downsample_frames(frames_np, target_w, target_h):
    """Downsample frames to target resolution using area interpolation.

    Args:
        frames_np: numpy array of shape (N, H, W) or (N, H, W, C)
        target_w: target width
        target_h: target height

    Returns:
        Downsampled numpy array with the same dtype and number of dims.
    """
    src_h, src_w = frames_np.shape[1], frames_np.shape[2]
    if src_w == target_w and src_h == target_h:
        return frames_np

    out = np.empty(
        (frames_np.shape[0], target_h, target_w) + frames_np.shape[3:],
        dtype=frames_np.dtype,
    )
    for i in range(frames_np.shape[0]):
        out[i] = cv2.resize(
            frames_np[i], (target_w, target_h), interpolation=cv2.INTER_AREA
        )
    return out


def _raw_bayer_to_y_downsampled(frames_np, target_w, target_h):
    """Convert RAW Bayer frames to luminance before downsampling.

    This preserves the calibrated RAW→Y domain semantics when shrinking
    520x692 Bayer mosaics to the DAVIS346 calibration resolution.
    """
    luminance_frames = file_read.bayer_mosaic_to_luminance(frames_np)
    return _downsample_frames(luminance_frames, target_w, target_h)


def process_batch(prefix, input_dir="train_class/test_data", output_dir="train_class/output",
                  sim_backend="auto", downsample_resolution=(346, 260)):
    # so that RAW frames match the calibration/DAVIS346 resolution before event generation.
    print(f"Processing batch: {prefix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Fuzzy match base_id (ignoring the exact seconds segment)
    base_id = prefix.rsplit('_', 1)[0]
    
    def find_file(pattern):
        matches = glob.glob(os.path.join(input_dir, pattern))
        if not matches:
            raise FileNotFoundError(f"Missing file matching: {pattern} in {input_dir}")
        return matches[0]

    files = {
        'rgb_frames': find_file(f"rgb_frames_{base_id}*_rgb.mkv"),
        'raw_frames': find_file(f"raw_frames_{base_id}*_raw_10bit.mkv"),
        'dv': find_file(f"dv_output_{base_id}*.aedat4"),
        'metadata': find_file(f"metadata_{base_id}*.dat")
    }

    print("Loading DV data...")
    dv_frames, dv_frames_timestamps = file_read.load_frames(files['dv'])
    dv_events_tensor = file_read.load_events(files['dv'])
    if dv_frames is None or dv_frames_timestamps is None:
        raise ValueError(
            f"DV frame stream is empty or unavailable for prefix '{prefix}': {files['dv']}"
        )
    if dv_events_tensor is None:
        raise ValueError(
            f"DV event stream is empty or unavailable for prefix '{prefix}': {files['dv']}"
        )
    
    # Filter out the known DAVIS346 hot pixel at (108, 108).
    # This pixel is permanently active and emits very high-frequency events
    # (~1% of the total per file), which would distort downstream stream analysis.
    dv_events_tensor = dv_events_tensor[~((dv_events_tensor[:, 1] == 108) & (dv_events_tensor[:, 2] == 108))]
    print(f"dv_frames: {dv_frames.shape}, dv_events: {dv_events_tensor.shape}")

    print("Loading Metadata...")
    pi_timestamps, real_timestamps = file_read.read_metadata(files['metadata'])
    if pi_timestamps is None or real_timestamps is None:
        raise ValueError(
            f"Metadata timestamps are empty or unavailable for prefix '{prefix}': {files['metadata']}"
        )

    print("Loading MKV Frames...")
    rgb_frames_np, _ = load_from_video(files['rgb_frames'], quiet=True)
    raw_frames_np, _ = load_from_video(files['raw_frames'], quiet=True)
    if rgb_frames_np is None:
        raise ValueError(
            f"RGB video frames are empty or unavailable for prefix '{prefix}': {files['rgb_frames']}"
        )
    if raw_frames_np is None:
        raise ValueError(
            f"RAW video frames are empty or unavailable for prefix '{prefix}': {files['raw_frames']}"
        )

    # Ensure lengths match
    min_len_rgb = min(len(pi_timestamps), len(rgb_frames_np))
    min_len_raw = min(len(pi_timestamps), len(raw_frames_np))
    
    # Use pi_timestamps for event generation to maintain alignment
    pi_timestamps_rgb = pi_timestamps[:min_len_rgb]
    pi_timestamps_raw = pi_timestamps[:min_len_raw]

    rgb_frames_trimmed = rgb_frames_np[:min_len_rgb]
    raw_frames_trimmed = raw_frames_np[:min_len_raw]

    raw_is_luminance = False

    # Downsample frames before event generation if requested
    if downsample_resolution is not None:
        ds_w, ds_h = downsample_resolution
        src_h, src_w = raw_frames_trimmed.shape[1], raw_frames_trimmed.shape[2]
        print(f"Downsampling frames: {src_w}x{src_h} -> {ds_w}x{ds_h}")
        raw_frames_trimmed = _raw_bayer_to_y_downsampled(raw_frames_trimmed, ds_w, ds_h)
        rgb_frames_trimmed = _downsample_frames(rgb_frames_trimmed, ds_w, ds_h)
        raw_is_luminance = True

    rgb_frames = _frame_array_to_tensor(rgb_frames_trimmed, name="rgb_frames")
    raw_frames = _frame_array_to_tensor(raw_frames_trimmed, name="raw_frames")
    print(f"rgb_frames: {rgb_frames.shape}, dtype: {rgb_frames.dtype}, min: {rgb_frames.float().min()}, max: {rgb_frames.float().max()}, mean: {rgb_frames.float().mean():.2f}")
    print(f"raw_frames: {raw_frames.shape}, dtype: {raw_frames.dtype}, min: {raw_frames.float().min()}, max: {raw_frames.float().max()}, mean: {raw_frames.float().mean():.2f}")

    # Set up threads taking system core limits into account
    n_workers = min(16, os.cpu_count() or 4)

    k_values_raw = get_K("Raw2DVS346")
    k_values_rgb = get_K("RGB2DVS346")

    print("Simulating RGB DVS events...")
    rgb_events_tensor = dvs_generate.generate_events_tensor(
        pi_timestamps_rgb, rgb_frames, is_rgb=True, k_values=k_values_rgb, sim_backend=sim_backend
    )
    print(f"RGB DVS events shape: {rgb_events_tensor.shape}")

    print("Simulating RAW DVS events...")
    raw_events_tensor = dvs_generate.generate_events_tensor(
        pi_timestamps_raw,
        raw_frames,
        is_rgb=False,
        raw_is_luminance=raw_is_luminance,
        k_values=k_values_raw,
        sim_backend=sim_backend,
    )
    print(f"RAW DVS events shape: {raw_events_tensor.shape}")

    print("Calculating time offset...")
    time_offset = file_read.calculate_time_offset(pi_timestamps, real_timestamps)
    dv_frames_timestamps = dv_frames_timestamps - time_offset
    dv_events_tensor[:, 0] = dv_events_tensor[:, 0] - time_offset

    # Tag Detector Parameters
    TAG_REF_WIDTH = 287      
    BARBARA_REF_SIZE = 861   
    BARBARA_GAP = 82         
    margin_ratio = 0.03

    margin_ratios = [margin_ratio] * n_workers
    tag_ref_widths = [TAG_REF_WIDTH] * n_workers
    barbara_ref_sizes = [BARBARA_REF_SIZE] * n_workers
    barbara_gaps = [BARBARA_GAP] * n_workers
    is_raws_rgb = [False] * n_workers
    is_raws_raw = [True] * n_workers
    # ``is_luminance`` mirrors ``raw_is_luminance`` in the event generation path:
    # once the RAW Bayer frames have been converted to Y before downsampling,
    # the tag detector must NOT apply a second Bayer→Gray decoding.
    is_luminances_rgb = [False] * n_workers
    is_luminances_raw = [raw_is_luminance] * n_workers
    is_luminances_dv = [False] * n_workers

    print("Running Tag Detector for RGB...")
    rgb_frame_batches = tag_detector.split_batches(rgb_frames.numpy(), n_workers)
    ts_batches_rgb = tag_detector.split_batches(pi_timestamps_rgb.numpy() if hasattr(pi_timestamps_rgb, 'numpy') else pi_timestamps_rgb, n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        rgb_all_results = list(tqdm(executor.map(
            tag_detector.process_batch, rgb_frame_batches, ts_batches_rgb,
            margin_ratios, tag_ref_widths, barbara_ref_sizes, barbara_gaps,
            is_raws_rgb, is_luminances_rgb
        ), total=n_workers))
    rgb_crops_info = [item for batch in rgb_all_results for item in batch if item[2] is not None]
    rgb_crops_info.sort(key=lambda x: x[2])

    print("Running Tag Detector for RAW...")
    raw_frame_batches = tag_detector.split_batches(raw_frames.numpy(), n_workers)
    ts_batches_raw = tag_detector.split_batches(pi_timestamps_raw.numpy() if hasattr(pi_timestamps_raw, 'numpy') else pi_timestamps_raw, n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        raw_all_results = list(tqdm(executor.map(
            tag_detector.process_batch, raw_frame_batches, ts_batches_raw,
            margin_ratios, tag_ref_widths, barbara_ref_sizes, barbara_gaps,
            is_raws_raw, is_luminances_raw
        ), total=n_workers))
    raw_crops_info = [item for batch in raw_all_results for item in batch if item[2] is not None]
    raw_crops_info.sort(key=lambda x: x[2])

    print("Running Tag Detector for DV...")
    dv_frame_batches = tag_detector.split_batches(dv_frames.numpy(), n_workers)
    dv_ts_batches = tag_detector.split_batches(dv_frames_timestamps.numpy() if hasattr(dv_frames_timestamps, 'numpy') else dv_frames_timestamps, n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        dv_all_results = list(tqdm(executor.map(
            tag_detector.process_batch, dv_frame_batches, dv_ts_batches,
            margin_ratios, tag_ref_widths, barbara_ref_sizes, barbara_gaps,
            is_raws_rgb, is_luminances_dv
        ), total=n_workers))
    dv_crops_info = [item for batch in dv_all_results for item in batch if item[2] is not None]
    dv_crops_info.sort(key=lambda x: x[2])

    # Dynamic target size from crops bounding box
    rgb_valid = [info[0] for info in rgb_crops_info if info[0] is not None]
    raw_valid = [info[0] for info in raw_crops_info if info[0] is not None]
    dv_valid = [info[0] for info in dv_crops_info if info[0] is not None]

    if not rgb_valid or not raw_valid or not dv_valid:
        print("Warning: Failed to detect Tags in some frames. Filtering may be incomplete.")
        
    rgb_box_size = max(info['polygon'].ptp(axis=0).max() for info in rgb_valid) if rgb_valid else 256
    raw_box_size = max(info['polygon'].ptp(axis=0).max() for info in raw_valid) if raw_valid else 256
    dv_box_size = max(info['polygon'].ptp(axis=0).max() for info in dv_valid) if dv_valid else 256

    RGB_BOX_SIZE_FOR_EVENT = round_up_to_10(rgb_box_size)
    RAW_BOX_SIZE_FOR_EVENT = round_up_to_10(raw_box_size)
    DV_BOX_SIZE_FOR_EVENT = round_up_to_10(dv_box_size)

    print(f"Target sizes - RGB: {RGB_BOX_SIZE_FOR_EVENT}, RAW: {RAW_BOX_SIZE_FOR_EVENT}, DV: {DV_BOX_SIZE_FOR_EVENT}")

    BATCH_SIZE_FOR_EVENT = 100000

    print("Filtering RGB Events...")
    filtered_events_rgb = event_filter.filter_events_parallel(
        events_tensor=rgb_events_tensor, crops_info=rgb_crops_info,
        target_size=RGB_BOX_SIZE_FOR_EVENT, transform=True,
        batch_size=BATCH_SIZE_FOR_EVENT, n_workers=n_workers
    )

    print("Filtering RAW Events...")
    filtered_events_raw = event_filter.filter_events_parallel(
        events_tensor=raw_events_tensor, crops_info=raw_crops_info,
        target_size=RAW_BOX_SIZE_FOR_EVENT, transform=True,
        batch_size=BATCH_SIZE_FOR_EVENT, n_workers=n_workers
    )

    print("Filtering DV Events...")
    filtered_events_dv = event_filter.filter_events_parallel(
        events_tensor=dv_events_tensor, crops_info=dv_crops_info,
        target_size=DV_BOX_SIZE_FOR_EVENT, transform=True,
        batch_size=BATCH_SIZE_FOR_EVENT, n_workers=n_workers
    )

    print("Saving filtered events...")
    # Use .npz by default to save disk space and read/write time
    out_raw = os.path.join(output_dir, f"{prefix}_filtered_raw.npz")
    out_rgb = os.path.join(output_dir, f"{prefix}_filtered_rgb.npz")
    out_dv = os.path.join(output_dir, f"{prefix}_filtered_dv.npz")

    np.savez_compressed(out_raw, events=filtered_events_raw.numpy() if isinstance(filtered_events_raw, torch.Tensor) else filtered_events_raw)
    np.savez_compressed(out_rgb, events=filtered_events_rgb.numpy() if isinstance(filtered_events_rgb, torch.Tensor) else filtered_events_rgb)
    np.savez_compressed(out_dv, events=filtered_events_dv.numpy() if isinstance(filtered_events_dv, torch.Tensor) else filtered_events_dv)

    print(f"✔ Data successfully processed and saved for prefix: {prefix}")
    print(f"✔ Output files stored in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process single batch of raw2event data.")
    parser.add_argument("--input", type=str, required=True, help="Path to input directory containing MKV and DAT files")
    parser.add_argument("--prefix", type=str, required=True, help="File prefix (e.g., 18900_cat_1_9006_20251225_165232)")
    parser.add_argument("--output", type=str, default="./data/unified80", help="Directory where filtered NPZ events will be saved")
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
    process_batch(args.prefix, args.input, args.output, args.sim_backend,
                  downsample_resolution=args.downsample)
