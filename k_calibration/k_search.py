"""
This script uses Optuna to search for optimal K values for Raw2Event k parameters.
"""
import os
import sys
import math
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import numpy as np

import torch
import optuna
import pykeops
from tqdm.notebook import tqdm
from geomloss import SamplesLoss

# Local module
sys.path.append('./src/process_data')
import src.process_data.file_read as file_read
import src.process_data.dvs_generate as dvs_generate
import src.process_data.event_filter as event_filter
import src.process_data.tag_detector as tag_detector

def parse_args():
    parser = argparse.ArgumentParser(description='DVS event generation parameter optimization')
    # File related parameters
    parser.add_argument('--input', type=str, default="barbara",
                      help='Input folder path')
    parser.add_argument('--file', type=str, default="133009",
                      help='File suffix')
    parser.add_argument('--output', type=str, default='barbara/k_search_grid_output_optuna',
                      help='Optuna result output path')
    # Mode parameter
    parser.add_argument('--mode', type=str, default='raw', choices=['raw', 'rgb'],
                      help='Processing mode: raw or rgb')
    # EMD related parameters
    parser.add_argument('--emd_blur', type=float, default=0.01,
                      help='Blur parameter for EMD calculation')
    parser.add_argument('--emd_scaling', type=float, default=0.9,
                      help='Scaling parameter for EMD calculation')
    parser.add_argument('--max_pts', type=int, default=int(5e6),
                      help='Maximum number of points limit')
    # Optuna optimization parameters
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of Optuna optimization trials')
    parser.add_argument('--timeout', type=int, default=None,
                      help='Optuna optimization timeout (seconds), None means unlimited')
    return parser.parse_args()

# ============= Hyperparameter configuration =============
# Image parameters
PI_IMAGE_WEIGHT = 692
PI_IMAGE_HEIGHT = 520
# AprilTag and Barbara parameters
TAG_REF_WIDTH = 287      # AprilTag reference width (pixels)
BARBARA_REF_SIZE = 861   # Barbara reference side length (pixels)
BARBARA_GAP = 82         # Gap between Barbara and tag (pixels)
MARGIN_RATIO = 0.03
# Parallel processing parameters
N_WORKERS = 4            # Number of threads
BATCH_SIZE_FOR_EVENT = 100000
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: CUDA not available. Running on CPU. EMD might be slow.")

# Random seed
np.random.seed(42)
torch.manual_seed(42)

def round_up_to_10(x):
    return int(math.ceil(x / 10.0) * 10)

def normalize_events(ev_tensor):
    """Normalize event data to [0,1] range"""
    ev_norm = ev_tensor.clone()
    if ev_norm.numel() == 0 or ev_norm.shape[1] < 3:
        return ev_norm
    for i in range(3):
        min_val = ev_norm[:, i].min()
        max_val = ev_norm[:, i].max()
        if max_val > min_val:
            ev_norm[:, i] = (ev_norm[:, i] - min_val) / (max_val - min_val)
        else:
            ev_norm[:, i] = 0.0
    return ev_norm

def process_frame_data(detector, frames, timestamps, is_raw):
    """Process frame data and extract box information"""
    # Process the first frame to get reference information
    barbara_info, cropped, ts = tag_detector.process_frame(
        frames[0].numpy(), timestamps[0], detector, 
        margin_ratio=MARGIN_RATIO, 
        tag_ref_width=TAG_REF_WIDTH, 
        barbara_ref_size=BARBARA_REF_SIZE, 
        barbara_gap=BARBARA_GAP,
        is_raw=is_raw
    )
    
    # Prepare batch parameters
    margin_ratios = [MARGIN_RATIO] * N_WORKERS
    tag_ref_widths = [TAG_REF_WIDTH] * N_WORKERS
    barbara_ref_sizes = [BARBARA_REF_SIZE] * N_WORKERS
    barbara_gaps = [BARBARA_GAP] * N_WORKERS
    is_raws = [is_raw] * N_WORKERS
    
    frame_batches = tag_detector.split_batches(frames.numpy(), N_WORKERS)
    ts_batches = tag_detector.split_batches(timestamps, N_WORKERS)
    
    # Batch processing
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        all_results = []
        for batch_result in tqdm(
            executor.map(
                tag_detector.process_batch,
                frame_batches, ts_batches,
                margin_ratios, tag_ref_widths, barbara_ref_sizes, barbara_gaps, is_raws
            ),
            total=N_WORKERS
        ):
            all_results.append(batch_result)
    
    # Merge results and sort
    crops_info = [item for batch in all_results for item in batch]
    crops_info.sort(key=lambda x: x[2])
    
    return barbara_info, crops_info

def objective(trial, frames, crops_info, box_size, is_rgb,
             dv_filtered, pi_timestamps, sinkhorn_loss_fn, args):
    """Optuna objective function"""
    # Optuna parameter suggestion
    # k_params = [
    #     trial.suggest_float("k1", 1e-4, 1e2, log=True),
    #     trial.suggest_float("k2", 1e-4, 1e2, log=True),
    #     trial.suggest_float("k3", 1e-6, 1e-4, log=True),
    #     trial.suggest_float("k4", 1e-8, 1e-6, log=True),
    #     trial.suggest_float("k5", 1e-11, 1e-7, log=True),
    #     trial.suggest_float("k6", 1e-7, 1e-3, log=True),
    # ] # warmup

    # k_params = [
    #     trial.suggest_float("k1", 1e-1, 1e3, log=True),
    #     trial.suggest_float("k2", 1e-3, 1e2, log=True),
    #     trial.suggest_float("k3", 1e-8, 1e-5, log=True),
    #     trial.suggest_float("k4", 1e-8, 1e-6, log=True),
    #     trial.suggest_float("k5", 1e-11, 1e-8, log=True),
    #     trial.suggest_float("k6", 1e-8, 1e-5, log=True),
    # ] # rgb

    k_params = [
        trial.suggest_float("k1", 1e-1, 1e2, log=True),
        trial.suggest_float("k2", 1e-8, 1e-6, log=True),
        trial.suggest_float("k3", 1e-8, 1e-4, log=True),
        trial.suggest_float("k4", 1e-8, 1e-6, log=True),
        trial.suggest_float("k5", 1e-13, 1e-5, log=True),
        trial.suggest_float("k6", 1e-7, 1e-3, log=True),
    ] # raw
    
    # Generate and filter events
    generated_events = dvs_generate.generate_events_tensor(
        pi_timestamps, frames, is_rgb=is_rgb, k_values=k_params
    )
    
    # Filter generated events, only keep events within DV timestamp range
    generated_events = generated_events[(generated_events[:, 0] <= dv_filtered[-1, 0]) & (generated_events[:, 0] >= dv_filtered[0, 0])]
    
    filtered_events = event_filter.filter_events_parallel(
        generated_events, crops_info, box_size,
        transform=True, batch_size=BATCH_SIZE_FOR_EVENT, n_workers=N_WORKERS
    )
    
    if filtered_events is None or dv_filtered is None or \
       filtered_events.shape[0] == 0 or dv_filtered.shape[0] == 0:
        return float("inf")
    
    # Normalize and downsample
    gen_pts = normalize_events(torch.from_numpy(filtered_events))[:, :3]
    dv_pts = normalize_events(torch.from_numpy(dv_filtered))[:, :3]
    
    print(f"  Calculating EMD: Synthetic points {gen_pts.shape[0]}, DV points {dv_pts.shape[0]}") # Print point count info

    if gen_pts.shape[0] > args.max_pts:
        gen_pts = gen_pts[torch.randperm(gen_pts.shape[0])[:args.max_pts]]
        return float('inf')
    if dv_pts.shape[0] > args.max_pts:
        dv_pts = dv_pts[torch.randperm(dv_pts.shape[0])[:args.max_pts]]
        print(f"ERROR: DV points exceed max_pts") # Runtime error
        return float('inf') # Set EMD to infinity
    
    gen_pts = gen_pts.to(device, dtype=torch.float32)
    dv_pts = dv_pts.to(device, dtype=torch.float32)

    # Calculate EMD
    start_time = time.time()
    try:
        w_gen = torch.full((gen_pts.shape[0],), 1.0/gen_pts.shape[0], device=device)
        w_dv = torch.full((dv_pts.shape[0],), 1.0/dv_pts.shape[0], device=device)
        emd = sinkhorn_loss_fn(w_gen, gen_pts, w_dv, dv_pts)
        emd_distance = emd.item()
        emd_calc_time = time.time() - start_time
        print(f"  Sinkhorn-EMD: {emd_distance}, Time: {emd_calc_time:.2f}s")
        return emd_distance
    except RuntimeError as e: # 捕获运行时错误
        print(f"  Runtime error during EMD calculation: {e}") # Runtime error
        return float('inf') # Set EMD to infinity
    

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    try:
        files = file_read.find_matching_files(args.input, args.file)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    dv_frames, dv_frames_timestamps = file_read.load_frames(files['dv'])
    dv_events_tensor = file_read.load_events(files['dv'])
    rgb_frames = file_read.read_rgb_frames(files['rgb_frames'], PI_IMAGE_HEIGHT, PI_IMAGE_WEIGHT)
    raw_frames = file_read.read_raw_frames(files['raw_frames'], PI_IMAGE_HEIGHT, PI_IMAGE_WEIGHT)
    pi_timestamps, real_timestamps = file_read.read_metadata(files['metadata'])
    
    # Timestamp alignment
    time_offset = file_read.calculate_time_offset(pi_timestamps, real_timestamps)
    print(f"Time offset: {time_offset} us")
    dv_frames_timestamps = dv_frames_timestamps - time_offset
    dv_events_tensor[:, 0] = dv_events_tensor[:, 0] - time_offset
    
    # Create detector
    detector = tag_detector.create_detector()
    
    # Process DV frame data
    barbara_info_dv, dv_crops_info = process_frame_data(
        detector, dv_frames, dv_frames_timestamps, is_raw=False
    )
    dv_box_size = round_up_to_10(max(barbara_info_dv['polygon'].ptp(axis=0)))
    
    # Calculate dv_filtered
    dv_filtered = event_filter.filter_events_parallel(
        dv_events_tensor, dv_crops_info, dv_box_size,
        transform=True, batch_size=BATCH_SIZE_FOR_EVENT, n_workers=N_WORKERS
    )
    
    # Process data according to mode
    if args.mode == 'raw':
        print("Processing RAW data...")
        barbara_info, crops_info = process_frame_data(
            detector, raw_frames, pi_timestamps, is_raw=True
        )
        box_size = round_up_to_10(max(barbara_info['polygon'].ptp(axis=0)))
        frames = raw_frames
        is_rgb = False
    else:  # rgb mode
        print("Processing RGB data...")
        barbara_info, crops_info = process_frame_data(
            detector, rgb_frames, pi_timestamps, is_raw=False
        )
        box_size = round_up_to_10(max(barbara_info['polygon'].ptp(axis=0)))
        frames = rgb_frames
        is_rgb = True
    
    # Create Sinkhorn loss function instance
    sinkhorn_loss_fn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=args.emd_blur,
        scaling=args.emd_scaling,
        backend="auto"
    )
    
    # Create Optuna study object
    study = optuna.create_study(
        direction="minimize",
        study_name=f"dvs_k_param_optimization_{args.mode}",
        storage=f"sqlite:///optuna_k_search_{args.mode}.db",
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, frames, crops_info, box_size, is_rgb,
                              dv_filtered, pi_timestamps, sinkhorn_loss_fn, args),
        n_trials=args.n_trials,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main()
