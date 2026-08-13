def generate_events_tensor(
    timestamps,
    frames,
    is_rgb=False,
    raw_is_luminance=False,
    k_values=None,
    sim_backend='auto',
):
    """Generate event data using DVS simulator and return torch tensor
    
    Args:
        timestamps: Timestamp tensor (microseconds)
        frames: Frame data tensor [N, H, W] or [N, H, W, 3]
        is_rgb: Whether it's RGB data
        raw_is_luminance: When ``is_rgb`` is False, indicates the RAW input
            frames are already single-channel luminance Y rather than Bayer
            mosaic, so Bayer demosaic must be skipped.
        k_values (list or None, optional): List of K parameters. Either 6 elements
                                     [k1..k6] (legacy) or 8 elements
                                     [k1..k6, k_on, k_off] (R1 polarity-aware).
                                     If provided, will override default values in
                                     config.py. 6D values are auto-padded with
                                     k_on=k_off=1.0 to match legacy behavior.
                                     Default is None.
        sim_backend (str): Simulator backend. One of {'auto', 'cuda', 'cpu', 'numpy'}.
        
    Returns:
        torch.Tensor: Event data tensor [N, 4] (timestamp, x, y, polarity)
    """
    from src.simulator import EventSim
    from src.config import cfg
    from src.process_data.file_read import bayer_mosaic_to_luminance
    import numpy as np
    import torch
    import cv2
    from tqdm import tqdm

    # Override default K parameters in config if k_values provided.
    # Accept both 6D and 8D (polarity-aware) K.
    # 6D inputs are auto-padded with k_on=k_off=1.0.
    if k_values is not None:
        if len(k_values) == 6:
            if not hasattr(cfg, 'SENSOR'):
                from easydict import EasyDict as edict # type: ignore
                cfg.SENSOR = edict()
            cfg.SENSOR.K = list(k_values) + [1.0, 1.0]
            print(f"Using custom K values (6D padded to 8D): {cfg.SENSOR.K}")
        elif len(k_values) == 8:
            if not hasattr(cfg, 'SENSOR'):
                from easydict import EasyDict as edict # type: ignore
                cfg.SENSOR = edict()
            cfg.SENSOR.K = list(k_values)
            print(f"Using custom K values (8D polarity-aware): {k_values}")
        else:
            print(
                f"Warning: k_values length is {len(k_values)}, expected 6 or 8. "
                "Falling back to default K values."
            )

    
    # Initialize simulator
    sim = EventSim(cfg=cfg, output_folder=None, sim_backend=sim_backend)
    
    # Prepare frame processing
    all_events = []
    prev_frame = None
    
    # Preprocess frames into a single-channel luminance domain that matches
    # the DV sensor's monochromatic response. Both RGB and RAW Bayer inputs
    # are converted via BT.601 weights; K is calibrated in this Y domain.
    processed_frames = []
    if is_rgb:
        for frame in frames:
            frame_np = frame.numpy()
            if frame_np.ndim >= 3 and frame_np.shape[2] >= 3:
                rgb_frame = frame_np[:, :, :3]
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                processed_frames.append(gray_frame)
            else:
                if frame_np.ndim == 3 and frame_np.shape[2] == 1:
                    frame_np = frame_np[:, :, 0]
                processed_frames.append(frame_np)
    else:
        for frame in frames:
            frame_np = frame.numpy()
            if frame_np.ndim == 3 and frame_np.shape[2] == 1:
                frame_np = frame_np[:, :, 0]
            if raw_is_luminance:
                processed_frames.append(frame_np)
            else:
                processed_frames.append(bayer_mosaic_to_luminance(frame_np))
    
    # Generate events for each frame
    for i in tqdm(range(len(timestamps)), desc="Generating events", unit="frame"):
        frame = processed_frames[i]
        timestamp = int(timestamps[i].item())
        
        # Save first frame, no events generated
        if prev_frame is None:
            prev_frame = frame
            continue
        
        # Generate events
        events = sim.generate_events(frame, timestamp)
        if events is not None and len(events) > 0:
            events[:, 0] = events[:, 0].astype(np.int64)  # Ensure timestamps are integers
            all_events.append(events)
        
        prev_frame = frame
    
    # Merge and convert to torch tensor if events exist
    if all_events:
        events_np = np.concatenate(all_events, axis=0)
        events_tensor = torch.from_numpy(events_np).to(torch.int64)
        return events_tensor
    else:
        return torch.zeros((0, 4), dtype=torch.int64)
    

# Move nested functions to module level
def _process_rgb(timestamps, frames, results_dict, k_values=None, sim_backend='auto'):
    """Process RGB frames to generate event data"""
    events = generate_events_tensor(
        timestamps, frames, is_rgb=True, k_values=k_values, sim_backend=sim_backend
    )
    results_dict['rgb'] = events
    
def _process_raw(timestamps, frames, results_dict, k_values=None, sim_backend='auto'):
    """Process RAW frames to generate event data"""
    events = generate_events_tensor(
        timestamps, frames, is_rgb=False, k_values=k_values, sim_backend=sim_backend
    )
    results_dict['raw'] = events

def parallel_generate_events(pi_ts, rgb_frames, raw_frames, k_values=None, sim_backend='auto'):
    """Generate RGB and RAW event data in parallel (using torch.multiprocessing)
    
    Args:
        pi_ts: Timestamp tensor
        rgb_frames: RGB frame data tensor
        raw_frames: RAW frame data tensor
        
    Returns:
        tuple: (rgb_events, raw_events) Generated event data
    """
    import torch.multiprocessing as mp
    from torch import multiprocessing as tmp
    
    # Ensure spawn mode to avoid issues on Windows
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # Create shared results dictionary
    manager = mp.Manager()
    results = manager.dict()
    
    # Create processes
    p1 = mp.Process(target=_process_rgb, args=(pi_ts, rgb_frames, results, k_values, sim_backend))
    p2 = mp.Process(target=_process_raw, args=(pi_ts, raw_frames, results, k_values, sim_backend))
    
    # Start processes
    print("Starting parallel event data processing...")
    p1.start()
    p2.start()
    
    # Wait for processes to complete
    p1.join()
    p2.join()
    
    # Get results
    rgb_events = results.get('rgb')
    raw_events = results.get('raw')
    
    print("Parallel processing completed!")
    return rgb_events, raw_events
