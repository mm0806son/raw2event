def generate_events_tensor(timestamps, frames, is_rgb=False, k_values=None):
    """Generate event data using DVS simulator and return torch tensor
    
    Args:
        timestamps: Timestamp tensor (microseconds)
        frames: Frame data tensor [N, H, W] or [N, H, W, 4]
        is_rgb: Whether it's RGB data
        k_values (list or None, optional): List containing 6 k parameters [k1, k2, k3, k4, k5, k6].
                                     If provided, will override default values in config.py.
                                     Default is None.
        
    Returns:
        torch.Tensor: Event data tensor [N, 4] (timestamp, x, y, polarity)
    """
    from src.simulator import EventSim
    from src.config import cfg
    import numpy as np
    import torch
    import cv2
    from tqdm import tqdm

    # Override default K parameters in config if k_values provided
    if k_values is not None:
        if len(k_values) == 6:
            if not hasattr(cfg, 'SENSOR'):
                from easydict import EasyDict as edict # type: ignore
                cfg.SENSOR = edict()
            cfg.SENSOR.K = k_values
            print(f"Using custom K values: {k_values}")
        else:
            print("Warning: k_values length is not 6, will use default K values.")

    
    # Initialize simulator
    sim = EventSim(cfg=cfg, output_folder=None)
    
    # Prepare frame processing
    all_events = []
    prev_frame = None
    
    # Preprocess frames (if RGB)
    processed_frames = []
    if is_rgb:
        for frame in frames:
            frame_np = frame.numpy()
            rgb_frame = frame_np[:, :, :3]  # Remove alpha channel
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
            processed_frames.append(gray_frame)
    else:
        processed_frames = [frame.numpy() for frame in frames]
    
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
def _process_rgb(timestamps, frames, results_dict, k_values=None):
    """Process RGB frames to generate event data"""
    events = generate_events_tensor(timestamps, frames, is_rgb=True, k_values=k_values)
    results_dict['rgb'] = events
    
def _process_raw(timestamps, frames, results_dict, k_values=None):
    """Process RAW frames to generate event data"""
    events = generate_events_tensor(timestamps, frames, is_rgb=False, k_values=k_values)
    results_dict['raw'] = events

def parallel_generate_events(pi_ts, rgb_frames, raw_frames, k_values=None):
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
    p1 = mp.Process(target=_process_rgb, args=(pi_ts, rgb_frames, results, k_values))
    p2 = mp.Process(target=_process_raw, args=(pi_ts, raw_frames, results, k_values))
    
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