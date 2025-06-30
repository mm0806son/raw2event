#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Luminance Conversion Tool (Full Screen Version)
Read RAW/RGB frame data and AEDAT4 event data, calculate luminance values for each event position
Process all events without trajectory filtering
"""



# ============= Import Libraries =============
import os
import sys
import numpy as np
import time
import torch
import multiprocessing
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import gc
import traceback
import threading


# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import track_function.common_utils as cu
from model_calibration import calib_function as cf


# ============= Parameter Configuration =============
# Image parameters
IMAGE_HEIGHT = 260  # Original image height
IMAGE_WIDTH = 346   # Original image width

# ROI parameters (Region of Interest)
ROI_HEIGHT = 100    # ROI region height
ROI_WIDTH = 100     # ROI region width
# Calculate ROI region center position and boundaries
ROI_CENTER_X = IMAGE_WIDTH // 2
ROI_CENTER_Y = IMAGE_HEIGHT // 2
ROI_LEFT = ROI_CENTER_X - ROI_WIDTH // 2
ROI_TOP = ROI_CENTER_Y - ROI_HEIGHT // 2
ROI_RIGHT = ROI_LEFT + ROI_WIDTH
ROI_BOTTOM = ROI_TOP + ROI_HEIGHT

# Input file paths
# ROOT_FOLDER = 'data/k_timeProcess_output'
ROOT_FOLDER = 'k_calib_20250509'

# Event processing parameters
MAX_EVENTS = 100000000  # Maximum number of events to process
BATCH_SIZE = 1000000    # Batch size for processing large numbers of events

# Multi-threading and GPU settings
USE_MULTIPROCESSING = True  # Whether to use multi-threading
MAX_WORKERS = 24            # Maximum number of threads
MAX_GPU_BATCH_SIZE = 10000000  # Maximum GPU batch size to prevent VRAM overflow

# Check GPU availability
USE_GPU = True
try:
    import torch
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_GPU else "cpu")
    if USE_GPU:
        print(f"GPU acceleration enabled! Using {torch.cuda.get_device_name(0)}")
        # Enable CUDA memory optimization
        torch.backends.cudnn.benchmark = True
        print("CUDA performance optimization enabled")
    else:
        print("No available GPU detected, will run in CPU mode")
except ImportError:
    print("PyTorch library not detected, will run in CPU mode")
    USE_GPU = False
    DEVICE = torch.device("cpu")

# === Batch processing helper functions ===

def find_file_by_prefix(folder_path, prefix):
    """Find the first file in a folder matching the prefix."""
    for fname in os.listdir(folder_path):
        if fname.startswith(prefix):
            return fname
    raise FileNotFoundError(f"No file starting with {prefix} found in {folder_path}")

def process_one_subfolder(folder_path):
    """Process a single subfolder (prepare all paths and call run_one_process)."""
    try:
        # Set up paths for this folder
        raw_frames_path = os.path.join(folder_path, find_file_by_prefix(folder_path, 'raw_frames'))
        rgb_frames_path = os.path.join(folder_path, find_file_by_prefix(folder_path, 'rgb_frames'))
        metadata_path = os.path.join(folder_path, find_file_by_prefix(folder_path, 'metadata'))
        aedat4_path = os.path.join(folder_path, find_file_by_prefix(folder_path, 'time_filtered_dv_events'))
        output_folder = os.path.join(folder_path, 'frames_analysis_full')
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Extract events
        events_tensor = extract_all_events_from_aedat4(aedat4_path, MAX_EVENTS)
        
        # Read frame data
        safe_print("\nReading frame data and timestamps...")
        raw_frames, rgb_frames, timestamps = cf.read_frame_data(
            raw_frames_path, rgb_frames_path, metadata_path, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        
        # Check data validity
        if not check_data_validity(events_tensor, raw_frames, rgb_frames, timestamps):
            return
        
        try:
            # Visualize ROI region
            visualize_roi(rgb_frames, output_folder)
        except Exception as e:
            safe_print(f"ROI visualization error: {str(e)}")
            traceback.print_exc()
        
        # Calculate frame statistics
        safe_print("\nCalculating frame statistics...")
        timestamps_array = np.array(timestamps) / 1e6
        
        # Process RAW and RGB frame statistics
        raw_stats = process_frame_batch(raw_frames, is_rgb=False)
        rgb_stats = process_frame_batch(rgb_frames, is_rgb=True)
        
        # Plot frame luminance comparison
        plot_frame_stats(
            timestamps_array, raw_stats,
            timestamps_array, rgb_stats,
            output_folder
        )
        
        # Parallel processing of RAW and RGB frame luminance calculation
        safe_print(f"\nSetting ROI region: {ROI_WIDTH}x{ROI_HEIGHT} @ ({ROI_LEFT},{ROI_TOP}) - ({ROI_RIGHT},{ROI_BOTTOM})")
        safe_print(f"Only process events in ROI region")
        
        raw_events_data, rgb_events_data = process_frames_parallel(
            events_tensor, raw_frames, rgb_frames, timestamps
        )
        
        # Save results
        if raw_events_data is not None:
            save_event_data(raw_events_data, os.path.join(output_folder, "events_with_luminance_raw.pt"))
        if rgb_events_data is not None:
            save_event_data(rgb_events_data, os.path.join(output_folder, "events_with_luminance_rgb.pt"))
        
        safe_print(f"✅ Successfully processed folder: {folder_path}")
        
    except Exception as e:
        safe_print(f"Error processing folder {folder_path}: {str(e)}")
        traceback.print_exc()
        raise  # Re-raise the exception to be caught by the process pool

def batch_process_all_subfolders(root_folder):
    """Batch process all subfolders under root_folder, with progress bar."""
    subfolders = [sub for sub in sorted(os.listdir(root_folder)) 
                  if os.path.isdir(os.path.join(root_folder, sub))]
    
    if not subfolders:
        safe_print(f"No subfolders found under {root_folder}.")
        return
    
    safe_print(f"\n📂 Found {len(subfolders)} subfolders. Start processing...\n")
    
    # Create a process pool with as many processes as there are subfolders
    # Note: You might want to limit this if you have many subfolders
    num_processes = min(len(subfolders), os.cpu_count())
    safe_print(f"Using {num_processes} processes for parallel processing")
    
    # Use ProcessPoolExecutor for parallel processing
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes, 
                                              mp_context=ctx) as executor:
        # Create a list to store all futures
        futures = []
        
        # Submit all folder processing tasks
        for subfolder in subfolders:
            subfolder_path = os.path.join(root_folder, subfolder)
            future = executor.submit(process_one_subfolder, subfolder_path)
            futures.append((subfolder, future))
        
        # Monitor progress
        completed = 0
        with tqdm(total=len(subfolders), desc="Processing Folders") as pbar:
            for subfolder, future in futures:
                try:
                    future.result()  # Wait for completion
                    safe_print(f"✅ Completed processing folder: {subfolder}")
                except Exception as e:
                    safe_print(f"❌ Failed to process {subfolder}: {str(e)}")
                    traceback.print_exc()
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(f"Completed: {completed}/{len(subfolders)}")

# ============= Thread-safe print function =============
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

# ============= Helper functions =============
def gpu_batch_process(func, data, batch_size=MAX_GPU_BATCH_SIZE, *args, **kwargs):
    """GPU batch processing function to prevent VRAM overflow"""
    if data is None or len(data) == 0:
        return None
        
    # If data size is small, process directly
    if len(data) <= batch_size or DEVICE.type != 'cuda':
        return func(data, *args, **kwargs)
        
    # Otherwise batch process
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="GPU batch processing"):
        batch = data[i:i+batch_size]
        batch_result = func(batch, *args, **kwargs)
        if batch_result is not None:
            results.append(batch_result)
        # Free VRAM
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
    # Merge results
    if not results:
        return None
    if isinstance(results[0], torch.Tensor):
        return torch.cat(results)
    return results

# ============= Simplified luminance calculation functions =============
def get_rgb_luminance(frame, x, y):
    """
    Simplified RGB luminance calculation - directly use OpenCV's BGR2GRAY conversion
    
    Parameters:
        frame: RGB frame (may be Tensor or NumPy array)
        x, y: coordinates
        
    Returns:
        grayscale value (0-255)
    """
    if frame is None:
        raise ValueError('Frame is None!')
    
    # Ensure frame is NumPy array
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()  # Convert from GPU tensor to CPU numpy array

    if not isinstance(frame, np.ndarray):
        raise TypeError(f'Frame is not a valid numpy array! Type: {type(frame)}')
    
    if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
        try:
            # Convert to grayscale
            if len(frame.shape) == 3 and frame.shape[2] == 3:  # Color image
                # Ensure frame is contiguous memory block
                frame = np.ascontiguousarray(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return int(gray[y, x])
            elif len(frame.shape) == 2:  # Already grayscale
                return int(frame[y, x])
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
        except Exception as e:
            safe_print(f"Error converting RGB luminance: {e}")
            return None
    return None

def get_raw_luminance(frame, x, y):
    """
    Simplified RAW luminance calculation - directly use 10-bit raw values
    
    Parameters:
        frame: RAW frame (may be Tensor or NumPy array)
        x, y: coordinates
        
    Returns:
        raw RAW value (0-1023)
    """
    # Ensure frame is NumPy array
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    
    if frame is None:
        return None
        
    try:
        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
            return int(frame[y, x])
    except Exception as e:
        safe_print(f"Error getting RAW luminance: {e}")
    return None

# ============= Event processing functions =============
def extract_all_events_from_aedat4(aedat4_path, max_events=MAX_EVENTS):
    """
    Extract all event data from AEDAT4 file
    
    Parameters:
        aedat4_path: AEDAT4 file path
        max_events: maximum number of events to process
        
    Returns:
        events_tensor: tensor containing event data [timestamp, x, y, polarity]
    """
    safe_print("\nExtracting all events from AEDAT4 file...")
    
    try:
        import dv_processing as dv
        
        if not os.path.exists(aedat4_path):
            safe_print(f"Error: Input file {aedat4_path} does not exist")
            return None
            
        # Open AEDAT4 file
        recording = dv.io.MonoCameraRecording(dv.Path(str(aedat4_path)))
        if not recording.isEventStreamAvailable():
            safe_print("Error: No event stream in AEDAT4 file")
            return None
            
        # Prepare data structure
        all_events = []
        event_count = 0
        
        # Process events in batches
        with tqdm(desc="Extracting event data", unit="batch") as pbar:
            while True:
                # Check if maximum event count reached
                if event_count >= max_events:
                    safe_print(f"Reached maximum event processing count: {max_events}")
                    break
                
                # Get next batch of events
                events = recording.getNextEventBatch()
                if events is None or len(events) == 0:
                    break  # No more events
                
                # Convert events to numpy array
                events_data = events.numpy()
                if len(events_data) == 0:
                    continue
                
                # Extract event information (timestamp, x, y, polarity)
                batch_events = np.column_stack((
                    events_data['timestamp'],
                    events_data['x'],
                    events_data['y'],
                    events_data['polarity']
                ))
                
                all_events.append(batch_events)
                event_count += len(batch_events)
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({"Total events": event_count})
                
                # If maximum event count reached, exit loop
                if event_count >= max_events:
                    safe_print(f"Reached maximum event count: {max_events}")
                    break
        
        # Merge all batches of events
        if all_events:
            all_events_np = np.vstack(all_events)
            events_tensor = torch.tensor(all_events_np, dtype=torch.float32)
            safe_print(f"Event extraction complete! Total events: {len(events_tensor)}")
            return events_tensor.to(DEVICE)
        else:
            safe_print("No events found")
            return None
            
    except ImportError:
        safe_print("Error: DV library not installed, cannot read AEDAT4 file")
        return None
    except Exception as e:
        safe_print(f"Error extracting events: {e}")
        traceback.print_exc()
        return None

def process_event_batch(batch_events, frames, timestamps, is_rgb=False):
    """
    Process a batch of event data, calculate corresponding luminance values
    
    Parameters:
        batch_events: batch event data [timestamp, x, y, polarity]
        frames: frame list (RAW or RGB)
        timestamps: frame timestamp list
        is_rgb: whether it's RGB frames
        
    Returns:
        valid_results: processing results [timestamp, x, y, polarity, prev_lum, next_lum, frame_time_diff], only valid events
        valid_count: number of valid events
    """
    frame_type = "RGB" if is_rgb else "RAW"
    
    # Convert timestamps to numpy array for easier indexing
    timestamps_array = np.array(timestamps)
    
    # Extract event data
    event_timestamps = batch_events[:, 0].cpu().numpy()
    x_coords = batch_events[:, 1].cpu().numpy()
    y_coords = batch_events[:, 2].cpu().numpy()
    
    # Create result list for valid events
    valid_results = []
    valid_count = 0
    roi_count = 0 # Count events in ROI region
    
    # Batch calculate frame indices
    indices = np.searchsorted(timestamps_array, event_timestamps)
    indices = np.clip(indices, 1, len(timestamps_array)-1)
    prev_indices = indices - 1
    next_indices = indices
    
    # Calculate interpolation weights
    prev_ts = timestamps_array[prev_indices]
    next_ts = timestamps_array[next_indices]
    
    # Calculate frame time difference
    frame_time_diffs = next_ts - prev_ts
    
    # Pre-fetch required frames
    unique_prev = np.unique(prev_indices)
    unique_next = np.unique(next_indices)
    unique_frames = np.union1d(unique_prev, unique_next)
    frames_dict = {idx: frames[idx] for idx in unique_frames if 0 <= idx < len(frames)}
    
    # Calculate luminance values
    for i in range(len(batch_events)):
        prev_idx = prev_indices[i]
        next_idx = next_indices[i]
        x, y = int(round(x_coords[i])), int(round(y_coords[i]))
        
        # Check if in ROI region
        in_roi = (ROI_LEFT <= x < ROI_RIGHT and ROI_TOP <= y < ROI_BOTTOM)
        
        if in_roi:
            roi_count += 1
            
            # Check if there are previous and next frames
            if prev_idx in frames_dict and next_idx in frames_dict:
                prev_frame = frames_dict[prev_idx]
                next_frame = frames_dict[next_idx]
                
                # Calculate luminance values - use simplified method
                if is_rgb:
                    prev_lum = get_rgb_luminance(prev_frame, x, y)
                    next_lum = get_rgb_luminance(next_frame, x, y)
                else:
                    prev_lum = get_raw_luminance(prev_frame, x, y)
                    next_lum = get_raw_luminance(next_frame, x, y)
                
                # Ensure luminance values are valid
                if prev_lum is not None and next_lum is not None:
                    # Create result tensor
                    result = torch.zeros(7, dtype=torch.float32, device=DEVICE)
                    result[:4] = batch_events[i]  # Copy original event data
                    result[4] = prev_lum
                    result[5] = next_lum
                    result[6] = frame_time_diffs[i]
                    
                    valid_results.append(result)
                    valid_count += 1
    
    # Merge valid results into tensor
    if valid_results:
        return torch.stack(valid_results), valid_count, roi_count
    else:
        # Return empty tensor, 0 valid count and ROI count
        return torch.zeros((0, 7), dtype=torch.float32, device=DEVICE), 0, roi_count

def calculate_luminance_for_events(events_tensor, frames, timestamps, is_rgb=False, batch_size=BATCH_SIZE):
    """
    Calculate corresponding frame luminance values for event data, use batch processing for large numbers of events
    
    Parameters:
        events_tensor: tensor containing event data [timestamp, x, y, polarity]
        frames: frame list (RAW or RGB)
        timestamps: frame timestamp list
        is_rgb: whether it's RGB frames
        batch_size: batch size
        
    Returns:
        result_tensor: tensor containing event data and luminance values [timestamp, x, y, polarity, prev_lum, next_lum, frame_time_diff]
    """
    # Use gpu_batch_process to handle data
    return gpu_batch_process(
        lambda data: _process_events_luminance(data, frames, timestamps, is_rgb, batch_size),
        events_tensor,
        MAX_GPU_BATCH_SIZE
    )

def _process_events_luminance(events_tensor, frames, timestamps, is_rgb=False, batch_size=BATCH_SIZE):
    """Actual processing function called internally by GPU batch processing"""
    frame_type = "RGB" if is_rgb else "RAW"
    safe_print(f"\nStarting {frame_type} frame luminance calculation...")
    
    num_events = events_tensor.shape[0]
    results = []
    total_valid = 0
    total_roi = 0  # Calculate total events in ROI region
    
    # Process events in batches
    for batch_start in tqdm(range(0, num_events, batch_size), desc=f"Processing {frame_type} luminance"):
        batch_end = min(batch_start + batch_size, num_events)
        batch_events = events_tensor[batch_start:batch_end]
        
        # Process batch, only get valid events
        batch_results, valid_count, roi_count = process_event_batch(batch_events, frames, timestamps, is_rgb)
        total_roi += roi_count
        
        # Only add to result list if there are valid results
        if valid_count > 0:
            results.append(batch_results.cpu())
            total_valid += valid_count
        
        # Periodically clean GPU memory
        if USE_GPU and batch_start % (batch_size * 10) == 0:
            torch.cuda.empty_cache()
    
    # Merge results
    if results and total_valid > 0:
        result_tensor = torch.cat(results, dim=0)
        safe_print(f"{frame_type} frame luminance calculation complete, {total_roi}/{num_events} events in ROI region ({ROI_WIDTH}x{ROI_HEIGHT})")
        safe_print(f"Successfully processed {total_valid}/{total_roi} ROI events, remaining events ignored due to missing frames or luminance calculation issues")
        return result_tensor
    else:
        safe_print(f"Could not calculate any {frame_type} frame luminance values, {total_roi}/{num_events} events in ROI region ({ROI_WIDTH}x{ROI_HEIGHT})")
        return None

def process_frames_parallel(events_tensor, raw_frames, rgb_frames, timestamps):
    """
    Parallel processing of RAW and RGB frame luminance calculation
    
    Parameters:
        events_tensor: tensor containing event data [timestamp, x, y, polarity]
        raw_frames: RAW frame list
        rgb_frames: RGB frame list
        timestamps: frame timestamp list
        
    Returns:
        raw_events_data: event data containing RAW frame luminance values
        rgb_events_data: event data containing RGB frame luminance values
    """
    safe_print("\nStarting parallel calculation of RAW and RGB frame corresponding luminance values...")
    
    # Define function to process RAW frames
    def process_raw_frames(events_tensor):
        return calculate_luminance_for_events(events_tensor, raw_frames, timestamps, is_rgb=False)
    
    # Define function to process RGB frames
    def process_rgb_frames(events_tensor):
        return calculate_luminance_for_events(events_tensor, rgb_frames, timestamps, is_rgb=True)
    
    raw_events_data = None
    rgb_events_data = None
    
    if USE_MULTIPROCESSING:
        # Use thread pool for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks
            raw_future = executor.submit(process_raw_frames, events_tensor)
            rgb_future = executor.submit(process_rgb_frames, events_tensor)
            
            # Get results (this will wait for threads to complete)
            raw_events_data = raw_future.result()
            rgb_events_data = rgb_future.result()
    else:
        # Sequential processing
        raw_events_data = process_raw_frames(events_tensor)
        # Free no longer needed data to save memory
        if USE_GPU:
            torch.cuda.empty_cache()
        gc.collect()
        
        rgb_events_data = process_rgb_frames(events_tensor)
    
    # Check processing results
    if raw_events_data is None or rgb_events_data is None:
        safe_print("Failed to process luminance values")
        return None, None
    
    # Move results back to device for saving
    if USE_GPU:
        raw_events_data = raw_events_data.to(DEVICE)
        rgb_events_data = rgb_events_data.to(DEVICE)
    
    safe_print(f"RAW frame luminance calculation complete, processed {len(raw_events_data)} events")
    safe_print(f"RGB frame luminance calculation complete, processed {len(rgb_events_data)} events")
    
    return raw_events_data, rgb_events_data

def save_event_data(event_data, output_file):
    """
    Save event data to file
    
    Parameters:
        event_data: event data tensor
        output_file: output file path
    """
    try:
        # Save event data as Torch tensor
        torch.save(event_data.cpu(), output_file)
        safe_print(f"Event data saved to: {output_file}")
        safe_print(f"Contains {len(event_data)} events, each event contains: [timestamp(microseconds), x_coordinate, y_coordinate, polarity, prev_luminance, next_luminance, time_diff_between_frames]")
        return True
    except Exception as e:
        safe_print(f"Error saving event data: {e}")
        return False

def check_data_validity(events_tensor, raw_frames, rgb_frames, timestamps):
    """
    Check if loaded data is valid
    
    Parameters:
        events_tensor: event data
        raw_frames: RAW frame data
        rgb_frames: RGB frame data
        timestamps: timestamp data
        
    Returns:
        valid: whether data is valid
    """
    valid = True
    
    if events_tensor is None or len(events_tensor) == 0:
        safe_print("Cannot extract event data or no valid events")
        valid = False
    else:
        safe_print(f"Successfully extracted {len(events_tensor)} events")
        safe_print("Event format: (timestamp, x, y, polarity)")
    
    if raw_frames is None or len(raw_frames) == 0:
        safe_print("Cannot read RAW frame data")
        valid = False
    
    if rgb_frames is None or len(rgb_frames) == 0:
        safe_print("Cannot read RGB frame data")
        valid = False
    
    if timestamps is None or len(timestamps) == 0:
        safe_print("Cannot read timestamp data")
        valid = False
    
    if valid:
        safe_print(f"Read {len(raw_frames)} RAW frames, {len(rgb_frames)} RGB frames, {len(timestamps)} timestamps")
    
    return valid

def process_frame_batch(frames, is_rgb=False):
    """
    Process a batch of frame data and calculate statistics
    
    Parameters:
        frames: frame list (may be Tensor or NumPy array)
        is_rgb: whether it's RGB frames
        
    Returns:
        stats: dictionary containing statistics
    """
    means = []
    vars = []
    ranges = []
    
    for frame in tqdm(frames, desc=f"{'RGB' if is_rgb else 'RAW'} frame processing"):
        # Convert Tensor to NumPy array (if needed)
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            
        if is_rgb:
            # Convert RGB frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Normalize
            mean_value = np.mean(gray_frame) / 255.0
            var_value = np.var(gray_frame) / (255.0 * 255.0)
            range_value = (np.max(gray_frame) - np.min(gray_frame)) / 255.0
        else:
            # RAW frame is already grayscale
            # Normalize
            mean_value = np.mean(frame) / 1023.0  # 10-bit raw values range 0-1023
            var_value = np.var(frame) / (1023.0 * 1023.0)
            range_value = (np.max(frame) - np.min(frame)) / 1023.0
        
        means.append(mean_value)
        vars.append(var_value)
        ranges.append(range_value)
    
    return {'mean': np.array(means), 'var': np.array(vars), 'range': np.array(ranges)}

def plot_frame_stats(raw_timestamps, raw_stats, rgb_timestamps, rgb_stats, output_dir):
    """Plot frame luminance statistics"""
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Subplot 1: Mean luminance
    ax = axes[0]
    ax.plot(raw_timestamps, raw_stats['mean'], 'r.-', label='RAW Frames')
    ax.plot(rgb_timestamps, rgb_stats['mean'], 'g.-', label='RGB Frames')
    ax.set_ylabel('Normalized Mean Luminance')
    ax.set_title('Mean Luminance Comparison of Different Frame Types')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # Subplot 2: Luminance variance
    ax = axes[1]
    ax.plot(raw_timestamps, raw_stats['var'], 'r.-', label='RAW Frames')
    ax.plot(rgb_timestamps, rgb_stats['var'], 'g.-', label='RGB Frames')
    ax.set_ylabel('Normalized Luminance Variance')
    ax.set_title('Luminance Variance Comparison of Different Frame Types')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    # Subplot 3: Luminance range
    ax = axes[2]
    ax.plot(raw_timestamps, raw_stats['range'], 'r.-', label='RAW Frames')
    ax.plot(rgb_timestamps, rgb_stats['range'], 'g.-', label='RGB Frames')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Normalized Luminance Range')
    ax.set_title('Luminance Range Comparison of Different Frame Types')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(os.path.join(output_dir, 'frame_luminance_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'frame_luminance_comparison.pdf'), format='pdf')
    
    plt.close()

def visualize_roi(frames, output_folder):
    """
    Visualize ROI region, select one frame and mark ROI position
    
    Parameters:
        frames: frame list (may be Tensor or NumPy array)
        output_folder: output directory
    """
    if not frames or len(frames) == 0:
        safe_print("No available frames, cannot visualize ROI")
        return
    
    safe_print("Generating ROI visualization image...")
    
    # Select middle frame as example
    frame_idx = len(frames) // 2
    
    # Handle possible Tensor type
    if isinstance(frames[frame_idx], torch.Tensor):
        # If it's a PyTorch Tensor, first convert to NumPy array
        frame = frames[frame_idx].cpu().numpy()
    else:
        # If already NumPy array, copy directly
        frame = frames[frame_idx].copy()
    
    # If it's an RGB frame, use directly; if it's a RAW frame, convert to visualization format
    if len(frame.shape) == 2:  # RAW frame
        # Normalize to 0-255
        frame_vis = (frame * 255.0 / 1023.0).astype(np.uint8)
        # Convert to color image to draw colored border
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_GRAY2BGR)
    else:  # RGB frame
        # Ensure BGR order (OpenCV default)
        if isinstance(frames[frame_idx], torch.Tensor):
            # If it's an RGB Tensor from PyTorch, may need to adjust channel order
            frame_vis = np.ascontiguousarray(frame) # Ensure data is contiguous
        else:
            frame_vis = frame
    
    # Mark ROI region (draw green rectangle)
    cv2.rectangle(
        frame_vis, 
        (ROI_LEFT, ROI_TOP), 
        (ROI_RIGHT, ROI_BOTTOM), 
        (0, 255, 0), 
        2
    )
    
    # Add text annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame_vis, 
        f"ROI: {ROI_WIDTH}x{ROI_HEIGHT}", 
        (ROI_LEFT, ROI_TOP - 10), 
        font, 
        0.5, 
        (0, 255, 0), 
        1, 
        cv2.LINE_AA
    )
    
    # Mark image center (draw red cross)
    center_x, center_y = IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2
    cv2.drawMarker(
        frame_vis, 
        (center_x, center_y), 
        (0, 0, 255), 
        cv2.MARKER_CROSS, 
        10, 
        2
    )
    
    # Save image
    output_path = os.path.join(output_folder, "roi_visualization.png")
    cv2.imwrite(output_path, frame_vis)
    
    safe_print(f"ROI visualization image saved: {output_path}")

# ============= Main function =============
def main():
    """Main function to start batch processing."""
    start_time = time.time()
    batch_process_all_subfolders(ROOT_FOLDER)
    elapsed_time = time.time() - start_time
    safe_print(f"\n🎉 All processing completed in {elapsed_time:.2f} seconds!")

if __name__ == "__main__":
    if hasattr(cu, 'check_module_imports'):
        cu.check_module_imports()
    main()
