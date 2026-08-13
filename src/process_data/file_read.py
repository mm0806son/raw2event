import torch
import numpy as np
import os
import cv2
import dv_processing as dv
from datetime import datetime
import glob


def bayer_mosaic_to_luminance(frames):
    """Debayer an RGGB mosaic stack to single-channel luminance Y.

    Converts Bayer-filtered RAW frames (as produced by ``read_raw_frames`` and
    ``generate_event.load_from_video`` for ``*_raw_10bit.mkv`` files) into the
    BT.601 luminance domain that matches the monochromatic DV sensor. This is
    the input domain against which ``Raw2DVS346`` K is calibrated since the
    Y-domain switch; passing a Bayer mosaic directly into the simulator is a
    spatial-spectral aliasing error.

    Args:
        frames: ndarray or torch.Tensor shaped (H, W) or (N, H, W). Any
            integer/float dtype is accepted; values are cast to uint16 before
            OpenCV demosaic.

    Returns:
        np.ndarray of the same shape as ``frames`` (minus any singleton
        channel dim) with dtype uint16.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    frames = np.asarray(frames)

    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[..., 0]

    if frames.ndim not in (2, 3):
        raise ValueError(
            f"bayer_mosaic_to_luminance expects (H, W) or (N, H, W), got shape {frames.shape}"
        )

    if frames.dtype != np.uint16:
        # cv2's BayerRG2GRAY supports uint8/uint16; cast preserves 10-bit range.
        frames = frames.astype(np.uint16, copy=False)

    if frames.ndim == 2:
        return cv2.cvtColor(frames, cv2.COLOR_BayerRG2GRAY)

    out = np.empty(frames.shape, dtype=np.uint16)
    for i in range(frames.shape[0]):
        out[i] = cv2.cvtColor(frames[i], cv2.COLOR_BayerRG2GRAY)
    return out


def load_frames(file_path: str) -> tuple:
    """Load frame data from AEDAT4 file
    
    Args:
        file_path: AEDAT4 file path
        
    Returns:
        tuple: (frames_tensor, frames_timestamps_tensor)
    """
    recording = dv.io.MonoCameraRecording(dv.Path(str(file_path)))
    if not recording.isFrameStreamAvailable():
        return None, None
        
    frames_list = []
    frames_timestamps = []
    
    while True:
        frame = recording.getNextFrame()
        if frame is None:
            break
        frames_list.append(torch.from_numpy(frame.image.copy()))
        frames_timestamps.append(frame.timestamp)
    
    if not frames_list:
        return None, None
        
    return torch.stack(frames_list), torch.tensor(frames_timestamps)

def load_events(file_path: str) -> torch.Tensor:
    """Load event data from AEDAT4 file
    
    Args:
        file_path: AEDAT4 file path
        
    Returns:
        torch.Tensor: Event data tensor [N, 4] (timestamp, x, y, polarity)
    """
    recording = dv.io.MonoCameraRecording(dv.Path(str(file_path)))
    if not recording.isEventStreamAvailable():
        return None
        
    events_list = []
    
    while True:
        events = recording.getNextEventBatch()
        if events is None or len(events) == 0:
            break
            
        events_data = events.numpy()
        events_data = np.column_stack((
            events_data['timestamp'],
            events_data['x'],
            events_data['y'],
            events_data['polarity']
        ))
        events_list.append(torch.from_numpy(events_data))
    
    if not events_list:
        return None
        
    return torch.cat(events_list, dim=0)

def read_metadata(file_path: str) -> tuple:
    """Read timestamps from metadata file
    
    Args:
        file_path: Metadata file path
        
    Returns:
        tuple: (sensor_timestamps, unix_timestamps) Two timestamp tensors
    """
    if not os.path.exists(file_path):
        return None, None
        
    try:
        # Define data types
        metadata_dtype = np.dtype([
            ('SensorTimestamp', 'float64'),
            ('RealTime', 'S30')
        ])
        
        metadata = np.memmap(file_path, dtype=metadata_dtype, mode='r')
        sensor_timestamps = torch.from_numpy(metadata['SensorTimestamp'].copy())
        
        # Convert real time to Unix timestamps
        real_timestamps = torch.tensor([
            int(datetime.strptime(rt.decode('utf-8'), '%Y-%m-%d %H:%M:%S.%f').timestamp() * 1e6)
            for rt in metadata['RealTime']
        ])
        
        return sensor_timestamps, real_timestamps
        
    except Exception as e:
        print(f"Error reading metadata: {str(e)}")
        return None, None


    
def read_raw_frames(file_path: str, image_height: int, image_width: int) -> torch.Tensor:
    """Read RAW frame data
    
    Args:
        file_path: RAW frame file path
        image_height: Image height
        image_width: Image width
        
    Returns:
        torch.Tensor: Frame data tensor [N, H, W]
    """
    if not os.path.exists(file_path):
        return None
        
    try:
        raw_data = np.memmap(file_path, dtype='uint16', mode='r')
        total_frames = len(raw_data) // (image_height * image_width)
        frames = np.copy(raw_data).reshape(total_frames, image_height, image_width)
        return torch.from_numpy(frames)
        
    except Exception as e:
        print(f"Error reading RAW frames: {str(e)}")
        return None

def read_rgb_frames(file_path: str, image_height: int, image_width: int) -> torch.Tensor:
    """Read RGB frame data from Pi camera XBGR8888 (BGRA on disk).

    The raw file stores 4 bytes per pixel in BGRA order (B=ch0, G=ch1,
    R=ch2, X=ch3).  This function converts to RGB order and drops the
    padding channel so downstream code can use standard RGB conventions.

    Args:
        file_path: RGB frame file path
        image_height: Image height
        image_width: Image width

    Returns:
        torch.Tensor: RGB frame data tensor [N, H, W, 3] in RGB order
    """
    if not os.path.exists(file_path):
        return None

    try:
        rgb_data = np.memmap(file_path, dtype='uint8', mode='r')
        total_frames = len(rgb_data) // (image_height * image_width * 4)
        bgra = np.copy(rgb_data).reshape(total_frames, image_height, image_width, 4)
        # BGRA -> RGB: reverse first three channels, drop alpha
        rgb = bgra[:, :, :, 2::-1]  # [B,G,R,X] -> [R,G,B]
        return torch.from_numpy(np.ascontiguousarray(rgb))

    except Exception as e:
        print(f"Error reading RGB frames: {str(e)}")
        return None
    

import asyncio

async def read_pi_data_async(metadata_path, raw_path, rgb_path, height, width):
    """Asynchronously read PI camera data
    
    Args:
        metadata_path: Metadata file path
        raw_path: RAW frame file path
        rgb_path: RGB frame file path
        height: Image height
        width: Image width
        
    Returns:
        tuple: (sensor_ts, unix_ts, raw_frames, rgb_frames)
    """
    try:
        (sensor_ts, real_ts), raw_f, rgb_f = await asyncio.gather(
            asyncio.to_thread(read_metadata, metadata_path),
            asyncio.to_thread(read_raw_frames, raw_path, height, width),
            asyncio.to_thread(read_rgb_frames, rgb_path, height, width),
        )
        return sensor_ts, real_ts, raw_f, rgb_f
    except Exception as e:
        return None, None, None, None


def calculate_time_offset(pi_timestamps, real_timestamps):
    """Calculate time offset between PI timestamps and real timestamps
    
    Args:
        pi_timestamps: PI camera timestamp array
        real_timestamps: Real timestamp array
        
    Returns:
        int: Time offset (microseconds)
    """
    pi_intervals = np.diff(pi_timestamps)
    real_intervals = np.diff(real_timestamps)
    interval_ratios = pi_intervals / real_intervals
    closest_idx = np.argmin(np.abs(interval_ratios - 1))
    time_offset = int(real_timestamps[closest_idx] - pi_timestamps[closest_idx])
    return time_offset


def find_matching_files(folder_path, suffix):
    """
    Find matching files in specified folder
    
    Args:
        folder_path: Input folder path
        suffix: File name suffix (e.g., '170406')
        
    Returns:
        dict: Dictionary containing four file paths
    """
    files = {
        'dv': None,
        'metadata': None,
        'rgb_frames': None,
        'raw_frames': None
    }
    
    all_files = glob.glob(os.path.join(folder_path, "*"))
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        
        if suffix in filename:
            if "dv_output_" in filename and filename.endswith(".aedat4"):
                files['dv'] = file_path
            elif "metadata_" in filename and filename.endswith(".dat"):
                files['metadata'] = file_path
            elif "rgb_frames_" in filename and filename.endswith(".dat"):
                files['rgb_frames'] = file_path
            elif "raw_frames_" in filename and filename.endswith(".dat"):
                files['raw_frames'] = file_path
    
    missing_files = [k for k, v in files.items() if v is None]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")
    
    return files