"""
Event Generation from Frames

This script converts frame sequences into event data.

Supported input formats:
- Video files (.mp4, etc.)
- Raw frame files (.npy, .dat) with metadata
- AEDAT4 files (for reading existing events)

Event generation methods:
- Naive: Simple frame difference thresholding
- DVS: Dynamic Vision Sensor simulation

Output formats:
- Text files (.txt) with event data
- AEDAT4 files (.aedat4) for compatibility with event camera tools
"""

import numpy as np
from src.simulator import EventSim
from src.config import cfg
import os
from tqdm import tqdm
import argparse
import cv2
import dv_processing as dv


# Optionally import pandas if you need CSV support.
import pandas as pd

def load_metadata(file_path):
    """
    Loads metadata from a file by inspecting the file extension.
    Supports: .npy, .dat, .txt, .csv.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.npy':
        return np.load(file_path, allow_pickle=True)
    elif ext == '.dat':
        # Define metadata data type
        metadata_dtype = np.dtype([
            ('SensorTimestamp', 'float64'),
            ('RealTime', 'S30')  # String type, maximum length 30
        ])
        # Get file size and calculate frame count
        file_size = os.path.getsize(file_path)
        frame_count = file_size // metadata_dtype.itemsize
        # Use memmap to read data
        return np.memmap(file_path, dtype=metadata_dtype, mode='r', shape=(frame_count,))
    elif ext in ['.txt']:
        try:
            metadata = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Could not load {file_path} using np.genfromtxt: {e}")
        return metadata

    elif ext == '.csv':
        # For CSV files, use pandas for robust parsing.
        try:
            metadata = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not load {file_path} using pandas.read_csv: {e}")
        return metadata

    else:
        raise ValueError(f"Unsupported metadata file type: {ext}")

def extract_timestamps(metadata):
    """
    Extracts timestamps from the loaded metadata.
    Supports NumPy structured arrays and pandas DataFrames.
    Looks for 'SensorTimestamp' or 'timestamp' field/column.
    """
    # Case 1: NumPy structured array (e.g., from np.genfromtxt)
    if isinstance(metadata, np.ndarray) and metadata.dtype.names is not None:
        if 'SensorTimestamp' in metadata.dtype.names:
            return [int(t) for t in metadata['SensorTimestamp']]
        elif 'timestamp' in metadata.dtype.names:
            return [int(t) for t in metadata['timestamp']]
        else:
            raise KeyError("No 'SensorTimestamp' or 'timestamp' field in metadata.")

    # Case 2: pandas DataFrame (e.g., from CSV)
    elif isinstance(metadata, pd.DataFrame):
        if 'SensorTimestamp' in metadata.columns:
            return metadata['SensorTimestamp'].astype(int).tolist()
        elif 'timestamp' in metadata.columns:
            return metadata['timestamp'].astype(int).tolist()
        else:
            raise KeyError("No 'SensorTimestamp' or 'timestamp' column in metadata.")

    # Otherwise, unsupported metadata type.
    raise TypeError("Unsupported metadata type. Please use .npy, .dat, .txt, or .csv with an appropriate format.")

def load_from_file(raw_frames_path, metadata_path, frame_width=None, frame_height=None, is_rgb=False):
    """
    Loads raw frames and timestamps from the provided file paths.
    Supports both .npy and .dat formats for raw frames.
    
    Args:
        raw_frames_path: Path to the raw frames file
        metadata_path: Path to the metadata file
        frame_width: Width of each frame (required for .dat files)
        frame_height: Height of each frame (required for .dat files)
        is_rgb: Whether the input is RGB data (4 channels)
    """
    # Select loading method based on file extension
    if raw_frames_path.endswith('.npy'):
        raw_frames = np.load(raw_frames_path)
        print(f"Input resolution: {raw_frames.shape[2]}x{raw_frames.shape[1]} (npy)")
    elif raw_frames_path.endswith('.dat'):
        if frame_width is None or frame_height is None:
            raise ValueError("frame_width and frame_height must be specified for .dat files")
            
        # Select dtype based on data type
        if is_rgb:
            dtype = np.dtype('uint8')
            channels = 4  # BGRA
        else:
            dtype = np.dtype('uint16')
            channels = 1
            
        # Get file size and calculate frame count
        file_size = os.path.getsize(raw_frames_path)
        frame_size = frame_width * frame_height * channels * dtype.itemsize
        frame_count = file_size // frame_size
        
        # Use memmap to read data
        if is_rgb:
            raw_frames = np.memmap(raw_frames_path, dtype=dtype, mode='r',
                                 shape=(frame_count, frame_height, frame_width, channels))
            print(f"Input resolution: {frame_width}x{frame_height} (RGB)")
            # Convert to grayscale
            gray_frames = []
            for frame in raw_frames:
                # Remove alpha channel
                rgb_frame = frame[:, :, :3]
                # Convert to grayscale
                gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
                gray_frames.append(gray_frame)
            raw_frames = np.array(gray_frames, dtype=np.uint16)
        else:
            raw_frames = np.memmap(raw_frames_path, dtype=dtype, mode='r',
                                 shape=(frame_count, frame_height, frame_width))
            print(f"Input resolution: {frame_width}x{frame_height} (dat)")
    else:
        raise ValueError(f"Unsupported raw frames format: {raw_frames_path}")
        
    metadata = load_metadata(metadata_path)
    timestamps_us = extract_timestamps(metadata)
    return raw_frames, timestamps_us

def load_from_video(video_path):
    """
    Loads raw frames and generates timestamps based on video FPS.
    """
    raw_frames = []
    timestamps_us = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input resolution: {width}x{height} (video)")
    dt_us = int(1e6 / fps)
    t_us = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_frames.append(gray)
        timestamps_us.append(int(t_us))
        t_us += dt_us
    cap.release()
    
    if not raw_frames:
        raise ValueError(f"No frames could be read from video: {video_path}")
    
    return np.array(raw_frames), timestamps_us

def save_to_aedat4(events, filename="events_output.aedat4", input_resolution=None):
    """
    Save events to AEDAT4 format using dv package.
    events: NumPy array of shape (N, 4) with columns: timestamp, x, y, polarity
    input_resolution: tuple of (width, height) for the input frames
    """
    # Convert polarity to boolean (in case it's stored as int)
    events[:, 3] = (events[:, 3] > 0).astype(np.uint8)

    # Sort events by timestamp
    events = events[events[:, 0].argsort()]
    
    # Check and filter invalid timestamps
    valid_events = []
    min_timestamp = 0  # Ensure timestamps are positive
    for event in events:
        t, x, y, p = event
        if t >= min_timestamp:  # Only keep valid timestamps
            valid_events.append(event)
        else:
            print(f"Warning: Skipping invalid timestamp event t={t}, x={x}, y={y}, p={p}")
    
    if not valid_events:
        raise ValueError("No valid events to save")
        
    events = np.array(valid_events)
    print(f"Valid events count: {len(events)}")
    print(f"Timestamp range: {events[0, 0]} - {events[-1, 0]}")

    store = dv.EventStore()

    # Add events to store
    for t, x, y, p in events:
        try:
            store.push_back(int(t), int(x), int(y), bool(p))
        except Exception as e:
            print(f"Error adding event: t={t}, x={x}, y={y}, p={p}")
            print(f"Error message: {str(e)}")
            raise

    # Debug: print first event
    for ev in store:
        print(f"Sliced event [{ev.timestamp()}, {ev.x()}, {ev.y()}, {ev.polarity()}]")
        break

    # Define resolution
    if input_resolution is not None:
        resolution = input_resolution
    else:
        resolution = (int(np.max(events[:, 1])) + 1, int(np.max(events[:, 2])) + 1)
    print(f"Resolution: {resolution}")

    # Setup writer config
    # API Ref: https://dv-processing.inivation.com/rel_1_7/python_api/classdv_1_1io_1_1MonoCameraWriter_1_1Config.html
    config = dv.io.MonoCameraWriter.Config("DVXplorer_sample")
    config.addEventStream(resolution)

    # Create the writer
    # API Ref: https://dv-processing.inivation.com/rel_1_7/python_api/classdv_1_1io_1_1MonoCameraWriter.html
    writer = dv.io.MonoCameraWriter(filename, config)
    print(f"Is event stream available? {str(writer.isEventStreamConfigured())}")

    # Write the events
    # API Ref: https://dv-processing.inivation.com/rel_1_7/python_api/classdv_1_1io_1_1MonoCameraWriter.html#a9c37fbbd8eb33b728a777b9b2a2128b4
    writer.writeEvents(store)
    print(f"✅ Saved {len(events)} events to {filename} (AEDAT 4)")

def generate_events_naive(prev_frame, curr_frame, timestamp, threshold=50):
    """
    Naive method: Generates events when the change is bigger than threshold
    """
    if prev_frame is None or curr_frame is None:
        return None
        
    # Ensure frames have the same shape
    if prev_frame.shape != curr_frame.shape:
        print(f"Warning: Frame shape mismatch: {prev_frame.shape} vs {curr_frame.shape}")
        return None
    
    diff = curr_frame.astype(np.int16) - prev_frame.astype(np.int16)

    # ON and OFF events
    on_mask = diff > threshold
    off_mask = diff < -threshold

    on_coords = np.argwhere(on_mask)
    off_coords = np.argwhere(off_mask)

    # Generate events: timestamp, x, y, polarity
    events = []
    for y, x in on_coords:
        events.append([timestamp, x, y, 1])
    for y, x in off_coords:
        events.append([timestamp, x, y, 0])

    return np.array(events, dtype=np.int64) if events else None

def load_from_aedat4(aedat4_path):
    """
    Loads raw frames and events from an AEDAT4 file.
    Returns raw_frames, timestamps_us, and events if available.
    """
    if not os.path.exists(aedat4_path):
        raise FileNotFoundError(f"AEDAT4 file not found: {aedat4_path}")
    
    recording = dv.io.MonoCameraRecording(aedat4_path)
    
    # Load frames if available
    raw_frames = []
    timestamps_us = []
    if recording.isFrameStreamAvailable():
        while True:
            frame = recording.getNextFrame()
            if frame is None:
                break
            # Convert frame to grayscale if it's not already
            if len(frame.image.shape) == 3:
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.image
            raw_frames.append(gray)
            timestamps_us.append(frame.timestamp)
    
    # Load events if available
    events = None
    if recording.isEventStreamAvailable():
        events_packets = []
        while True:
            event_batch = recording.getNextEventBatch()
            if event_batch is None:
                break
            events_packets.append(event_batch.numpy())
        if events_packets:
            events = np.concatenate(events_packets, axis=0)
    
    return np.array(raw_frames), timestamps_us, events

def main(args):
    # Select input mode based on provided arguments
    if args.aedat4:
        raw_frames, timestamps_us, events = load_from_aedat4(args.aedat4)
        if events is not None:
            print(f"📼 Read {len(events)} events from AEDAT4 file")
            if args.save_txt:
                output_txt = f"events_from_aedat4.txt"
                np.savetxt(output_txt, events, fmt="%d")
            if args.save_aedat4:
                output_aedat4 = f"events_from_aedat4.aedat4"
                save_to_aedat4(events, filename=output_aedat4)
            return
    elif args.video:
        raw_frames, timestamps_us = load_from_video(args.video)
        input_resolution = (raw_frames.shape[2], raw_frames.shape[1])
    else:
        raw_frames, timestamps_us = load_from_file(args.raw_frames, args.metadata, 
                                                 args.frame_width, args.frame_height,
                                                 args.is_rgb)
        input_resolution = (args.frame_width, args.frame_height)

    print(f"📼 Read {len(timestamps_us)} frames")

    # Initialize the EventSim if using the 'dvs' method.
    if args.method == 'dvs':
        sim = EventSim(cfg=cfg, output_folder='.')

    # Generate events for each consecutive frame.
    all_events = []
    prev_frame = None
    for i in tqdm(range(len(timestamps_us)), desc="Generating events", unit="frame"):
        frame = raw_frames[i]
        timestamp = timestamps_us[i]
        if prev_frame is None:
            prev_frame = frame
            continue

        if args.method == 'naive':
            events = generate_events_naive(prev_frame, frame, timestamp, threshold=20)
        elif args.method == 'dvs':
            events = sim.generate_events(frame, timestamp)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        prev_frame = frame
        if events is not None:
            all_events.append(events)

    # Save the generated events if any were found.
    if all_events:
        all_events = np.concatenate(all_events, axis=0)
        print(f"Generated {all_events.shape[0]} events, saving...")
        if args.save_txt:
            output_txt = f"events_from_{'video' if args.video else 'file'}_{args.method}.txt"
            np.savetxt(output_txt, all_events, fmt="%d")
        if args.save_aedat4:
            output_aedat4 = f"events_from_{'video' if args.video else 'file'}_{args.method}.aedat4"
            save_to_aedat4(all_events, filename=output_aedat4, input_resolution=input_resolution)
            print(f"Saved AEDAT4 file to {output_aedat4}")
    else:
        print("No events detected.")

    print("✅ Generation finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['naive', 'dvs'], default='naive', help='Event generation method')
    parser.add_argument('--video', type=str, help='Path to input video (mp4)')
    parser.add_argument('--aedat4', type=str, help='Path to input AEDAT4 file')
    parser.add_argument('--raw_frames', type=str, default='raw_frames.npy', help='Path to raw_frames file (.npy, .dat, .csv, .txt)')
    parser.add_argument('--metadata', type=str, default='metadata.dat', help='Path to metadata file (.npy, .dat, .csv, .txt)')
    parser.add_argument('--frame_width', type=int, default=692, help='Width of each frame (required for .dat files)')
    parser.add_argument('--frame_height', type=int, default=520, help='Height of each frame (required for .dat files)')
    parser.add_argument('--is_rgb', action='store_true', help='Whether the input is RGB data (4 channels)')
    parser.add_argument('--save_aedat4', action='store_true', help='Also save output in AEDAT 4 format')
    parser.add_argument('--save_txt', action='store_true', help='Also save output in txt format')
    args = parser.parse_args()
    main(args)