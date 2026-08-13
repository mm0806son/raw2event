# mkv_to_dat.py

import numpy as np
import cv2
import os
import subprocess
import argparse

def extract_frames_from_video(video_path, output_dir, is_rgb):
    os.makedirs(output_dir, exist_ok=True)
    ext = 'png' if is_rgb else 'tiff'
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        os.path.join(output_dir, f"frame_%04d.{ext}")
    ])
    print(f"Frames extracted to: {output_dir}")

def convert_raw_frames_to_dat(frame_dir, output_path, width, height, mode):
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.tiff')])
    num_frames = len(files)
    raw_data = np.memmap(output_path, dtype='uint16', mode='w+', shape=(num_frames, height, width))

    for i, fname in enumerate(files):
        frame = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError(f"Error reading frame {fname}")
        # Restore original 16-bit alignment (<< 6)
        if mode == 'raw_10bit':
            raw_data[i] = (frame.astype(np.uint16) << 6)
        else:
            raw_data[i] = frame.astype(np.uint16)

    raw_data.flush()
    print(f"RAW .dat saved to: {output_path}")

def convert_rgb_frames_to_dat(frame_dir, output_path, width, height):
    files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    num_frames = len(files)
    rgb_data = np.memmap(output_path, dtype='uint8', mode='w+', shape=(num_frames, height, width, 4))

    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(frame_dir, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Error reading frame {fname}")
        # Handle 4-channel (BGRA) images
        if img.ndim == 3 and img.shape[2] == 4:
            rgba = img.copy()
            rgba[:, :, 3] = 0  # Set alpha channel to 0
        # Handle 3-channel (BGR) images
        elif img.ndim == 3 and img.shape[2] == 3:
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = img
            rgba[:, :, 3] = 0
        else:
            raise ValueError(f"Frame {fname} is not valid RGB or RGBA image")
        rgb_data[i] = rgba

    rgb_data.flush()
    print(f"RGB .dat saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .mkv video back to .dat format.")
    parser.add_argument('--input', type=str, required=True, help='Input .mkv file path')
    parser.add_argument('--output', type=str, required=True, help='Output .dat file path')
    parser.add_argument('--temp_dir', type=str, default='frames_temp', help='Temporary directory to hold extracted frames')
    parser.add_argument('--width', type=int, default=692, help='Frame width')
    parser.add_argument('--height', type=int, default=520, help='Frame height')
    parser.add_argument('--mode', type=str, choices=['raw', 'rgb', 'raw_10bit'], required=True, help='Frame type: raw or rgb')

    args = parser.parse_args()

    is_rgb = args.mode == 'rgb'
    extract_frames_from_video(args.input, args.temp_dir, is_rgb)

    if args.mode == 'raw' or args.mode == 'raw_10bit':
        convert_raw_frames_to_dat(args.temp_dir, args.output, args.width, args.height, args.mode)
    elif args.mode == 'rgb':
        convert_rgb_frames_to_dat(args.temp_dir, args.output, args.width, args.height)

    # Optional: clean up
    import shutil
    shutil.rmtree(args.temp_dir)
