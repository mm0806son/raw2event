# dat_to_mkv.py

import numpy as np
import cv2
import os
import subprocess
import argparse

def convert_raw_to_mkv(raw_path, output_path, width, height, num_frames, fps=60, mode='raw', temp_dir="temp_raw_frames"):
    raw = np.memmap(raw_path, dtype='uint16', mode='r', shape=(num_frames, height, width))
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(num_frames):
        frame = raw[i]
        # Convert to 16-bit single channel grayscale TIFF
        if mode == 'raw_10bit':
            frame = (frame >> 6) & 0x03FF
        else:
            frame = frame
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.tiff", frame)

    # Create FFV1 video using ffmpeg
    base_name = os.path.splitext(os.path.basename(raw_path))[0]
    mkv_path = os.path.join(output_path, f"{base_name}_{mode}.mkv")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{temp_dir}/frame_%04d.tiff",
        "-c:v", "ffv1", "-pix_fmt", "gray16le",
        mkv_path
    ])

    # Clean up
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"RAW video saved to {mkv_path}")

def convert_rgb_to_mkv(rgb_path, output_path, width, height, num_frames, fps=60, temp_dir="temp_rgb_frames"):
    rgb = np.memmap(rgb_path, dtype='uint8', mode='r', shape=(num_frames, height, width, 4))
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(num_frames):
        frame = rgb[i]
        frame_bgr = frame[:, :, :3]
        cv2.imwrite(f"{temp_dir}/frame_{i:04d}.png", frame_bgr)

    base_name = os.path.splitext(os.path.basename(rgb_path))[0]
    mkv_path = os.path.join(output_path, f"{base_name}_rgb.mkv")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", f"{temp_dir}/frame_%04d.png",
        # "-c:v", "ffv1", "-pix_fmt", "rgba",
        # "-c:v", "ffv1", "-pix_fmt", "bgra",
        # input: BGR PNG, with no alpha; output: bgr0 to save space
        "-c:v", "ffv1", "-pix_fmt", "bgr0",
        mkv_path
    ])

    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    print(f"RGB video saved to {mkv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .dat raw/rgb frames to .mkv video using FFV1 codec.")
    parser.add_argument('--input', type=str, required=True, help='Input .dat file path')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for .mkv file')
    parser.add_argument('--width', type=int, default=692, help='Frame width')
    parser.add_argument('--height', type=int, default=520, help='Frame height')
    parser.add_argument('--fps', type=int, default=60, help='Frames per second for output video')
    parser.add_argument('--mode', type=str, choices=['raw', 'rgb', 'raw_10bit'], required=True, help='Frame type: "raw" for 16-bit grayscale, "rgb" for 4-channel RGBA')
    parser.add_argument('--temp_dir', type=str, default=None, help='Temporary directory for frames')

    args = parser.parse_args()

    # Calculate num_frames automatically
    import os
    file_size = os.path.getsize(args.input)
    if args.mode == 'raw' or args.mode == 'raw_10bit':
        # uint16
        frame_size = args.width * args.height * 2
        num_frames = file_size // frame_size
        print(f"Auto-calculated num_frames: {num_frames}")
        
        temp_dir = args.temp_dir if args.temp_dir else "temp_raw_frames"
        convert_raw_to_mkv(args.input, args.output_dir, args.width, args.height, num_frames, args.fps, args.mode, temp_dir)
    elif args.mode == 'rgb':
        # uint8, channels = 4
        frame_size = args.width * args.height * 4
        num_frames = file_size // frame_size
        print(f"Auto-calculated num_frames: {num_frames}")
        
        temp_dir = args.temp_dir if args.temp_dir else "temp_rgb_frames"
        convert_rgb_to_mkv(args.input, args.output_dir, args.width, args.height, num_frames, args.fps, temp_dir)
    else:
        print("Invalid mode. Use 'raw' or 'rgb'.")
