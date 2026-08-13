"""
Convert .dat file to .csv format for sanity check.
"""
import numpy as np
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Convert .dat file to .csv format')
    parser.add_argument('--input_dat', type=str, required=True,
                        help='Path to the input .dat file')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to the output .csv file (optional, default: same as input)')
    parser.add_argument('--file_type', type=str, choices=['raw', 'metadata', 'rgb'], required=True,
                        help='File type: raw (raw frame data), metadata, or rgb (RGB frame data)')
    parser.add_argument('--height', type=int, default=520,
                        help='Image height (default 520, only for raw type)')
    parser.add_argument('--width', type=int, default=692,
                        help='Image width (default 692, only for raw type)')
    return parser.parse_args()

def convert_metadata_to_csv(args):
    # If output file is not specified, use the input file name
    if args.output_csv is None:
        base_name = os.path.splitext(args.input_dat)[0]
        args.output_csv = f"{base_name}.csv"

    print(f"Reading metadata file: {args.input_dat}")
    print(f"Will save as: {args.output_csv}")

    # Define metadata dtype
    metadata_dtype = np.dtype([
            ('SensorTimestamp', 'float64'),
            ('RealTime', 'S30')  # String type, max length 30
        ])
    dtype_size = metadata_dtype.itemsize

    # Get file size and calculate frame count
    file_size = os.path.getsize(args.input_dat)
    frame_count = file_size // dtype_size
    print(f"Auto-calculated frame count: {frame_count}")

    # Read metadata file
    data = np.memmap(args.input_dat, dtype=metadata_dtype, mode='r',
                    shape=(frame_count,))

    # Create CSV file
    with open(args.output_csv, 'w') as f:
        # Write header
        f.write("Frame,SensorTimestamp,RealTime\n")
        
        # Write data
        for frame in range(frame_count):
            timestamp = data[frame]['SensorTimestamp']
            realtime = data[frame]['RealTime'].decode('utf-8').strip()  # Convert bytes to string and strip whitespace
            f.write(f"{frame},{timestamp},{realtime}\n")
            
            if (frame + 1) % 50 == 0:
                print(f"Processed {frame + 1}/{frame_count} frames")

    print(f"Conversion complete! File saved to: {args.output_csv}")

def convert_raw_to_csv(args):
    # If output file is not specified, use the input file name
    if args.output_csv is None:
        base_name = os.path.splitext(args.input_dat)[0]
        args.output_csv = f"{base_name}.csv"

    print(f"Reading raw frame data file: {args.input_dat}")
    print(f"Will save as: {args.output_csv}")

    # Define data type
    dtype = np.dtype('uint16')
    dtype_size = dtype.itemsize
    frame_size = args.height * args.width * dtype_size

    # Get file size and calculate frame count
    file_size = os.path.getsize(args.input_dat)
    frame_count = file_size // frame_size
    print(f"Auto-calculated frame count: {frame_count}")

    # Read .dat file
    data = np.memmap(args.input_dat, dtype=dtype, mode='r',
                    shape=(frame_count, args.height, args.width))

    # Create CSV file
    with open(args.output_csv, 'w') as f:
        # Write header
        f.write("Frame,Values\n")
        
        # Write data
        for frame in range(frame_count):
            # Flatten the entire frame data to 1D array
            values = data[frame].flatten()
            # Convert values to string and join with commas
            values_str = ','.join(map(str, values))
            f.write(f"{frame},{values_str}\n")
            
            if (frame + 1) % 50 == 0:
                print(f"Processed {frame + 1}/{frame_count} frames")

    print(f"Conversion complete! File saved to: {args.output_csv}")

def convert_rgb_to_csv(args):
    # If output file is not specified, use the input file name
    if args.output_csv is None:
        base_name = os.path.splitext(args.input_dat)[0]
        args.output_csv = f"{base_name}.csv"

    print(f"Reading RGB frame data file: {args.input_dat}")
    print(f"Will save as: {args.output_csv}")

    # Define data type and shape
    dtype = np.dtype('uint8')
    dtype_size = dtype.itemsize
    frame_size = args.height * args.width * 4 * dtype_size  # 4 channels

    # Get file size and frame count
    file_size = os.path.getsize(args.input_dat)
    frame_count = file_size // frame_size
    print(f"Auto-calculated frame count: {frame_count}")

    # Read .dat file
    data = np.memmap(args.input_dat, dtype=dtype, mode='r',
                    shape=(frame_count, args.height, args.width, 4))

    # Create CSV file
    with open(args.output_csv, 'w') as f:
        # Write header
        f.write("Frame,Values\n")
        # Write data
        for frame in range(frame_count):
            values = data[frame].flatten()
            values_str = ','.join(map(str, values))
            f.write(f"{frame},{values_str}\n")
            if (frame + 1) % 10 == 0:
                print(f"Processed {frame + 1}/{frame_count} frames")
                break
    print(f"Conversion complete! File saved to: {args.output_csv}")

def main():
    args = parse_args()
    if args.file_type == 'metadata':
        convert_metadata_to_csv(args)
    elif args.file_type == 'raw':
        convert_raw_to_csv(args)
    elif args.file_type == 'rgb':
        convert_rgb_to_csv(args)

if __name__ == "__main__":
    main()
