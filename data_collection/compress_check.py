import numpy as np
import argparse
import os

def load_dat(path, mode, shape, dtype):
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)

def verify_raw(original, restored, allow_bit_difference=True):
    diff = np.abs(original.astype(np.int32) - restored.astype(np.int32))
    max_diff = np.max(diff)
    mismatch_count = np.count_nonzero(diff)

    if allow_bit_difference:
        # If data was originally 10-bit aligned to 16-bit, low 6 bits may differ
        mask = 0xFFC0  # upper 10 bits
        identical = np.all((original & mask) == (restored & mask))
        print(f"Bit-masked comparison passed: {identical}")
    else:
        identical = np.array_equal(original, restored)

    print(f"Max absolute difference: {max_diff}")
    print(f"Number of mismatched elements: {mismatch_count}")
    return identical

def verify_rgb(original, restored):
    identical = np.array_equal(original, restored)
    diff = np.abs(original.astype(np.int16) - restored.astype(np.int16))
    max_diff = np.max(diff)
    mismatch_count = np.count_nonzero(diff)
    print(f"Exact match: {identical}")
    print(f"Max absolute difference: {max_diff}")
    print(f"Number of mismatched pixels: {mismatch_count}")
    return identical

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two .dat files for raw/rgb consistency.")
    parser.add_argument('--original', required=True, help='Original .dat file path')
    parser.add_argument('--restored', required=True, help='Restored .dat file path to verify')
    parser.add_argument('--mode', choices=['raw', 'rgb'], required=True, help='Data mode: raw or rgb')
    parser.add_argument('--width', type=int, default=692, help='Image width')
    parser.add_argument('--height', type=int, default=520, help='Image height')
    parser.add_argument('--bit10_mask', action='store_true', help='(raw only) Enable 10-bit mask comparison')
    
    args = parser.parse_args()

    # Auto-calculate frame count from file size
    orig_size = os.path.getsize(args.original)
    if args.mode == 'raw':
        bytes_per_frame = args.width * args.height * 2
        dtype = 'uint16'
        shape = (-1, args.height, args.width)
    else:  # rgb
        bytes_per_frame = args.width * args.height * 4
        dtype = 'uint8'
        shape = (-1, args.height, args.width, 4)

    num_frames = orig_size // bytes_per_frame
    shape = (num_frames,) + shape[1:]

    print(f"Comparing {num_frames} frames of shape {shape[1:]}, dtype={dtype}")

    original = load_dat(args.original, args.mode, shape, dtype)
    restored = load_dat(args.restored, args.mode, shape, dtype)

    if args.mode == 'raw':
        verify_raw(original, restored, allow_bit_difference=args.bit10_mask)
    else:
        verify_rgb(original, restored)
