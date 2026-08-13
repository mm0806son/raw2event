# Imports.
import dv_processing as dv  # dv_processing: event-camera I/O.
import numpy as np
import cv2 as cv
import os
import glob
from tqdm import tqdm

def read_frame(file_path):
    """Read the first and last frames from an AEDAT4 file.

    Returns (first_image, last_image, frame_count) or (None, None) on failure.
    """
    recording = dv.io.MonoCameraRecording(file_path)

    if not recording.isFrameStreamAvailable():
        return None, None

    frame = recording.getNextFrame()
    first_image = frame.image
    frame_count = 1
    while frame is not None:
        last_image = frame.image
        frame = recording.getNextFrame()
        frame_count += 1

    return first_image, last_image, frame_count

def batch_process_aedat4_files(input_folder, output_folder):
    """Batch-process every AEDAT4 file in a folder.

    Args:
        input_folder: directory containing AEDAT4 files.
        output_folder: directory for the output PNG files.
    """

    # Ensure the output folder exists.
    os.makedirs(output_folder, exist_ok=True)

    # Enumerate the AEDAT4 files.
    aedat4_pattern = os.path.join(input_folder, "*.aedat4")
    aedat4_files = glob.glob(aedat4_pattern)
    print(f"Found {len(aedat4_files)} AEDAT4 files.")

    success_count = 0
    error_count = 0
    failed_files = []

    # Progress bar.
    for file_path in tqdm(aedat4_files, desc="Processing AEDAT4", unit="file"):
        try:
            # Derive the basename without extension.
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_folder, f"{file_name}_frame.png")

            # Read first and last frames.
            first_img, last_img, frame_count = read_frame(file_path)
            if first_img is None or last_img is None:
                raise ValueError("frame stream missing or empty")

            # combined = np.hstack([first_img, last_img])
            # cv.imwrite(output_path, combined)
            cv.imwrite(output_path, last_img)
            # print(f"file_name: {file_name}, frame_count: {frame_count}")
            success_count += 1

        except Exception as e:
            error_count += 1
            failed_files.append((file_path, str(e)))

    print("\nBatch processing complete.")
    print(f"Succeeded: {success_count} files")
    print(f"Failed:    {error_count} files")
    if failed_files:
        print("Failed file list:")
        for file_path, reason in failed_files:
            print(f"- {file_path} : {reason}")

if __name__ == "__main__":
    # Configuration.
    # input_folder = "cifar10_xdvs_preview"  # directory containing AEDAT4 files
    input_folder = "./data"  # directory containing AEDAT4 files
    output_folder = "aedat_preview_lastframe"  # output PNG directory

    # Run batch processing.
    batch_process_aedat4_files(input_folder, output_folder)
