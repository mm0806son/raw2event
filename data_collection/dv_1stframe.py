# Imports.
import dv_processing as dv  # dv_processing: event-camera I/O.
import numpy as np
import cv2 as cv
import os

def read_1st_frame(file_path):
    """
    Read the first frame from an AEDAT4 file.
    """
   
    # Open the AEDAT4 recording.
    recording = dv.io.MonoCameraRecording(file_path)
    
    # Check that the frame stream is present.
    if not recording.isFrameStreamAvailable():
        print("Error: file has no frame stream.")
        return None
    
    frame = recording.getNextFrame()
    
    return frame.image

# Example usage — replace with your AEDAT4 path.
file_path = "cifar10_xdvs_preview/dv_output_20001_cat_2_9875_20250714_141817.aedat4"  # Replace with your AEDAT4 file path.
output_dir = "aedat_preview"  # Output directory.

# Derive the basename without extension.
file_name = os.path.splitext(os.path.basename(file_path))[0]
output_path = f"{output_dir}/{file_name}_first_frame.png"

frame = read_1st_frame(file_path)
cv.imwrite(output_path, frame)
