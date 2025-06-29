"""
Event Display Module for Raspberry Pi Camera (Naive Version)

This module implements a real-time event camera simulation using a Raspberry Pi camera.
It captures raw Bayer sensor data and converts frame differences into event-like data.

This is a naive version, which only compare the difference between two frames.

Usage:
Run this script to start real-time event visualization. Press 'q' to quit.
"""

import numpy as np
import cv2
from picamera2 import Picamera2
import time
from libcamera import controls

sensor_width, sensor_height = 1536, 864  # sensor resolution
crop_width, crop_height = 692, 520  # target resolution
crop_x = (sensor_width - crop_width) // 2
crop_y = (sensor_height - crop_height) // 2

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    sensor={"output_size": (sensor_width, sensor_height), "bit_depth": 10},
    raw={"format": "SRGGB10", "size": (346, 260)}
)
picam2.configure(config)
picam2.set_controls({"ScalerCrop": (493, 365, 346, 260)})
picam2.set_controls({"FrameDurationLimits": (8333, 8333)})  # 1 / 120s = 8333us
picam2.start()

# Disable auto exposure time
# picam2.set_controls({"AeEnable": True})
picam2.set_controls({"AeEnable": False})
picam2.set_controls({"AnalogueGain": 10.0})
# picam2.set_controls({"ExposureTime": 1000000}) # Set exposure time (ms)

# Set Auto White Balance
picam2.set_controls({"AwbEnable": False}) 
# picam2.set_controls({"ColourGains": (1.5, 2.0)}) 
picam2.set_controls({"Brightness": 0.0, "Contrast": 10.0, "Saturation": 1.0})
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

picam2.set_controls({"ScalerCrop": (100, 100, 346, 260)})

# Calculate FPS
frame_count = 0
start_time = time.time()

# Capture first frame
(frame_prev,), metadata = picam2.capture_arrays(["raw"])
frame_prev = frame_prev.view("uint16")  # Convert Bayer data to uint16
# Cut data to target resolution
center_x = frame_prev.shape[1] // 2
center_y = frame_prev.shape[0] // 2
frame_prev = frame_prev[center_y - crop_height // 2:center_y + crop_height // 2, center_x - crop_width // 2:center_x + crop_width // 2]

# Convert to grayscale intensity (debayering)
# frame_prev_gray = cv2.cvtColor(frame_prev.astype("uint8"), cv2.COLOR_BAYER_RG2GRAY)

while True:
    # Capture new frame
    (frame_curr,), metadata = picam2.capture_arrays(["raw"])
    frame_curr = frame_curr.view("uint16")
    # Cut data to target resolution
    frame_curr = frame_curr[center_y - crop_height // 2:center_y + crop_height // 2, center_x - crop_width // 2:center_x + crop_width // 2]

    # Compute intensity difference
    # diff = frame_curr_gray.astype(np.int16) - frame_prev_gray.astype(np.int16)
    diff = frame_curr.astype(np.int16) - frame_prev.astype(np.int16)

    # Define event threshold (adjust as needed)
    threshold = 100  # Change threshold

    # Generate event map
    event_map = np.zeros_like(diff, dtype=np.int8)
    event_map[diff > threshold] = 1   # Positive events (brightness increase)
    event_map[diff < -threshold] = -1 # Negative events (brightness decrease)

    # Convert event map to a colored visualization
    event_display = np.zeros((event_map.shape[0], event_map.shape[1], 3), dtype=np.uint8)
    event_display[event_map == 1] = [0, 0, 255]   # Red for positive events (brightness increase)
    event_display[event_map == -1] = [255, 0, 0]  # Blue for negative events (brightness decrease)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Generate overlay_texts
    overlay_texts = [f"FPS: {fps:.2f}"]  # FPS
    for key, value in metadata.items():  # metadata
        overlay_texts.append(f"{key}: {value}")

    # Display
    for i, text in enumerate(overlay_texts):
        y_pos = 30 + i * 20  # display position
        cv2.putText(event_display, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # event_display = np.flip(event_display, axis=(0, 1))  # flip both vertically and horizontally

    cv2.imshow("Event Data", event_display)

    # Store current frame for next iteration
    # frame_prev_gray = frame_curr_gray.copy()
    frame_prev = frame_curr.copy()

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
