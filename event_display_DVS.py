"""
Event Display Module for Raspberry Pi Camera

This module implements a real-time event camera simulation using a Raspberry Pi camera.
It captures raw Bayer sensor data and converts frame differences into event-like data, 
using the model of DVS-Voltmeter.

Usage:
Run this script to start real-time event visualization. Press 'q' to quit.
Select the camera type in "src/config.py".
"""

import numpy as np
import cv2
from picamera2 import Picamera2
import time
from src.simulator import EventSim
from src.config import cfg
from libcamera import controls

sensor_width, sensor_height = 1536, 864  # sensor resolution
crop_width, crop_height = 346, 220  # target resolution
crop_x = (sensor_width - crop_width) // 2
crop_y = (sensor_height - crop_height) // 2
DECAY_FACTOR = 0.9  # for cv2.addWeighted

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

sim = EventSim(cfg=cfg, output_folder='.')

# Disable auto exposure time
# picam2.set_controls({"AeEnable": True})
picam2.set_controls({"AeEnable": False})
picam2.set_controls({"AnalogueGain": 5.0})
# picam2.set_controls({"ExposureTime": 1000000}) # Set exposure time (ms)
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

(frame_first,), metadata_first = picam2.capture_arrays(["raw"])
time_first = metadata_first["SensorTimestamp"]
last_timestamp = 0

while True:
    # Capture new frame
    (frame_curr,), metadata = picam2.capture_arrays(["raw"])
    metadata["SensorTimestamp"] = (metadata["SensorTimestamp"] - time_first)/1000

    frame_curr = frame_curr.view("uint16")
    # Cut data to target resolution
    frame_curr = frame_curr[center_y - crop_height // 2:center_y + crop_height // 2, center_x - crop_width // 2:center_x + crop_width // 2]

    # Timestamp
    timestamp = int(metadata["SensorTimestamp"])  # unit：us

    events = sim.generate_events(frame_curr, timestamp)

    # Display events
    event_display = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    if events is not None:
        events = events[events[:, 0] > last_timestamp]
        last_timestamp = timestamp

        x = events[:, 1].astype(np.int32)
        y = events[:, 2].astype(np.int32)
        p = events[:, 3]

        event_display = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)

        pos_mask = p == 1
        neg_mask = p == 0

        event_display[y[pos_mask], x[pos_mask]] = [0, 0, 255]    # red for positive
        event_display[y[neg_mask], x[neg_mask]] = [0, 255, 0]    # green for negative

        last_timestamp = timestamp
    # event_display = cv2.addWeighted(event_display, FADE_ALPHA, event_display, 1 - FADE_ALPHA, 0)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(fps)

    # Generate overlay_texts and display
    overlay_texts = [f"FPS: {fps:.2f}"]  # FPS
    for key, value in metadata.items():  # metadata
        overlay_texts.append(f"{key}: {value}")

    for i, text in enumerate(overlay_texts):
        y_pos = 30 + i * 20  # display position
        cv2.putText(event_display, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # event_display = np.flip(event_display, axis=(0, 1))  # flip both vertically and horizontally

    cv2.imshow("Event Data", event_display)

    # Store current frame for next iteration
    frame_prev = frame_curr.copy()

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
