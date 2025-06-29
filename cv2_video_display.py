from picamera2 import Picamera2
from libcamera import controls
import time
import cv2
import numpy as np
import csv
import argparse

def check_camera_config(picam2):
    """Check and print camera configuration information"""
    print("\n=== Camera Configuration Info ===")
    
    # Get camera control parameters
    # controls = picam2.camera_controls
    # print("\nCamera control parameters:")
    # for control, value in controls.items():
    #     print(f"{control}: {value}")
    
    # Get camera properties
    # properties = picam2.camera_properties
    # print("\nCamera properties:")
    # for prop, value in properties.items():
    #     print(f"{prop}: {value}")
    
    # Get current configuration
    config = picam2.camera_config
    print("\nCurrent configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Get metadata fields
    # metadata = picam2.capture_metadata()
    # print("\nAvailable metadata fields:")
    # for key in metadata.keys():
    #     print(f"- {key}")
    
    print("\n=== Configuration check complete ===\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Camera display program')
    
    # Display parameters
    parser.add_argument('--mode', type=str, choices=['raw', 'rgb'], default='raw',
                        help='Select output mode: raw (raw data) or rgb (RGB image)')
    parser.add_argument('--display_metadata', action='store_true', default=False,
                        help='Whether to display metadata')
    parser.add_argument('--show_fps', action='store_true', default=False,
                        help='Whether to display FPS')
    
    # Image parameters
    parser.add_argument('--crop_width', type=int, default=692,
                        help='Crop width')
    parser.add_argument('--crop_height', type=int, default=520,
                        help='Crop height')
    
    return parser.parse_args()

def crop(data, crop_width, crop_height):
    center_x = data.shape[1] // 2
    center_y = data.shape[0] // 2
    data = data[
        center_y - crop_height // 2:center_y + crop_height // 2,
        center_x - crop_width // 2:center_x + crop_width // 2
    ]
    return data

def main():
    args = parse_args()
    
    # Initialize camera
    picam2 = Picamera2()
    if args.mode == 'raw':
        config = picam2.create_video_configuration(
            raw={"format": "SRGGB10", "size": (args.crop_width, args.crop_height)}
        )
    else:  # rgb mode
        config = picam2.create_video_configuration(
            main={"size": (args.crop_width, args.crop_height), "format": "XBGR8888"},
            raw=None,
        )
    picam2.configure(config)
    
    # Set camera control parameters
    picam2.set_controls({
        "FrameDurationLimits": (8333, 8333),  # 120FPS
        "AeEnable": False,
        "AnalogueGain": 10.0,
        "AfMode": controls.AfModeEnum.Continuous,
    })
    
    picam2.start()
    
    # Check camera configuration
    check_camera_config(picam2)
    sensor_width, sensor_height = picam2.camera_config['sensor']['output_size']
    print(f"\nActual sensor resolution: {sensor_width}x{sensor_height}")
    
    # Get current color space
    current_colorspace = str(picam2.camera_config['colour_space']).split("'")[1]
    print(f"\nCurrent color space: {current_colorspace}")

    # Calculate FPS
    frame_count = 0
    start_time = time.time()

    (frame_first,), metadata_first = picam2.capture_arrays(["raw" if args.mode == 'raw' else "main"])
    time_first = metadata_first["SensorTimestamp"]

    while True:
            
        (frame,), metadata = picam2.capture_arrays(["raw" if args.mode == 'raw' else "main"])
        # print(f"Image size: {frame.shape}")
        
        if args.mode == 'raw':
            data = frame.view('uint16')
            # Crop
            if args.crop_width != sensor_width and args.crop_height != sensor_height:
                data = crop(data, args.crop_width, args.crop_height)
                # print(f"Image size after crop: {data.shape}")
            max_value = 2**10-1
            data = (data >> 6) & 0x03FF  # ! Use this conversion for Pi 5
            display_data = ((data.astype('uint32') * 255) / max_value).astype('uint8')

        else:  # rgb mode
            display_data = frame
            display_data = cv2.cvtColor(display_data, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Generate overlay_texts
        overlay_texts = []
        if args.show_fps:
            overlay_texts.append(f"FPS: {fps:.2f}")
        if args.display_metadata:
            metadata["SensorTimestamp"] = (metadata["SensorTimestamp"] - time_first)/1000
            for key, value in metadata.items():
                overlay_texts.append(f"{key}: {value}")

        # Display
        if args.display_metadata:
            for i, text in enumerate(overlay_texts):
                y_pos = 30 + i * 20  # display position
                cv2.putText(display_data, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Camera', display_data)

        if args.show_fps and frame_count % 50 == 0:
            print(f"Frame {frame_count}/{args.max_frames if args.max_frames else '∞'}, FPS: {fps:.2f}")
        
        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main()