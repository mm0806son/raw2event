"""
Client script for recording raw/RGB frame camera data using Pi Camera.

This script listens for MQTT commands to start/stop recording, captures and saves image data and transfers recorded data to a remote server. 

Example usage:
python dataset_control_client.py --num_images 10 --server_path user@server:/path/to/server

This will record 10 frames per session and transfer the recorded data to the specified server path after completion.
"""
from picamera2 import Picamera2
from libcamera import controls
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import subprocess
import argparse
import os
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Camera recording control program for dataset collection')
    
    # MQTT configuration
    parser.add_argument('--mqtt_broker', type=str, default="localhost",
                        help='MQTT broker address')
    parser.add_argument('--mqtt_port', type=int, default=1883,
                        help='MQTT broker port')
    parser.add_argument('--mqtt_topic', type=str, default="record",
                        help='MQTT topic')
    
    # Recording parameters
    parser.add_argument('--num_images', type=int, default=350,
                        help='Number of images to process')
    parser.add_argument('--save_raw', action='store_true', default=True,
                        help='Save Raw frames')
    parser.add_argument('--display', action='store_true', default=False,
                        help='Show real-time preview')
    parser.add_argument('--display_metadata', action='store_true', default=False,
                        help='Show metadata')
    parser.add_argument('--show_fps', action='store_true', default=False,
                        help='Show FPS')
    parser.add_argument('--save_rgb', action='store_true', default=True,
                        help='Save RGB frames')
    
    # Path configuration
    parser.add_argument('--output_path', type=str, default='/dev/shm',
                        help='Local temporary path')
    parser.add_argument('--server_path', type=str, default='.',
                        help='Server path for storing recorded data')
    
    # Image parameters
    parser.add_argument('--crop_width', type=int, default=692,
                        help='Crop width')
    parser.add_argument('--crop_height', type=int, default=520,
                        help='Crop height')
    
    return parser.parse_args()

recording = False
should_stop = False
current_tag = ""

def crop(data, crop_width, crop_height):
    """Crop image data to specified size"""
    center_x = data.shape[1] // 2
    center_y = data.shape[0] // 2
    return data[
        center_y - crop_height // 2:center_y + crop_height // 2,
        center_x - crop_width // 2:center_x + crop_width // 2
    ]

def transfer_to_host(args, timestamp, mqtt_client):
    rsync_cmd = f"rsync --progress {args.output_path}/*_{timestamp}.dat {args.output_path}/preview_{current_tag}_{timestamp}.png {args.server_path} && rm -f {args.output_path}/*_{timestamp}.dat {args.output_path}/preview_{current_tag}_{timestamp}.png"
    process = subprocess.Popen(rsync_cmd, shell=True)
    process.wait()
    message = f"transfer_{'complete' if process.returncode == 0 else 'failed'}_{current_tag}_{timestamp}"
    mqtt_client.publish(args.mqtt_topic, message)
    print(f"{'' if process.returncode == 0 else ''} Transfer {'completed' if process.returncode == 0 else 'failed'} (timestamp: {timestamp})")

def save_preview_image(data, args, timestamp):
    # Apply correct debayer conversion for Pi 5
    data_preview = (data >> 6) & 0x03FF  # ! Use this conversion for Pi 5
    max_value = 2**10 - 1
    preview_image = ((data_preview.astype('uint32') * 255) / max_value).astype('uint8')
    preview_path = os.path.join(args.output_path, f'preview_{current_tag}_{timestamp}.png')
    cv2.imwrite(preview_path, preview_image)
    print(f"Preview image saved: {preview_path}")

def record_frames(picam2, args, mqtt_client):
    """Record frames"""
    global should_stop
    sensor_width, sensor_height = picam2.camera_config['sensor']['output_size']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create memmap with timestamp and identifier
    if args.save_raw:
        mmap_shape = (args.num_images, args.crop_height, args.crop_width)
        frame_memmap = np.memmap(args.output_path + f'/raw_frames_{current_tag}_{timestamp}.dat', dtype='uint16', mode='w+', shape=mmap_shape)
        metadata_fields = ['SensorTimestamp', 'RealTime']
        metadata_dtype = np.dtype([
            ('SensorTimestamp', 'float64'),
            ('RealTime', 'S30')  # String type, max length 30
            ])
        metadata_memmap = np.memmap(args.output_path + f"/metadata_{current_tag}_{timestamp}.dat", dtype=metadata_dtype, mode='w+', shape=(args.num_images,))

    if args.save_rgb:
        rgb_shape = (args.num_images, args.crop_height, args.crop_width, 4)
        rgb_memmap = np.memmap(args.output_path + f"/rgb_frames_{current_tag}_{timestamp}.dat", dtype='uint8', mode='w+', shape=rgb_shape)

    # FPS
    frame_count = 0
    start_time = time.time()

    # Capture first frame
    (frame_first,), metadata_first = picam2.capture_arrays(["raw"])
    time_first = metadata_first["SensorTimestamp"]
    data_first = frame_first.view('uint16')
    
    # Save first frame as PNG preview
    if args.crop_width != sensor_width and args.crop_height != sensor_height:
        data_first = crop(data_first, args.crop_width, args.crop_height)
    
    save_preview_image(data_first, args, timestamp)

    # Recording loop
    while frame_count < args.num_images and not should_stop:
        (frame,), metadata = picam2.capture_arrays(["raw"])
        data = frame.view('uint16')

        # Crop
        if args.crop_width != sensor_width and args.crop_height != sensor_height:
            data = crop(data, args.crop_width, args.crop_height)

        # Save to memmap
        if args.save_raw:
            frame_memmap[frame_count] = data
            sensor_time = (metadata["SensorTimestamp"] - time_first)/1000
            current_real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            metadata_memmap[frame_count] = (sensor_time, current_real_time)

        if args.save_rgb:
            rgb_image = picam2.capture_array("main")
            if args.crop_width != sensor_width and args.crop_height != sensor_height:
                rgb_image = crop(rgb_image, args.crop_width, args.crop_height)

            rgb_memmap[frame_count] = rgb_image

        frame_count += 1
        
        if args.show_fps:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Show Metadata
        if args.display:
            # Apply correct debayer conversion for Pi 5
            data_display = (data >> 6) & 0x03FF  # ! Use this conversion for Pi 5
            max_value = 2**10 - 1
            display_img = ((data_display.astype('uint32') * 255) / max_value).astype('uint8')
            overlay_texts = []
            if args.show_fps:
                overlay_texts.append(f"FPS: {fps:.2f}")
            if args.display_metadata:
                for key in metadata_fields:
                    if key in metadata:
                        overlay_texts.append(f"{key}: {metadata[key]}")

            if args.display_metadata:
                for i, text in enumerate(overlay_texts):
                    cv2.putText(display_img, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

            cv2.imshow("Camera", display_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                should_stop = True

        # if frame_count % 50 == 0:
        #     status = f"Captured Frame {frame_count}/{args.max-frames}"
        #     if args.show-fps:
        #         status += f", FPS: {fps:.2f}"
        #     print(status)

    print(f"Actual recorded frames: {frame_count}/{args.num_images}")
    cv2.destroyAllWindows()

    # Save and clean up data
    if args.save_raw:
        frame_memmap.flush()
        del frame_memmap
        metadata_memmap.flush()
        del metadata_memmap

        # compute how many bytes are really used
        bytes_per_frame = args.crop_height * args.crop_width * np.dtype('uint16').itemsize
        new_raw_size  = frame_count * bytes_per_frame

        bytes_per_meta = metadata_dtype.itemsize
        new_meta_size = frame_count * bytes_per_meta

        # truncate the files down to that size
        raw_path  = os.path.join(args.output_path, f'raw_frames_{current_tag}_{timestamp}.dat')
        meta_path = os.path.join(args.output_path, f'metadata_{current_tag}_{timestamp}.dat')
        os.truncate(raw_path,  new_raw_size)
        os.truncate(meta_path, new_meta_size)

    if args.save_rgb:
        rgb_memmap.flush()
        del rgb_memmap

        # compute bytes per RGB frame (frame, height, width, channels=4), dtype=uint8 ⇒ 1 byte per element
        bytes_per_rgb_frame = args.crop_height * args.crop_width * 4 * np.dtype('uint8').itemsize
        new_rgb_size = frame_count * bytes_per_rgb_frame

        # truncate the RGB .dat file
        rgb_path = os.path.join(args.output_path, f"rgb_frames_{current_tag}_{timestamp}.dat")
        os.truncate(rgb_path, new_rgb_size)
    # Asynchronous data transfer
    transfer_to_host(args, timestamp, mqtt_client)

def wait_for_focus(picam2, timeout=10):
    """Wait for camera focus
    
    Args:
        picam2: Camera instance
        timeout: Timeout in seconds
    
    Returns:
        bool: Whether focus was successful
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        (_,), metadata = picam2.capture_arrays(["raw"])
        if "AfState" in metadata and metadata["AfState"] == controls.AfStateEnum.Focused:
            print("Camera focused.")
            return True
        time.sleep(0.1)
    raise RuntimeError("Camera focus timeout.")

# MQTT callback functions
def on_connect(client, userdata, flags, rc, args):
    print(f"Connected to MQTT Broker, status code: {rc}")
    client.subscribe(args.mqtt_topic)

def on_message(client, userdata, msg, picam2):
    global recording, should_stop, current_tag
    message = msg.payload.decode('utf-8')
    print(f"Received message: {message}")
    
    if message.startswith("start_") and not recording:
        recording = True
        should_stop = False
        current_tag = message[6:]
        print(f"Starting recording... Tag: {current_tag}")
        
        # If in calibration mode, set focus mode to auto and wait for focus
        if current_tag.startswith("0_calibration"):
            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
            print("Auto focus mode enabled")
            wait_for_focus(picam2)
        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual})
        print("Focus locked")
    elif message == "stop" and recording:
        should_stop = True

def main():
    global recording, should_stop, current_tag
    args = parse_args()
    for file in os.listdir(args.output_path):
        if file.endswith('.dat'):
            os.remove(os.path.join(args.output_path, file))
            print(f"Deleted: {file}")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        raw={"format": "SRGGB10", "size": (args.crop_width, args.crop_height)},
        main={"size": (args.crop_width, args.crop_height), "format": "XBGR8888"}
    )
    picam2.configure(config)
    picam2.set_controls({
        "FrameDurationLimits": (8333, 8333),  # 120FPS
        "AeEnable": False,
        "AnalogueGain": 1.0,
        "AfMode": controls.AfModeEnum.Continuous,
        "AwbEnable": False
    })
    picam2.start()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = lambda client, userdata, flags, rc, properties: on_connect(client, userdata, flags, rc, args)
    client.on_message = lambda client, userdata, msg: on_message(client, userdata, msg, picam2)
    client.connect(args.mqtt_broker, args.mqtt_port, 60)
    client.loop_start()
    print("Waiting for start recording command...")
    try:
        while True:
            if recording:
                record_frames(picam2, args, client)
                recording = False
                should_stop = False
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program interrupted by user")
    client.loop_stop()
    client.disconnect()
    picam2.stop()

if __name__ == "__main__":
    main()
    
