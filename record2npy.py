from picamera2 import Picamera2
from libcamera import controls
import time
import cv2
import numpy as np

# Params
MAX_FRAMES = 500
SAVE_DAT = True
DISPLAY = False
DISPLAY_METADATA = False
SHOW_FPS = True
CONVERT_NPY = False
CONVERT_CSV = False
SAVE_RGB = True
SAVE_GRAY = False
HOST_SHARE_PATH = 'host_shared'
LOCAL_TEMP_PATH = '/dev/shm'
SERVER_PATH = 'user@server:/ubuntu/user/folder'

# SEND_DATA = True

sensor_width, sensor_height = 1536, 864
# CROP_WIDTH, CROP_HEIGHT = 346, 260
CROP_WIDTH, CROP_HEIGHT = 692, 520

def crop(data):
    center_x = data.shape[1] // 2
    center_y = data.shape[0] // 2
    data = data[
        center_y - CROP_HEIGHT // 2:center_y + CROP_HEIGHT // 2,
        center_x - CROP_WIDTH // 2:center_x + CROP_WIDTH // 2
    ]
    return data

# Initialization
picam2 = Picamera2()
config = picam2.create_video_configuration(
    sensor={"output_size": (sensor_width, sensor_height), "bit_depth": 10},
    raw={"format": "SRGGB10", "size": (CROP_WIDTH, CROP_HEIGHT)}
)
picam2.configure(config)
picam2.set_controls({
    "FrameDurationLimits": (8333, 8333),  # 120FPS
    "AeEnable": False,
    "AnalogueGain": 10.0,
    "AfMode": controls.AfModeEnum.Continuous,
    "AwbEnable": False,
    "Brightness": 0.0,
    "Contrast": 1.0,
    "Saturation": 1.0,
})

picam2.start()

# ! Synchonize
(temp,), _ = picam2.capture_arrays(["raw"])
temp = temp.view('uint16')
if CROP_WIDTH != sensor_width and CROP_HEIGHT != sensor_height:
    temp = crop(temp)
temp = ((temp.astype('uint32') * 255) / (2**10 - 1)).astype('uint8')
cv2.imshow("Camera", temp)
while True:
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break
cv2.destroyAllWindows()

# Create memmap
if SAVE_DAT:
    mmap_shape = (MAX_FRAMES, CROP_HEIGHT, CROP_WIDTH)
    mmap_file = np.memmap(LOCAL_TEMP_PATH + '/raw_frames.dat', dtype='uint16', mode='w+', shape=mmap_shape)
    metadata_list = []
    metadata_fields = ['SensorTimestamp']
    metadata_dtype = np.dtype([
        ('SensorTimestamp', 'float64')
        ])
    metadata_memmap = np.memmap(LOCAL_TEMP_PATH + "/metadata.dat", dtype=metadata_dtype, mode='w+', shape=(MAX_FRAMES,))

if SAVE_RGB:
    rgb_shape = (MAX_FRAMES, CROP_HEIGHT, CROP_WIDTH, 4)
    rgb_memmap = np.memmap(LOCAL_TEMP_PATH + "/rgb_frames.dat", dtype='uint8', mode='w+', shape=rgb_shape)

if SAVE_GRAY:
    gray_mmap_shape = (MAX_FRAMES, CROP_HEIGHT, CROP_WIDTH)
    gray_memmap = np.memmap(LOCAL_TEMP_PATH + "/gray_frames.dat", dtype='uint8', mode='w+', shape=gray_mmap_shape)


# FPS
frame_count = 0
start_time = time.time()

# Capture first frame
(frame_first,), metadata_first = picam2.capture_arrays(["raw"])
time_first = metadata_first["SensorTimestamp"]

# Recording loop
while frame_count < MAX_FRAMES:
    (frame,), metadata = picam2.capture_arrays(["raw"])
    data = frame.view('uint16')

    # Crop
    if CROP_WIDTH != sensor_width and CROP_HEIGHT != sensor_height:
        data = crop(data)

    # Save to memmap
    if SAVE_DAT:
        mmap_file[frame_count] = data
        # sensor_time = (metadata["SensorTimestamp"] - time_first) / 1000
        sensor_time = metadata["SensorTimestamp"]
        metadata_memmap[frame_count] = sensor_time

    if SAVE_RGB:
        rgb_image = picam2.capture_array("main")
        if CROP_WIDTH != sensor_width and CROP_HEIGHT != sensor_height:
            rgb_image = crop(rgb_image)
        rgb_memmap[frame_count] = rgb_image

    if SAVE_GRAY:
        gray_frame = ((data.astype('uint32') * 255) / (2**10 - 1)).astype('uint8')
        gray_memmap[frame_count] = gray_frame


    frame_count += 1
    # Calculate FPS
    if SHOW_FPS:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # Show Metadata
    if DISPLAY:
        display_img = ((data.astype('uint32') * 255) / (2**10 - 1)).astype('uint8')
        overlay_texts = []
        if SHOW_FPS:
            overlay_texts.append(f"FPS: {fps:.2f}")
        if DISPLAY_METADATA:
            for key in metadata_fields:
                if key in metadata:
                    overlay_texts.append(f"{key}: {metadata[key]}")


        if DISPLAY_METADATA:
            for i, text in enumerate(overlay_texts):
                cv2.putText(display_img, text, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

        cv2.imshow("Camera", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if SHOW_FPS:
        print(f"Captured Frame {frame_count}/{MAX_FRAMES}, FPS: {fps:.2f}")
    else:
        print(f"Captured Frame {frame_count}/{MAX_FRAMES}")

# Stop
cv2.destroyAllWindows()
picam2.stop()

if SAVE_DAT:
    mmap_file.flush()
    del mmap_file  # Release lock
    metadata_memmap.flush()
    del metadata_memmap  

if SAVE_RGB:
    rgb_memmap.flush()
    del rgb_memmap

if SAVE_GRAY:
    gray_memmap.flush()
    del gray_memmap


if CONVERT_NPY:
    # Convert .dat to .npy
    final_array = np.memmap('raw_frames.dat', dtype='uint16', mode='r', shape=(MAX_FRAMES, CROP_HEIGHT, CROP_WIDTH))
    np.save(LOCAL_TEMP_PATH + "/raw_frames.npy", np.array(final_array))
    print("✅ Frame data saved to raw_frames.npy.")

if CONVERT_CSV:
    import csv
    # Read memmap
    print(f"Loading metadata from {'metadata.dat'}...")
    metadata = np.memmap('metadata.dat', dtype=metadata_dtype, mode='r', shape=(frame_count,))

    # Write to CSV
    csv_filename = LOCAL_TEMP_PATH + '/metadata.csv'
    print(f"Saving to CSV: {csv_filename}...")
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # header
        header = ["FrameIndex"] + list(metadata_dtype.names)
        writer.writerow(header)

        # Frames
        for i, row in enumerate(metadata):
            writer.writerow([i] + list(row))
            if i % 50 == 0:
                print(f"Saved frame {i}/{frame_count}")

    print("✅ Metadata CSV saved to {'metadata.dat'}.")


def transfer_to_host():
    import subprocess

    print("🚀 Sending data to host/cloud...")
    # subprocess.run(f"rsync -av {LOCAL_TEMP_PATH}/*dat {HOST_SHARE_PATH}", shell=True)
    subprocess.run(f"rsync -av {LOCAL_TEMP_PATH}/*dat {SERVER_PATH}", shell=True)
    subprocess.run(f"rm -f {LOCAL_TEMP_PATH}/*dat", shell=True)
    print("✅ Finish. Local memory released.")
    

transfer_to_host()

