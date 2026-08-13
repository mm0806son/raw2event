# CIFAR10-XDVS

CIFAR10-XDVS: Extended CIFAR10-DVS with Raw, RGB, and Event Streams via a Unified Capture System

## How to use

### Environment

1. Clone this repository to both Raspberry Pi and the host PC.
2. Download and prepare the CIFAR-10 dataset using `cifar10_extract.py`. It is recommended to save the paths of all images to a text file so that the collection can be paused and resumed.

For Raspberry Pi

```
sudo apt update
sudo apt install libcap-dev
sudo apt install python3-picamera2
sudo apt install libcamera-dev libcamera-apps python3-libcamera
```

### System setup

1. Connect Pi Cam to Raspberry Pi, connect Dobot and DVS to host PC.
2. Connect Pi and PC to the network, check the communication using `mqtt_helloworld_host/client.py`.
3. Run `set_home.py` to initiate Dobot.
4. Run `cv2_video_display.py` on Pi, and run `dataset_control_host.py --preview` to check if the image and the tag can always be seen during the movement.
5. Adjust the position of Dobot and repeat step 3 and 4, if needed.
6. Run `dataset_control_client.py` on Pi, and then run `dataset_control_host.py` on host PC.

## Scripts

### Main functions

- `dataset_control_host.py`: Host script for recording event camera data using a DV camera. Displays images, controls a robotic arm, communicates with the client, records data, and transfers results to a server.
- `dataset_control_client.py`: Client script for recording raw/RGB frame data using Pi Camera. Listens for MQTT commands, captures and saves images, and transfers data to a remote server.

### Helper functions

- `aedat42csv.py`: Converts AEDAT4 event files to CSV format.
- `cifar10_extract.py`: Prepare the CIFAR10 dataset. Extracts and saves CIFAR-10 images by class into separate folders as PNG files.
- `csv2npy.py`: Converts CSV files to NPY (NumPy) format.
- `cv2_video_display.py`: Displays camera output (raw or RGB) using Pi Camera.
- `dat2csv.py`: Converts .dat files (raw or metadata) to .csv format for checking.
- `dvs_preview.py`: Previews DVS (event camera) stream using the dv_processing library.
- `file_playback.py`: Event stream playback tool. Plays back AEDAT4, TXT, or CSV event stream files and visualizes them using OpenCV.
- `mqtt_helloworld_client.py`: Simple MQTT client example for testing message sending/receiving.
- `mqtt_helloworld_host.py`: Simple MQTT host/server example for testing message sending/receiving.
- `npy2csv.py`: Converts NPY (NumPy) files to CSV format.
- `read_aedat4.py`: Read and print AEDAT4 event data for sanity check.
- `set_home.py`: Sets the home position for the Dobot Magician robotic arm.

### Image Resources

- The `image/` folder contains calibration and tag images used for display and robotic arm calibration.
