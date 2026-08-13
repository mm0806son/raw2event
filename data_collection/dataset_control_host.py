"""
Host script for recording event camera data using DV camera.

This script displays images, triggers robotic arm motion, communicates with client to start/stop recording, records data, and transfers results to a server.
Example usage:
python dataset_control_host.py --num-images 10 --server-path /path/to/server

This will process 10 images and transfer the recorded data to the specified server path.
"""
import time
import datetime
import threading
import paho.mqtt.client as mqtt
import dobot.DobotDllType as dType
import dv_processing as dv
import cv2 as cv
import argparse
import subprocess
import os
import numpy as np
from screeninfo import get_monitors
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel
import sys
import re

# MQTT message constants
MQTT_MESSAGE_START = "start"  # Start recording command
MQTT_MESSAGE_STOP = "stop"    # Stop recording command

# Display parameters
DISPLAY_CONFIG = {
    "screen": 1,            # Display screen number (0 for main screen, 1 for secondary screen, etc.)
    "screen_width": -1,     # Screen width (if set to -1, will use actual width of selected screen)
    "screen_height": -1,    # Screen height (if set to -1, will use actual height of selected screen)
    "canvas_width_ratio": 0.95,   # Canvas width ratio to screen width
    "canvas_height_ratio": 0.95,  # Canvas height ratio to screen height
    "image_width_ratio": 0.28,    # Image width ratio to canvas width
    "image_height_ratio": 0.28,   # Image height ratio to canvas height
    "image_x": -1,          # Image position (-1 for center)
    "image_y": 450,          # Image position (-1 for center)
    "fullscreen": True,     # Whether to display in fullscreen
}

# Image relative size parameters (based on reference values)
IMAGE_RATIO = {
    "barbara_ref_size": 861,  # Barbara reference side length
    "tag_ref_width": 287,     # Tag reference width
    "gap": 82,               # Gap between images
}

def to_qt(img_np):
    """Convert numpy image to QPixmap"""
    # Convert to RGB if not already
    if len(img_np.shape) == 2:  # Grayscale image
        img_rgb = cv.cvtColor(img_np, cv.COLOR_GRAY2RGB)
    else:  # Color image
        img_rgb = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)  # OpenCV uses BGR order
    
    h, w, _ = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, 3*w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class FullScreenPlayer(QLabel):
    def __init__(self, pixmap, screen_idx=0):
        super().__init__()
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        
        # Get available screens list
        screens = QApplication.screens()
        
        # Ensure screen_idx is within valid range
        if screen_idx >= len(screens):
            print(f"Warning: Screen {screen_idx} does not exist, using main screen")
            screen_idx = 0
        
        # Get target screen
        self.target_screen = screens[screen_idx]
        screen_geom = self.target_screen.geometry()
        
        print(f"Target screen {screen_idx}: Position=({screen_geom.x()}, {screen_geom.y()}), "
              f"Size={screen_geom.width()}x{screen_geom.height()}")
        
        # Set window properties
        self.setWindowTitle("Image Display (Press ESC to exit)")
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        
        # Set geometry to match target screen
        self.setGeometry(screen_geom)
        
        # Generate window handle and bind to target screen
        self.setAttribute(Qt.WA_NativeWindow, True)
        if self.windowHandle() is not None:
            self.windowHandle().setScreen(self.target_screen)
        else:
            print("Warning: Unable to create window handle, trying alternative method")
            self.show()
            self.hide()
            if self.windowHandle() is not None:
                self.windowHandle().setScreen(self.target_screen)
        
        if DISPLAY_CONFIG["fullscreen"]:
            self.showFullScreen()
        else:
            self.show()
        
        # Ensure window is in correct position
        QTimer.singleShot(100, self.check_screen_position)
    
    def check_screen_position(self):
        """Ensure window is on the correct screen"""
        if self.windowHandle() and self.windowHandle().screen() != self.target_screen:
            print("Correcting window screen position...")
            self.windowHandle().setScreen(self.target_screen)
            self.showFullScreen()
    
    def close_window(self):
        """Close window"""
        self.hide()
        self.close()
        if self.windowHandle():
            self.windowHandle().close()

def get_screen_info(screen_index):
    """Get information about specified screen"""
    monitors = get_monitors()
    if screen_index < len(monitors):
        return monitors[screen_index]
    else:
        print(f"Warning: Screen {screen_index} does not exist, using main screen")
        return monitors[0]

def calculate_sizes(config, screen_info):
    """Calculate actual sizes based on ratios"""
    # Use actual screen size or configured size
    screen_width = screen_info.width if config["screen_width"] == -1 else config["screen_width"]
    screen_height = screen_info.height if config["screen_height"] == -1 else config["screen_height"]
    
    # Calculate canvas size
    canvas_width = int(screen_width * config["canvas_width_ratio"])
    canvas_height = int(screen_height * config["canvas_height_ratio"])
    
    # Calculate target image size
    target_width = int(canvas_width * config["image_width_ratio"])
    target_height = int(canvas_height * config["image_height_ratio"])
    
    return (canvas_width, canvas_height), (target_width, target_height)

def calculate_relative_sizes(barbara_size):
    """Calculate relative size of Tag image"""
    # Calculate scale
    scale = barbara_size[0] / IMAGE_RATIO["barbara_ref_size"]
    
    # Calculate Tag size
    tag_width = int(IMAGE_RATIO["tag_ref_width"] * scale)
    tag_height = tag_width  # Tag is square
    
    # Calculate gap
    gap = int(IMAGE_RATIO["gap"] * scale)
    
    return (tag_width, tag_height), gap

class DVRecorder:
    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.camera = None
        self.current_tag = None
        self.dobot_api = None
        self.writer = None
        self.transfer_complete = threading.Event()
        
        # Load CIFAR10 dataset
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')
        self.images = []
        self.labels = []
        
        # Add barbara.jpeg as calibration image
        barbara_path = "image/barbara.jpeg"
        if os.path.exists(barbara_path):
            self.images.append(barbara_path)
            self.labels.append(-1)  # Use -1 to represent calibration image
            print("Calibration image added: barbara.jpeg")
        else:
            print("Warning: Calibration image barbara.jpeg not found")
        
        # Load images from local directory
        # for class_idx, class_name in enumerate(self.classes):
        #     class_dir = os.path.join('cifar10', class_name)
        #     if os.path.exists(class_dir):
        #         for img_name in os.listdir(class_dir):
        #             if img_name.endswith(('.png', '.jpg', '.jpeg')):
        #                 img_path = os.path.join(class_dir, img_name)
        #                 self.images.append(img_path)
        #                 self.labels.append(class_idx)
        
        # Load image paths from TXT
        txt_path = self.args.paths_file
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path:  # Ignore empty lines
                        self.images.append(img_path)
                        # Infer class
                        parts = img_path.split('/')
                        if len(parts) >= 2:
                            class_name = parts[1]
                            if class_name in self.classes:
                                class_idx = self.classes.index(class_name)
                                self.labels.append(class_idx)
                            else:
                                print(f"Warning: Unknown class {class_name}, skipping label")
                                self.labels.append(-1)
                        else:
                            print(f"Warning: Unable to infer class from path {img_path}")
                            self.labels.append(-1)
            print(f"Loaded {len(self.images)-1} images from {txt_path}")
        else:
            print(f"Error: Path file {txt_path} not found")

        print(f"Loaded {len(self.images)} images")
        
        # Get screen information
        self.screen_info = get_screen_info(DISPLAY_CONFIG["screen"])
        self.canvas_size, self.target_size = calculate_sizes(DISPLAY_CONFIG, self.screen_info)
        
        # Load tag image
        self.tag_image = cv.imread("image/AprilTag_tag36h11.png")
        if self.tag_image is None:
            print("Warning: Unable to load tag image")
        
        # Initialize image display
        self.init_image_display()

    def init_image_display(self):
        """Initialize image display related settings"""
        # Create Qt application instance
        self.app = QApplication.instance() or QApplication(sys.argv)
        
        # Calculate Tag size and gap (using first image as reference)
        if self.images:
            img = cv.imread(self.images[0])
            aspect_ratio = img.shape[1] / img.shape[0]
            if aspect_ratio > 1:  # Wide image
                new_width = min(self.target_size[0], int(self.target_size[1] * aspect_ratio))
                new_height = int(new_width / aspect_ratio)
            else:  # Tall image
                new_height = min(self.target_size[1], int(self.target_size[0] / aspect_ratio))
                new_width = int(new_height * aspect_ratio)
            
            barbara_size = (new_width, new_height)
            self.tag_size, self.gap = calculate_relative_sizes(barbara_size)
            
            # Calculate image position
            x, y = DISPLAY_CONFIG["image_x"], DISPLAY_CONFIG["image_y"]
            if x == -1:
                # Consider Tag width and gap, center overall
                total_width = new_width + self.gap + self.tag_size[0]
                self.image_x = (self.canvas_size[0] - total_width) // 2
            else:
                self.image_x = x
                
            if y == -1:
                self.image_y = (self.canvas_size[1] - new_height) // 2
            else:
                self.image_y = y
            
            # Calculate Tag position
            self.tag_x = self.image_x + new_width + self.gap
            self.tag_y = self.image_y + new_height - self.tag_size[1]  # Bottom edge aligned
            
            # Create blank canvas
            canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255
            pixmap = to_qt(canvas)
            
            # Create display window
            self.player = FullScreenPlayer(pixmap, screen_idx=DISPLAY_CONFIG["screen"])
            self.player.showFullScreen()

    def on_connect(self, client, userdata, flags, rc, properties=None):
        """MQTT connection callback function"""
        print("Connected to MQTT Broker, status code:", rc)
        client.subscribe(self.args.mqtt_topic)

    def on_message(self, client, userdata, msg):
        """MQTT message receive callback function"""
        message = msg.payload.decode('utf-8')
        print("Received message:", message)
        
        if message.startswith("transfer_complete_"):
            self.transfer_complete.set()

    def send_mqtt_message(self, message):
        """
        Send message to MQTT Broker
        
        Args:
            message (str): Message content to send
        """
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.args.mqtt_broker, self.args.mqtt_port, 60)
        client.publish(self.args.mqtt_topic, payload=message, qos=0)
        print(f"MQTT message sent: {message}")
        client.disconnect()

    def init_camera(self):
        """
        Initialize DV camera
        """
        if self.camera is None:
            self.camera = dv.io.CameraCapture(self.args.camera_id)
            print("DV camera initialization complete")
        return self.camera is not None

    def update_image(self, img_path):
        """Update displayed image"""
        # Display image
        img = cv.imread(img_path)
        
        # Calculate target size of image (maintain aspect ratio)
        aspect_ratio = img.shape[1] / img.shape[0]
        if aspect_ratio > 1:  # Wide image
            new_width = min(self.target_size[0], int(self.target_size[1] * aspect_ratio))
            new_height = int(new_width / aspect_ratio)
        else:  # Tall image
            new_height = min(self.target_size[1], int(self.target_size[0] / aspect_ratio))
            new_width = int(new_height * aspect_ratio)
        
        # Resize image
        img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        
        # Resize Tag
        tag_img = cv.resize(self.tag_image, self.tag_size, interpolation=cv.INTER_LINEAR)
        
        # Create white canvas
        canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 255
        
        # Paste image and Tag onto canvas
        canvas[self.image_y:self.image_y+new_height, self.image_x:self.image_x+new_width] = img
        canvas[self.tag_y:self.tag_y+self.tag_size[1], self.tag_x:self.tag_x+self.tag_size[0]] = tag_img
        
        # Update display
        pixmap = to_qt(canvas)
        self.player.setPixmap(pixmap)
        self.app.processEvents()

    def process_image(self, img_path, label, counter):
        """
        Process single image and trigger data collection process
        
        Args:
            img_path (str): Image file path
            label (int): Image label
            counter (int): Current image count
        """
        # Update displayed image
        self.update_image(img_path)
        
        # Generate label
        if label == -1:  # Calibration image
            self.current_tag = f"{counter}_calibration"
            time.sleep(1)
        else:
            img_filename = os.path.splitext(os.path.basename(img_path))[0]
            self.current_tag = f"{counter}_{self.classes[label]}_{img_filename}"
        
        # Set output path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(self.args.output_path)
        self.output_path = os.path.join(output_dir, f"dv_output_{self.current_tag}_{timestamp}.aedat4")
        
        # Reset transfer complete event
        self.transfer_complete.clear()
        
        # Send start message
        self.send_mqtt_message(f"{MQTT_MESSAGE_START}_{self.current_tag}")
        
        # Start recording
        dv_thread = self.start_recording()
        if dv_thread is None:
            print("Recording start failed")
            return
        
        # Execute robotic arm motion
        self.move_dobot()
        time.sleep(4)
        
        # Stop recording
        self.stop_recording()
        dv_thread.join()
        
        # Send stop message
        self.send_mqtt_message(MQTT_MESSAGE_STOP)
        
        # Transfer data
        self.transfer_to_server()
        
        # Wait for client to complete transfer
        print(f"Waiting for client to complete transfer: {self.current_tag}")
        self.transfer_complete.wait(timeout=120)  # Set 120 seconds timeout

    def start_recording(self):
        """Start DV recording"""
        if not self.init_camera():
            print("DV camera initialization failed")
            return None
        self.stop_event.clear()
        dv_thread = threading.Thread(target=self.dv_record_thread)
        dv_thread.start()
        return dv_thread

    def stop_recording(self):
        """Stop DV recording"""
        self.stop_event.set()
        if self.writer is not None:
            self.writer = None
        if self.camera is not None:
            self.camera = None
        time.sleep(0.5)  # Wait for recording thread to end

    def init_dobot(self):
        """
        Initialize Dobot robotic arm
        """
        self.dobot_api = dType.load()
        connectResult = dType.ConnectDobot(self.dobot_api, self.args.dobot_port, self.args.dobot_baudrate)
        if connectResult[0] != dType.DobotConnect.DobotConnect_NoError:
            print("Failed to connect to Dobot")
            return False

        print("Connected to Dobot")
        dType.ClearAllAlarmsState(self.dobot_api)
        dType.SetQueuedCmdForceStopExec(self.dobot_api)
        dType.SetQueuedCmdClear(self.dobot_api)
        dType.SetQueuedCmdStartExec(self.dobot_api)

        dType.SetARCParams(self.dobot_api, 50, 50, 50, 50, isQueued=0)
        dType.SetPTPCommonParams(self.dobot_api, 15, 15, isQueued=0)
        return True

    def move_dobot(self):
        """
        Control Dobot robotic arm to perform circular motion
        """
        if self.dobot_api is None:
            print("Dobot not initialized")
            return

        # Initiate position
        start_x, start_y, start_z, start_r = self.args.dobot_pos
        if self.args.init_pos:
            self.args.init_pos = False # Set to False after initiating 
            dType.SetPTPCmd(self.dobot_api, dType.PTPMode.PTPMOVJXYZMode,
                            start_x, start_y, start_z, start_r, isQueued=1)
            time.sleep(2)

        mid = (start_x + self.args.radius, start_y + self.args.radius, 100, 90)
        end = (start_x + 2 * self.args.radius, start_y, 100, 90)

        dType.SetCircleCmd(self.dobot_api, mid, end, count=self.args.circle_count, isQueued=0)
        print("Starting circular motion")

    def disconnect_dobot(self):
        """
        Disconnect Dobot
        """
        if self.dobot_api is not None:
            dType.SetQueuedCmdStopExec(self.dobot_api)
            dType.DisconnectDobot(self.dobot_api)
            print("Disconnected from Dobot")

    def preview_mode(self):
        """
        Preveiw mode: only display event stream, no recording
        """
        # Initialize camera
        self.camera = dv.io.CameraCapture(self.args.camera_id)

        if self.images:
            self.update_image(self.images[0])

        visualizer = dv.visualization.EventVisualizer(self.camera.getEventResolution())
        cv.namedWindow("Preview", cv.WINDOW_NORMAL)
        slicer = dv.EventStreamSlicer()
        def preview_events(event_slice):
            cv.imshow("Preview", visualizer.generateImage(event_slice))
            cv.waitKey(1)
        slicer.doEveryTimeInterval(datetime.timedelta(milliseconds=33), preview_events)
        # Start robotic arm motion
        self.init_dobot()
        self.move_dobot()
        # Display for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            events = self.camera.getNextEventBatch()
            if events is not None:
                slicer.accept(events)
        cv.destroyAllWindows()
        self.disconnect_dobot()
        print("Preview ended, program exiting")

    def run(self):
        """
        Run dataset collection process
        """
        if self.args.preview:
            self.preview_mode()
            return
        # Initialize Dobot
        if not self.init_dobot():
            print("Dobot initialization failed, program exiting")
            return

        # Initialize MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(self.args.mqtt_broker, self.args.mqtt_port, 60)
        client.loop_start()

        entries = list(zip(self.images, self.labels))
        calibration_entries = [(p, l) for p, l in entries if l == -1]
        dataset_entries = [(p, l) for p, l in entries if l != -1]

        # Calibration should always be executed with index 0 if available
        if calibration_entries:
            calib_path, calib_label = calibration_entries[0]
            self.process_image(calib_path, calib_label, 0)

        index_to_entry = {}
        for idx, (img_path, label) in enumerate(dataset_entries, start=self.args.start_index):
            index_to_entry[idx] = (img_path, label)

        if self.args.index_list:
            indices = load_index_list(self.args.index_list)
            for idx in indices:
                if idx == 0:
                    continue
                if idx not in index_to_entry:
                    print(f"Warning: index {idx} not found in paths file, skipping")
                    continue
                img_path, label = index_to_entry[idx]
                self.process_image(img_path, label, idx)
        else:
            counter = self.args.start_index
            processed = 0
            for img_path, label in dataset_entries:
                self.process_image(img_path, label, counter)
                counter += 1
                processed += 1
                if processed >= self.args.num_images:
                    break
                
        print("Dataset collection completed")
        client.loop_stop()
        client.disconnect()
        self.disconnect_dobot()
        if self.player:
                self.player.close_window()
    def transfer_to_server(self):
        """
        Asynchronously transfer recorded data to server
        
        Returns:
            subprocess.Popen: Transfer process object
        """
        rsync_cmd = f"rsync -z --progress {self.output_path} {self.args.server_path} && rm -f {self.output_path}"
        process = subprocess.Popen(rsync_cmd, shell=True)
        return process

    def dv_record_thread(self):
        """
        Event camera recording thread
        """
        if self.camera is None:
            print("DV camera not initialized")
            return
        current_camera = self.camera
        eventsAvailable = self.camera.isEventStreamAvailable()
        framesAvailable = self.camera.isFrameStreamAvailable()
        self.writer = dv.io.MonoCameraWriter(self.output_path, self.camera)
        print("Event camera recording started, saving to: {}".format(self.output_path))
        
        while not self.stop_event.is_set() and self.camera is not None:
            try:
                if self.camera.isEventStreamAvailable():
                    events = self.camera.getNextEventBatch()
                    if events is not None:
                        self.writer.writeEvents(events, streamName='events')
                if self.camera.isFrameStreamAvailable():
                    frame = self.camera.getNextFrame()
                    if frame is not None:
                        self.writer.writeFrame(frame, streamName='frames')
                time.sleep(0.01)
            except Exception as e:
                print(f"Error: {e}")
                break
        print("Event camera recording ended")
        self.writer = None

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Control Dobot robotic arm and record event camera data')
    
    # MQTT parameters
    parser.add_argument('--mqtt_broker', type=str, default="localhost",
                        help='MQTT server address')
    parser.add_argument('--mqtt_port', type=int, default=1883,
                        help='MQTT server port')
    parser.add_argument('--mqtt_topic', type=str, default="record",
                        help='MQTT topic')
    
    # Dobot parameters
    parser.add_argument('--dobot_port', type=str, default="/dev/ttyUSB0",
                        help='Dobot serial port device path')
    parser.add_argument('--dobot_baudrate', type=int, default=115200,
                        help='Dobot serial port baud rate')
    parser.add_argument('--dobot_pos', type=float, nargs=4, default=(240, 0, 100, 90),
                        help='Dobot position (x y z r)')
    parser.add_argument('--radius', type=int, default=12,
                        help='Circular motion radius')
    parser.add_argument('--circle_count', type=int, default=1,
                        help='Number of circular motions')
    parser.add_argument('--init_pos', action='store_true',
                        help='Initiate position before circular motion')
    
    # Camera parameters
    parser.add_argument('--output_path', type=str, default="./record",
                        help='Data save path')
    parser.add_argument('--camera_id', type=str, default="",
                        help='Camera device ID')
    parser.add_argument('--preview', action='store_true',
                        help='Enable preview mode (do not record data)')
    
    # Server transfer parameters
    parser.add_argument('--server_path', type=str, default='.',
                        help='Server path for storing recorded data')
    
    # Dataset parameters
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to process')
    parser.add_argument('--paths_file', type=str, default="cifar10_paths.txt",
                        help='CIFAR10 paths file (order defines index mapping)')
    parser.add_argument('--start_index', type=int, default=1,
                        help='Start index for tagging/ordering')
    parser.add_argument('--index_list', type=str, default=None,
                        help='Text file with non-continuous indices/tags to collect')
    
    return parser.parse_args()

def load_index_list(file_path):
    """
    Load indices from a text file. Supports lines like:
    - 52073 ship/ship_5_1647.png
    - 52073_ship_5_1647
    - 52073_ship_5_1647 missing=...
    """
    indices = []
    seen = set()
    tag_regex = re.compile(r"\b(\d+_[A-Za-z]+_[^ \t]+)")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = tag_regex.search(line)
            if match:
                tag = match.group(1)
                idx_part = tag.split("_", 1)[0]
                if idx_part.isdigit():
                    idx = int(idx_part)
                    if idx not in seen:
                        indices.append(idx)
                        seen.add(idx)
                continue
            parts = line.split()
            if parts and parts[0].isdigit():
                idx = int(parts[0])
                if idx not in seen:
                    indices.append(idx)
                    seen.add(idx)
    if not indices:
        print("Warning: index_list is empty or no valid indices found")
    return indices

def main():
    """
    Main function
    """
    args = parse_args()
    recorder = DVRecorder(args)
    recorder.run()

if __name__ == "__main__":
    main()
