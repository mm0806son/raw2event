"""
Event Stream Playback Tool

This script plays back AEDAT4, TXT, or CSV event stream files and visualizes them using OpenCV.

Usage:
    python file_playback.py -f path/to/your_file.aedat4
    python file_playback.py -f path/to/your_file.txt
    python file_playback.py -f path/to/your_file.csv

Arguments:
    -f, --file    Path to an AEDAT4 / TXT / CSV file
"""
import dv_processing as dv
import cv2 as cv
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='Playback AEDAT4 / TXT / CSV event stream.')
parser.add_argument('-f', '--file',
                    dest='file',
                    type=str,
                    required=True,
                    metavar='path/to/file',
                    help='Path to an AEDAT4 / TXT / CSV file')
args = parser.parse_args()

file_ext = os.path.splitext(args.file)[-1].lower()

# Set time window (30ms)
time_slice_us = 30000

# Create preview window
cv.namedWindow("Event Playback", cv.WINDOW_NORMAL)

# Initialize variables
acc = None
events = None
current_time = 0
resolution = (692, 520)  # Default resolution (for DVS346)

def load_events_from_text_file(path):
    print(f"Reading event data from text file: {path}")
    data = np.loadtxt(path, delimiter=',')
    timestamps = data[:, 0].astype(np.int64)
    x = data[:, 1].astype(np.uint16)
    y = data[:, 2].astype(np.uint16)
    polarity = data[:, 3].astype(np.bool_)

    event_store = dv.EventStore()
    for i in range(len(timestamps)):
        event_store.push_back(dv.Event(timestamps[i], x[i], y[i], polarity[i]))
    return event_store, timestamps[0], timestamps[-1]

def load_events_from_csv_file(path):
    print(f"Reading event data from CSV file: {path}")
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    timestamps = data['timestamp'].astype(np.int64)
    x = data['x'].astype(np.uint16)
    y = data['y'].astype(np.uint16)
    polarity = data['polarity'].astype(np.bool_)

    event_store = dv.EventStore()
    for i in range(len(timestamps)):
        event_store.push_back(dv.Event(timestamps[i], x[i], y[i], polarity[i]))
    return event_store, timestamps[0], timestamps[-1]

# -------------------- TXT file processing --------------------
if file_ext =='.txt':
    event_store, start_time, end_time = load_events_from_text_file(args.file)
    current_time = start_time

    acc = dv.Accumulator(resolution)
    acc.setMaxPotential(1.0)
    acc.setEventContribution(0.12)

    while current_time < end_time:
        sliced = event_store.sliceTime(current_time, current_time + time_slice_us)
        if sliced.isEmpty():
            break

        acc.accept(sliced)
        frame = acc.generateFrame()
        cv.imshow("Event Playback", frame.image)

        key = cv.waitKey(int(time_slice_us / 1000))
        if key == 27:
            break

        current_time += time_slice_us

# -------------------- CSV file processing --------------------
if file_ext == '.csv':
    event_store, start_time, end_time = load_events_from_csv_file(args.file)
    current_time = start_time

    acc = dv.Accumulator(resolution)
    acc.setMaxPotential(1.0)
    # acc.setMaxPotential(0.01)
    acc.setEventContribution(0.12)

    while current_time < end_time:
        sliced = event_store.sliceTime(current_time, current_time + time_slice_us)
        if sliced.isEmpty():
            break

        acc.accept(sliced)
        frame = acc.generateFrame()
        cv.imshow("Event Playback", frame.image)
        
        key = cv.waitKey(int(time_slice_us / 1000))
        if key == 27:
            break

        current_time += time_slice_us
# -------------------- AEDAT4 file processing --------------------
elif file_ext == '.aedat4':
    print("Reading event data from AEDAT4 file...")

    recording = dv.io.MonoCameraRecording(args.file)
    assert recording.isEventStreamAvailable()

    acc = dv.Accumulator(resolution)
    acc.setMaxPotential(1.0)
    # acc.setMaxPotential(0.01)
    acc.setEventContribution(0.12)

    events = recording.getNextEventBatch()
    print(f"First event: timestamp {events[0].timestamp()}, coordinates: ({events[0].x()}, {events[0].y()})")

    start_time = events[0].timestamp()
    current_time = start_time

    while True:
        events = recording.getEventsTimeRange(current_time, current_time + time_slice_us)
        if events is None or len(events) == 0:
            break

        acc.accept(events)
        frame = acc.generateFrame()
        cv.imshow("Event Playback", frame.image)

        print(f"Number of events in time window [{current_time}, {current_time+time_slice_us}]: {len(events)}")
        # Assuming frame.image is a float type image matrix, print its statistics:
        print(f"Accumulated frame max: {np.max(frame.image)}, mean: {np.mean(frame.image)}")
        key = cv.waitKey(int(time_slice_us / 1000))
        if key == 27:
            break

        current_time += time_slice_us

else:
    print("Error: Only .aedat4 / .txt / .csv file formats are supported!")

cv.destroyAllWindows()
