"""
Show AEDAT4 file for sanity check.
"""
import dv_processing as dv
import argparse
parser = argparse.ArgumentParser(description='Read aedat4.')
parser.add_argument('-f,--file',
                    dest='file',
                    type=str,
                    required=True,
                    metavar='path/to/file',
                    help='Path to an AEDAT4 file')

args = parser.parse_args()
# Open a file
reader = dv.io.MonoCameraRecording(args.file)

# Run the loop while camera is still connected
while reader.isRunning():
    # Read batch of events
    events = reader.getNextEventBatch()
    if events is not None:
        # Print received packet time range
        print(f"{events}")
    for ev in events:
        print(f"Sliced event [{ev.timestamp()}, {ev.x()}, {ev.y()}, {ev.polarity()}]")
        break
    break