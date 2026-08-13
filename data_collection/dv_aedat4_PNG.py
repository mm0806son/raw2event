# pip install dv-processing opencv-python
import os
from datetime import timedelta
import cv2 as cv
import dv_processing as dv

AEDAT_PATH = "cifar10_xdvs_preview/dv_output_20001_cat_2_9875_20250714_141817.aedat4"  
OUT_DIR    = "aedat_preview"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Open the AEDAT4 file.
reader = dv.io.MonoCameraRecording(AEDAT_PATH)  # docs: open file and read event batches.

# 2) Initialize the accumulator at the event resolution.
acc = dv.Accumulator(reader.getEventResolution())
# Common configuration (can be tuned to taste).
acc.setMinPotential(0.0)
acc.setMaxPotential(1.0)
acc.setNeutralPotential(0.5)
acc.setEventContribution(0.15)
acc.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
acc.setDecayParam(1e6)           # tau (us): controls afterglow / decay rate
acc.setIgnorePolarity(False)
acc.setSynchronousDecay(False)   # See docs for these APIs and the 8-bit grayscale output.

slicer = dv.EventStreamSlicer()
frame_idx = 0

def dump_frame_from_events(events: dv.EventStore):
    global frame_idx
    acc.accept(events)
    frame = acc.generateFrame()          # dv.Frame.image is the grayscale frame.
    # Save as PNG (change the extension for JPEG).
    cv.imwrite(os.path.join(OUT_DIR, f"frame_{frame_idx:06d}.png"), frame.image)
    frame_idx += 1

# 3) Read every event and accumulate them into a single frame.
all_events = []
while reader.isRunning():
    batch = reader.getNextEventBatch()   # docs: read events in batches.
    if batch is not None:
        all_events.append(batch)

# Merge all event batches and produce one frame.
if all_events:
    # Merge all event batches.
    combined_events = dv.EventStore()
    for batch in all_events:
        combined_events.add(batch)
    
    # Generate the final frame.
    acc.accept(combined_events)
    frame = acc.generateFrame()
    cv.imwrite(os.path.join(OUT_DIR, "all_events_frame.png"), frame.image)
    print(f"Wrote single-frame image of all events: all_events_frame.png")

print(f"done. All events merged into a single image at: {OUT_DIR}")
