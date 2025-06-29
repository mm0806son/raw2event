# Raw2Event: Convert Raw Frame Camera into Event Camera

Raw2event is a toolkit for converting traditional camera into event camera. It supports generating event data from video files, raw frame data, or real-time camera streams, and can output in multiple formats.

This project aims to to emulate the behavior of an event camera using affordable hardware such as a Raspberry Pi camera.

## Environment Setup

### System Dependencies
```bash
sudo apt install libcap-dev
sudo apt install python3-picamera2
sudo apt install libcamera-dev libcamera-apps python3-libcamera
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Scripts

- **`generate_event.py`** - Offline event generation
  
- **`record2npy.py`**- Raspberry Pi camera recording script

- **`cv2_video_display.py`**  - Pi Camera preview tool
- **`dvs_preview.py`** - DVS event preview tool
- **`event_display.py`**  - Event data display in real time (naive version)
- **`event_display_DVS.py`**  - Event data data display tool (realistic version)

## Configuration

In addition to the parameters calibrated by the authors of DVS-Voltmeter, we provide the parameters for Pi Camera Module 3. The configurations can be modified in `src/config.py`:
- DVS346: 346x260 resolution event camera (from its frame sensor to event)
- DVS240: 240x180 resolution event camera (from its frame sensor to event)
- Raw2DVS346: Raw data of Pi cam to DVS346
- RGB2DVS346: RGB data of Pi cam to DVS346

## Acknowledgement

Part of the code from this project is based on the DVS-Voltmeter paper:

```bibtex
@inproceedings{lin2022dvsvoltmeter,
  title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
  author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
  booktitle={ECCV},
  year={2022}
}
```
