# Body Measurement System

This project uses computer vision to calculate a person's height and waist measurements using a standard webcam.

## Requirements

- Python 3.6 or higher
- Webcam
- The person should stand 6-7 feet away from the camera
- Good lighting conditions
- Wearing fitted clothing for accurate measurements

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the YOLOv8 model:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Usage

1. Run the main application:
```bash
python main.py
```

2. Follow the on-screen instructions:
   - Stand 6-7 feet away from the camera
   - Position yourself between the green guide lines
   - Press 'c' to capture and calculate measurements
   - Press 'q' to quit

## How it Works

1. **Distance Calculation**: 
   - Uses MediaPipe Pose detection to identify body landmarks
   - Calculates the distance based on the person's height in pixels

2. **Height Measurement**:
   - Detects nose and ankle points using MediaPipe
   - Uses the calculated distance to convert pixel measurements to centimeters

3. **Waist Measurement**:
   - Uses YOLOv8 for person detection and segmentation
   - Identifies the waist region using hip landmarks
   - Calculates approximate waist circumference

## Limitations

- Measurements are approximations and may need calibration
- Accuracy depends on:
  - Camera quality
  - Lighting conditions
  - Person's pose
  - Clothing fit
- Best results when wearing fitted clothing and standing in a neutral pose

## Files

- `main.py`: Main application file
- `camera_utils.py`: Camera handling and distance calculation
- `body_measurements.py`: Height and waist measurement calculations
- `requirements.txt`: Required Python packages
