# YOLO11 3D Object Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-red.svg)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computer vision application that uses YOLOv8 object detection with depth estimation to create 3D bounding boxes around detected objects in real-time video. The system projects 2D detections into 3D space using depth information and visualizes objects in a Bird's Eye View (BEV).

This project was inspired by and builds upon [Nicolai Nielsen's](https://www.youtube.com/watch?v=wAKmKsZ9PSw) YouTube tutorials.

![3D Object Detection Example](https://via.placeholder.com/800x400?text=3D+Object+Detection+Example)

## Features

- 🚀 Real-time object detection using YOLOv8
- 📏 Depth estimation using Depth Anything v2
- 📦 3D bounding box estimation
- 🐦 Bird's Eye View (BEV) visualization
- 🔄 Object tracking across frames
- 🎛️ Performance optimization with frame skipping and downsampling
- 📝 Detailed visualization of detection results

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- PIL

## Installation

1. Clone this repository:
```bash
git clone https://github.com/pakagronglb/yolo11-3d-object-detection.git
cd yolo11-3d-object-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install PyTorch (see [PyTorch website](https://pytorch.org/get-started/locally/) for options specific to your system):
```bash
pip install torch torchvision
```

## Usage

Run the application with:

```bash
python run.py --source 0  # Use webcam
# or
python run.py --source path/to/video.mp4  # Use video file
```

### Command-line Options

- `--source`: Input source (0 for webcam, or path to video file)
- `--output`: Output video path (default: output.mp4)
- `--device`: Device to run models on ('cpu', 'cuda', 'mps')
- `--model`: YOLO model size ('nano', 'small', 'medium', 'large', 'extra_large')
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--skip`: Frame skip value (process 1 out of N+1 frames for speed)
- `--scale`: Processing scale factor (smaller = faster, but less accurate)
- `--depth_scale`: Scale factor for depth estimation (smaller = faster)

### Performance Optimization

For better performance on less powerful hardware:

```bash
python run.py --device cpu --skip 2 --scale 0.5 --depth_scale 0.3
```

## Project Structure

```
├── run.py                  # Main application
├── detection_model.py      # YOLO object detection model
├── depth_model.py          # Depth estimation model
├── bbox3d_utils.py         # 3D bounding box utilities
├── requirements.txt        # Required packages
└── README.md               # This file
```

## How It Works

1. **Object Detection**: YOLOv8 detects objects in each video frame
2. **Depth Estimation**: A depth map is generated for the frame
3. **3D Estimation**: Depth information is used to project 2D boxes into 3D space
4. **Tracking**: Objects are tracked across frames (optional)
5. **Visualization**: Results are displayed with 3D boxes and Bird's Eye View

## Key Features Explained

### Bird's Eye View (BEV)

The BEV provides a top-down perspective of detected objects, showing their relative positions and distances. This is particularly useful for applications like autonomous driving or surveillance.

### Depth Estimation

The system uses the Depth Anything v2 model to estimate depth in monocular images. This depth information is crucial for placing objects in 3D space without requiring specialized hardware like LiDAR or stereo cameras.

### Performance Optimizations

- **Frame Skipping**: Process only a subset of frames for higher FPS
- **Resolution Scaling**: Downscale frames for faster processing
- **Depth Map Caching**: Cache depth maps for similar frames to reduce computation
- **Selective Processing**: Advanced object tracking reduces the need for detection on every frame

## Credits

This project was inspired by and builds upon the work of [Nicolai Nielsen](https://www.youtube.com/watch?v=wAKmKsZ9PSw) and his excellent YouTube tutorials on computer vision and object detection.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
