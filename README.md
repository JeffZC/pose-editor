# Pose Editor: An Intuitive Tool for Human Pose Keypoint Refinement

Pose Editor is a user-friendly application for editing and refining human pose keypoint data, which bridges the gap between automated pose detection and human refinement, allowing users to manually adjust AI-generated keypoints that may be imprecise or incorrect. This versatile tool allows researchers, animators, and motion analysts to precisely correct and improve keypoints recognized by human pose estimation algorithms such as MediaPipe, OpenPose, etc.

## Key Features

- **Easy Video Navigation**: Browse through video frames with intuitive playback controls.
- **Interactive Pose Editing**: Click and drag keypoints directly on the video frame.
- **Black and White Toggle**: Switch between color and grayscale modes for better visibility.
- **Precise Coordinate Control**: Manually enter exact coordinates for perfect positioning.
- **Dynamic Zoom**: Zoom in for detailed edits and zoom out for full context.
- **Simple File Management**: Load videos, import existing pose data, and save your edited work with ease.

## Perfect For

- Motion analysis researchers
- Physical therapy assessment
- Sports performance analysis
- Computer vision dataset preparation

## Requirements
```bash
python>=3.9
PyQt5>=5.15.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.19.0
```

## Installation

### Quick Start

1. Download the ZIP file from the GitHub repository and extract it.
2. (Conditional:) [Install Python](https://www.python.org/downloads/) if you don’t have it yet.
3. Open a terminal (or Command Prompt) inside the extracted folder.
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python pose_annotator.py
   ```

### Using Conda (Recommended)

1. Open your Anaconda Prompt.
2. Create or activate a conda environment:
   ```bash
   conda create -n pose-editor python=3.9
   conda activate pose-editor
   ```
3. Clone or download the repository and navigate into it:
   ```bash
   git clone <repository-url>
   cd pose-editor
   ```
4. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the application:
   ```bash
   python pose_annotator.py
   ```

## Controls

- Left click to select points.
- Drag to move selected points.
- Right click to deselect points.
- Use +/- buttons to zoom in/out.
- Use arrow buttons or slider for frame navigation.
