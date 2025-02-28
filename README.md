# Pose Editor: An Intuitive Tool for Human Pose Annotation

Pose Editor is a user-friendly application for editing and annotating human pose data in videos. This versatile tool allows researchers, animators, and motion analysts to precisely mark and adjust key points on the human body throughout video sequences.

## Key Features

- **Easy Video Navigation**: Browse through video frames with intuitive playback controls
- **Interactive Pose Editing**: Click and drag keypoints directly on the video frame
- **Black and White Toggle**: Switch between color and grayscale modes for better visibility
- **Precise Coordinate Control**: Manually enter exact coordinates for perfect positioning
- **Dynamic Zoom**: Zoom in for detailed edits and zoom out for full context
- **Synchronized Interface**: All controls stay in sync - clicking points updates the dropdown menu and coordinate displays automatically
- **Simple File Management**: Load videos, import existing pose data, and save your edited work with ease

## Perfect For

- Motion analysis researchers
- Animation reference creation
- Physical therapy assessment
- Sports performance analysis
- Computer vision dataset preparation

This application bridges the gap between raw video footage and precise human pose data, allowing for efficient annotation and correction without requiring technical expertise in computer vision or programming.

## Requirements
```bash
python>=3.9
PyQt5>=5.15.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.19.0
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pose-editor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python pose_annotator.py
```

## Controls
- Left click to select points
- Drag to move selected points
- Right click to deselect points
- Use +/- buttons to zoom in/out
- Use arrow buttons or slider for frame navigation
