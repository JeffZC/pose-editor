# Pose Editor

A PyQt5-based application for editing huamn pose data with videos.

## Features
- Load and display video files
- Load and edit pose data from CSV files
- Interactive point selection and editing
- Zoom and pan functionality
- Frame-by-frame navigation
- Save modified pose data

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