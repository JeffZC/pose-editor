# Pose Editor: Human Pose Annotation Tool

An intuitive PyQt5-based application for editing and refining human pose keypoint data. This tool is specifically designed to correct and improve keypoints recognized by human pose estimation algorithms such as MediaPipe, OpenPose, etc.

## Overview

Pose Editor bridges the gap between automated pose detection and human refinement, allowing users to manually adjust AI-generated keypoints that may be imprecise or incorrect. It's the perfect tool for creating high-quality pose datasets or refining motion capture data.

## Features
- Load and display video
- Import pose data from CSV files generated by pose estimation algorithms
- Interactive point selection and editing
- Black and white toggle for better visibility of difficult frames
- Precise coordinate input for exact keypoint positioning
- Intuitive zoom and pan functionality for detailed editing
- Frame-by-frame navigation for accurate temporal adjustments
- Save refined pose data for further analysis or machine learning tasks

## Requirements
```bash
python>=3.9
PyQt5>=5.15.0
opencv-python>=4.5.0
pandas>=1.3.0
numpy>=1.19.0

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