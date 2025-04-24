import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from .body_format import get_keypoint_coordinates

def create_plot_widget(parent=None):
    """
    Creates a widget containing a matplotlib plot for visualizing keypoint data.
    
    Args:
        parent: Parent widget
        
    Returns:
        Widget containing the plot
    """
    # Create widget
    plot_widget = QWidget(parent)
    layout = QVBoxLayout(plot_widget)
    
    # Create matplotlib figure
    figure = Figure(figsize=(4, 3), dpi=100, tight_layout=True)
    canvas = FigureCanvas(figure)
    layout.addWidget(canvas)
    
    # Store figure and canvas in the widget for future access
    plot_widget.figure = figure
    plot_widget.canvas = canvas
    plot_widget.axes = None  # Will be created when data is plotted
    
    return plot_widget

def plot_keypoint_trajectory(widget, pose_data, keypoint_name, format_name="mediapipe33"):
    """
    Plots the trajectory of a specific keypoint over all frames.
    
    Args:
        widget: Plot widget created by create_plot_widget
        pose_data: DataFrame containing pose data
        keypoint_name: Name of the keypoint to plot
        format_name: Format name
    """
    if pose_data is None:
        return
        
    # Clear figure and create axes if needed
    widget.figure.clear()
    widget.axes = widget.figure.add_subplot(111)
    
    # Extract X and Y coordinates for all frames
    num_frames = len(pose_data)
    x_values = []
    y_values = []
    
    for frame_idx in range(num_frames):
        coords = get_keypoint_coordinates(pose_data, frame_idx, keypoint_name, format_name)
        if coords is not None:
            x, y, visibility = coords
            if visibility > 0.5:  # Only include points with good visibility
                x_values.append(x)
                y_values.append(y)
    
    # Plot trajectory
    widget.axes.plot(x_values, y_values, 'b-', linewidth=1)  # Line
    widget.axes.scatter(x_values, y_values, c='r', s=10)     # Points
    
    # Invert Y axis because image coordinates have Y increasing downward
    widget.axes.invert_yaxis()
    
    # Set labels
    widget.axes.set_title(f'Trajectory of {keypoint_name}')
    widget.axes.set_xlabel('X coordinate')
    widget.axes.set_ylabel('Y coordinate')
    
    # Update canvas
    widget.canvas.draw()

def calculate_angle(point1, point2, point3):
    """
    Calculates the angle between three points (in degrees).
    
    Args:
        point1: First point (x, y) tuple
        point2: Second point (vertex) (x, y) tuple
        point3: Third point (x, y) tuple
        
    Returns:
        Angle in degrees
    """
    if point1 is None or point2 is None or point3 is None:
        return None
    
    # Extract coordinates
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    x3, y3 = point3[0], point3[1]
    
    # Calculate vectors
    vector1 = [x1 - x2, y1 - y2]
    vector2 = [x3 - x2, y3 - y2]
    
    # Calculate dot product and magnitudes
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
    magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
    
    # Calculate angle in radians
    if magnitude1 * magnitude2 == 0:
        return None
        
    cos_angle = max(-1, min(1, dot_product / (magnitude1 * magnitude2)))
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_ankle_angle(pose_data, frame_idx, format_name="mediapipe33", side="left"):
    """
    Calculates the angle at the ankle joint.
    
    Args:
        pose_data: DataFrame containing pose data
        frame_idx: Frame index
        format_name: Format name
        side: "left" or "right"
        
    Returns:
        Angle in degrees
    """
    # Define keypoint names based on side
    if side.lower() == "left":
        knee_name = "LEFT_KNEE"
        ankle_name = "LEFT_ANKLE"
        foot_name = "LEFT_FOOT_INDEX"
    else:
        knee_name = "RIGHT_KNEE"
        ankle_name = "RIGHT_ANKLE"
        foot_name = "RIGHT_FOOT_INDEX"
    
    # Get coordinates
    knee = get_keypoint_coordinates(pose_data, frame_idx, knee_name, format_name)
    ankle = get_keypoint_coordinates(pose_data, frame_idx, ankle_name, format_name)
    foot = get_keypoint_coordinates(pose_data, frame_idx, foot_name, format_name)
    
    # Check if points are valid
    if knee is None or ankle is None or foot is None:
        return None
    
    # Extract coordinates (ignore visibility for angle calculation)
    knee_point = (knee[0], knee[1])
    ankle_point = (ankle[0], ankle[1])
    foot_point = (foot[0], foot[1])
    
    # Calculate angle
    return calculate_angle(knee_point, ankle_point, foot_point)

def plot_angular_data(widget, pose_data, joint_name, format_name="mediapipe33"):
    """
    Plots angular data for a specific joint across all frames.
    
    Args:
        widget: Plot widget created by create_plot_widget
        pose_data: DataFrame containing pose data
        joint_name: Name of the joint (e.g., "ankle", "knee")
        format_name: Format name
    """
    if pose_data is None:
        return
        
    # Clear figure and create axes if needed
    widget.figure.clear()
    widget.axes = widget.figure.add_subplot(111)
    
    # Extract angular data for all frames
    num_frames = len(pose_data)
    frame_indices = list(range(num_frames))
    left_angles = []
    right_angles = []
    
    for frame_idx in range(num_frames):
        if joint_name.lower() == "ankle":
            left_angle = calculate_ankle_angle(pose_data, frame_idx, format_name, "left")
            right_angle = calculate_ankle_angle(pose_data, frame_idx, format_name, "right")
            left_angles.append(left_angle)
            right_angles.append(right_angle)
    
    # Plot data
    widget.axes.plot(frame_indices, left_angles, 'r-', linewidth=1, label='Left')
    widget.axes.plot(frame_indices, right_angles, 'b-', linewidth=1, label='Right')
    
    # Set labels and legend
    widget.axes.set_title(f'{joint_name.capitalize()} Angle Over Time')
    widget.axes.set_xlabel('Frame')
    widget.axes.set_ylabel('Angle (degrees)')
    widget.axes.legend()
    
    # Update canvas
    widget.canvas.draw()