import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import pandas as pd

class KeypointPlot(FigureCanvasQTAgg):
    """Plot canvas for displaying keypoint trajectories"""
    
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.tight_layout()
        self.axes_x = self.fig.add_subplot(211)  # X coordinate plot
        self.axes_y = self.fig.add_subplot(212)  # Y coordinate plot
        
        # Set up plots
        self.axes_x.set_ylabel('X Position')
        self.axes_y.set_ylabel('Y Position')
        self.axes_y.set_xlabel('Frame')
        
        # Hide x labels on top plot
        self.axes_x.set_xticklabels([])
        
        super(KeypointPlot, self).__init__(self.fig)
        self.setMinimumHeight(200)  # Set minimum height for the plot

    def plot_keypoint_trajectory(self, pose_data, keypoint_idx, current_frame_idx, total_frames):
        """
        Plot the trajectory of a specific keypoint across all frames
        
        Args:
            pose_data: DataFrame containing pose data
            keypoint_idx: Index of the keypoint to plot
            current_frame_idx: Current frame index to highlight
            total_frames: Total number of frames
        """
        if pose_data is None or keypoint_idx is None:
            self.clear_plot()
            return
            
        try:
            # Clear previous plots
            self.axes_x.clear()
            self.axes_y.clear()
            
            # Set up axes
            self.axes_x.set_ylabel('X Position')
            self.axes_y.set_ylabel('Y Position')
            self.axes_y.set_xlabel('Frame')
            
            # Hide x labels on top plot
            self.axes_x.set_xticklabels([])
            
            # Extract x and y coordinates for the selected keypoint across all frames
            x_coords = []
            y_coords = []
            frames = []
            
            for i in range(len(pose_data)):
                row = pose_data.iloc[i]
                try:
                    x = row[keypoint_idx * 2]
                    y = row[keypoint_idx * 2 + 1]
                    if not np.isnan(x) and not np.isnan(y):
                        x_coords.append(x)
                        y_coords.append(y)
                        frames.append(i)
                except (IndexError, ValueError):
                    continue
                    
            if not frames:
                self.clear_plot()
                return
                
            # Plot trajectories
            self.axes_x.plot(frames, x_coords, 'b-', alpha=0.7)
            self.axes_y.plot(frames, y_coords, 'g-', alpha=0.7)
            
            # Add point for current frame
            if current_frame_idx in frames:
                idx = frames.index(current_frame_idx)
                self.axes_x.plot(current_frame_idx, x_coords[idx], 'ro', ms=6)
                self.axes_y.plot(current_frame_idx, y_coords[idx], 'ro', ms=6)
            
            # Set limits
            padding = max(20, int(total_frames * 0.05))  # 5% padding or at least 20 frames
            self.axes_x.set_xlim(max(0, current_frame_idx - padding), 
                               min(total_frames, current_frame_idx + padding))
            self.axes_y.set_xlim(max(0, current_frame_idx - padding), 
                               min(total_frames, current_frame_idx + padding))
            
            # Draw vertical line at current frame
            self.axes_x.axvline(x=current_frame_idx, color='r', linestyle='--', alpha=0.5)
            self.axes_y.axvline(x=current_frame_idx, color='r', linestyle='--', alpha=0.5)
            
            self.fig.tight_layout()
            self.draw()
            
        except Exception as e:
            print(f"Error plotting keypoint trajectory: {e}")
            self.clear_plot()
            
    def clear_plot(self):
        """Clear the plot"""
        self.axes_x.clear()
        self.axes_y.clear()
        self.axes_x.set_ylabel('X Position')
        self.axes_y.set_ylabel('Y Position')
        self.axes_y.set_xlabel('Frame')
        self.axes_x.set_xticklabels([])
        self.fig.tight_layout()
        self.draw()


def create_plot_widget():
    """Create a widget containing the keypoint plot"""
    plot_widget = QWidget()
    plot_layout = QVBoxLayout(plot_widget)
    plot = KeypointPlot()
    plot_layout.addWidget(plot)
    plot_layout.setContentsMargins(0, 0, 0, 0)
    
    return plot_widget, plot