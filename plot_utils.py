import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from PyQt5.QtCore import Qt, QTimer
import pandas as pd

class KeypointPlot(FigureCanvasQTAgg):
    """Plot canvas for displaying keypoint trajectories"""
    
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        super().__init__()  # Add this line to call the parent class initializer
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
        
        # Add interactivity elements
        self.frame_callback = None
        self.hover_line = None
        self.hover_text = None
        self.click_enabled = True
        self.current_frame_marker = None
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('figure_leave_event', self.on_mouse_leave)
        
        # Set cursor to pointing hand to indicate clickable area
        self.setCursor(Qt.PointingHandCursor)
        
        # Add help annotation
        self.help_text = self.fig.text(0.5, 0.98, "Click on plot to navigate to frame", 
                                     ha='center', fontsize=8, color='gray', 
                                     bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        super(KeypointPlot, self).__init__(self.fig)
        self.setMinimumHeight(200)  # Set minimum height for the plot
    
    def on_plot_click(self, event):
        """Handle mouse clicks on the plot to select a frame"""
        if not self.click_enabled or event.inaxes not in [self.axes_x, self.axes_y] or not self.frame_callback:
            return
            
        # Get the frame index from the x-coordinate of the click
        frame_idx = int(round(event.xdata))
        
        # Provide visual feedback - highlight clicked position
        self.draw_vertical_indicator(frame_idx, color='blue', linestyle='-', alpha=0.8)
        self.draw()
        
        # Disable clicks temporarily to prevent rapid multiple clicks
        self.click_enabled = False
        
        # Call callback after a short delay to allow visual feedback
        QTimer.singleShot(100, lambda: self.frame_callback(frame_idx))
        
        # Re-enable clicks after a reasonable delay
        QTimer.singleShot(300, lambda: setattr(self, 'click_enabled', True))
    
    def on_mouse_move(self, event):
        """Show hover indicator when mouse moves over the plot"""
        if event.inaxes not in [self.axes_x, self.axes_y]:
            # Remove hover line when mouse leaves plot area
            if self.hover_line is not None:
                self.clear_hover_line()
                self.draw_idle()  # Use draw_idle() for better performance
            return
            
        # Get the frame index from the x-coordinate
        frame_idx = int(round(event.xdata))
        
        # Draw a vertical line at hover position
        self.draw_hover_indicator(frame_idx)
        self.draw_idle()  # Use draw_idle() for better performance
    
    def on_mouse_leave(self, event):
        """Clear hover indicator when mouse leaves the figure"""
        if self.hover_line is not None:
            self.clear_hover_line()
            self.draw_idle()
    
    def draw_hover_indicator(self, frame_idx):
        """Draw a vertical line indicating hover position"""
        self.clear_hover_line()
        
        # Draw dotted gray line at hover position
        self.hover_line = [
            self.axes_x.axvline(x=frame_idx, color='gray', linestyle=':', alpha=0.6),
            self.axes_y.axvline(x=frame_idx, color='gray', linestyle=':', alpha=0.6)
        ]
        
        # Add frame number tooltip
        ymin_x, ymax_x = self.axes_x.get_ylim()
        
        self.hover_text = [
            self.axes_x.text(frame_idx, ymax_x, f"Frame: {frame_idx}", 
                           fontsize=8, ha='center', va='top', alpha=0.8,
                           bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
        ]
    
    def clear_hover_line(self):
        """Remove the hover indicator line"""
        if self.hover_line is not None:
            for line in self.hover_line:
                line.set_visible(False)  # Use set_visible instead of remove
            self.hover_line = None
            
        if hasattr(self, 'hover_text') and self.hover_text is not None:
            for text in self.hover_text:
                text.set_visible(False)  # Use set_visible instead of remove
            self.hover_text = None
    
    def draw_vertical_indicator(self, frame_idx, color='red', linestyle='--', alpha=0.5):
        """Draw a vertical line at the given frame index"""
        # Clear any existing hover line
        if self.hover_line is not None:
            self.clear_hover_line()
            
        # Draw vertical line in both plots
        self.axes_x.axvline(x=frame_idx, color=color, linestyle=linestyle, alpha=alpha)
        self.axes_y.axvline(x=frame_idx, color=color, linestyle=linestyle, alpha=alpha)
    
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
                    x = row.iloc[keypoint_idx * 2]
                    y = row.iloc[keypoint_idx * 2 + 1]
                    if not np.isnan(x) and not np.isnan(y):
                        x_coords.append(x)
                        y_coords.append(y)
                        frames.append(i)
                except (IndexError, ValueError):
                    continue
                    
            if not frames:
                self.clear_plot()
                return
                
            # Plot trajectories with distinct styling
            self.axes_x.plot(frames, x_coords, 'b-', alpha=0.7, linewidth=1.2)
            self.axes_y.plot(frames, y_coords, 'g-', alpha=0.7, linewidth=1.2)
            
            # Add point for current frame
            if current_frame_idx in frames:
                idx = frames.index(current_frame_idx)
                self.axes_x.plot(current_frame_idx, x_coords[idx], 'ro', ms=6)
                self.axes_y.plot(current_frame_idx, y_coords[idx], 'ro', ms=6)
                
                # Add coordinate value text at current frame
                self.axes_x.text(current_frame_idx+2, x_coords[idx], f"{int(x_coords[idx])}", 
                               fontsize=8, color='blue', ha='left', va='center',
                               bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))
                self.axes_y.text(current_frame_idx+2, y_coords[idx], f"{int(y_coords[idx])}", 
                               fontsize=8, color='green', ha='left', va='center',
                               bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))
            
            # Set limits with focus on current frame
            padding = max(20, int(total_frames * 0.05))  # 5% padding or at least 20 frames
            self.axes_x.set_xlim(max(0, current_frame_idx - padding), 
                               min(total_frames, current_frame_idx + padding))
            self.axes_y.set_xlim(max(0, current_frame_idx - padding), 
                               min(total_frames, current_frame_idx + padding))
            
            # Add grid for better readability
            self.axes_x.grid(True, alpha=0.3, linestyle=':')
            self.axes_y.grid(True, alpha=0.3, linestyle=':')
            
            # Draw vertical line at current frame
            self.axes_x.axvline(x=current_frame_idx, color='r', linestyle='--', alpha=0.5)
            self.axes_y.axvline(x=current_frame_idx, color='r', linestyle='--', alpha=0.5)
            
            # Add keypoint name as title
            if keypoint_idx < len(pose_data.columns) // 2:
                keypoint_name = pose_data.columns[keypoint_idx * 2].replace('_X', '')
                self.axes_x.set_title(f"Trajectory for keypoint: {keypoint_name}", 
                                   fontsize=9, pad=2)
            
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