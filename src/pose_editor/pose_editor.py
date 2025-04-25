import sys
import cv2
import pandas as pd
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSlider,
                           QScrollArea, QGroupBox, QComboBox, QLineEdit, QShortcut,
                           QProgressDialog, QMessageBox)
from PyQt5.QtCore import Qt, QPoint, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon, QKeySequence
from .plot_utils import create_plot_widget, calculate_ankle_angle
from .mediapipe_utils import get_pose_landmarks_from_frame, process_video_with_mediapipe
from .pose_format_utils import process_mediapipe_to_rr21, load_pose_data, save_pose_data
from .keypoint_formats import SUPPORTED_FORMATS

def detect_pose_current_frame(self):
    """Detect pose on the current frame using MediaPipe"""
    if self.current_frame is None:
        QMessageBox.warning(self, "No Frame", "Please load a video first.")
        return
    
    try:
        # Process the current frame with the MediaPipe utilities
        landmarks_list, annotated_frame = get_pose_landmarks_from_frame(self.current_frame)
        
        if not landmarks_list:
            QMessageBox.warning(self, "No Pose Detected", "MediaPipe couldn't detect a pose in this frame.")
            return
            
        # Update the current frame to show annotations
        self.current_frame = annotated_frame
        self._needs_redraw = True
        
        # Convert MediaPipe landmarks to RR21 format
        rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
        
        # Scale normalized coordinates (0-1) to pixel coordinates
        frame_h, frame_w = self.current_frame.shape[:2]
        for i in range(0, len(rr21_landmarks), 2):
            # X coordinate (even indices)
            rr21_landmarks[i] = rr21_landmarks[i] * frame_w
            
            # Y coordinate (odd indices)
            if i + 1 < len(rr21_landmarks):
                rr21_landmarks[i + 1] = rr21_landmarks[i + 1] * frame_h
        
        # Store old pose data for undo functionality
        old_pose_data = None
        if self.pose_data is not None and self.current_frame_idx < len(self.pose_data):
            old_pose_data = self.pose_data.iloc[self.current_frame_idx].copy()
        
        # If no pose data exists yet, create a blank DataFrame
        if self.pose_data is None:
            # Create column names for RR21 format
            column_names = []
            for name in SUPPORTED_FORMATS["rr21"]:
                column_names.extend([f'{name}_X', f'{name}_Y'])
                
            # Create a DataFrame with empty values
            if hasattr(self, 'cap'):
                num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.pose_data = pd.DataFrame(np.zeros((num_frames, len(column_names))), columns=column_names)
            else:
                # If there's no video loaded somehow, just create a single row
                self.pose_data = pd.DataFrame([np.zeros(len(column_names))], columns=column_names)
                
            # Update keypoint names
            self.keypoint_names = SUPPORTED_FORMATS["rr21"]
            
            # Update keypoint dropdown
            self.keypoint_dropdown.blockSignals(True)
            self.keypoint_dropdown.clear()
            self.keypoint_dropdown.addItems(self.keypoint_names)
            self.keypoint_dropdown.blockSignals(False)
            
            # Set format information
            self.original_format = "mediapipe33"
            self.pose_format = "rr21"
        
        # Update pose data for current frame with RR21 landmarks
        for i in range(0, len(rr21_landmarks), 2):
            if i+1 < len(rr21_landmarks) and i//2 < len(self.pose_data.columns)//2:
                self.pose_data.iloc[self.current_frame_idx, i] = rr21_landmarks[i]
                self.pose_data.iloc[self.current_frame_idx, i+1] = rr21_landmarks[i+1]
        
        # Create and add command for undo/redo
        new_pose_data = self.pose_data.iloc[self.current_frame_idx].copy()
        command = MediaPipeDetectionCommand(self, self.current_frame_idx, old_pose_data, new_pose_data)
        self.add_command(command)
        
        # Update current pose
        self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
        
        # Update display
        self.display_frame()
        self.update_coordinate_inputs()
        self.update_plot()
        
        # Show success message
        QMessageBox.information(self, "Detection Complete", "Pose detected and converted to RR21 format successfully!")
        
    except Exception as e:
        QMessageBox.critical(self, "Error", f"An error occurred during pose detection: {str(e)}")

def detect_pose_video(self):
    """Detect pose on the entire video using MediaPipe"""
    if not hasattr(self, 'video_path') or not self.video_path:
        QMessageBox.warning(self, "No Video", "Please load a video first.")
        return
    
    try:
        # Create progress dialog
        progress = QProgressDialog("Processing video with MediaPipe...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Processing Video")
        progress.show()
        
        # Process the video using our utility function
        new_pose_data, success = process_video_with_mediapipe(self.video_path, progress)
        
        # Make sure to close the progress dialog when done
        progress.setValue(100)  # Set to 100% to ensure it closes
        progress.close()
        
        if not success:
            if progress.wasCanceled():
                QMessageBox.information(self, "Canceled", "Video processing was canceled.")
            else:
                QMessageBox.warning(self, "Processing Failed", "MediaPipe couldn't process the video.")
            return
        
        # Update pose data (it's already in RR21 format from our utility)
        self.pose_data = new_pose_data
        
        # Update keypoint names for RR21 format
        self.keypoint_names = SUPPORTED_FORMATS["rr21"]
        
        # Update keypoint dropdown
        self.keypoint_dropdown.blockSignals(True)
        self.keypoint_dropdown.clear()
        self.keypoint_dropdown.addItems(self.keypoint_names)
        self.keypoint_dropdown.blockSignals(False)
        
        # Set format information
        self.original_format = "mediapipe33"
        self.pose_format = "rr21"
        
        # Update current pose
        if self.current_frame_idx < len(self.pose_data):
            self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
        
        # Update display
        self._needs_redraw = True  # Force redraw
        self.display_frame()
        self.update_coordinate_inputs()
        self.update_plot()
        
        # Show success message
        QMessageBox.information(self, "Detection Complete", "Video processed successfully with poses in RR21 format!")
        
    except Exception as e:
        # Close progress dialog if it's still open during an exception
        if 'progress' in locals() and progress is not None:
            progress.close()
        QMessageBox.critical(self, "Error", f"An error occurred during video processing: {str(e)}")