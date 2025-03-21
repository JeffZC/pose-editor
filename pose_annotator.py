import sys
import cv2
import pandas as pd
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSlider,
                           QScrollArea, QGroupBox, QComboBox, QLineEdit)
from PyQt5.QtCore import Qt, QPoint, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon, QKeySequence
from plot_utils import create_plot_widget
from mediapipe_utils import get_pose_landmarks_from_frame, process_video_with_mediapipe
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
from pose_format_utils import load_pose_data, save_pose_data, SUPPORTED_FORMATS, process_mediapipe_to_rr21

# Command class for undo/redo operations
class KeypointCommand:
    def __init__(self, editor, frame_idx, point_idx, old_x, old_y, new_x, new_y):
        self.editor = editor
        self.frame_idx = frame_idx
        self.point_idx = point_idx
        self.old_x = old_x
        self.old_y = old_y
        self.new_x = new_x
        self.new_y = new_y
        
    def undo(self):
        """Restore the previous state"""
        # Check if we need to change frames
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        # Check if we need to change selected point
        if self.point_idx != self.editor.selected_point:
            self.editor.selected_point = self.point_idx
            self.editor.keypoint_dropdown.blockSignals(True)
            self.editor.keypoint_dropdown.setCurrentIndex(self.point_idx)
            self.editor.keypoint_dropdown.blockSignals(False)
        
        # Restore old coordinates
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2] = self.old_x
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2 + 1] = self.old_y
        
        # Update current pose
        if self.editor.current_pose is not None:
            self.editor.current_pose[self.point_idx] = [self.old_x, self.old_y]
        
        # Update UI
        self.editor._needs_redraw = True
        self.editor.update_coordinate_inputs()
        self.editor.display_frame()
        self.editor.update_plot()
        
    def redo(self):
        """Apply the change again"""
        # Check if we need to change frames
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        # Check if we need to change selected point
        if self.point_idx != self.editor.selected_point:
            self.editor.selected_point = self.point_idx
            self.editor.keypoint_dropdown.blockSignals(True)
            self.editor.keypoint_dropdown.setCurrentIndex(self.point_idx)
            self.editor.keypoint_dropdown.blockSignals(False)
        
        # Apply new coordinates
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2] = self.new_x
        self.editor.pose_data.iloc[self.frame_idx, self.point_idx * 2 + 1] = self.new_y
        
        # Update current pose
        if self.editor.current_pose is not None:
            self.editor.current_pose[self.point_idx] = [self.new_x, self.new_y]
        
        # Update UI
        self.editor._needs_redraw = True
        self.editor.update_coordinate_inputs()
        self.editor.display_frame()
        self.editor.update_plot()

class PoseEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Editor")
        self.setGeometry(100, 100, 1024, 768)

        # Initialize variables
        self.video_path = None
        self.pose_data = None
        self.current_frame = None
        self.current_pose = None
        self.selected_point = None
        self.zoom_level = 1.0
        self.current_frame_idx = 0
        self.max_zoom_level = 5.0
        self.zoom_center = QPoint(0, 0)
        self.dragging = False
        self.black_and_white = False  # Initialize black and white mode
        self.playing = False
        self.play_speed = 30  # frames per second
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.advance_frame)
        self.rotation_angle = 0  # Initialize rotation angle (0, 90, 180, 270)
        self.play_button_size = 35  # Size in pixels for play button
        self.play_icon_size = 30    # Size in pixels for play icon
        
        # Initialize command history for undo/redo operations
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 50

        # Add keypoint names (default, to be updated based on pose data)
        self.keypoint_names = []

        self.initUI()
    
    def initUI(self):
        # Create main container with horizontal layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
    
        # Create left panel for video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
    
        # Create file controls
        self.file_controls = QHBoxLayout()
        left_layout.addLayout(self.file_controls)
    
        # Create scroll area for video
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
        # Create video container and label
        self.video_container = QWidget()
        self.video_layout = QVBoxLayout(self.video_container)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.installEventFilter(self)
        self.label.setMinimumSize(400, 300)
        self.video_layout.addWidget(self.label)
    
        # Add video container to scroll area
        self.scroll_area.setWidget(self.video_container)
        left_layout.addWidget(self.scroll_area)
    
        # Add plot widget
        self.plot_widget, self.keypoint_plot = create_plot_widget()
        # Connect plot click callback
        self.keypoint_plot.frame_callback = self.set_frame_from_plot
        left_layout.addWidget(self.plot_widget)
    
        # Create navigation controls with play/pause
        self.nav_controls = QHBoxLayout()
        
        # Add play button (will toggle between play/pause)
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.setIconSize(QSize(self.play_icon_size, self.play_icon_size))
        self.play_button.setFixedSize(self.play_button_size, self.play_button_size)
        self.play_button.setStyleSheet("padding: 0px;")  # Remove padding to maximize icon space
        self.play_button.clicked.connect(self.toggle_playback)
        
        self.prev_frame_button = QPushButton("←")
        self.prev_frame_button.setFixedWidth(25)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        self.frame_slider.sliderPressed.connect(self.on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_slider_released)
        
        self.next_frame_button = QPushButton("→")
        self.next_frame_button.setFixedWidth(25)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.frame_counter = QLabel("Frame: 0/0")
    
        self.nav_controls.addWidget(self.play_button)
        self.nav_controls.addWidget(self.prev_frame_button)
        self.nav_controls.addWidget(self.frame_slider)
        self.nav_controls.addWidget(self.next_frame_button)
        self.nav_controls.addWidget(self.frame_counter)
        left_layout.addLayout(self.nav_controls)
    
        # Create right panel for zoom controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
    
        # Create video controls group box
        self.video_control_group_box = QGroupBox("Video Controls")
        self.video_control_layout = QVBoxLayout()

        # Add Load Video button
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self.load_video)
        self.video_control_layout.addWidget(self.load_video_button)

        # Add the rotate button
        self.rotate_button = QPushButton("Rotate View (90°)")
        self.rotate_button.clicked.connect(self.rotate_video)
        self.video_control_layout.addWidget(self.rotate_button)

        # Add black and white toggle button
        self.bw_button = QPushButton("Toggle Black and White")
        self.bw_button.clicked.connect(self.toggle_black_and_white)
        self.video_control_layout.addWidget(self.bw_button)

        # Create zoom controls in a horizontal layout
        self.zoom_controls = QHBoxLayout()
        self.zoom_out_button = QPushButton("Zoom Out (-)")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button = QPushButton("Zoom In (+)")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_controls.addWidget(self.zoom_out_button)
        self.zoom_controls.addWidget(self.zoom_in_button)

        # Add the zoom controls horizontal layout to the vertical layout
        self.video_control_layout.addLayout(self.zoom_controls)
        self.zoom_label = QLabel("Zoom: 100%")
        self.video_control_layout.addWidget(self.zoom_label)
        self.video_control_group_box.setLayout(self.video_control_layout)
        right_layout.addWidget(self.video_control_group_box)
    
        # Create keypoint selection group box
        self.keypoint_ops_group_box = QGroupBox("Keypoint Operations")
        self.keypoint_ops_layout = QVBoxLayout()

        # Add keypoint selection with label
        self.keypoint_selection_layout = QHBoxLayout()
        self.keypoint_selection_label = QLabel("Select Keypoint:")
        self.keypoint_dropdown = QComboBox()
        self.keypoint_dropdown.addItems(self.keypoint_names)
        self.keypoint_dropdown.currentIndexChanged.connect(self.on_keypoint_selected)
        self.keypoint_selection_layout.addWidget(self.keypoint_selection_label)
        self.keypoint_selection_layout.addWidget(self.keypoint_dropdown)
        self.keypoint_ops_layout.addLayout(self.keypoint_selection_layout)

        # Add coordinate inputs with labels
        self.x_coord_layout = QHBoxLayout()
        self.x_coord_label = QLabel("X Coordinate:")
        self.x_coord_input = QLineEdit()
        self.x_coord_layout.addWidget(self.x_coord_label)
        self.x_coord_layout.addWidget(self.x_coord_input)
        self.keypoint_ops_layout.addLayout(self.x_coord_layout)

        self.y_coord_layout = QHBoxLayout()
        self.y_coord_label = QLabel("Y Coordinate:")
        self.y_coord_input = QLineEdit()
        self.y_coord_layout.addWidget(self.y_coord_label)
        self.y_coord_layout.addWidget(self.y_coord_input)
        self.keypoint_ops_layout.addLayout(self.y_coord_layout)

        # Add confirm button
        self.confirm_button = QPushButton("Update Coordinates")
        self.confirm_button.clicked.connect(self.update_keypoint_coordinates)
        self.keypoint_ops_layout.addWidget(self.confirm_button)

        # Set the layout and add to right panel
        self.keypoint_ops_group_box.setLayout(self.keypoint_ops_layout)
        right_layout.addWidget(self.keypoint_ops_group_box)
    
        # Create undo/redo group box (new)
        self.history_group_box = QGroupBox("Edit History")
        self.history_layout = QHBoxLayout()  # Horizontal layout for buttons side by side
        self.undo_button = QPushButton("Undo")
        self.undo_button.setShortcut(QKeySequence("Ctrl+Z"))
        self.undo_button.clicked.connect(self.undo_last_command)
        self.undo_button.setEnabled(False)  # Disabled by default
        self.redo_button = QPushButton("Redo")
        self.redo_button.setShortcut(QKeySequence("Ctrl+Y"))
        self.redo_button.clicked.connect(self.redo_last_command)
        self.redo_button.setEnabled(False)  # Disabled by default
        self.history_layout.addWidget(self.undo_button)
        self.history_layout.addWidget(self.redo_button)
        self.history_group_box.setLayout(self.history_layout)
        right_layout.addWidget(self.history_group_box)
        
        # Create Pose Options group box (renamed from MediaPipe)
        self.pose_options_group_box = QGroupBox("Pose Options")
        self.pose_options_layout = QVBoxLayout()

        # Add Load Pose button at the top
        self.load_pose_button = QPushButton("Load Pose (from csv)")
        self.load_pose_button.clicked.connect(self.load_pose)
        self.pose_options_layout.addWidget(self.load_pose_button)

        # Add pose detection buttons
        self.detect_current_frame_button = QPushButton("Run Pose Current Frame")
        self.detect_current_frame_button.clicked.connect(self.detect_pose_current_frame)
        self.detect_video_button = QPushButton("Run Pose Entire Video")
        self.detect_video_button.clicked.connect(self.detect_pose_video)

        # Add widgets to layout
        self.pose_options_layout.addWidget(self.detect_current_frame_button)
        self.pose_options_layout.addWidget(self.detect_video_button)

        # Add Save Pose button at the bottom
        self.save_button = QPushButton("Save Poses (to csv)")
        self.save_button.clicked.connect(self.save_pose)
        self.pose_options_layout.addWidget(self.save_button)

        self.pose_options_group_box.setLayout(self.pose_options_layout)
        right_layout.addWidget(self.pose_options_group_box)
                
        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=4)
        main_layout.addWidget(right_panel, stretch=1)
    
        self.setMinimumSize(800, 600)
        self.show()

    def toggle_black_and_white(self):
        self.black_and_white = not self.black_and_white
        self.display_frame()

    def display_frame(self):
        if self.current_frame is not None:
            # Use cached pixmap when possible
            if not hasattr(self, '_cached_frame') or self._cached_frame_idx != self.current_frame_idx or self._needs_redraw:
                # Only do the minimum needed for visual feedback
                frame = self.current_frame.copy()
                
                if self.black_and_white:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    
                if self.pose_data is not None:
                    self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
                    for i, point in enumerate(self.current_pose):
                        radius = 8 if i == self.selected_point else 5
                        color = (255, 0, 0) if i == self.selected_point else (0, 255, 0)
                        cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)

                # Apply rotation if needed
                if self.rotation_angle > 0:
                    h, w = frame.shape[:2]
                    center = (w // 2, h // 2)
                    
                    # Get rotation matrix
                    if self.rotation_angle == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif self.rotation_angle == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif self.rotation_angle == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Convert to QPixmap only when necessary
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self._base_pixmap = QPixmap.fromImage(q_img)
                self._cached_frame_idx = self.current_frame_idx
                self._needs_redraw = False
                
            # Apply zoom to cached base pixmap
            scaled_width = int(self._base_pixmap.width() * self.zoom_level)
            scaled_height = int(self._base_pixmap.height() * self.zoom_level)

            # Use faster transformation when dragging
            transformation = Qt.FastTransformation if self.dragging else Qt.SmoothTransformation
            scaled_pixmap = self._base_pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                transformation
            )

            self.label.setPixmap(scaled_pixmap)
            self.label.setFixedSize(scaled_width, scaled_height)

    def update_coordinate_inputs(self):
        if self.selected_point is not None and self.current_pose is not None:
            if 0 <= self.selected_point < len(self.current_pose):
                point = self.current_pose[self.selected_point]
                self.x_coord_input.setText(str(int(point[0])))
                self.y_coord_input.setText(str(int(point[1])))
        else:
            self.x_coord_input.clear()
            self.y_coord_input.clear()

    def load_pose(self):
        pose_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open Pose Data", 
            "", 
            "Pose Files (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if not pose_path:
            return
            
        # Get expected frame count if video is loaded
        expected_frame_count = None
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            expected_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # First try to load without forcing
        pose_data, format_name, keypoint_names, success, message = load_pose_data(
            pose_path, 
            expected_frame_count=expected_frame_count,
            force_import=False
        )
        
        # If frame count mismatch, ask user if they want to force import
        if not success and "Frame count mismatch" in message and expected_frame_count is not None:
            reply = QMessageBox.question(
                self, 
                "Frame Count Mismatch", 
                f"{message}\n\nDo you want to force import? If yes, the pose data will be adjusted to match the video frame count.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Try again with force import
                pose_data, format_name, keypoint_names, success, message = load_pose_data(
                    pose_path, 
                    expected_frame_count=expected_frame_count,
                    force_import=True
                )
        
        # Handle other errors
        if not success:
            QMessageBox.warning(self, "Load Failed", message)
            return
        
        # Clean NaN values - replace them with 0
        pose_data = pose_data.fillna(0)
        
        # Update pose data
        self.pose_data = pose_data
        self.pose_format = "rr21"  # Always use RR21 internally
        self.original_format = format_name
        
        # Update keypoint names
        self.keypoint_names = keypoint_names
        
        # Update keypoint dropdown
        self.keypoint_dropdown.blockSignals(True)
        self.keypoint_dropdown.clear()
        self.keypoint_dropdown.addItems(self.keypoint_names)
        self.keypoint_dropdown.blockSignals(False)
        
        # Reset selection
        self.selected_point = None
        
        # Update display
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.update_frame()
            
        # Show information about loaded data
        QMessageBox.information(
            self, 
            "Pose Data Loaded", 
            f"Loaded {len(self.pose_data)} frames of {format_name} format pose data (converted to RR21) with {len(self.keypoint_names)} keypoints."
        )

    def update_frame(self):
        if hasattr(self, 'cap') and self.cap:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                # Update the current frame
                self.current_frame = frame
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
                
                # Mark for redraw and force cache update
                self._needs_redraw = True
                
                # Update the display
                self.display_frame()
                self.update_coordinate_inputs()
                
                # Only update plot if not in the middle of a playback
                if not self.playing:
                    self.update_plot()
            else:
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")

    def load_video(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(total_frames - 1)
            self.current_frame_idx = 0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width <= 0 or height <= 0:
                self.cap.release()
                self.cap = None
                return
                
            # Calculate aspect ratio and set fixed size for label
            aspect_ratio = width / height
            label_height = 480  # Fixed height
            label_width = int(label_height * aspect_ratio)
            self.label.setFixedSize(label_width, label_height)
            
            # Update play button to show correct icon
            self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
            self.playing = False
            
            # Set smooth stepping for the slider based on total frames
            self.frame_slider.setPageStep(max(1, total_frames // 100))
            
            # Reset cached frame data
            if hasattr(self, '_cached_frame_idx'):
                delattr(self, '_cached_frame_idx')
            self._needs_redraw = True
            
            # Get actual video FPS
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.play_speed = fps
            
            self.update_frame()
    
    
    def zoom_in(self):
        if self.zoom_level < self.max_zoom_level:
            # Get cursor position relative to label
            cursor_pos = self.label.mapFromGlobal(QCursor.pos())
            
            # Calculate zoom
            old_zoom = self.zoom_level
            self.zoom_level = min(self.max_zoom_level, round(self.zoom_level * 1.1, 1))
            
            if self.zoom_level != old_zoom:
                # Update zoom center
                self.zoom_center = cursor_pos
                self.zoom_label.setText(f"Zoom: {int(self.zoom_level * 100)}%")
                self.display_frame()
                
                # Adjust scroll position to keep zoom center
                self.adjust_scroll_position(old_zoom)
    
    def zoom_out(self):
        if self.zoom_level > 0.1:
            # Get cursor position relative to label
            cursor_pos = self.label.mapFromGlobal(QCursor.pos())
            
            # Calculate zoom
            old_zoom = self.zoom_level
            self.zoom_level = max(0.1, round(self.zoom_level / 1.1, 1))
            
            if self.zoom_level != old_zoom:
                # Update zoom center
                self.zoom_center = cursor_pos
                self.zoom_label.setText(f"Zoom: {int(self.zoom_level * 100)}%")
                self.display_frame()
                
                # Adjust scroll position to keep zoom center
                self.adjust_scroll_position(old_zoom)
    
    def adjust_scroll_position(self, old_zoom):
        # Calculate new scroll position to maintain zoom center
        scroll_x = self.scroll_area.horizontalScrollBar().value()
        scroll_y = self.scroll_area.verticalScrollBar().value()
        
        # Calculate new positions based on zoom center
        factor = self.zoom_level / old_zoom
        new_x = int(factor * scroll_x + self.zoom_center.x() * (factor - 1))
        new_y = int(factor * scroll_y + self.zoom_center.y() * (factor - 1))
        
        # Set new scroll positions
        self.scroll_area.horizontalScrollBar().setValue(new_x)
        self.scroll_area.verticalScrollBar().setValue(new_y)
    
    def eventFilter(self, source, event):
        if source is self.label:
            if event.type() == event.MouseButtonPress:
                pos = event.pos()
                scaled_pos = QPoint(int(pos.x() / self.zoom_level), 
                                   int(pos.y() / self.zoom_level))
                new_selected_point = self.get_selected_point(scaled_pos)
                
                if event.button() == Qt.LeftButton:
                    if new_selected_point is not None:
                        self.selected_point = new_selected_point
                        # Synchronize dropdown
                        self.keypoint_dropdown.blockSignals(True)
                        self.keypoint_dropdown.setCurrentIndex(self.selected_point)
                        self.keypoint_dropdown.blockSignals(False)
                        self.dragging = True
                        self.update_coordinate_inputs()
                        self._needs_redraw = True
                        self.display_frame()
                        return True
                elif event.button() == Qt.RightButton:
                    self.selected_point = None
                    self.dragging = False
                    # Reset dropdown
                    self.keypoint_dropdown.blockSignals(True)
                    self.keypoint_dropdown.setCurrentIndex(-1)
                    self.keypoint_dropdown.blockSignals(False)
                    self.update_coordinate_inputs()
                    self._needs_redraw = True
                    self.display_frame()
                    # Only update plot when not dragging
                    self.update_plot()
                    return True
                    
            elif event.type() == event.MouseMove and self.dragging and self.selected_point is not None:
                pos = event.pos()
                scaled_pos = QPoint(int(pos.x() / self.zoom_level), 
                                   int(pos.y() / self.zoom_level))
                # Update point position
                self.move_point(scaled_pos)
                # Update display without updating plot for performance
                self.display_frame()
                return True
                
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.dragging:
                    self.dragging = False
                    
                    # Create command when drag completes
                    if hasattr(self, '_drag_start_pos'):
                        start_x, start_y = self._drag_start_pos
                        current_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
                        current_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
                        
                        # Only add command if position actually changed
                        if abs(start_x - current_x) > 0 or abs(start_y - current_y) > 0:
                            self.create_move_command(
                                self.selected_point,
                                start_x, start_y,
                                current_x, current_y
                            )
                        
                        delattr(self, '_drag_start_pos')
                    
                    # Only update plot when done dragging
                    self.update_plot()
                    return True
        
        return super().eventFilter(source, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def get_selected_point(self, pos):
        if self.current_pose is not None:
            # Transform mouse coordinates based on rotation
            transformed_pos = self.transform_coordinates(pos)
            for i, point in enumerate(self.current_pose):
                # Scale the detection radius with zoom level
                detect_radius = 10 / self.zoom_level
                if np.linalg.norm(np.array([point[0], point[1]]) - 
                                np.array([transformed_pos.x(), transformed_pos.y()])) < detect_radius:
                    return i
        return None

    def save_pose(self):
        if self.pose_data is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Pose Data", 
                "", 
                "CSV Files (*.csv);;JSON Files (*.json)"
            )
            if file_path:
                if save_pose_data(self.pose_data, file_path, "rr21"):
                    QMessageBox.information(
                        self, 
                        "Save Successful", 
                        "Pose data saved successfully in RR21 format."
                    )
                else:
                    QMessageBox.warning(self, "Save Failed", "Failed to save pose data.")

    def on_frame_change(self, value):
        # Always update the frame counter text
        self.current_frame_idx = value
        self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
        
        # If we're dragging the slider, provide visual feedback but with lighter processing
        if self.frame_slider.isSliderDown():
            # Update frame with lightweight preview (skip plot updates during dragging)
            self.preview_frame_at_position(value)
        else:
            # Full update when slider is released or changed via buttons
            self.update_frame()
            self.update_plot()

    def preview_frame_at_position(self, frame_idx):
        """Provides a fast preview while dragging the slider"""
        if hasattr(self, 'cap') and self.cap:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                # Store current frame but with reduced processing
                self.current_frame = frame
                
                # Use simplified frame display for better performance during dragging
                frame_preview = frame.copy()
                
                # Convert to black and white if needed (this is fast)
                if self.black_and_white:
                    frame_preview = cv2.cvtColor(frame_preview, cv2.COLOR_BGR2GRAY)
                    frame_preview = cv2.cvtColor(frame_preview, cv2.COLOR_GRAY2RGB)
                
                # Add keypoints with simplified rendering
                if self.pose_data is not None:
                    # Update current pose data for this frame
                    self.current_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                    
                    # Draw simplified keypoints (faster)
                    for i, point in enumerate(self.current_pose):
                        # Use uniform color and size during dragging for speed
                        cv2.circle(frame_preview, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                
                # Convert to QPixmap with fast transformation
                frame_rgb = cv2.cvtColor(frame_preview, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Scale with fast transformation
                scaled_width = int(pixmap.width() * self.zoom_level)
                scaled_height = int(pixmap.height() * self.zoom_level)
                scaled_pixmap = pixmap.scaled(
                    scaled_width, 
                    scaled_height,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation  # Always use fast transformation during dragging
                )
                
                # Update display
                self.label.setPixmap(scaled_pixmap)
                self.label.setFixedSize(scaled_width, scaled_height)
                
                # Update coordinates display
                self.update_coordinate_inputs()

    def on_slider_pressed(self):
        # Pause playback if we're scrubbing with the slider
        if self.playing:
            self.pause_playback()
            self._was_playing = True
        else:
            self._was_playing = False
        
        # Store starting frame for performance optimization
        self._slider_start_frame = self.current_frame_idx

    def on_slider_released(self):
        # Resume playback if it was playing before scrubbing
        if getattr(self, '_was_playing', False):
            self.start_playback()
        
        # Force a full update if the frame actually changed
        if self._slider_start_frame != self.current_frame_idx:
            self._needs_redraw = True  # Force redraw after slider release
            self.update_frame()  # Full update with proper rendering
            self.update_plot()   # Update the plot now that we've settled on a frame

    def next_frame(self):
        if hasattr(self, 'cap') and self.cap and self.current_frame_idx < self.frame_slider.maximum():
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)
    
    def prev_frame(self):
        if hasattr(self, 'cap') and self.cap and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.setValue(self.current_frame_idx)
    
    def on_keypoint_selected(self, index):
        if 0 <= index < len(self.keypoint_names):
            self.selected_point = index
            self.update_coordinate_inputs()
            self.update_plot()  # Add this line
            self.display_frame()
        else:
            self.selected_point = None
            self.update_coordinate_inputs()
            self.update_plot()  # Add this line
            self.display_frame()
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # Note: We'll handle this in the eventFilter instead to avoid duplicate handling
        pass

    def move_point(self, pos):
        if self.selected_point is not None and self.pose_data is not None:
            # Transform mouse coordinates based on rotation
            transformed_pos = self.transform_coordinates(pos)
            x, y = transformed_pos.x(), transformed_pos.y()
            
            # Quick bounds check
            if x < 0 or y < 0:
                return
                
            # Skip update if position hasn't changed significantly
            current_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
            current_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
            
            # Only update if position changed by at least 1 pixel to avoid unnecessary redraws
            if abs(x - current_x) < 1 and abs(y - current_y) < 1:
                return
            
            # Store initial position for undo when first starting to drag
            if not hasattr(self, '_drag_start_pos'):
                self._drag_start_pos = (current_x, current_y)
            
            # Update data
            self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2] = x
            self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1] = y
            self.current_pose[self.selected_point] = [x, y]
            
            # Update coordinate inputs
            self.x_coord_input.setText(str(int(x)))
            self.y_coord_input.setText(str(int(y)))
            
            # Mark for redraw
            self._needs_redraw = True

    def update_keypoint_coordinates(self):
        if self.selected_point is not None and self.pose_data is not None:
            try:
                new_x = int(self.x_coord_input.text())
                new_y = int(self.y_coord_input.text())
            except ValueError:
                return
            
            # Validate coordinates
            if new_x < 0 or new_y < 0:
                return
            
            # Get current position
            old_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
            old_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
            
            # Only create command if position changed
            if old_x != new_x or old_y != new_y:
                # Create command for this change
                self.create_move_command(
                    self.selected_point,
                    old_x, old_y,
                    new_x, new_y
                )
                
                # Update the pose data
                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2] = new_x
                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1] = new_y
                
                # Update current_pose to reflect the changes
                if self.current_pose is not None and self.selected_point < len(self.current_pose):
                    self.current_pose[self.selected_point] = [new_x, new_y]
                    
                # Display the updated frame
                self.display_frame()
                
                # Update the plot
                self.update_plot()
            
    def update_plot(self):
        # Only update plot if we have data and aren't dragging (for responsiveness)
        if self.dragging:
            return
            
        if hasattr(self, 'keypoint_plot') and self.pose_data is not None and self.selected_point is not None:
            total_frames = len(self.pose_data)
            self.keypoint_plot.plot_keypoint_trajectory(
                self.pose_data, 
                self.selected_point, 
                self.current_frame_idx, 
                total_frames
            )
        elif hasattr(self, 'keypoint_plot'):
            self.keypoint_plot.clear_plot()
    
    def mouseMoveEvent(self, event):
        # Note: We'll handle this in the eventFilter instead to avoid duplicate handling
        pass

    def toggle_playback(self):
        if not hasattr(self, 'cap') or not self.cap:
            return
            
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        self.playing = True
        # Use a custom icon or text that clearly indicates the pause state
        self.play_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        
        # Calculate frame interval based on video FPS if available
        if hasattr(self, 'cap') and self.cap:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.play_speed = fps
        
        frame_interval = int(1000 / self.play_speed)  # Convert FPS to milliseconds
        self.play_timer.start(frame_interval)
        
        # Disable manual frame controls during playback
        self.frame_slider.setEnabled(False)
        self.prev_frame_button.setEnabled(False)
        self.next_frame_button.setEnabled(False)

    def pause_playback(self):
        self.playing = False
        # Use a custom icon or text that clearly indicates the play state
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_timer.stop()
        
        # Re-enable manual frame controls
        self.frame_slider.setEnabled(True)
        self.prev_frame_button.setEnabled(True)
        self.next_frame_button.setEnabled(True)
        
        # Update plot once playback stops
        self.update_plot()
    
    def advance_frame(self):
        if self.current_frame_idx >= self.frame_slider.maximum():
            self.pause_playback()  # Stop playback at end of video
            return
            
        self.current_frame_idx += 1
        
        # Update slider without triggering on_frame_change
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)
        
        # Update frame but skip plot update during playback
        self.update_frame()
    
    def closeEvent(self, event):
        if hasattr(self, 'play_timer') and self.play_timer.isActive():
            self.play_timer.stop()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        super().closeEvent(event)

    def add_command(self, command):
        """Add a command to the history and execute it"""
        self.undo_stack.append(command)
        # Clear the redo stack when a new command is added
        self.redo_stack = []
        self.redo_button.setEnabled(False)
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        
        self.undo_button.setEnabled(True)

    def undo_last_command(self):
        """Undo the most recent command"""
        if self.undo_stack:
            command = self.undo_stack.pop()
            self.redo_stack.append(command)
            command.undo()
            
            # Update button states
            self.redo_button.setEnabled(True)
            self.undo_button.setEnabled(len(self.undo_stack) > 0)

    def redo_last_command(self):
        """Redo the most recently undone command"""
        if self.redo_stack:
            command = self.redo_stack.pop()
            self.undo_stack.append(command)
            command.redo()
            
            # Update button states
            self.undo_button.setEnabled(True)
            self.redo_button.setEnabled(len(self.redo_stack) > 0)

    def create_move_command(self, point_idx, old_x, old_y, new_x, new_y):
        """Create and register a move command"""
        # Only create a command if something actually changed
        if old_x != new_x or old_y != new_y:
            command = KeypointCommand(
                self, 
                self.current_frame_idx,
                point_idx, 
                old_x, old_y, 
                new_x, new_y
            )
            self.add_command(command)

    def detect_pose_current_frame(self):
        """Detect pose on the current frame using MediaPipe"""
        if self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "Please load a video first.")
            return
        
        try:
            # Process the current frame
            landmarks_list, annotated_frame = get_pose_landmarks_from_frame(self.current_frame)
            
            if not landmarks_list:
                QMessageBox.warning(self, "No Pose Detected", "MediaPipe couldn't detect a pose in this frame.")
                return
                
            # Update the current frame to show annotations
            self.current_frame = annotated_frame
            self._needs_redraw = True
            
            # Convert MediaPipe landmarks to RR21 format
            rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
            
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
            
            # Process the video
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
            
            # Update pose data
            self.pose_data = new_pose_data
            
            # Update keypoint names
            self.keypoint_names = []
            columns = self.pose_data.columns
            for i in range(0, len(columns), 2):
                if i+1 < len(columns):
                    name = columns[i].replace('_X', '')
                    self.keypoint_names.append(name)
            
            # Update keypoint dropdown
            self.keypoint_dropdown.blockSignals(True)
            self.keypoint_dropdown.clear()
            self.keypoint_dropdown.addItems(self.keypoint_names)
            self.keypoint_dropdown.blockSignals(False)
            
            # Update current pose
            if self.current_frame_idx < len(self.pose_data):
                self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            
            # Update display
            self._needs_redraw = True  # Force redraw
            self.display_frame()
            self.update_coordinate_inputs()
            self.update_plot()
            
            # Show success message
            QMessageBox.information(self, "Detection Complete", "Video processed successfully!")
            
        except Exception as e:
            # Close progress dialog if it's still open during an exception
            if 'progress' in locals() and progress is not None:
                progress.close()
            QMessageBox.critical(self, "Error", f"An error occurred during video processing: {str(e)}")

    def rotate_video(self):
        """Rotate the video display by 90 degrees clockwise"""
        # Update rotation angle (0 -> 90 -> 180 -> 270 -> 0)
        self.rotation_angle = (self.rotation_angle + 90) % 360
        
        # Force redraw of the frame with rotation
        self._needs_redraw = True
        self.display_frame()

    def transform_coordinates(self, pos):
        """Transform coordinates based on rotation angle"""
        # If no rotation, return original position
        if self.rotation_angle == 0:
            return pos
        
        # Get original dimensions before rotation
        if hasattr(self, 'current_frame'):
            original_h, original_w = self.current_frame.shape[:2]
        else:
            # Default fallback
            original_w = self._base_pixmap.width() if hasattr(self, '_base_pixmap') else 640
            original_h = self._base_pixmap.height() if hasattr(self, '_base_pixmap') else 480
        
        x, y = pos.x(), pos.y()
        
        # Apply inverse transformation based on rotation angle
        if self.rotation_angle == 90:  # 90° clockwise rotation
            # For 90° clockwise: new_x = y, new_y = width - x
            new_x = y
            new_y = original_w - x
        elif self.rotation_angle == 180:  # 180° rotation
            # For 180°: new_x = width - x, new_y = height - y
            new_x = original_w - x
            new_y = original_h - y
        elif self.rotation_angle == 270:  # 270° clockwise (90° counterclockwise)
            # For 270° clockwise: new_x = height - y, new_y = x
            new_x = original_h - y
            new_y = x
        else:
            # This shouldn't happen, but just in case
            new_x, new_y = x, y
        
        return QPoint(int(new_x), int(new_y))

    def rotate_pose_data(self):
        """Rotate the actual pose data by 90 degrees clockwise"""
        if self.pose_data is None or self.current_frame is None:
            QMessageBox.warning(self, "No Pose Data", "Please load a video and detect poses first.")
            return
        
        try:
            # Get frame dimensions
            h, w = self.current_frame.shape[:2]
            
            # Make a backup for undo functionality
            old_data = self.pose_data.copy()
            
            # Rotate all keypoints in all frames
            for frame_idx in range(len(self.pose_data)):
                frame_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                
                for point_idx in range(len(frame_pose)):
                    x, y = frame_pose[point_idx]
                    
                    # 90° clockwise rotation: new_x = y, new_y = width - x
                    new_x = y
                    new_y = w - x
                    
                    # Update data
                    self.pose_data.iloc[frame_idx, point_idx * 2] = new_x
                    self.pose_data.iloc[frame_idx, point_idx * 2 + 1] = new_y
            
            # Update current pose
            self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            
            # Force redraw
            self._needs_redraw = True
            self.display_frame()
            self.update_coordinate_inputs()
            self.update_plot()
            
            QMessageBox.information(self, "Rotation Complete", "Pose data rotated 90° clockwise.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during pose rotation: {str(e)}")

    def set_frame_from_plot(self, frame_idx):
        """Navigate to a specific frame when the user clicks on the plot"""
        if hasattr(self, 'cap') and self.cap:
            # Make sure the frame is within valid range
            frame_idx = max(0, min(frame_idx, self.frame_slider.maximum()))
            
            # If playback is active, pause it
            if self.playing:
                self.pause_playback()
            
            # Temporarily disable the plot click to avoid multiple rapid clicks
            self.keypoint_plot.click_enabled = False
            
            # Update frame index and slider
            self.current_frame_idx = frame_idx
            self.frame_slider.setValue(frame_idx)
            
            # Visual feedback - make the slider flash briefly to indicate the new position
            original_style = self.frame_slider.styleSheet()
            self.frame_slider.setStyleSheet("QSlider::handle:horizontal { background-color: #ff5555; }")
            QTimer.singleShot(300, lambda: self.frame_slider.setStyleSheet(original_style))
            
            # Re-enable plot clicks after a short delay
            QTimer.singleShot(300, lambda: setattr(self.keypoint_plot, 'click_enabled', True))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseEditor()
    sys.exit(app.exec_())