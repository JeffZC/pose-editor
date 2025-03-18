import sys
import cv2
import pandas as pd
import numpy as np
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSlider,
                           QScrollArea, QGroupBox, QComboBox, QLineEdit, QShortcut)
from PyQt5.QtCore import Qt, QPoint, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon, QKeySequence
from plot_utils import create_plot_widget

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
        self.load_video_button = QPushButton("Load Video")
        self.load_video_button.clicked.connect(self.load_video)
        self.load_pose_button = QPushButton("Load Pose")
        self.load_pose_button.clicked.connect(self.load_pose)
        self.save_button = QPushButton("Save Poses")
        self.save_button.clicked.connect(self.save_pose)
    
        self.file_controls.addWidget(self.load_video_button)
        self.file_controls.addWidget(self.load_pose_button)
        self.file_controls.addWidget(self.save_button)
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
        left_layout.addWidget(self.plot_widget)
    
        # Create navigation controls with play/pause
        self.nav_controls = QHBoxLayout()
        
        # Add play button (will toggle between play/pause)
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.setFixedSize(32, 32)
        self.play_button.clicked.connect(self.toggle_playback)
        
        self.prev_frame_button = QPushButton("←")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        self.frame_slider.sliderPressed.connect(self.on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_slider_released)
        
        self.next_frame_button = QPushButton("→")
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
    
        # Create zoom controls group box
        self.zoom_group_box = QGroupBox("Zoom Controls")
        self.zoom_group_box.setFixedHeight(100)  # Set fixed height for the zoom controls box
        self.zoom_controls = QVBoxLayout()  # Change to vertical layout
        self.zoom_buttons_layout = QHBoxLayout()  # Horizontal layout for buttons
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_label = QLabel("Zoom: 100%")
    
        self.zoom_buttons_layout.addWidget(self.zoom_out_button)
        self.zoom_buttons_layout.addWidget(self.zoom_in_button)
        self.zoom_controls.addLayout(self.zoom_buttons_layout)
        self.zoom_controls.addWidget(self.zoom_label)
        self.zoom_controls.addStretch()
        self.zoom_group_box.setLayout(self.zoom_controls)
        right_layout.addWidget(self.zoom_group_box)
    
        # Create keypoint selection group box
        self.keypoint_group_box = QGroupBox("Select Keypoint")
        self.keypoint_layout = QVBoxLayout()
        self.keypoint_dropdown = QComboBox()
        self.keypoint_dropdown.addItems(self.keypoint_names)
        self.keypoint_dropdown.currentIndexChanged.connect(self.on_keypoint_selected)
        self.keypoint_layout.addWidget(self.keypoint_dropdown)
        self.keypoint_group_box.setLayout(self.keypoint_layout)
        right_layout.addWidget(self.keypoint_group_box)
    
        # Create keypoint coordinates group box
        self.coordinates_group_box = QGroupBox("Keypoint Coordinates")
        self.coordinates_layout = QVBoxLayout()
        self.x_coord_input = QLineEdit()
        self.x_coord_input.setPlaceholderText("X Coordinate")
        self.y_coord_input = QLineEdit()
        self.y_coord_input.setPlaceholderText("Y Coordinate")
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.update_keypoint_coordinates)
        self.coordinates_layout.addWidget(self.x_coord_input)
        self.coordinates_layout.addWidget(self.y_coord_input)
        self.coordinates_layout.addWidget(self.confirm_button)
        self.coordinates_group_box.setLayout(self.coordinates_layout)
        right_layout.addWidget(self.coordinates_group_box)

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
    
        # Create black and white switch group box
        self.bw_group_box = QGroupBox("Black and White Switch")
        self.bw_layout = QVBoxLayout()
        self.bw_button = QPushButton("Toggle Black and White")
        self.bw_button.clicked.connect(self.toggle_black_and_white)
        self.bw_layout.addWidget(self.bw_button)
        self.bw_group_box.setLayout(self.bw_layout)
        right_layout.addWidget(self.bw_group_box)
        
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
                    
                if self.pose_data is not None and self.current_frame_idx < len(self.pose_data):
                    try:
                        self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
                        for i, point in enumerate(self.current_pose):
                            radius = 8 if i == self.selected_point else 5
                            color = (255, 0, 0) if i == self.selected_point else (0, 255, 0)
                            cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)
                    except IndexError:
                        self.current_pose = None
                else:
                    self.current_pose = None

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
        pose_path, _ = QFileDialog.getOpenFileName(self, "Open Pose CSV")
        if pose_path:
            try:
                # Load the pose data
                self.pose_data = pd.read_csv(pose_path)
                
                # Fill empty/NaN values with zeros
                self.pose_data = self.pose_data.fillna(0)
                
                # Get all columns
                columns = self.pose_data.columns
                processed_columns = []
                self.keypoint_names = []
                
                # First look for columns ending with _x to identify keypoints
                x_columns = [col for col in columns if col.lower().endswith('_x')]
                
                if x_columns:
                    print(f"Found {len(x_columns)} keypoints with _x suffix")
                    
                    # For each x column, find the corresponding y column
                    for x_col in x_columns:
                        base_name = x_col[:-2]  # Remove '_x'
                        y_col = f"{base_name}_y" 
                        
                        if y_col in columns:
                            self.keypoint_names.append(base_name)
                            processed_columns.append(x_col)
                            processed_columns.append(y_col)
                
                # If no keypoints found with _x suffix, try x_ prefix
                if not processed_columns:
                    x_columns = [col for col in columns if col.lower().startswith('x_')]
                    
                    if x_columns:
                        print(f"Found {len(x_columns)} keypoints with x_ prefix")
                        
                        for x_col in x_columns:
                            base_name = x_col[2:]  # Remove 'x_'
                            y_col = f"y_{base_name}"
                            
                            if y_col in columns:
                                self.keypoint_names.append(base_name)
                                processed_columns.append(x_col)
                                processed_columns.append(y_col)
                
                # If still no keypoints found, assume columns come in pairs (even/odd)
                if not processed_columns:
                    print("No standard naming pattern found, assuming pairs of columns")
                    
                    # Skip every third column if it looks like a triplet format (x,y,conf)
                    # Check if number of columns is divisible by 3 and the third column looks like conf/z
                    if len(columns) % 3 == 0:
                        # Check a sample of third columns to see if they contain conf/z keywords
                        third_cols = [columns[i] for i in range(2, len(columns), 3)]
                        conf_indicators = ['conf', 'score', 'z', 'vis', 'visibility']
                        
                        if any(any(indicator in col.lower() for indicator in conf_indicators) for col in third_cols):
                            print("Detected triplet format with confidence/z values")
                            # Use only the first two columns of each triplet
                            for i in range(0, len(columns), 3):
                                if i+1 < len(columns):
                                    self.keypoint_names.append(f"keypoint{i//3+1}")
                                    processed_columns.append(columns[i])  # x column
                                    processed_columns.append(columns[i+1])  # y column
                        else:
                            # Just use pairs of columns
                            for i in range(0, len(columns), 2):
                                if i+1 < len(columns):
                                    self.keypoint_names.append(f"keypoint{i//2+1}")
                                    processed_columns.append(columns[i])
                                    processed_columns.append(columns[i+1])
                    else:
                        # Just use pairs of columns
                        for i in range(0, len(columns), 2):
                            if i+1 < len(columns):
                                self.keypoint_names.append(f"keypoint{i//2+1}")
                                processed_columns.append(columns[i])
                                processed_columns.append(columns[i+1])
                
                # Extract only the processed columns
                if processed_columns:
                    self.pose_data = self.pose_data[processed_columns]
                
                    # Round all values to integers for consistent display
                    numeric_columns = self.pose_data.select_dtypes(include=['float64', 'int64']).columns
                    self.pose_data[numeric_columns] = self.pose_data[numeric_columns].round().astype(int)
                    
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
                    
                    print(f"Loaded pose data with {len(self.keypoint_names)} keypoints")
                else:
                    print("Error: Could not identify keypoint columns in the CSV")

            except Exception as e:
                print(f"Error loading pose data: {e}")
                import traceback
                traceback.print_exc()
                self.pose_data = None
                self.keypoint_names = []
                self.keypoint_dropdown.clear()

    def update_frame(self):
        if hasattr(self, 'cap') and self.cap:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                # Ensure frame has the correct orientation - sometimes needed for certain video files
                # This explicitly ignores any rotation metadata and uses the raw frame data
                if frame.shape[0] > 0 and frame.shape[1] > 0:
                    frame = frame.copy()  # Work with a copy to avoid any reference issues
                
                # Update the current frame
                self.current_frame = frame
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
                
                # Mark for redraw and force cache update
                self._needs_redraw = True
                
                # Update the display
                self.display_frame()
                
                # Only update pose data if it exists for this frame
                if self.pose_data is not None and self.current_frame_idx < len(self.pose_data):
                    try:
                        # Update current pose data
                        self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
                    except IndexError:
                        self.current_pose = None
                else:
                    self.current_pose = None
                    
                self.update_coordinate_inputs()
                
                # Only update plot if not in the middle of a playback
                if not self.playing:
                    self.update_plot()
            else:
                # Handle frame read failure
                print(f"Failed to read frame {self.current_frame_idx}")
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")

    def load_video(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            
            # Force the video to maintain its default orientation
            # CV_CAP_PROP_ORIENTATION_AUTO = 49
            self.cap.set(49, 0)  # Disable auto-orientation
            
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(total_frames - 1)
            self.current_frame_idx = 0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width <= 0 or height <= 0:
                print("Error: Could not determine video dimensions")
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
            
            print(f"Loaded video: {self.video_path}")
            print(f"Dimensions: {width}x{height}, Total frames: {total_frames}, FPS: {fps}")
            
            # Read the first frame to ensure correct orientation
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
            for i, point in enumerate(self.current_pose):
                # Scale the detection radius with zoom level
                detect_radius = 10 / self.zoom_level
                if np.linalg.norm(np.array([point[0], point[1]]) - 
                                np.array([pos.x(), pos.y()])) < detect_radius:
                    return i
        return None

    def save_pose(self):
        if self.pose_data is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Pose Data", 
                "", 
                "CSV Files (*.csv)"
            )
            if file_path:
                self.pose_data.to_csv(file_path, index=False)

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
                if self.pose_data is not None and frame_idx < len(self.pose_data):
                    try:
                        # Update current pose data for this frame
                        self.current_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                        
                        # Draw simplified keypoints (faster)
                        for i, point in enumerate(self.current_pose):
                            # Use uniform color and size during dragging for speed
                            cv2.circle(frame_preview, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                    except IndexError:
                        # Handle the case where frame_idx is out of bounds for pose data
                        self.current_pose = None
                else:
                    self.current_pose = None
                
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
            x, y = pos.x(), pos.y()
            
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseEditor()
    sys.exit(app.exec_())