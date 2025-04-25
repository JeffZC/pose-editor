import sys
import cv2
import pandas as pd
import numpy as np
import time
import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSlider,
                           QScrollArea, QGroupBox, QComboBox, QLineEdit, QShortcut, 
                           QProgressDialog, QMessageBox, QToolTip)
from PyQt5.QtCore import Qt, QPoint, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QCursor, QIcon, QKeySequence
from .plot import create_plot_widget
from .mediapipe_io import (get_pose_landmarks_from_frame, process_video_with_mediapipe,
                           process_mediapipe_to_rr21)
from .body_format import load_pose_data, save_pose_data, SUPPORTED_FORMATS
from .hand_format import (process_video_with_mediapipe_hands, get_hand_landmarks_from_frame, 
                              HAND_LANDMARK_NAMES, draw_hand_landmarks, save_hand_data, load_hand_data)
from .face_format import (process_video_with_mediapipe_face, get_face_landmarks_from_frame,
                              FACE_LANDMARK_COUNT, draw_face_landmarks, save_face_data, load_face_data)


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

# New command class for MediaPipe pose detection
class MediaPipeDetectionCommand:
    def __init__(self, editor, frame_idx, old_pose_data, new_pose_data):
        self.editor = editor
        self.frame_idx = frame_idx
        self.old_pose_data = old_pose_data.copy() if old_pose_data is not None else None
        self.new_pose_data = new_pose_data.copy()
        
    def undo(self):
        """Restore the previous state"""
        # Check if we need to change frames
        if self.frame_idx != self.editor.current_frame_idx:
            self.editor.current_frame_idx = self.frame_idx
            self.editor.frame_slider.setValue(self.frame_idx)
        
        # Restore old pose data for the current frame
        if self.old_pose_data is not None:
            self.editor.pose_data.iloc[self.frame_idx] = self.old_pose_data
        
        # Update current pose
        self.editor.current_pose = self.editor.pose_data.iloc[self.frame_idx].values.reshape(-1, 2)
        
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
        
        # Apply new pose data for the current frame
        self.editor.pose_data.iloc[self.frame_idx] = self.new_pose_data
        
        # Update current pose
        self.editor.current_pose = self.editor.pose_data.iloc[self.frame_idx].values.reshape(-1, 2)
        
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
        self.current_keypoint_list = []  # current category keypoints
        self._hovered_point = None       # hovered keypoint index
        self._current_keypoint_type = "Body"  # Track current keypoint type
        self.keypoint_type_mapping = {}  # Map dropdown index to keypoint type and index
        self.update_interval_ms = 10  # Only update every 10ms during dragging

        self.last_update_time = 0
        self.gc_counter = 0

        # Initialize caching system properties
        self._cached_frame = None
        self._cached_frame_idx = -1  # Use -1 to ensure it's different from any valid frame idx
        self._needs_redraw = True

        # Initialize separate data structures for hands and face
        self.hand_data_left = None  # DataFrame for left hand landmarks
        self.hand_data_right = None  # DataFrame for right hand landmarks
        self.face_data = None  # DataFrame for face landmarks
        
        # Track if hand/face detection is enabled
        self.hand_detection_enabled = False
        self.face_detection_enabled = False

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
        self.zoom_label = QLabel("Zoom: 100%")  # Move zoom label definition here

        self.nav_controls.addWidget(self.play_button)
        self.nav_controls.addWidget(self.prev_frame_button)
        self.nav_controls.addWidget(self.frame_slider)
        self.nav_controls.addWidget(self.next_frame_button)
        self.nav_controls.addWidget(self.frame_counter)
        self.nav_controls.addWidget(self.zoom_label)  # Add zoom label to navigation controls
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
        self.video_control_group_box.setLayout(self.video_control_layout)
        right_layout.addWidget(self.video_control_group_box)
    
        # Create keypoint selection group box
        self.keypoint_ops_group_box = QGroupBox("Keypoint Operations")
        self.keypoint_ops_layout = QVBoxLayout()

        # Add keypoint type selector above keypoint dropdown
        self.keypoint_category_layout = QHBoxLayout()
        self.keypoint_category_label = QLabel("Keypoint Type:")
        self.keypoint_category_dropdown = QComboBox()
        self.keypoint_category_dropdown.addItems(["Body", "Left Hand", "Right Hand", "Face"])
        self.keypoint_category_dropdown.currentIndexChanged.connect(self.on_category_changed)
        self.keypoint_category_layout.addWidget(self.keypoint_category_label)
        self.keypoint_category_layout.addWidget(self.keypoint_category_dropdown)
        self.keypoint_ops_layout.addLayout(self.keypoint_category_layout)

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

        # Add to your initUI method after creating the coordinate inputs
        self.x_coord_input.returnPressed.connect(self.update_keypoint_coordinates)
        self.y_coord_input.returnPressed.connect(self.update_keypoint_coordinates)

        # Add these connections
        self.x_coord_input.textEdited.connect(self.preview_coordinate_update)
        self.y_coord_input.textEdited.connect(self.preview_coordinate_update)

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
        
        # Create Pose Model Selection group box
        self.model_selection_group_box = QGroupBox("Pose Model Selection")
        self.model_selection_layout = QVBoxLayout()

        # Body Model row
        body_layout = QHBoxLayout()
        body_label = QLabel("Body Model:")
        self.body_model_combo = QComboBox()
        self.body_model_combo.addItems([
            "MediaPipe Pose Small", 
            "MediaPipe Pose Large", 
            "Skip"
        ])
        body_layout.addWidget(body_label)
        body_layout.addWidget(self.body_model_combo)
        self.model_selection_layout.addLayout(body_layout)

        # Hand Model row
        hand_layout = QHBoxLayout()
        hand_label = QLabel("Hand Model:")
        self.hand_model_combo = QComboBox()
        self.hand_model_combo.addItems([
            "MediaPipe Hands", 
            "Skip"
        ])
        hand_layout.addWidget(hand_label)
        hand_layout.addWidget(self.hand_model_combo)
        self.model_selection_layout.addLayout(hand_layout)

        # Face Model row
        face_layout = QHBoxLayout()
        face_label = QLabel("Face Model:")
        self.face_model_combo = QComboBox()
        self.face_model_combo.addItems([
            "MediaPipe Face", 
            "Skip"
        ])
        face_layout.addWidget(face_label)
        face_layout.addWidget(self.face_model_combo)
        self.model_selection_layout.addLayout(face_layout)

        self.model_selection_group_box.setLayout(self.model_selection_layout)
        
        # Insert above pose_options_group_box
        right_layout.addWidget(self.model_selection_group_box)
    
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

        # Add keyboard shortcuts that work globally
        self.prev_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.prev_shortcut.activated.connect(self.prev_frame)
        
        self.next_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.next_shortcut.activated.connect(self.next_frame)
        
        # Make the navigation buttons more obvious with tooltips
        self.prev_frame_button.setToolTip("Previous frame (Left arrow key)")
        self.next_frame_button.setToolTip("Next frame (Right arrow key)")

    def toggle_black_and_white(self):
        self.black_and_white = not self.black_and_white
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
        self._cached_frame = self.current_frame.copy()
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
        """Update coordinate input fields based on selected keypoint"""
        if self.selected_point is not None:
            category = self._current_keypoint_type
            point = None
            
            if category == "Body" and self.current_pose is not None:
                if 0 <= self.selected_point < len(self.current_pose):
                    point = self.current_pose[self.selected_point]
            elif category == "Left Hand" and self.hand_data_left is not None:
                if 0 <= self.selected_point < 21:
                    x = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point]
                    y = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21]
                    point = [x, y]
            elif category == "Right Hand" and self.hand_data_right is not None:
                if 0 <= self.selected_point < 21:
                    x = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point]
                    y = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21]
                    point = [x, y]
            elif category == "Face" and self.face_data is not None:
                if 0 <= self.selected_point < 468:
                    x = self.face_data.iloc[self.current_frame_idx, self.selected_point]
                    y = self.face_data.iloc[self.current_frame_idx, self.selected_point + 468]
                    point = [x, y]
            
            if point is not None:
                self.x_coord_input.setText(str(int(point[0])))
                self.y_coord_input.setText(str(int(point[1])))
                return
        
        # Clear inputs if no valid point
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
            # Clear previous pose data to avoid carrying over to the new video
            self.pose_data = None
            self.hand_data_left = None
            self.hand_data_right = None
            self.face_data = None
            self.selected_point = None
            self._hovered_point = None
            
            # Clear keypoints dropdown
            self.keypoint_dropdown.blockSignals(True)
            self.keypoint_dropdown.clear()
            self.keypoint_dropdown.blockSignals(False)
            
            # Reset keypoint list
            self.keypoint_names = []
            self.current_keypoint_list = []
            
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
            self._cached_frame_idx = -1  # Reset to invalid value instead of deleting
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
            # Process hover events to show keypoint labels
            if event.type() == event.MouseMove:
                pos = event.pos()
                scaled_pos = QPoint(int(pos.x() / self.zoom_level), int(pos.y() / self.zoom_level))
                
                # Transform coordinates based on rotation
                transformed_pos = self.transform_coordinates(scaled_pos)
                
                # Check which keypoint is hovered
                hovered_idx = self.get_hovered_keypoint(transformed_pos)
                
                if hovered_idx != self._hovered_point:
                    self._hovered_point = hovered_idx
                    self._needs_redraw = True
                    
                    # Show tooltip if hovering over a keypoint
                    if hovered_idx is not None and self.current_keypoint_list:
                        if hovered_idx < len(self.current_keypoint_list):
                            QToolTip.showText(
                                self.label.mapToGlobal(pos), 
                                self.current_keypoint_list[hovered_idx]
                            )
                    else:
                        QToolTip.hideText()
                    
                    # Redraw with highlighted keypoint
                    self.display_frame()
            
            # Process mouse press events
            if event.type() == event.MouseButtonPress:
                pos = event.pos()
                scaled_pos = QPoint(int(pos.x() / self.zoom_level), int(pos.y() / self.zoom_level))
                
                # Get the selected keypoint based on current category
                new_selected_point = self.get_selected_point(scaled_pos)
                
                if event.button() == Qt.LeftButton:
                    if new_selected_point is not None:
                        self.selected_point = new_selected_point
                        # Synchronize dropdown
                        self.keypoint_dropdown.blockSignals(True)
                        self.keypoint_dropdown.setCurrentIndex(self.selected_point)
                        self.keypoint_dropdown.blockSignals(False)
                        self.dragging = True
                        
                        # Store the initial position for the drag operation
                        category = self._current_keypoint_type
                        if category == "Body" and self.pose_data is not None:
                            self._drag_start_pos = (
                                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2],
                                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
                            )
                        elif category == "Left Hand" and self.hand_data_left is not None:
                            self._drag_start_pos = (
                                self.hand_data_left.iloc[self.current_frame_idx, self.selected_point],
                                self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21]
                            )
                        elif category == "Right Hand" and self.hand_data_right is not None:
                            self._drag_start_pos = (
                                self.hand_data_right.iloc[self.current_frame_idx, self.selected_point],
                                self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21]
                            )
                        elif category == "Face" and self.face_data is not None:
                            self._drag_start_pos = (
                                self.face_data.iloc[self.current_frame_idx, self.selected_point],
                                self.face_data.iloc[self.current_frame_idx, self.selected_point + 468]
                            )
                        
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
                # Transform coordinates based on rotation for accurate dragging
                transformed_pos = self.transform_coordinates(scaled_pos)
                
                # Update point position
                current_time = time.time() * 1000
                if current_time - self.last_update_time > self.update_interval_ms:
                    self.move_point(transformed_pos)
                    self.last_update_time = current_time
                # Update display without updating plot for performance
                self.display_frame()
                return True
                
            elif event.type() == event.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.dragging:
                    self.dragging = False
                    
                    # Create command when drag completes
                    if hasattr(self, '_drag_start_pos') and self.selected_point is not None:
                        start_x, start_y = self._drag_start_pos
                        category = self._current_keypoint_type
                        
                        # Get current position based on keypoint category
                        try:
                            current_x = None
                            current_y = None
                            
                            if category == "Body" and self.pose_data is not None:
                                current_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
                                current_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
                            elif category == "Left Hand" and self.hand_data_left is not None:
                                current_x = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point]
                                current_y = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21]
                            elif category == "Right Hand" and self.hand_data_right is not None:
                                current_x = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point]
                                current_y = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21]
                            elif category == "Face" and self.face_data is not None:
                                current_x = self.face_data.iloc[self.current_frame_idx, self.selected_point]
                                current_y = self.face_data.iloc[self.current_frame_idx, self.selected_point + 468]
                            
                            # Only add command if position actually changed and we have valid coordinates
                            if current_x is not None and current_y is not None and (abs(start_x - current_x) > 0 or abs(start_y - current_y) > 0):
                                # Create the appropriate command
                                command = KeypointCommand(
                                    self,
                                    self.current_frame_idx,
                                    self.selected_point,
                                    start_x, start_y,
                                    current_x, current_y
                                )
                                self.add_command(command)
                        except (AttributeError, IndexError, TypeError) as e:
                            print(f"Warning: Could not create move command: {e}")
                        
                        delattr(self, '_drag_start_pos')
                    
                    # Only update plot when done dragging
                    self.update_plot()
                    return True
        
        return super().eventFilter(source, event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def get_hovered_keypoint(self, pos):
        """Get index of keypoint being hovered over"""
        # Get the current pose data based on selected category
        current_pose = self.get_current_pose_data()
        
        if current_pose is not None:
            for i, point in enumerate(current_pose):
                # Scale the detection radius with zoom level
                detect_radius = 10 / self.zoom_level
                if np.linalg.norm(np.array([point[0], point[1]]) - 
                                np.array([pos.x(), pos.y()])) < detect_radius:
                    return i
        return None
    
    def get_selected_point(self, pos):
        """Get index of keypoint selected by click"""
        # Transform mouse coordinates based on rotation
        transformed_pos = self.transform_coordinates(pos)
        
        # Use the same logic as get_hovered_keypoint but with transformed position
        return self.get_hovered_keypoint(transformed_pos)

    def get_current_pose_data(self):
        """Get the current pose data based on selected category"""
        category = self._current_keypoint_type
        
        if category == "Body" and self.pose_data is not None:
            # Check if we have 3 values per point (x, y, visibility) or just 2 (x, y)
            total_columns = self.pose_data.shape[1]
            
            if total_columns % 3 == 0:  # If we have visibility data (x, y, v) format
                # Extract just x and y coordinates, ignore visibility
                x_cols = self.pose_data.iloc[self.current_frame_idx, 0:total_columns:3].values
                y_cols = self.pose_data.iloc[self.current_frame_idx, 1:total_columns:3].values
                return np.column_stack((x_cols, y_cols))
            else:  # Simple (x, y) format
                return self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
        elif category == "Left Hand" and self.hand_data_left is not None:
            # Extract x and y coordinates from the dataframe
            left_hand_x = self.hand_data_left.iloc[self.current_frame_idx, :21].values
            left_hand_y = self.hand_data_left.iloc[self.current_frame_idx, 21:].values
            # Combine into point pairs
            return np.column_stack((left_hand_x, left_hand_y))
        elif category == "Right Hand" and self.hand_data_right is not None:
            # Extract x and y coordinates from the dataframe
            right_hand_x = self.hand_data_right.iloc[self.current_frame_idx, :21].values
            right_hand_y = self.hand_data_right.iloc[self.current_frame_idx, 21:].values
            # Combine into point pairs
            return np.column_stack((right_hand_x, right_hand_y))
        elif category == "Face" and self.face_data is not None:
            # Extract x and y coordinates from the dataframe
            face_x = self.face_data.iloc[self.current_frame_idx, :468].values
            face_y = self.face_data.iloc[self.current_frame_idx, 468:].values
            # Combine into point pairs
            return np.column_stack((face_x, face_y))
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
        """Display frame at the given index without changing the current frame"""
        if not hasattr(self, 'cap') or self.cap is None:
            return False
            
        if frame_idx < 0 or frame_idx >= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            return False
            
        # Set position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, preview_frame = self.cap.read()
        
        if not ret:
            return False
            
        # Store the current frame
        self.current_frame = preview_frame
        self.current_frame_idx = frame_idx
        
        # Update data about current frame
        if hasattr(self, 'pose_data') and self.pose_data is not None and frame_idx < len(self.pose_data):
            # Check if we have 3 values per point (x, y, visibility) or just 2 (x, y)
            total_columns = self.pose_data.shape[1]
            
            if total_columns % 3 == 0:  # If we have visibility data (x, y, v) format
                num_points = total_columns // 3
                # Extract just x and y coordinates, ignore visibility
                x_coords = self.pose_data.iloc[frame_idx, [i*3 for i in range(num_points)]].values
                y_coords = self.pose_data.iloc[frame_idx, [i*3+1 for i in range(num_points)]].values
                self.current_pose = np.column_stack((x_coords, y_coords))
            else:  # Simple (x, y) format
                self.current_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
        
        # Display the current frame
        self.display_frame()
        
        # Update coordinate inputs
        self.update_coordinate_inputs()
        
        # Update trajectory plot if needed
        self.update_plot()
        
        return True

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
            # Pause playback if active
            if self.playing:
                self.pause_playback()
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)
    
    def prev_frame(self):
        if hasattr(self, 'cap') and self.cap and self.current_frame_idx > 0:
            # Pause playback if active
            if self.playing:
                self.pause_playback()
            self.current_frame_idx -= 1
            self.frame_slider.setValue(self.current_frame_idx)
    
    def on_keypoint_selected(self, index):
        if index < 0:
            self.selected_point = None
            self._needs_redraw = True
            self.update_coordinate_inputs()
            self.display_frame()
            self.update_plot()
            return
            
        # Get the mapping information for this dropdown index
        if index in self.keypoint_type_mapping:
            # Save the selected point index for the current keypoint type
            self.selected_point = index
            
            # This is the correct index for lookup based on the keypoint type
            # (it maps directly to the index for accessing data in the respective dataframe)
            self._needs_redraw = True  # Force redraw
            self.update_coordinate_inputs()
            self.display_frame()  # Must come before update_plot for visual feedback
            self.update_plot()
        else:
            self.selected_point = None
            self._needs_redraw = True  # Force redraw
            self.update_coordinate_inputs()
            self.display_frame()
            self.update_plot()
    
    def on_category_changed(self, index):
        """Update keypoint dropdown when category changes"""
        category = self.keypoint_category_dropdown.currentText()
        self._current_keypoint_type = category
        
        if category == "Body":
            items = self.keypoint_names
            self.current_keypoint_list = items
            # Store keypoint type mapping - for Body, index in dropdown is the same as the actual keypoint index
            self.keypoint_type_mapping = {i: {"type": "Body", "index": i} for i in range(len(items))}
        elif category == "Left Hand" and self.hand_data_left is not None:
            # Use proper names for MediaPipe hand landmarks
            items = [
                "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                "PINKY_FINGER_MCP", "PINKY_FINGER_PIP", "PINKY_FINGER_DIP", "PINKY_FINGER_TIP"
            ]
            self.current_keypoint_list = items
            # Store keypoint type mapping - for hand landmarks, map dropdown index to hand keypoint index
            self.keypoint_type_mapping = {i: {"type": "Left Hand", "index": i} for i in range(len(items))}
        elif category == "Right Hand" and self.hand_data_right is not None:
            # Use proper names for MediaPipe hand landmarks
            items = [
                "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
                "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
                "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
                "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                "PINKY_FINGER_MCP", "PINKY_FINGER_PIP", "PINKY_FINGER_DIP", "PINKY_FINGER_TIP"
            ]
            self.current_keypoint_list = items
            # Store keypoint type mapping - for hand landmarks, map dropdown index to hand keypoint index
            self.keypoint_type_mapping = {i: {"type": "Right Hand", "index": i} for i in range(len(items))}
        elif category == "Face" and self.face_data is not None:
            items = [f"Face_{i}" for i in range(468)]
            self.current_keypoint_list = items
            # Store keypoint type mapping - for face landmarks, map dropdown index to face keypoint index
            self.keypoint_type_mapping = {i: {"type": "Face", "index": i} for i in range(len(items))}
        else:
            # Empty or default list
            items = []
            self.current_keypoint_list = items
            self.keypoint_type_mapping = {}
            
        self.keypoint_dropdown.blockSignals(True)
        self.keypoint_dropdown.clear()
        self.keypoint_dropdown.addItems(items)
        self.keypoint_dropdown.blockSignals(False)
        
        # Reset selection
        self.selected_point = None
        self._hovered_point = None
        self._needs_redraw = True
        self.display_frame()
        self.update_coordinate_inputs()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        # Note: We'll handle this in the eventFilter instead to avoid duplicate handling
        pass

    def move_point(self, pos):
        """Move selected keypoint to new position"""
        if self.selected_point is None:
            return
            
        # Transform mouse coordinates based on rotation
        transformed_pos = self.transform_coordinates(pos)
        x, y = transformed_pos.x(), transformed_pos.y()
        
        # Quick bounds check
        if x < 0 or y < 0:
            return
        
        category = self._current_keypoint_type
        
        if category == "Body" and self.pose_data is not None:
            # Skip update if position hasn't changed significantly
            current_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
            current_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
            
            # Only update if position changed by at least 1 pixel
            if abs(x - current_x) < 1 and abs(y - current_y) < 1:
                return
            
            # Store initial position for undo when first starting to drag
            if not hasattr(self, '_drag_start_pos'):
                self._drag_start_pos = (current_x, current_y)
            
            # Update data
            self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2] = x
            self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1] = y
            
            # Update current pose if using body keypoints
            if self.current_pose is not None:
                self.current_pose[self.selected_point] = [x, y]
                
        elif category == "Left Hand" and self.hand_data_left is not None:
            # Use selected_point as index for hand landmark
            # Update left hand keypoint
            current_x = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point]
            current_y = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21]
            
            if abs(x - current_x) < 1 and abs(y - current_y) < 1:
                return
                
            if not hasattr(self, '_drag_start_pos'):
                self._drag_start_pos = (current_x, current_y)
                
            self.hand_data_left.iloc[self.current_frame_idx, self.selected_point] = x
            self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21] = y
            
        elif category == "Right Hand" and self.hand_data_right is not None:
            # Use selected_point as index for hand landmark
            current_x = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point]
            current_y = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21]
            
            if abs(x - current_x) < 1 and abs(y - current_y) < 1:
                return
                
            if not hasattr(self, '_drag_start_pos'):
                self._drag_start_pos = (current_x, current_y)
                
            self.hand_data_right.iloc[self.current_frame_idx, self.selected_point] = x
            self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21] = y
            
        elif category == "Face" and self.face_data is not None:
            # Use selected_point as index for face landmark
            current_x = self.face_data.iloc[self.current_frame_idx, self.selected_point]
            current_y = self.face_data.iloc[self.current_frame_idx, self.selected_point + 468]
            
            if abs(x - current_x) < 1 and abs(y - current_y) < 1:
                return
                
            if not hasattr(self, '_drag_start_pos'):
                self._drag_start_pos = (current_x, current_y)
                
            self.face_data.iloc[self.current_frame_idx, self.selected_point] = x
            self.face_data.iloc[self.current_frame_idx, self.selected_point + 468] = y
            
        # Update coordinate inputs
        self.x_coord_input.setText(str(int(x)))
        self.y_coord_input.setText(str(int(y)))
        
        # Mark for redraw
        self._needs_redraw = True

    def update_keypoint_coordinates(self):
        """Update keypoint coordinates based on input fields"""
        if self.selected_point is None:
            return
            
        try:
            new_x = int(self.x_coord_input.text())
            new_y = int(self.y_coord_input.text())
        except ValueError:
            return
        
        # Validate coordinates
        if new_x < 0 or new_y < 0:
            return
        
        category = self._current_keypoint_type
        old_x = None
        old_y = None
        
        if category == "Body" and self.pose_data is not None:
            old_x = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2]
            old_y = self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1]
            
            if old_x != new_x or old_y != new_y:
                # Create command for undo/redo
                self.create_move_command(
                    self.selected_point,
                    old_x, old_y,
                    new_x, new_y
                )
                
                # Update the pose data
                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2] = new_x
                self.pose_data.iloc[self.current_frame_idx, self.selected_point * 2 + 1] = new_y
                
                # Update current_pose
                if self.current_pose is not None and self.selected_point < len(self.current_pose):
                    self.current_pose[self.selected_point] = [new_x, new_y]
        
        elif category == "Left Hand" and self.hand_data_left is not None:
            old_x = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point]
            old_y = self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21]
            
            if old_x != new_x or old_y != new_y:
                # Create command for hand keypoint undo/redo
                command = KeypointCommand(
                    self,
                    self.current_frame_idx,
                    self.selected_point,
                    old_x, old_y,
                    new_x, new_y
                )
                self.add_command(command)
                
                # Update left hand data
                self.hand_data_left.iloc[self.current_frame_idx, self.selected_point] = new_x
                self.hand_data_left.iloc[self.current_frame_idx, self.selected_point + 21] = new_y
        
        elif category == "Right Hand" and self.hand_data_right is not None:
            old_x = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point]
            old_y = self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21]
            
            if old_x != new_x or old_y != new_y:
                # Create command for hand keypoint undo/redo
                command = KeypointCommand(
                    self,
                    self.current_frame_idx,
                    self.selected_point,
                    old_x, old_y,
                    new_x, new_y
                )
                self.add_command(command)
                
                # Update right hand data
                self.hand_data_right.iloc[self.current_frame_idx, self.selected_point] = new_x
                self.hand_data_right.iloc[self.current_frame_idx, self.selected_point + 21] = new_y
                
        elif category == "Face" and self.face_data is not None:
            old_x = self.face_data.iloc[self.current_frame_idx, self.selected_point]
            old_y = self.face_data.iloc[self.current_frame_idx, self.selected_point + 468]
            
            if old_x != new_x or old_y != new_y:
                # Create command for face keypoint undo/redo 
                command = KeypointCommand(
                    self,
                    self.current_frame_idx,
                    self.selected_point,
                    old_x, old_y,
                    new_x, new_y
                )
                self.add_command(command)
                
                # Update face data
                self.face_data.iloc[self.current_frame_idx, self.selected_point] = new_x
                self.face_data.iloc[self.current_frame_idx, self.selected_point + 468] = new_y
        
        # Force redraw and update
        if old_x != new_x or old_y != new_y:
            self._needs_redraw = True
            self.display_frame()
            self.update_plot()

    def update_plot(self):
        """Update the keypoint trajectory plot based on selected keypoint"""
        # Only update plot if we have data and aren't dragging (for responsiveness)
        if self.dragging:
            return
            
        if hasattr(self, 'keypoint_plot') and self.selected_point is not None:
            category = self._current_keypoint_type
            plot_data = None
            keypoint_idx = self.selected_point
            
            if category == "Body" and self.pose_data is not None:
                plot_data = self.pose_data
                # Calculate ankle angles for body data
                ankle_angles = {'left': [], 'right': []}
                
                try:
                    # Find keypoint indices using RR21 uppercase format
                    l_knee_idx = self.keypoint_names.index("LEFT_KNEE") if "LEFT_KNEE" in self.keypoint_names else None
                    l_ankle_idx = self.keypoint_names.index("LEFT_ANKLE") if "LEFT_ANKLE" in self.keypoint_names else None
                    l_foot_idx = self.keypoint_names.index("LEFT_FOOT") if "LEFT_FOOT" in self.keypoint_names else None
                    
                    r_knee_idx = self.keypoint_names.index("RIGHT_KNEE") if "RIGHT_KNEE" in self.keypoint_names else None
                    r_ankle_idx = self.keypoint_names.index("RIGHT_ANKLE") if "RIGHT_ANKLE" in self.keypoint_names else None
                    r_foot_idx = self.keypoint_names.index("RIGHT_FOOT") if "RIGHT_FOOT" in self.keypoint_names else None
                                
                    # Calculate angles for all frames if we have all required keypoints
                    have_left = all(idx is not None for idx in [l_knee_idx, l_ankle_idx, l_foot_idx])
                    have_right = all(idx is not None for idx in [r_knee_idx, r_ankle_idx, r_foot_idx])

                    # Prepare empty lists of the right length
                    ankle_angles['left'] = [None] * len(self.pose_data)
                    ankle_angles['right'] = [None] * len(self.pose_data)
                    
                    # Calculate angles for each frame
                    for frame_idx in range(len(self.pose_data)):
                        # Get current frame's pose
                        frame_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                        
                        # Left ankle angle
                        if have_left:
                            l_knee = frame_pose[l_knee_idx]
                            l_ankle = frame_pose[l_ankle_idx]
                            l_foot = frame_pose[l_foot_idx]
                            
                            # Calculate angle
                            l_angle = calculate_ankle_angle(l_knee, l_ankle, l_foot)
                            ankle_angles['left'][frame_idx] = l_angle
                        
                        # Right ankle angle
                        if have_right:
                            r_knee = frame_pose[r_knee_idx]
                            r_ankle = frame_pose[r_ankle_idx]
                            r_foot = frame_pose[r_foot_idx]
                            
                            # Calculate angle
                            r_angle = calculate_ankle_angle(r_knee, r_ankle, r_foot)
                            ankle_angles['right'][frame_idx] = r_angle
                
                except Exception as e:
                    print(f"Error calculating ankle angles: {e}")
                    import traceback
                    traceback.print_exc()
                    
                # Plot trajectory with ankle angles
                if plot_data is not None:
                    total_frames = len(plot_data)
                    self.keypoint_plot.plot_keypoint_trajectory(
                        plot_data, 
                        keypoint_idx, 
                        self.current_frame_idx, 
                        total_frames,
                        ankle_angles,
                        show_angles=True
                    )
                
            elif category == "Left Hand" and self.hand_data_left is not None:
                # Create a temporary DataFrame in the format expected by the plotter
                # We need to convert from separate X/Y columns to alternating X/Y format
                try:
                    num_frames = len(self.hand_data_left)
                    hand_data_plot = pd.DataFrame(np.zeros((num_frames, 21*2)))
                    
                    # Copy data to the format expected by plotter
                    for i in range(21):
                        hand_data_plot.iloc[:, i*2] = self.hand_data_left.iloc[:, i]        # X values
                        hand_data_plot.iloc[:, i*2+1] = self.hand_data_left.iloc[:, i+21]   # Y values
                    
                    # Plot trajectory (without ankle angles)
                    self.keypoint_plot.plot_keypoint_trajectory(
                        hand_data_plot, 
                        keypoint_idx, 
                        self.current_frame_idx, 
                        num_frames,
                        None,
                        show_angles=False
                    )
                except Exception as e:
                    print(f"Error plotting left hand data: {e}")
                
            elif category == "Right Hand" and self.hand_data_right is not None:
                # Create a temporary DataFrame in the format expected by the plotter
                try:
                    num_frames = len(self.hand_data_right)
                    hand_data_plot = pd.DataFrame(np.zeros((num_frames, 21*2)))
                    
                    # Copy data to the format expected by plotter
                    for i in range(21):
                        hand_data_plot.iloc[:, i*2] = self.hand_data_right.iloc[:, i]        # X values
                        hand_data_plot.iloc[:, i*2+1] = self.hand_data_right.iloc[:, i+21]   # Y values
                    
                    # Plot trajectory (without ankle angles)
                    self.keypoint_plot.plot_keypoint_trajectory(
                        hand_data_plot, 
                        keypoint_idx, 
                        self.current_frame_idx, 
                        num_frames,
                        None,
                        show_angles=False
                    )
                except Exception as e:
                    print(f"Error plotting right hand data: {e}")
                
            elif category == "Face" and self.face_data is not None:
                # Create a temporary DataFrame in the format expected by the plotter
                try:
                    num_frames = len(self.face_data)
                    face_data_plot = pd.DataFrame(np.zeros((num_frames, 468*2)))
                    
                    # Copy data to the format expected by plotter
                    for i in range(468):
                        face_data_plot.iloc[:, i*2] = self.face_data.iloc[:, i]         # X values
                        face_data_plot.iloc[:, i*2+1] = self.face_data.iloc[:, i+468]   # Y values
                    
                    # Plot trajectory (without ankle angles)
                    self.keypoint_plot.plot_keypoint_trajectory(
                        face_data_plot, 
                        keypoint_idx, 
                        self.current_frame_idx, 
                        num_frames,
                        None,
                        show_angles=False
                    )
                except Exception as e:
                    print(f"Error plotting face data: {e}")
            
        elif hasattr(self, 'keypoint_plot'):
            # Clear the plot if no keypoint is selected
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

    def _get_selected_models(self):
        """Get the currently selected detection models from the UI"""
        body_model = self.body_model_combo.currentText()
        hand_model = self.hand_model_combo.currentText()
        face_model = self.face_model_combo.currentText()
        
        # Check if at least one model is enabled
        if body_model == "Skip" and hand_model == "Skip" and face_model == "Skip":
            return None
            
        return {
            "body": None if body_model == "Skip" else body_model,
            "hand": None if hand_model == "Skip" else hand_model,
            "face": None if face_model == "Skip" else face_model
        }

    def _process_detection_results(self, frame_idx, results):
        """Process and update detection results for a specific frame"""
        models_run = []
        
        # Body pose detection results
        if 'body' in results and results['body']:
            # Process body landmarks
            rr21_landmarks = results['body']
            
            # Create or update pose_data if not initialized
            if self.pose_data is None:
                # Create empty DataFrame with appropriate columns
                columns = []
                for name in SUPPORTED_FORMATS["rr21"]:
                    columns.extend([f'{name}_X', f'{name}_Y'])
                
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.pose_data = pd.DataFrame(np.zeros((frame_count, len(columns))), columns=columns)
                self.keypoint_names = SUPPORTED_FORMATS["rr21"]
                
                # Update keypoint dropdown
                self.keypoint_dropdown.blockSignals(True)
                self.keypoint_dropdown.clear()
                self.keypoint_dropdown.addItems(self.keypoint_names)
                self.keypoint_dropdown.blockSignals(False)
            
            # Update pose data for current frame
            for i in range(0, len(rr21_landmarks), 2):
                if i+1 < len(rr21_landmarks):
                    col_idx = i
                    self.pose_data.iloc[frame_idx, col_idx] = rr21_landmarks[i]
                    self.pose_data.iloc[frame_idx, col_idx+1] = rr21_landmarks[i+1]
            
            # Update current pose if we're on the current frame
            if frame_idx == self.current_frame_idx:
                self.current_pose = self.pose_data.iloc[frame_idx].values.reshape(-1, 2)
                
            models_run.append("Body")
        
        # Hand detection results
        if 'left_hand' in results and results['left_hand']:
            left_hand = results['left_hand']
            
            # Initialize hand data if needed
            if self.hand_data_left is None:
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                columns = [f'HandL_{i}_X' for i in range(21)] + [f'HandL_{i}_Y' for i in range(21)]
                self.hand_data_left = pd.DataFrame(np.zeros((frame_count, len(columns))), columns=columns)
            
            # Update hand data
            for i in range(0, len(left_hand), 2):
                if i//2 < 21:
                    self.hand_data_left.iloc[frame_idx, i//2] = left_hand[i]
                    self.hand_data_left.iloc[frame_idx, i//2 + 21] = left_hand[i+1]
                    
            self.hand_detection_enabled = True
            models_run.append("Left Hand")
        
        if 'right_hand' in results and results['right_hand']:
            right_hand = results['right_hand']
            
            # Initialize hand data if needed
            if self.hand_data_right is None:
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                columns = [f'HandR_{i}_X' for i in range(21)] + [f'HandR_{i}_Y' for i in range(21)]
                self.hand_data_right = pd.DataFrame(np.zeros((frame_count, len(columns))), columns=columns)
            
            # Update hand data
            for i in range(0, len(right_hand), 2):
                if i//2 < 21:
                    self.hand_data_right.iloc[frame_idx, i//2] = right_hand[i]
                    self.hand_data_right.iloc[frame_idx, i//2 + 21] = right_hand[i+1]
                    
            self.hand_detection_enabled = True
            models_run.append("Right Hand")
        
        # Face detection results
        if 'face' in results and results['face']:
            face_landmarks = results['face']
            
            # Initialize face data if needed
            if self.face_data is None:
                frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Create columns for 468 facial landmarks (x and y coordinates)
                columns = [f'Face_{i}_X' for i in range(468)] + [f'Face_{i}_Y' for i in range(468)]
                self.face_data = pd.DataFrame(np.zeros((frame_count, len(columns))), columns=columns)
            
            # Update face data
            for i in range(0, len(face_landmarks), 2):
                if i//2 < 468:
                    self.face_data.iloc[frame_idx, i//2] = face_landmarks[i]
                    self.face_data.iloc[frame_idx, i//2 + 468] = face_landmarks[i+1]
                    
            self.face_detection_enabled = True
            models_run.append("Face")
        
        return models_run

    def detect_pose_current_frame(self):
        """Detect pose on the current frame using selected MediaPipe models"""
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            QMessageBox.warning(self, "No Frame", "No current frame to process.")
            return
        
        # Get selected models
        models = self._get_selected_models()
        
        if models is None:
            QMessageBox.warning(self, "No Model Selected", "Please select at least one detection model.")
            return
        
        try:
            # Store old data for undo
            old_data = {}
            
            if hasattr(self, 'pose_data') and self.pose_data is not None and self.current_frame_idx < len(self.pose_data):
                old_data['body'] = self.pose_data.iloc[self.current_frame_idx].copy()
            
            if hasattr(self, 'left_hand_data') and self.left_hand_data is not None and self.current_frame_idx < len(self.left_hand_data):
                old_data['left_hand'] = self.left_hand_data.iloc[self.current_frame_idx].copy()
            
            if hasattr(self, 'right_hand_data') and self.right_hand_data is not None and self.current_frame_idx < len(self.right_hand_data):
                old_data['right_hand'] = self.right_hand_data.iloc[self.current_frame_idx].copy()
            
            if hasattr(self, 'face_data') and self.face_data is not None and self.current_frame_idx < len(self.face_data):
                old_data['face'] = self.face_data.iloc[self.current_frame_idx].copy()
            
            # Results dict to store all detection results
            results = {}
            annotated_frame = self.current_frame.copy()
            
            # Body detection
            if models["body"]:
                # Set model complexity based on selection
                model_complexity = 0  # Default to small
                if "Medium" in models["body"]:
                    model_complexity = 1
                elif "Large" in models["body"]:
                    model_complexity = 2
                
                # Detect body landmarks
                landmarks_list, body_annotated = get_pose_landmarks_from_frame(
                    self.current_frame, 
                    model_complexity=model_complexity
                )
                
                if landmarks_list:
                    # Convert to RR21 format
                    results['body'] = process_mediapipe_to_rr21(landmarks_list)
                    # Use annotated frame for visualization
                    annotated_frame = body_annotated
            
            # Hand detection
            if models["hand"]:
                # Call hand detection function
                left_landmarks, right_landmarks, hand_annotated = get_hand_landmarks_from_frame(self.current_frame)
                
                if left_landmarks or right_landmarks:
                    results['left_hand'] = left_landmarks
                    results['right_hand'] = right_landmarks
                    # If no body detection, use hand annotated frame
                    if 'body' not in results:
                        annotated_frame = hand_annotated
                    else:
                        # Overlay hand landmarks on body frame
                        # This is a simple overlay, might not be ideal for all cases
                        alpha = 0.5
                        annotated_frame = cv2.addWeighted(annotated_frame, alpha, hand_annotated, 1-alpha, 0)
            
            # Face detection
            if models["face"]:
                # Call face detection function
                face_landmarks, face_annotated = get_face_landmarks_from_frame(self.current_frame)
                
                if face_landmarks:
                    results['face'] = face_landmarks
                    # If no previous detections, use face annotated frame
                    if 'body' not in results and 'left_hand' not in results and 'right_hand' not in results:
                        annotated_frame = face_annotated
                    else:
                        # Overlay face landmarks on previous frame
                        alpha = 0.5
                        annotated_frame = cv2.addWeighted(annotated_frame, alpha, face_annotated, 1-alpha, 0)
            
            # Update data with detection results
            new_data = {}
            
            # Update pose data if detected
            if 'body' in results and hasattr(self, 'pose_data') and self.pose_data is not None:
                # Create a flattened list for RR21 format
                rr21_data = results['body']
                
                # Convert to columns for DataFrame
                keypoints = SUPPORTED_FORMATS["rr21"]
                
                for i, kp in enumerate(keypoints):
                    if i*3+2 < len(rr21_data):
                        self.pose_data.loc[self.current_frame_idx, f'{kp}_X'] = rr21_data[i*3]
                        self.pose_data.loc[self.current_frame_idx, f'{kp}_Y'] = rr21_data[i*3+1]
                        self.pose_data.loc[self.current_frame_idx, f'{kp}_V'] = rr21_data[i*3+2]
                
                new_data['body'] = self.pose_data.iloc[self.current_frame_idx].copy()
            
            # Update hand data if detected
            if 'left_hand' in results and hasattr(self, 'left_hand_data') and self.left_hand_data is not None:
                left_landmarks = results['left_hand']
                
                for i in range(21):  # MediaPipe tracks 21 hand landmarks
                    if i*2+1 < len(left_landmarks):
                        self.left_hand_data.loc[self.current_frame_idx, f'HandL_{i}_X'] = left_landmarks[i*2]
                        self.left_hand_data.loc[self.current_frame_idx, f'HandL_{i}_Y'] = left_landmarks[i*2+1]
                
                new_data['left_hand'] = self.left_hand_data.iloc[self.current_frame_idx].copy()
            
            if 'right_hand' in results and hasattr(self, 'right_hand_data') and self.right_hand_data is not None:
                right_landmarks = results['right_hand']
                
                for i in range(21):  # MediaPipe tracks 21 hand landmarks
                    if i*2+1 < len(right_landmarks):
                        self.right_hand_data.loc[self.current_frame_idx, f'HandR_{i}_X'] = right_landmarks[i*2]
                        self.right_hand_data.loc[self.current_frame_idx, f'HandR_{i}_Y'] = right_landmarks[i*2+1]
                
                new_data['right_hand'] = self.right_hand_data.iloc[self.current_frame_idx].copy()
            
            # Update face data if detected
            if 'face' in results and hasattr(self, 'face_data') and self.face_data is not None:
                face_landmarks = results['face']
                
                for i in range(468):  # MediaPipe tracks 468 face landmarks
                    if i*2+1 < len(face_landmarks):
                        self.face_data.loc[self.current_frame_idx, f'Face_{i}_X'] = face_landmarks[i*2]
                        self.face_data.loc[self.current_frame_idx, f'Face_{i}_Y'] = face_landmarks[i*2+1]
                
                new_data['face'] = self.face_data.iloc[self.current_frame_idx].copy()
            
            # Add command for undo/redo
            if new_data:
                cmd = MediaPipeDetectionCommand(self, self.current_frame_idx, old_data, new_data)
                self.undo_stack.push(cmd)
            
            # Display annotated frame
            self.display_frame(annotated_frame)
            
            # Refresh GUI with updated data
            self.update_coordinate_inputs()
            self.update_plot()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error detecting pose: {str(e)}")
            import traceback
            traceback.print_exc()

    def detect_pose_video(self):
        """Detect pose on the entire video using selected MediaPipe models"""
        if not hasattr(self, 'video_path') or not self.video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        # Get selected models
        models = self._get_selected_models()
        
        if models is None:
            QMessageBox.warning(self, "No Model Selected", "Please select at least one detection model.")
            return
        
        try:
            # Create progress dialog
            progress = QProgressDialog("Processing video...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("Processing Video")
            progress.show()
            
            # Store which models will run
            models_run = []
            
            # Body pose detection
            if models["body"]:
                progress.setLabelText("Processing video with MediaPipe Pose...")
                # Set model complexity based on selection
                model_complexity = 0  # Default to small (was incorrectly set to 2)
                if "Medium" in models["body"]:
                    model_complexity = 1
                elif "Large" in models["body"]:
                    model_complexity = 2
                
                # Process video with selected complexity
                pose_data, success = process_video_with_mediapipe(
                    self.video_path, 
                    progress_dialog=progress,
                    model_complexity=model_complexity
                )
                
                if success:
                    # Update pose data
                    self.pose_data = pose_data
                    self.keypoint_names = SUPPORTED_FORMATS["rr21"]
                    
                    # Update dropdown
                    self.keypoint_dropdown.blockSignals(True)
                    self.keypoint_dropdown.clear()
                    self.keypoint_dropdown.addItems(self.keypoint_names)
                    self.keypoint_dropdown.blockSignals(False)
                    
                    models_run.append("Body")
            
            # Hand detection
            if models["hand"]:
                progress.setLabelText("Processing video with MediaPipe Hands...")
                
                # Process video for hands
                left_hand_data, right_hand_data, success = process_video_with_mediapipe_hands(
                    self.video_path, 
                    progress_dialog=progress
                )
                
                if success:
                    # Update hand data
                    self.hand_data_left = left_hand_data
                    self.hand_data_right = right_hand_data
                    self.hand_detection_enabled = True
                    
                    models_run.append("Hands")
            
            # Face detection
            if models["face"]:
                progress.setLabelText("Processing video with MediaPipe Face...")
                
                # Process video for face
                face_data, success = process_video_with_mediapipe_face(
                    self.video_path, 
                    progress_dialog=progress
                )
                
                if success:
                    # Update face data
                    self.face_data = face_data
                    self.face_detection_enabled = True
                    
                    models_run.append("Face")
            
            # Close progress dialog
            progress.close()
            
            # Update display
            if models_run:
                self._needs_redraw = True
                self.display_frame()
                self.update_coordinate_inputs()
                self.update_plot()
                
                QMessageBox.information(
                    self, 
                    "Processing Complete", 
                    f"Successfully processed video with the following models: {', '.join(models_run)}"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "Processing Failed", 
                    "No models were successfully processed. Please check the video file and try again."
                )
        
        except Exception as e:
            # Close progress dialog if open
            if 'progress' in locals() and progress is not None:
                progress.close()
            
            QMessageBox.critical(self, "Error", f"Error processing video: {str(e)}")
            import traceback
            traceback.print_exc()

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

    # Add this new method
    def preview_coordinate_update(self):
        """Preview coordinate changes without committing them to the undo history"""
        if self.selected_point is not None and self.pose_data is not None:
            try:
                x = int(self.x_coord_input.text())
                y = int(self.y_coord_input.text())
                
                # Create temporary display without changing underlying data
                self._preview_coords = (self.selected_point, x, y)
                self._needs_redraw = True
                self.display_frame()
            except ValueError:
                pass

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for frame navigation"""
        if event.key() == Qt.Key_Left:
            # Left arrow key - previous frame
            self.prev_frame()
        elif event.key() == Qt.Key_Right:
            # Right arrow key - next frame
            self.next_frame()
        else:
            # Pass other key events to parent class
            super().keyPressEvent(event)

    def display_frame(self, frame=None):
        """Render and display the current frame with all appropriate transformations"""
        if self.current_frame is None:
            return
            
        # More robust check that won't crash if attribute is missing
        cached_frame_idx = getattr(self, '_cached_frame_idx', -999)
        needs_redraw = getattr(self, '_needs_redraw', True)
        
        if frame is not None or not hasattr(self, '_cached_frame') or cached_frame_idx != self.current_frame_idx or needs_redraw:
            # Start with a fresh copy of the provided frame or the current frame
            if frame is not None:
                # Use the provided frame (like an annotated frame from detection)
                frame = frame.copy()
            else:
                # Use the current frame
                frame = self.current_frame.copy()
            
            # Apply black and white transformation if enabled
            if self.black_and_white:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Draw keypoints on frame based on current category
            category = self._current_keypoint_type
            
            # Draw body keypoints
            if category == "Body" and self.pose_data is not None:
                try:
                    # Check if we have 3 values per point (x, y, visibility) or just 2 (x, y)
                    total_columns = self.pose_data.shape[1]
                    points_count = total_columns // 2 if total_columns % 2 == 0 else total_columns // 3
                    
                    # Reshape correctly based on data format
                    if total_columns % 3 == 0:  # If we have visibility data (x, y, v) format
                        # Extract just x and y, ignore visibility
                        x_cols = self.pose_data.iloc[self.current_frame_idx, 0:total_columns:3]
                        y_cols = self.pose_data.iloc[self.current_frame_idx, 1:total_columns:3]
                        self.current_pose = np.column_stack((x_cols, y_cols))
                    else:  # Simple (x, y) format
                        self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
                    
                    for i, point in enumerate(self.current_pose):
                        radius = 8 if i == self.selected_point else 5
                        # Blue for selected, yellow for hovered, green for normal
                        if i == self.selected_point:
                            color = (255, 0, 0)  # Blue (selected)
                        elif i == self._hovered_point:
                            color = (0, 255, 255)  # Yellow (hovered)
                        else:
                            color = (0, 255, 0)  # Green (normal)
                        cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)
                except Exception as e:
                    print(f"Error drawing body keypoints: {e}")
            
            # Draw hand landmarks if available
            # ... existing code ...
            
            # Always draw other keypoint categories in the background with reduced visibility
            self._draw_background_keypoints(frame)
            
            # Apply rotation if needed
            if self.rotation_angle > 0:
                if self.rotation_angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotation_angle == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotation_angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Convert to QPixmap
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self._base_pixmap = QPixmap.fromImage(q_img)
            self._cached_frame = self.current_frame.copy()
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

    def _draw_background_keypoints(self, frame):
        """Draw other keypoint categories in the background with reduced visibility"""
        # Current category is handled separately with full visibility
        category = self._current_keypoint_type
        
        # Draw body keypoints in background if not current category
        if category != "Body" and self.pose_data is not None:
            pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            for point in pose:
                # Draw with low opacity
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 150, 0), -1)
        
        # Draw left hand in background if not current category
        if category != "Left Hand" and self.hand_data_left is not None:
            try:
                left_hand_x = self.hand_data_left.iloc[self.current_frame_idx, :21].values
                left_hand_y = self.hand_data_left.iloc[self.current_frame_idx, 21:].values
                
                for i in range(21):
                    x, y = int(left_hand_x[i]), int(left_hand_y[i])
                    if x > 0 or y > 0:
                        cv2.circle(frame, (x, y), 2, (0, 150, 150), -1)
            except:
                pass
        
        # Draw right hand in background if not current category
        if category != "Right Hand" and self.hand_data_right is not None:
            try:
                right_hand_x = self.hand_data_right.iloc[self.current_frame_idx, :21].values
                right_hand_y = self.hand_data_right.iloc[self.current_frame_idx, 21:].values
                
                for i in range(21):
                    x, y = int(right_hand_x[i]), int(right_hand_y[i])
                    if x > 0 or y > 0:
                        cv2.circle(frame, (x, y), 2, (150, 0, 150), -1)
            except:
                pass
        
        # Don't draw face in background - too many points

    def get_exact_keypoint_at_position(self, pos):
        """
        Get the exact keypoint at the given position without any mapping confusion.
        Returns a tuple (category, index, point_data) of the keypoint under the cursor.
        """
        category = self._current_keypoint_type
        detect_radius = 10 / self.zoom_level
        
        # Check body keypoints first regardless of current category
        if self.pose_data is not None:
            pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
            for i, point in enumerate(pose):
                if np.linalg.norm(np.array([point[0], point[1]]) - np.array([pos.x(), pos.y()])) < detect_radius:
                    return "Body", i, point
        
        # Check left hand keypoints
        if self.hand_data_left is not None:
            left_hand_x = self.hand_data_left.iloc[self.current_frame_idx, :21].values
            left_hand_y = self.hand_data_left.iloc[self.current_frame_idx, 21:].values
            for i in range(len(left_hand_x)):
                point = [left_hand_x[i], left_hand_y[i]]
                if np.linalg.norm(np.array(point) - np.array([pos.x(), pos.y()])) < detect_radius:
                    return "Left Hand", i, point
        
        # Check right hand keypoints
        if self.hand_data_right is not None:
            right_hand_x = self.hand_data_right.iloc[self.current_frame_idx, :21].values
            right_hand_y = self.hand_data_right.iloc[self.current_frame_idx, 21:].values
            for i in range(len(right_hand_x)):
                point = [right_hand_x[i], right_hand_y[i]]
                if np.linalg.norm(np.array(point) - np.array([pos.x(), pos.y()])) < detect_radius:
                    return "Right Hand", i, point
        
        # Check face keypoints
        if self.face_data is not None:
            face_x = self.face_data.iloc[self.current_frame_idx, :468].values
            face_y = self.face_data.iloc[self.current_frame_idx, 468:].values
            for i in range(len(face_x)):
                point = [face_x[i], face_y[i]]
                if np.linalg.norm(np.array(point) - np.array([pos.x(), pos.y()])) < detect_radius:
                    return "Face", i, point
        
        return None, None, None

def main():
    """
    Main function to initialize and run the Pose Editor application.
    """
    app = QApplication(sys.argv)
    window = PoseEditor()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()