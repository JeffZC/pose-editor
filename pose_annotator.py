
import sys
import cv2
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QSlider,
                           QScrollArea)
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QCursor


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

        # Add keypoint names (default, to be updated based on pose data)
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        self.initUI()

    def initUI(self):
        # Create main container with horizontal layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create left panel for video and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

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
        self.label.setMinimumSize(400, 300)
        self.video_layout.addWidget(self.label)

        # Add video container to scroll area
        self.scroll_area.setWidget(self.video_container)
        left_layout.addWidget(self.scroll_area)

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

        # Create navigation controls
        self.nav_controls = QHBoxLayout()
        self.prev_frame_button = QPushButton("←")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        self.next_frame_button = QPushButton("→")
        self.next_frame_button.clicked.connect(self.next_frame)
        self.frame_counter = QLabel("Frame: 0/0")

        self.nav_controls.addWidget(self.prev_frame_button)
        self.nav_controls.addWidget(self.frame_slider)
        self.nav_controls.addWidget(self.next_frame_button)
        self.nav_controls.addWidget(self.frame_counter)
        left_layout.addLayout(self.nav_controls)

        # Create zoom controls
        self.zoom_controls = QHBoxLayout()
        self.zoom_out_button = QPushButton("-")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_label = QLabel("Zoom: 100%")

        self.zoom_controls.addWidget(self.zoom_out_button)
        self.zoom_controls.addWidget(self.zoom_in_button)
        self.zoom_controls.addWidget(self.zoom_label)
        self.zoom_controls.addStretch()
        left_layout.addLayout(self.zoom_controls)

        # Create info panel
        self.info_label = QLabel()
        self.info_label.setStyleSheet("QLabel { background-color : #f0f0f0; padding: 10px; }")
        self.info_label.setMinimumWidth(200)
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setText("No point selected")

        # Add panels to main layout
        main_layout.addWidget(left_panel, stretch=4)
        main_layout.addWidget(self.info_label, stretch=1)

        self.setMinimumSize(800, 600)
        self.show()

    def load_pose(self):
        pose_path, _ = QFileDialog.getOpenFileName(self, "Open Pose CSV")
        if pose_path:
            try:
                self.pose_data = pd.read_csv(pose_path)
            except Exception as e:
                self.info_label.setText(f"Error loading pose data: {e}")
                return
            columns = self.pose_data.columns
            self.keypoint_names = [col[:-2] for col in columns[::2]]  # Remove _x suffix
            
            # Update info label with new keypoint names
            if self.selected_point is not None and self.selected_point < len(self.keypoint_names):
                self.update_info_label()
            
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.update_frame()

    def update_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
            else:
                self.frame_counter.setText(f"Frame: {self.current_frame_idx}/{self.frame_slider.maximum()}")
            self.display_frame()

    def load_video(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video")
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.info_label.setText("Error: Unable to open video file.")
                return
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_slider.setMaximum(total_frames - 1)
            self.current_frame_idx = 0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                self.info_label.setText("Error: Invalid video dimensions.")
                self.cap.release()
                self.cap = None
                return
            label_height = min(480, height)  # Adaptive height based on video dimensions
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate aspect ratio and set fixed size for label
            aspect_ratio = width / height
            label_height = 480  # Fixed height
            label_width = int(label_height * aspect_ratio)
            self.label.setFixedSize(label_width, label_height)
            
            self.update_frame()
    
    def display_frame(self):
        if self.current_frame is not None:
            frame = self.current_frame.copy()
            if self.pose_data is not None:
                self.current_pose = self.pose_data.iloc[self.current_frame_idx].values.reshape(-1, 2)
                for i, point in enumerate(self.current_pose):
                    # Draw larger circles for selected point
                    radius = 8 if i == self.selected_point else 5
                    color = (255, 0, 0) if i == self.selected_point else (0, 255, 0)
                    cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)

            # Update info label with selected point info
            if self.selected_point is not None and self.selected_point < len(self.keypoint_names):
                point = self.current_pose[self.selected_point]
                info_text = f"Selected Point:\n\n"
                info_text += f"Name: {self.keypoint_names[self.selected_point]}\n"
                info_text += f"Position: ({int(point[0])}, {int(point[1])})"
                self.info_label.setText(info_text)
            else:
                self.info_label.setText("No point selected")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Calculate zoom transformation
            scaled_width = int(width * self.zoom_level)
            scaled_height = int(height * self.zoom_level)
            
            # Create scaled pixmap
            scaled_pixmap = pixmap.scaled(
                scaled_width,
                scaled_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.label.setPixmap(scaled_pixmap)
            self.label.setFixedSize(scaled_width, scaled_height)
    
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
    
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Convert global coordinates to label coordinates
            pos = self.label.mapFrom(self, event.pos())
            # Convert coordinates based on zoom level
            scaled_pos = QPoint(int(pos.x() / self.zoom_level), 
                              int(pos.y() / self.zoom_level))
            self.selected_point = self.get_selected_point(scaled_pos)
            if self.selected_point is not None:
                self.dragging = True
                self.display_frame()
        elif event.button() == Qt.RightButton:
            self.selected_point = None
            self.dragging = False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_point is not None:
            # Convert global coordinates to label coordinates
            pos = self.label.mapFrom(self, event.pos())
            # Convert coordinates based on zoom level
            scaled_pos = QPoint(int(pos.x() / self.zoom_level), 
                              int(pos.y() / self.zoom_level))
            self.move_point(scaled_pos)
            self.display_frame()

    def get_selected_point(self, pos):
        if self.current_pose is not None:
            for i, point in enumerate(self.current_pose):
                # Scale the detection radius with zoom level
                detect_radius = 10 / self.zoom_level
                if np.linalg.norm(np.array([point[0], point[1]]) - 
                                np.array([pos.x(), pos.y()])) < detect_radius:
                    return i
        return None

    def move_point(self, pos):
        if self.selected_point is not None and self.pose_data is not None:
            # Update the pose data directly
            self.pose_data.iloc[self.current_frame_idx, 
                              self.selected_point * 2] = pos.x()
            self.pose_data.iloc[self.current_frame_idx, 
                              self.selected_point * 2 + 1] = pos.y()

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
        self.current_frame_idx = value
        self.update_frame()

    def next_frame(self):
        if self.cap and self.current_frame_idx < self.frame_slider.maximum():
            self.current_frame_idx += 1
            self.frame_slider.setValue(self.current_frame_idx)

    def prev_frame(self):
        if self.cap and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_slider.setValue(self.current_frame_idx)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseEditor()
    sys.exit(app.exec_())