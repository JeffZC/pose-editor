import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_pose_landmarks_from_frame(frame):
    """
    Extract pose landmarks from a single frame.
    
    Args:
        frame: BGR image
    
    Returns:
        landmarks_list: List of landmarks in pixel coordinates [x1, y1, x2, y2, ...]
        annotated_frame: Frame with landmarks visualized
    """
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize pose detector for a single frame
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        # Process the image
        results = pose.process(image_rgb)
        
        # Prepare output
        landmarks_list = []
        annotated_image = frame.copy()
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Extract landmarks
            for landmark in results.pose_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                landmarks_list.extend([x_px, y_px])
                
    return landmarks_list, annotated_image

def process_video_with_mediapipe(video_path, progress_dialog=None):
    """
    Process a video with MediaPipe pose detection.
    
    Args:
        video_path: Path to the video file
        progress_dialog: QProgressDialog for showing progress (optional)
    
    Returns:
        pose_data: DataFrame containing pose data for all frames
        success: Boolean indicating if processing was successful
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize pose detector for video
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        all_landmarks = []
        
        # Update progress dialog if provided
        if progress_dialog:
            progress_dialog.setMaximum(total_frames)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Update progress
            if progress_dialog:
                # Check if canceled
                if progress_dialog.wasCanceled():
                    cap.release()
                    return None, False
                progress_dialog.setValue(frame_idx)
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = pose.process(image_rgb)
            
            # Extract and store landmarks
            frame_landmarks = []
            
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x_px = int(landmark.x * width)
                    y_px = int(landmark.y * height)
                    frame_landmarks.extend([x_px, y_px])
                    
            else:
                # If no landmarks detected, fill with zeros or NaN
                frame_landmarks = [0] * 33 * 2  # 33 landmarks, each with x and y
            
            all_landmarks.append(frame_landmarks)
            frame_idx += 1
    
    cap.release()
    
    # Create a DataFrame from the landmarks
    pose_data = pd.DataFrame(all_landmarks)
    
    # Generate column names
    landmark_names = [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
        'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
        'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]
    
    # Create column names in format NOSE_X, NOSE_Y, etc.
    column_names = []
    for name in landmark_names:
        column_names.extend([f'{name}_X', f'{name}_Y'])
    
    # Set column names if dimensions match
    if len(column_names) == pose_data.shape[1]:
        pose_data.columns = column_names
    
    return pose_data, True