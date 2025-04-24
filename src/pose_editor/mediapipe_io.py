import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from typing import Tuple, Dict, List, Optional
from PyQt5.QtWidgets import QProgressDialog
from .body_format import create_empty_pose_dataframe, SUPPORTED_FORMATS

def get_pose_landmarks_from_frame(frame, min_detection_confidence=0.5):
    """
    Extract pose landmarks from a single frame.
    
    Args:
        frame: Input frame
        min_detection_confidence: MediaPipe detection confidence threshold
        
    Returns:
        Tuple containing landmarks and annotated frame
    """
    try:
        # Initialize MediaPipe pose detection
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence
        )
        mp_drawing = mp.solutions.drawing_utils
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = pose.process(frame_rgb)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )
        
        return results.pose_landmarks, annotated_frame
        
    except Exception as e:
        print(f"Error in get_pose_landmarks_from_frame: {e}")
        return None, frame

def process_video_with_mediapipe(video_path, progress_dialog=None):
    """
    Process a video with MediaPipe Pose and extract pose landmarks.
    
    Args:
        video_path: Path to the video
        progress_dialog: QProgressDialog instance
        
    Returns:
        DataFrame containing pose data and success flag
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create empty dataframe for pose data
        pose_data = create_empty_pose_dataframe(frame_count, "mediapipe33")
        
        # Setup MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Process frames
        frame_idx = 0
        start_time = time.time()
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_dialog is not None:
                progress = int((frame_idx / frame_count) * 100)
                progress_dialog.setValue(progress)
                
                # Update processing info
                elapsed = time.time() - start_time
                fps_processing = frame_idx / elapsed if elapsed > 0 else 0
                frames_left = frame_count - frame_idx
                time_left = frames_left / fps_processing if fps_processing > 0 else "calculating..."
                
                if isinstance(time_left, float):
                    time_left = f"{time_left:.1f} seconds"
                    
                progress_dialog.setLabelText(
                    f"Processing pose: {frame_idx}/{frame_count} frames "
                    f"({progress}%, {fps_processing:.1f} fps, est. time left: {time_left})"
                )
                
                # Check if user canceled
                if progress_dialog.wasCanceled():
                    cap.release()
                    return None, False
            
            # Convert to RGB (MediaPipe uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(frame_rgb)
            
            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = SUPPORTED_FORMATS["mediapipe33"]
                
                # Map to DataFrame
                for i, kp in enumerate(keypoints):
                    if i < len(landmarks):
                        lm = landmarks[i]
                        x = lm.x * width
                        y = lm.y * height
                        visibility = lm.visibility
                        
                        pose_data.iloc[frame_idx][f"{kp}_X"] = x
                        pose_data.iloc[frame_idx][f"{kp}_Y"] = y
                        pose_data.iloc[frame_idx][f"{kp}_V"] = visibility
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        
        # Return processed data
        return pose_data, True
        
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, False

def process_mediapipe_to_rr21(pose_data):
    """
    Convert MediaPipe pose data to RR21 format.
    
    Args:
        pose_data: DataFrame in MediaPipe format
        
    Returns:
        DataFrame in RR21 format
    """
    
    # Define mapping from MediaPipe to RR21 format
    mapping = {
        'NOSE': 'NOSE',
        'LEFT_EYE': 'LEFT_EYE',
        'RIGHT_EYE': 'RIGHT_EYE',
        'LEFT_EAR': 'LEFT_EAR',
        'RIGHT_EAR': 'RIGHT_EAR',
        'LEFT_SHOULDER': 'LEFT_SHOULDER',
        'RIGHT_SHOULDER': 'RIGHT_SHOULDER',
        'LEFT_ELBOW': 'LEFT_ELBOW',
        'RIGHT_ELBOW': 'RIGHT_ELBOW',
        'LEFT_WRIST': 'LEFT_WRIST',
        'RIGHT_WRIST': 'RIGHT_WRIST',
        'LEFT_HIP': 'LEFT_HIP',
        'RIGHT_HIP': 'RIGHT_HIP',
        'LEFT_KNEE': 'LEFT_KNEE',
        'RIGHT_KNEE': 'RIGHT_KNEE',
        'LEFT_ANKLE': 'LEFT_ANKLE',
        'RIGHT_ANKLE': 'RIGHT_ANKLE',
        'LEFT_HEEL': 'LEFT_HEEL',
        'RIGHT_HEEL': 'RIGHT_HEEL',
        'LEFT_FOOT_INDEX': 'LEFT_FOOT_INDEX',
        'RIGHT_FOOT_INDEX': 'RIGHT_FOOT_INDEX'
    }
    
    # Create RR21 DataFrame
    num_frames = len(pose_data)
    rr21_data = create_empty_pose_dataframe(num_frames, "rr21")
    
    # Copy data from MediaPipe format to RR21
    for mp_kp, rr_kp in mapping.items():
        x_col_mp = f"{mp_kp}_X"
        y_col_mp = f"{mp_kp}_Y"
        v_col_mp = f"{mp_kp}_V"
        
        x_col_rr = f"{rr_kp}_X"
        y_col_rr = f"{rr_kp}_Y"
        v_col_rr = f"{rr_kp}_V"
        
        if x_col_mp in pose_data.columns and x_col_rr in rr21_data.columns:
            rr21_data[x_col_rr] = pose_data[x_col_mp]
            rr21_data[y_col_rr] = pose_data[y_col_mp]
            rr21_data[v_col_rr] = pose_data[v_col_mp]
    
    return rr21_data