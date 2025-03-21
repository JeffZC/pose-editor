import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pose_format_utils import process_mediapipe_to_rr21, SUPPORTED_FORMATS

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_pose_landmarks_from_frame(frame):
    """
    Detect pose landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
    
    Returns:
        tuple: (landmarks_list as flat [x1,y1,x2,y2...], annotated_frame)
    """
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return [], frame
        
        # Create a copy for annotations
        annotated_frame = frame.copy()
        
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Extract landmarks as flat list [x1,y1,x2,y2...]
        landmarks_list = []
        h, w, _ = frame.shape
        for landmark in results.pose_landmarks.landmark:
            # Normalize coordinates to image dimensions
            landmarks_list.append(landmark.x * w)
            landmarks_list.append(landmark.y * h)
        
        return landmarks_list, annotated_frame

def process_video_with_mediapipe(video_path, progress_dialog=None):
    """
    Process an entire video with MediaPipe pose detection
    
    Args:
        video_path: Path to the video file
        progress_dialog: Optional PyQt progress dialog
    
    Returns:
        tuple: (DataFrame with pose data in RR21 format, success flag)
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize pose detector
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            # Create empty DataFrame for RR21 format
            column_names = []
            for name in SUPPORTED_FORMATS["rr21"]:
                column_names.extend([f'{name}_X', f'{name}_Y'])
            
            # Initialize with zeros
            pose_data = pd.DataFrame(np.zeros((frame_count, len(column_names))), columns=column_names)
            
            # Process frames
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                if progress_dialog is not None:
                    progress_percent = min(100, int((frame_idx / frame_count) * 100))
                    progress_dialog.setValue(progress_percent)
                    
                    # Handle cancel button
                    if progress_dialog.wasCanceled():
                        break
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                # Extract landmarks if detected
                if results.pose_landmarks:
                    landmarks_list = []
                    h, w, _ = frame.shape
                    for landmark in results.pose_landmarks.landmark:
                        landmarks_list.append(landmark.x * w)
                        landmarks_list.append(landmark.y * h)
                    
                    # Convert to RR21 format
                    rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
                    
                    # Update data
                    for i in range(0, len(rr21_landmarks), 2):
                        if i+1 < len(rr21_landmarks) and i//2 < len(column_names)//2:
                            pose_data.iloc[frame_idx, i] = rr21_landmarks[i]
                            pose_data.iloc[frame_idx, i+1] = rr21_landmarks[i+1]
                
                frame_idx += 1
            
            # Clean up
            cap.release()
            
            return pose_data, True
            
    except Exception as e:
        print(f"Error processing video: {e}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, False