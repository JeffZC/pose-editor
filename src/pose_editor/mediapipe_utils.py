import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from .pose_format_utils import process_mediapipe_to_rr21
from .body_format import SUPPORTED_FORMATS, create_empty_pose_dataframe

# Initialize MediaPipe Pose and other components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_pose_landmarks_from_frame(frame, model_complexity=1):
    """
    Process a single frame with MediaPipe Pose and return the landmarks.
    
    Args:
        frame: Image frame to process
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        
    Returns:
        Tuple of (landmarks_list, annotated_frame)
    """
    # Ensure we're working with a copy of the frame
    image = frame.copy()
    
    # Convert BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Pose with specified complexity
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=0.5) as pose:
        
        # Process the image
        results = pose.process(image_rgb)
        
        # Get landmark coordinates
        landmarks_list = None
        if results.pose_landmarks:
            # Draw the pose annotations on the image
            annotated_frame = image.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Get the landmark coordinates
            landmarks_list = results.pose_landmarks
            
            return landmarks_list, annotated_frame
        
        # No pose detected
        return None, image

def process_video_with_mediapipe(video_path, progress_dialog=None, model_complexity=1):
    """
    Process a video with MediaPipe Pose and return a DataFrame with pose data.
    
    Args:
        video_path: Path to the video file
        progress_dialog: QProgressDialog for UI feedback (optional)
        model_complexity: MediaPipe model complexity (0, 1, or 2)
        
    Returns:
        Tuple of (pose_data_DataFrame, success_boolean)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, False
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a DataFrame to store the pose data in RR21 format
    pose_data_columns = []
    for name in SUPPORTED_FORMATS["rr21"]:
        pose_data_columns.extend([f'{name}_X', f'{name}_Y'])
    
    pose_data = pd.DataFrame(np.zeros((total_frames, len(pose_data_columns))), columns=pose_data_columns)
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        
        # Process each frame
        current_frame = 0
        while current_frame < total_frames:
            # Check for user cancellation
            if progress_dialog and progress_dialog.wasCanceled():
                cap.release()
                return None, False
            
            # Update progress
            if progress_dialog:
                progress_value = int((current_frame / total_frames) * 100)
                progress_dialog.setValue(progress_value)
            
            # Read the next frame
            success, frame = cap.read()
            if not success:
                break
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = pose.process(frame_rgb)
            
            # If landmarks detected, update pose data
            if results.pose_landmarks:
                # Convert to RR21 format
                rr21_landmarks = process_mediapipe_to_rr21(results.pose_landmarks)
                
                # Scale normalized coordinates to pixel coordinates
                for i in range(0, len(rr21_landmarks), 2):
                    # X coordinate (even indices)
                    rr21_landmarks[i] = rr21_landmarks[i] * frame_width
                    
                    # Y coordinate (odd indices)
                    if i + 1 < len(rr21_landmarks):
                        rr21_landmarks[i + 1] = rr21_landmarks[i + 1] * frame_height
                
                # Update pose data for current frame
                for i in range(0, len(rr21_landmarks), 2):
                    if i//2 < len(pose_data.columns)//2:
                        pose_data.iloc[current_frame, i] = rr21_landmarks[i]
                        pose_data.iloc[current_frame, i+1] = rr21_landmarks[i+1]
            
            # Move to next frame
            current_frame += 1
    
    # Release the video
    cap.release()
    
    # Final progress update
    if progress_dialog:
        progress_dialog.setValue(100)
    
    return pose_data, True