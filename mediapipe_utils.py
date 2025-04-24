import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pose_format_utils import process_mediapipe_to_rr21, SUPPORTED_FORMATS

# Initialize MediaPipe Pose and other components
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_pose_landmarks_from_frame(frame, model_complexity=2):
    """
    Detect pose landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
        model_complexity: MediaPipe pose model complexity (1=small, 2=large)
    
    Returns:
        tuple: (landmarks_list as flat [x1,y1,x2,y2...], annotated_frame)
    """
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
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

def get_hand_landmarks_from_frame(frame):
    """
    Detect hand landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
    
    Returns:
        tuple: (left_hand_landmarks, right_hand_landmarks, annotated_frame)
    """
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return [], [], frame
        
        # Create a copy for annotations
        annotated_frame = frame.copy()
        
        # Draw hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        
        # Extract landmarks for left and right hands
        left_landmarks = []
        right_landmarks = []
        h, w, _ = frame.shape
        
        # Process each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine if left or right hand
            handedness = results.multi_handedness[idx].classification[0].label
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x * w)
                landmarks.append(landmark.y * h)
            
            # Assign to correct hand
            if handedness == "Left":
                left_landmarks = landmarks
            else:
                right_landmarks = landmarks
        
        return left_landmarks, right_landmarks, annotated_frame

def get_face_landmarks_from_frame(frame):
    """
    Detect face landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
    
    Returns:
        tuple: (face_landmarks, annotated_frame)
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.5) as face_mesh:
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return [], frame
        
        # Create a copy for annotations
        annotated_frame = frame.copy()
        
        # Draw face landmarks
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        
        # Extract landmarks
        landmarks = []
        h, w, _ = frame.shape
        
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append(landmark.x * w)
            landmarks.append(landmark.y * h)
        
        return landmarks, annotated_frame

def process_video_with_mediapipe(video_path, progress_dialog=None, model_complexity=1):
    """
    Process an entire video with MediaPipe pose detection
    
    Args:
        video_path: Path to the video file
        progress_dialog: Optional PyQt progress dialog
        model_complexity: MediaPipe pose model complexity (1=small, 2=large)
    
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
            model_complexity=model_complexity,
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

def process_video_with_mediapipe_hands(video_path, progress_dialog=None):
    """
    Process entire video with MediaPipe hands detection
    
    Args:
        video_path: Path to the video file
        progress_dialog: Optional PyQt progress dialog
    
    Returns:
        tuple: (left_hand_data, right_hand_data, success)
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize MediaPipe Hands
        # Create empty DataFrames for hands data
        left_columns = [f'HandL_{i}_X' for i in range(21)] + [f'HandL_{i}_Y' for i in range(21)]
        right_columns = [f'HandR_{i}_X' for i in range(21)] + [f'HandR_{i}_Y' for i in range(21)]
        
        left_hand_data = pd.DataFrame(np.zeros((frame_count, len(left_columns))), columns=left_columns)
        right_hand_data = pd.DataFrame(np.zeros((frame_count, len(right_columns))), columns=right_columns)
        
        # Track if we've detected each hand at least once
        left_detected = False
        right_detected = False
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            
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
                results = hands.process(frame_rgb)
                
                # Extract landmarks if detected
                if results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    
                    # Process each detected hand
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Determine if left or right hand
                        handedness = results.multi_handedness[idx].classification[0].label
                        
                        # Extract coordinates
                        landmarks = []
                        for landmark in hand_landmarks.landmark:
                            landmarks.append(landmark.x * w)
                            landmarks.append(landmark.y * h)
                        
                        # Update appropriate DataFrame
                        if handedness == "Left":
                            left_detected = True
                            # Update left hand data
                            for i in range(21):
                                if i*2+1 < len(landmarks):
                                    left_hand_data.iloc[frame_idx, i] = landmarks[i*2]        # X
                                    left_hand_data.iloc[frame_idx, i+21] = landmarks[i*2+1]   # Y
                        else:  # Right hand
                            right_detected = True
                            # Update right hand data
                            for i in range(21):
                                if i*2+1 < len(landmarks):
                                    right_hand_data.iloc[frame_idx, i] = landmarks[i*2]       # X
                                    right_hand_data.iloc[frame_idx, i+21] = landmarks[i*2+1]  # Y
                
                frame_idx += 1
            
            # Clean up
            cap.release()
            
            # Only return data for hands that were detected at least once
            left_result = left_hand_data if left_detected else None
            right_result = right_hand_data if right_detected else None
            
            return left_result, right_result, True
            
    except Exception as e:
        print(f"Error processing video for hands: {e}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, None, False

def process_video_with_mediapipe_face(video_path, progress_dialog=None):
    """
    Process entire video with MediaPipe face detection
    
    Args:
        video_path: Path to the video file
        progress_dialog: Optional PyQt progress dialog
    
    Returns:
        tuple: (face_data, success)
    """
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize MediaPipe Face Mesh
        # Number of landmarks in MediaPipe Face Mesh
        num_face_landmarks = 468
        
        # Create columns for face landmarks
        face_columns = [f'Face_{i}_X' for i in range(num_face_landmarks)] + [f'Face_{i}_Y' for i in range(num_face_landmarks)]
        
        # Create empty DataFrame
        face_data = pd.DataFrame(np.zeros((frame_count, len(face_columns))), columns=face_columns)
        
        # Track if face was detected at least once
        face_detected = False
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            
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
                results = face_mesh.process(frame_rgb)
                
                # Extract landmarks if detected
                if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                    face_detected = True
                    h, w, _ = frame.shape
                    
                    # Get first face (we're only tracking one)
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # Update face data
                    for i, landmark in enumerate(face_landmarks.landmark):
                        if i < num_face_landmarks:
                            face_data.iloc[frame_idx, i] = landmark.x * w               # X
                            face_data.iloc[frame_idx, i+num_face_landmarks] = landmark.y * h  # Y
                
                frame_idx += 1
            
            # Clean up
            cap.release()
            
            # Only return data if face was detected at least once
            result = face_data if face_detected else None
            
            return result, True
            
    except Exception as e:
        print(f"Error processing video for face: {e}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, False