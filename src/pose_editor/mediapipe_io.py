import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from .body_format import SUPPORTED_FORMATS, create_empty_pose_dataframe

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
        model_complexity: MediaPipe pose model complexity (0=small, 1=medium, 2=large)
    
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
        
        # Extract both coordinates and visibility
        for landmark in results.pose_landmarks.landmark:
            # Normalize coordinates to image dimensions
            landmarks_list.append(landmark.x * w)
            landmarks_list.append(landmark.y * h)
            landmarks_list.append(landmark.visibility)
        
        # Convert to RR21 format with proper scaling
        rr21_landmarks = process_mediapipe_to_rr21(landmarks_list, frame_width=w, frame_height=h)
        
        return rr21_landmarks, annotated_frame

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
        model_complexity: MediaPipe pose model complexity (0=small, 1=medium, 2=large)
    
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
            
            # Create empty DataFrame for MediaPipe format
            mediapipe_keypoints = [
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
            
            # Create columns for each keypoint
            column_names = []
            for name in mediapipe_keypoints:
                column_names.extend([f'{name}_X', f'{name}_Y', f'{name}_V'])
            
            # Initialize with zeros
            mp_data = pd.DataFrame(np.zeros((frame_count, len(column_names))), columns=column_names)
            
            # Process frames
            frame_idx = 0
            pose_detected = False
            
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
                    pose_detected = True
                    h, w, _ = frame.shape
                    
                    # Update MediaPipe data
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        if i < len(mediapipe_keypoints):
                            kp_name = mediapipe_keypoints[i]
                            x_col = f"{kp_name}_X"
                            y_col = f"{kp_name}_Y"
                            v_col = f"{kp_name}_V"
                            
                            mp_data.loc[frame_idx, x_col] = landmark.x * w
                            mp_data.loc[frame_idx, y_col] = landmark.y * h
                            mp_data.loc[frame_idx, v_col] = landmark.visibility
                
                frame_idx += 1
            
            # Clean up
            cap.release()
            
            if not pose_detected:
                return None, False
            
            # Convert to RR21 format after processing all frames
            rr21_data = process_mediapipe_to_rr21(mp_data, frame_width=w, frame_height=h)
            
            return rr21_data, True
            
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
    
def process_mediapipe_to_rr21(input_data, frame_width=None, frame_height=None):
    """
    Convert MediaPipe pose data to RR21 format.
    
    Args:
        input_data: Either a flat list of landmarks from single frame detection
                    or a DataFrame in MediaPipe format from video processing
        frame_width: Width of the video frame for coordinate conversion (optional)
        frame_height: Height of the video frame for coordinate conversion (optional)
        
    Returns:
        Either a flat list of landmarks in RR21 format or a DataFrame in RR21 format
    """
    # Check if input is a flat list (from single frame detection)
    if isinstance(input_data, list):
        # Define mapping from MediaPipe33 indices to RR21 indices
        # Format: mediapipe_idx: rr21_idx
        mapping = {
            0: 0,   # NOSE -> NOSE
            2: 1,   # LEFT_EYE -> LEFT_EYE
            5: 2,   # RIGHT_EYE -> RIGHT_EYE
            7: 3,   # LEFT_EAR -> LEFT_EAR
            8: 4,   # RIGHT_EAR -> RIGHT_EAR
            11: 5,  # LEFT_SHOULDER -> LEFT_SHOULDER
            12: 6,  # RIGHT_SHOULDER -> RIGHT_SHOULDER
            13: 7,  # LEFT_ELBOW -> LEFT_ELBOW
            14: 8,  # RIGHT_ELBOW -> RIGHT_ELBOW
            15: 9,  # LEFT_WRIST -> LEFT_WRIST
            16: 10, # RIGHT_WRIST -> RIGHT_WRIST
            23: 11, # LEFT_HIP -> LEFT_HIP
            24: 12, # RIGHT_HIP -> RIGHT_HIP
            25: 13, # LEFT_KNEE -> LEFT_KNEE
            26: 14, # RIGHT_KNEE -> RIGHT_KNEE
            27: 15, # LEFT_ANKLE -> LEFT_ANKLE
            28: 16, # RIGHT_ANKLE -> RIGHT_ANKLE
            29: 17, # LEFT_HEEL -> LEFT_HEEL
            30: 18, # RIGHT_HEEL -> RIGHT_HEEL
            31: 19, # LEFT_FOOT_INDEX -> LEFT_FOOT
            32: 20  # RIGHT_FOOT_INDEX -> RIGHT_FOOT
        }
        
        # Create RR21 landmarks array (x, y, visibility) for 21 keypoints
        rr21_landmarks = np.zeros(21 * 3)
        
        # Check if we have a flat list with 33 landmarks (x,y,z,visibility) = 132 values
        # or 33 landmarks (x,y,visibility) = 99 values
        landmarks_per_point = 3  # Default: (x, y, visibility)
        if len(input_data) >= 132:  # Full 3D landmarks
            landmarks_per_point = 4  # (x, y, z, visibility)
        
        # Copy coordinates from MediaPipe to RR21
        for mp_idx, rr_idx in mapping.items():
            # Calculate source indices in the flat list
            src_x_idx = mp_idx * landmarks_per_point
            src_y_idx = mp_idx * landmarks_per_point + 1
            src_v_idx = mp_idx * landmarks_per_point + (landmarks_per_point - 1)  # Visibility is last
            
            # Calculate destination indices in the RR21 array
            dst_x_idx = rr_idx * 3
            dst_y_idx = rr_idx * 3 + 1
            dst_v_idx = rr_idx * 3 + 2
            
            # Copy coordinates and visibility
            if src_x_idx + landmarks_per_point <= len(input_data):
                # MediaPipe returns normalized coordinates [0..1], convert to pixel coordinates
                if frame_width is not None and frame_height is not None:
                    rr21_landmarks[dst_x_idx] = input_data[src_x_idx] * frame_width
                    rr21_landmarks[dst_y_idx] = input_data[src_y_idx] * frame_height
                else:
                    rr21_landmarks[dst_x_idx] = input_data[src_x_idx]
                    rr21_landmarks[dst_y_idx] = input_data[src_y_idx]
                
                rr21_landmarks[dst_v_idx] = input_data[src_v_idx]
        
        return rr21_landmarks.tolist()
        
    # Otherwise, the input is a DataFrame (from video processing)
    else:
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
            'LEFT_FOOT_INDEX': 'LEFT_FOOT',
            'RIGHT_FOOT_INDEX': 'RIGHT_FOOT'
        }
        
        # Create RR21 DataFrame
        num_frames = len(input_data)
        rr21_data = create_empty_pose_dataframe(num_frames, "rr21")
        
        # Copy data from MediaPipe format to RR21
        for mp_kp, rr_kp in mapping.items():
            x_col_mp = f"{mp_kp}_X"
            y_col_mp = f"{mp_kp}_Y"
            v_col_mp = f"{mp_kp}_V"
            
            x_col_rr = f"{rr_kp}_X"
            y_col_rr = f"{rr_kp}_Y"
            v_col_rr = f"{rr_kp}_V"
            
            if x_col_mp in input_data.columns and x_col_rr in rr21_data.columns:
                # Convert coordinates if frame dimensions are provided
                if frame_width is not None and frame_height is not None:
                    # MediaPipe returns normalized coordinates [0..1], convert to pixel coordinates
                    rr21_data[x_col_rr] = input_data[x_col_mp] * frame_width
                    rr21_data[y_col_rr] = input_data[y_col_mp] * frame_height
                else:
                    rr21_data[x_col_rr] = input_data[x_col_mp]
                    rr21_data[y_col_rr] = input_data[y_col_mp]
                    
                if v_col_mp in input_data.columns and v_col_rr in rr21_data.columns:
                    rr21_data[v_col_rr] = input_data[v_col_mp]
        
        return rr21_data

def convert_list_to_dataframe(landmarks_list, format_name="mediapipe33"):
    """
    Convert a list of landmarks to a DataFrame

    Args:
        landmarks_list: List of landmarks [x1, y1, x2, y2, ...]
        format_name: Format name for column creation

    Returns:
        pandas.DataFrame: DataFrame with landmark coordinates
    """
    # Ensure we have landmarks data
    if landmarks_list and len(landmarks_list) > 0:
        # Create column names based on the format
        columns = []
        if format_name == "mediapipe33":
            # MediaPipe has 33 landmarks with x, y, and visibility
            for name in SUPPORTED_FORMATS.get(format_name, []):
                columns.extend([f'{name}_X', f'{name}_Y', f'{name}_V'])
        elif format_name == "rr21":
            # RR21 has 21 landmarks with x, y
            for name in SUPPORTED_FORMATS.get(format_name, []):
                columns.extend([f'{name}_X', f'{name}_Y'])
        else:
            # Generic format
            num_landmarks = len(landmarks_list) // 2
            columns = []
            for i in range(num_landmarks):
                columns.extend([f'point{i}_X', f'point{i}_Y'])

        # Prepare data based on what we received
        data = []

        # Check whether we have visibility data
        # For MediaPipe, we expect triplets (x, y, visibility)
        step = 3 if format_name == "mediapipe33" and len(landmarks_list) % 3 == 0 else 2

        for i in range(0, len(landmarks_list), step):
            if i+step-1 < len(landmarks_list):
                # Add coordinates
                for j in range(step):
                    if i+j < len(landmarks_list):
                        data.append(landmarks_list[i+j])

        # Create DataFrame with the data
        df = pd.DataFrame([data], columns=columns[:len(data)])
        return df
    else:
        # Create empty DataFrame with correct columns
        columns = []
        if format_name == "mediapipe33":
            for name in SUPPORTED_FORMATS.get(format_name, []):
                columns.extend([f'{name}_X', f'{name}_Y', f'{name}_V'])
        elif format_name == "rr21":
            for name in SUPPORTED_FORMATS.get(format_name, []):
                columns.extend([f'{name}_X', f'{name}_Y'])

        df = pd.DataFrame(columns=columns)
        return df

def process_detection_result(result):
    """
    Process the detection result to ensure it's a proper DataFrame

    Args:
        result: Detection result (DataFrame or list)

    Returns:
        pandas.DataFrame: Properly formatted pose data
    """
    # Check if result is already a DataFrame
    if isinstance(result, pd.DataFrame):
        return result

    # If result is a list, convert to DataFrame
    if isinstance(result, list):
        # Assuming the list comes from mediapipe single frame detection
        return convert_list_to_dataframe(result, format_name="mediapipe33")

    # If neither, create an empty DataFrame based on mediapipe33 format
    columns = []
    if "mediapipe33" in SUPPORTED_FORMATS:
        for kp in SUPPORTED_FORMATS["mediapipe33"]:
            columns.append(f"{kp}_X")
            columns.append(f"{kp}_Y")
            columns.append(f"{kp}_V")

    return pd.DataFrame(columns=columns)