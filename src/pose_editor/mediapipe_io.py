import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
from .body_keypoints import SUPPORTED_FORMATS, create_empty_pose_keypoints, process_mediapipe_to_rr21

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
        model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
    
    Returns:
        tuple: (landmarks_list, annotated_frame)
    """
    image = frame.copy()
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=0.5) as pose:
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            # No pose detected
            return None, image
        
        # Create annotated image for visualization
        annotated_frame = frame.copy()
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # Extract landmarks as flat list [x1,y1,v1,x2,y2,v2...]
        landmarks_list = []
        h, w, _ = frame.shape
        
        # Extract coordinates and visibility
        for landmark in results.pose_landmarks.landmark:
            landmarks_list.append(landmark.x)
            landmarks_list.append(landmark.y)
            landmarks_list.append(landmark.visibility)
        
        # Convert to RR21 format with proper scaling
        rr21_landmarks = process_mediapipe_to_rr21(landmarks_list, frame_width=w, frame_height=h)
        
        return rr21_landmarks, annotated_frame

def get_hand_landmarks_from_frame(frame, min_detection_confidence=0.5):
    """
    Detect hand landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
        min_detection_confidence: Detection confidence threshold
    
    Returns:
        tuple: (left_hand_landmarks, right_hand_landmarks, annotated_frame)
    """
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence) as hands:
        
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

def get_face_landmarks_from_frame(frame, min_detection_confidence=0.5):
    """
    Detect face landmarks for a single frame using MediaPipe
    
    Args:
        frame: OpenCV frame (BGR)
        min_detection_confidence: Detection confidence threshold
    
    Returns:
        tuple: (face_landmarks, annotated_frame)
    """
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence) as face_mesh:
        
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
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
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

def process_frame_with_mediapipe_all(frame, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Process single frame with Pose, Hands, and Face sequentially."""
    annotated_frame = frame.copy()
    results = {}
    # Pose
    landmarks_list, body_annotated = get_pose_landmarks_from_frame(frame, model_complexity=model_complexity)
    if landmarks_list:
        results['body'] = process_mediapipe_to_rr21(landmarks_list)
        annotated_frame = body_annotated
    # Hands
    left_landmarks, right_landmarks, hand_annotated = get_hand_landmarks_from_frame(frame, min_detection_confidence)
    if left_landmarks or right_landmarks:
        results['left_hand'] = left_landmarks
        results['right_hand'] = right_landmarks
        alpha = 0.5
        if 'body' not in results:
            annotated_frame = hand_annotated
        else:
            annotated_frame = cv2.addWeighted(annotated_frame, alpha, hand_annotated, 1-alpha, 0)
    # Face
    face_landmarks, face_annotated = get_face_landmarks_from_frame(frame, min_detection_confidence)
    if face_landmarks:
        results['face'] = face_landmarks
        alpha = 0.5
        if not any(k in results for k in ['body','left_hand','right_hand']):
            annotated_frame = face_annotated
        else:
            annotated_frame = cv2.addWeighted(annotated_frame, alpha, face_annotated, 1-alpha, 0)
    return results, annotated_frame

def process_video_with_mediapipe(video_path, progress_dialog=None, model_complexity=1):
    """
    Process a video with MediaPipe Pose and extract pose landmarks
    
    Args:
        video_path: Path to the video file
        progress_dialog: Progress dialog for UI feedback
        model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
        
    Returns:
        tuple: (pose_dataframe, success_flag)
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return None, False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create empty DataFrame for RR21 format
        pose_data = create_empty_pose_keypoints(frame_count, "rr21").to_dataframe()
        
        # Initialize MediaPipe pose
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            # Process each frame
            current_frame = 0
            start_time = time.time()
            
            while True:
                # Read frame
                success, frame = cap.read()
                if not success:
                    break
                
                # Update progress dialog
                if progress_dialog is not None:
                    # Calculate progress
                    progress = min(int(100.0 * current_frame / frame_count), 100)
                    
                    # Calculate estimated time remaining
                    elapsed = time.time() - start_time
                    frames_processed = current_frame + 1
                    frames_remaining = frame_count - frames_processed
                    
                    if frames_processed > 0:
                        time_per_frame = elapsed / frames_processed
                        time_left = frames_remaining * time_per_frame
                        
                        # Format as minutes:seconds
                        mins_left = int(time_left // 60)
                        secs_left = int(time_left % 60)
                        
                        current_fps = frames_processed / elapsed if elapsed > 0 else 0
                        
                        progress_dialog.setLabelText(
                            f"Processing video frame {current_frame+1}/{frame_count} "
                            f"({progress}%, {current_fps:.1f} fps, est. time left: {mins_left}m {secs_left}s)"
                        )
                    
                    progress_dialog.setValue(progress)
                    
                    # Check if canceled
                    if progress_dialog.wasCanceled():
                        cap.release()
                        return None, False
                
                # Convert to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the image
                results = pose.process(frame_rgb)
                
                # If landmarks detected, update pose data
                if results.pose_landmarks:
                    # Extract landmarks as flat list for RR21 conversion
                    landmarks_list = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks_list.extend([lm.x, lm.y, lm.visibility])
                    rr21_landmarks = process_mediapipe_to_rr21(landmarks_list, frame_width, frame_height)
                    
                    # Update pose data for current frame - RR21 format has x, y, visibility for each keypoint
                    keypoints = SUPPORTED_FORMATS["rr21"]
                    for i, kp in enumerate(keypoints):
                        if i*3+2 < len(rr21_landmarks):
                            pose_data.loc[current_frame, f'{kp}_X'] = rr21_landmarks[i*3]
                            pose_data.loc[current_frame, f'{kp}_Y'] = rr21_landmarks[i*3+1]
                            pose_data.loc[current_frame, f'{kp}_V'] = rr21_landmarks[i*3+2]
                
                # Move to next frame
                current_frame += 1
        
        # Release the video
        cap.release()
        
        # Final progress update
        if progress_dialog:
            progress_dialog.setValue(100)
        
        return pose_data, True
        
    except Exception as e:
        print(f"Error processing video with MediaPipe: {e}")
        import traceback
        traceback.print_exc()
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, False

def process_video_with_mediapipe_hands(video_path, progress_dialog=None, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """
    Process a video with MediaPipe Hands and extract hand landmarks.
    
    Args:
        video_path: Path to the video file
        progress_dialog: PyQt progress dialog
        min_detection_confidence: MediaPipe detection confidence threshold
        min_tracking_confidence: MediaPipe tracking confidence threshold
        
    Returns:
        Tuple containing left hand data, right hand data, and success flag
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, None, False
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create empty DataFrames for storing hand landmarks
        left_hand_data, right_hand_data = create_hand_dataframe(frame_count)
        
        # Initialize MediaPipe hands solution
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Process each frame
        frame_idx = 0
        left_hand_detected = False
        right_hand_detected = False
        
        start_time = time.time()
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update progress
            if progress_dialog is not None:
                # Calculate progress
                progress = min(int(100.0 * frame_idx / frame_count), 100)
                
                # Calculate FPS and remaining time
                elapsed = time.time() - start_time
                if frame_idx > 0 and elapsed > 0:
                    fps = frame_idx / elapsed
                    frames_left = frame_count - frame_idx - 1
                    time_left = frames_left / fps if fps > 0 else 0
                    
                    # Format as minutes:seconds
                    mins_left = int(time_left // 60)
                    secs_left = int(time_left % 60)
                    
                    progress_dialog.setLabelText(
                        f"Processing hand tracking frame {frame_idx+1}/{frame_count} "
                        f"({progress}%, {fps:.1f} fps, est. time left: {mins_left}m {secs_left}s)"
                    )
                
                progress_dialog.setValue(progress)
                
                # Check if canceled
                if progress_dialog.wasCanceled():
                    cap.release()
                    return None, None, False
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = hands.process(frame_rgb)
            
            # Extract hand landmarks if detected
            if results.multi_hand_landmarks:
                left_hand, right_hand = extract_hand_landmarks(results, width, height)
                
                # Update left hand DataFrame
                if left_hand:
                    update_hand_dataframe(left_hand_data, frame_idx, left_hand, True)
                    left_hand_detected = True
                    
                # Update right hand DataFrame
                if right_hand:
                    update_hand_dataframe(right_hand_data, frame_idx, right_hand, False)
                    right_hand_detected = True
            
            frame_idx += 1
        
        cap.release()
        
        # Return None for hands that were never detected
        left_result = left_hand_data if left_hand_detected else None
        right_result = right_hand_data if right_hand_detected else None
        
        # Return True if processing was completed successfully
        return left_result, right_result, True
        
    except Exception as e:
        print(f"Error processing video with MediaPipe Hands: {e}")
        import traceback
        traceback.print_exc()
        if 'cap' in locals() and cap is not None:
            cap.release()
        return None, None, False

def process_video_with_mediapipe_face(video_path, progress_dialog=None, min_detection_confidence=0.5):
    """
    Process a video with MediaPipe FaceMesh and extract face landmarks.
    
    Args:
        video_path: Path to the video file
        progress_dialog: PyQt progress dialog
        min_detection_confidence: MediaPipe detection confidence threshold
        
    Returns:
        Tuple containing face landmarks dataframe and success flag
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, False
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # MediaPipe tracks 468 face landmarks
        num_face_landmarks = 468
        
        # Create empty DataFrame for face landmarks (468 landmarks x 2 coordinates)
        face_data = pd.DataFrame(index=range(frame_count), 
                                columns=[f'Face_{i}_{coord}' for i in range(num_face_landmarks) for coord in ['X', 'Y']])
        
        # Track if face was detected at least once
        face_detected = False
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence) as face_mesh:
            
            # Process frames
            frame_idx = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update progress
                if progress_dialog is not None:
                    # Calculate progress
                    progress = min(int(100.0 * frame_idx / frame_count), 100)
                    
                    # Calculate FPS and remaining time
                    elapsed = time.time() - start_time
                    if frame_idx > 0 and elapsed > 0:
                        fps = frame_idx / elapsed
                        frames_left = frame_count - frame_idx - 1
                        time_left = frames_left / fps if fps > 0 else 0
                        
                        # Format as minutes:seconds
                        mins_left = int(time_left // 60)
                        secs_left = int(time_left % 60)
                        
                        progress_dialog.setLabelText(
                            f"Processing face tracking frame {frame_idx+1}/{frame_count} "
                            f"({progress}%, {fps:.1f} fps, est. time left: {mins_left}m {secs_left}s)"
                        )
                    
                    progress_dialog.setValue(progress)
                    
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
        import traceback
        traceback.print_exc()
        return None, False

def extract_hand_landmarks(results, image_width, image_height):
    """
    Extract hand landmarks from MediaPipe results
    
    Args:
        results: MediaPipe hands processing results
        image_width: Width of the frame
        image_height: Height of the frame
        
    Returns:
        tuple: (left_hand_landmarks, right_hand_landmarks)
    """
    left_hand = None
    right_hand = None
    
    # Process each detected hand
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
        # Get handedness (left or right)
        handedness = results.multi_handedness[i].classification[0].label
        
        # Extract landmarks as flat list [x1, y1, x2, y2, ...]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * image_width
            y = landmark.y * image_height
            
            landmarks.append(x)
            landmarks.append(y)
        
        # Assign to correct hand
        if handedness == "Left":
            left_hand = landmarks
        else:
            right_hand = landmarks
            
    return left_hand, right_hand

def update_hand_dataframe(df, frame_idx, landmarks, is_left):
    """
    Update hand DataFrame with landmarks
    
    Args:
        df: DataFrame to update
        frame_idx: Frame index
        landmarks: List of landmarks [x1, y1, x2, y2, ...]
        is_left: Whether it's the left hand
    """
    prefix = 'HandL_' if is_left else 'HandR_'
    
    for i in range(21):  # 21 hand landmarks
        if i*2+1 < len(landmarks):
            df.loc[frame_idx, f'{prefix}{i}_X'] = landmarks[i*2]
            df.loc[frame_idx, f'{prefix}{i}_Y'] = landmarks[i*2+1]

def create_hand_dataframe(num_frames):
    """
    Create empty DataFrames for hand landmarks
    
    Args:
        num_frames: Number of frames
        
    Returns:
        tuple: (left_hand_dataframe, right_hand_dataframe)
    """
    left_hand_data = pd.DataFrame(index=range(num_frames), columns=[f'HandL_{i}_{coord}' for i in range(21) for coord in ['X', 'Y']])
    right_hand_data = pd.DataFrame(index=range(num_frames), columns=[f'HandR_{i}_{coord}' for i in range(21) for coord in ['X', 'Y']])
    
    return left_hand_data, right_hand_data

__all__ = [
    "get_pose_landmarks_from_frame",
    "get_hand_landmarks_from_frame",
    "get_face_landmarks_from_frame",
    "process_video_with_mediapipe",
    "process_video_with_mediapipe_hands",
    "process_video_with_mediapipe_face",
    "extract_hand_landmarks",
    "update_hand_dataframe",
    "create_hand_dataframe",
    "process_frame_with_mediapipe_all"
]