import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Dict, List, Optional, Union
import time

# MediaPipe hand landmark names
HAND_LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_FINGER_MCP", "PINKY_FINGER_PIP", "PINKY_FINGER_DIP", "PINKY_FINGER_TIP"
]

def extract_hand_landmarks(results: Dict, image_width: int, image_height: int) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Extracts hand landmarks from MediaPipe results and converts to image coordinates.
    
    Args:
        results: The MediaPipe hand detection results
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        Tuple of (left hand landmarks, right hand landmarks)
    """
    left_hand = None
    right_hand = None
    
    # Check if hands were detected
    if results.multi_hand_landmarks:
        # Process each detected hand
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine hand type (left or right)
            if hand_idx < len(results.multi_handedness):
                handedness = results.multi_handedness[hand_idx]
                is_left = handedness.classification[0].label.lower() == "left"
                
                # Extract and scale landmarks to image coordinates
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * image_width
                    y = landmark.y * image_height
                    landmarks_list.append(x)
                    landmarks_list.append(y)
                
                # Assign to correct hand
                if is_left:
                    left_hand = landmarks_list
                else:
                    right_hand = landmarks_list
    
    return left_hand, right_hand

def create_hand_dataframe(total_frames: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates empty DataFrames for storing hand landmark data.
    
    Args:
        total_frames: Number of frames in the video
        
    Returns:
        Tuple of (left hand DataFrame, right hand DataFrame)
    """
    # Create columns for hand landmarks (x and y coordinates)
    x_columns = [f'Hand_{i}_X' for i in range(21)]
    y_columns = [f'Hand_{i}_Y' for i in range(21)]
    
    # Create DataFrames filled with zeros
    left_hand_data = pd.DataFrame(np.zeros((total_frames, len(x_columns) + len(y_columns))), 
                                 columns=x_columns + y_columns)
    right_hand_data = pd.DataFrame(np.zeros((total_frames, len(x_columns) + len(y_columns))), 
                                  columns=x_columns + y_columns)
    
    return left_hand_data, right_hand_data

def update_hand_dataframe(
    hand_data: pd.DataFrame, 
    frame_idx: int, 
    hand_landmarks: List[float], 
    is_left: bool
) -> None:
    """
    Updates hand DataFrame with detected landmarks for a specific frame.
    
    Args:
        hand_data: DataFrame to update
        frame_idx: Frame index
        hand_landmarks: List of hand landmark coordinates [x1, y1, x2, y2, ...]
        is_left: Whether this is the left hand
    """
    if hand_landmarks is None:
        return
        
    # Process landmark coordinates (x and y values)
    for i in range(0, len(hand_landmarks), 2):
        if i//2 < 21:  # 21 landmarks for each hand
            # Update X coordinate
            hand_data.iloc[frame_idx, i//2] = hand_landmarks[i]
            # Update Y coordinate (offset by 21 columns)
            hand_data.iloc[frame_idx, i//2 + 21] = hand_landmarks[i+1]

def draw_hand_landmarks(
    image: np.ndarray, 
    hand_data: pd.DataFrame, 
    frame_idx: int,
    selected_point: Optional[int] = None,
    hovered_point: Optional[int] = None,
    is_left: bool = True
) -> np.ndarray:
    """
    Draws hand landmarks on an image.
    
    Args:
        image: Image to draw on
        hand_data: DataFrame containing hand landmark data
        frame_idx: Current frame index
        selected_point: Index of selected landmark
        hovered_point: Index of hovered landmark
        is_left: Whether this is the left hand
        
    Returns:
        Image with drawn landmarks
    """
    try:
        # Get data for current frame
        hand_x = hand_data.iloc[frame_idx, :21].values
        hand_y = hand_data.iloc[frame_idx, 21:].values
        
        # Default color for landmarks
        default_color = (0, 255, 0) if is_left else (0, 255, 255)  # Green for left, Yellow for right
        connection_color = default_color
        
        # Draw connections between landmarks
        mp_hands = mp.solutions.hands
        for connection in mp_hands.HAND_CONNECTIONS:
            idx1, idx2 = connection
            x1, y1 = int(hand_x[idx1]), int(hand_y[idx1])
            x2, y2 = int(hand_x[idx2]), int(hand_y[idx2])
            # Check if points are valid (non-zero)
            if (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
                cv2.line(image, (x1, y1), (x2, y2), connection_color, 2)
        
        # Draw each landmark
        for i in range(21):
            x, y = int(hand_x[i]), int(hand_y[i])
            if x > 0 or y > 0:  # Only draw if point is valid
                radius = 8 if i == selected_point else 5
                if i == selected_point:
                    color = (255, 0, 0)  # Blue (selected)
                elif i == hovered_point:
                    color = (0, 255, 255)  # Yellow (hovered)
                else:
                    color = default_color
                cv2.circle(image, (x, y), radius, color, -1)
    except Exception as e:
        print(f"Error drawing hand: {e}")
    
    return image

def save_hand_data(
    left_hand_data: Optional[pd.DataFrame], 
    right_hand_data: Optional[pd.DataFrame], 
    file_path: str
) -> bool:
    """
    Saves hand landmark data to a CSV file.
    
    Args:
        left_hand_data: DataFrame for left hand landmarks
        right_hand_data: DataFrame for right hand landmarks
        file_path: Path to save the file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if left_hand_data is None and right_hand_data is None:
            return False
            
        # Prepare data for saving
        if left_hand_data is not None and right_hand_data is not None:
            # Rename columns to differentiate between left and right hands
            left_columns = {col: f'Left_{col}' for col in left_hand_data.columns}
            right_columns = {col: f'Right_{col}' for col in right_hand_data.columns}
            
            left_hand_data = left_hand_data.rename(columns=left_columns)
            right_hand_data = right_hand_data.rename(columns=right_columns)
            
            # Combine data
            combined_data = pd.concat([left_hand_data, right_hand_data], axis=1)
            combined_data.to_csv(file_path, index=False)
        elif left_hand_data is not None:
            # Rename columns to indicate left hand
            left_columns = {col: f'Left_{col}' for col in left_hand_data.columns}
            left_hand_data = left_hand_data.rename(columns=left_columns)
            left_hand_data.to_csv(file_path, index=False)
        else:
            # Rename columns to indicate right hand
            right_columns = {col: f'Right_{col}' for col in right_hand_data.columns}
            right_hand_data = right_hand_data.rename(columns=right_columns)
            right_hand_data.to_csv(file_path, index=False)
            
        return True
    except Exception as e:
        print(f"Error saving hand data: {e}")
        return False

def load_hand_data(file_path: str, expected_frame_count: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool, str]:
    """
    Loads hand landmark data from a CSV file.
    
    Args:
        file_path: Path to the file
        expected_frame_count: Expected number of frames
        
    Returns:
        Tuple containing left hand data, right hand data, success flag, and message
    """
    try:
        # Read CSV file
        data = pd.read_csv(file_path)
        
        # Check if the data contains hand landmarks
        left_hand_columns = [col for col in data.columns if col.startswith('Left_Hand_')]
        right_hand_columns = [col for col in data.columns if col.startswith('Right_Hand_')]
        
        left_hand_data = None
        right_hand_data = None
        
        # Extract left hand data if available
        if left_hand_columns:
            # Rename columns back to standard format for internal processing
            left_data = data[left_hand_columns]
            rename_dict = {col: col.replace('Left_', '') for col in left_hand_columns}
            left_hand_data = left_data.rename(columns=rename_dict)
        
        # Extract right hand data if available
        if right_hand_columns:
            # Rename columns back to standard format for internal processing
            right_data = data[right_hand_columns]
            rename_dict = {col: col.replace('Right_', '') for col in right_hand_columns}
            right_hand_data = right_data.rename(columns=rename_dict)
        
        if not left_hand_data and not right_hand_data:
            return None, None, False, "No hand landmark data found in the file"
            
        # Check frame count if expected_frame_count is provided
        if expected_frame_count is not None:
            actual_frame_count = len(data)
            
            if actual_frame_count != expected_frame_count:
                return left_hand_data, right_hand_data, False, f"Frame count mismatch: expected {expected_frame_count}, got {actual_frame_count}"
        
        return left_hand_data, right_hand_data, True, "Hand data loaded successfully"
        
    except Exception as e:
        return None, None, False, f"Error loading hand data: {e}"

def process_video_with_mediapipe_hands(
    video_path: str,
    progress_dialog=None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
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
            if progress_dialog:
                progress = int((frame_idx / frame_count) * 100)
                progress_dialog.setValue(progress)
                
                # Update processing speed info
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                frames_left = frame_count - frame_idx
                time_left = frames_left / fps if fps > 0 else "calculating..."
                
                if isinstance(time_left, float):
                    time_left = f"{time_left:.1f} seconds"
                    
                progress_dialog.setLabelText(
                    f"Processing hand landmarks: {frame_idx}/{frame_count} frames "
                    f"({progress}%, {fps:.1f} fps, est. time left: {time_left})"
                )
                
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

def get_hand_landmarks_from_frame(
    frame: np.ndarray,
    min_detection_confidence: float = 0.5
) -> Tuple[Optional[List[float]], Optional[List[float]], np.ndarray]:
    """
    Extract hand landmarks from a single frame.
    
    Args:
        frame: Input frame
        min_detection_confidence: MediaPipe detection confidence threshold
        
    Returns:
        Tuple containing left hand landmarks, right hand landmarks, and annotated frame
    """
    try:
        # Initialize MediaPipe hands solution
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence
        )
        mp_drawing = mp.solutions.drawing_utils
        
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(frame_rgb)
        
        # Initialize results
        left_hand = None
        right_hand = None
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Extract hand landmarks if detected
        if results.multi_hand_landmarks:
            # Draw landmarks on annotated frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
            # Extract landmarks
            left_hand, right_hand = extract_hand_landmarks(results, width, height)
        
        return left_hand, right_hand, annotated_frame
        
    except Exception as e:
        print(f"Error getting hand landmarks from frame: {e}")
        return None, None, frame