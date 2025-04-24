import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple, Dict, List, Optional, Union
import time

# Total number of face mesh landmarks in MediaPipe
FACE_LANDMARK_COUNT = 468

def extract_face_landmarks(results: Dict, image_width: int, image_height: int) -> Optional[List[float]]:
    """
    Extracts face landmarks from MediaPipe results and converts to image coordinates.
    
    Args:
        results: The MediaPipe face detection results
        image_width: Width of the image
        image_height: Height of the image
        
    Returns:
        List of face landmark coordinates [x1, y1, x2, y2, ...]
    """
    if not results.multi_face_landmarks or not results.multi_face_landmarks[0]:
        return None
    
    # Extract face landmarks (take the first face if multiple are detected)
    face_landmarks = results.multi_face_landmarks[0]
    
    # Extract and scale landmarks to image coordinates
    landmarks_list = []
    for landmark in face_landmarks.landmark:
        # Convert normalized coordinates to pixel coordinates
        x = landmark.x * image_width
        y = landmark.y * image_height
        landmarks_list.append(x)
        landmarks_list.append(y)
    
    return landmarks_list

def create_face_dataframe(total_frames: int) -> pd.DataFrame:
    """
    Creates an empty DataFrame for storing face landmark data.
    
    Args:
        total_frames: Number of frames in the video
        
    Returns:
        DataFrame for face landmarks
    """
    # Create columns for 468 face landmarks (x and y coordinates)
    x_columns = [f'Face_{i}_X' for i in range(FACE_LANDMARK_COUNT)]
    y_columns = [f'Face_{i}_Y' for i in range(FACE_LANDMARK_COUNT)]
    
    # Create DataFrame filled with zeros
    face_data = pd.DataFrame(np.zeros((total_frames, len(x_columns) + len(y_columns))), 
                            columns=x_columns + y_columns)
    
    return face_data

def update_face_dataframe(
    face_data: pd.DataFrame, 
    frame_idx: int, 
    face_landmarks: List[float]
) -> None:
    """
    Updates face DataFrame with detected landmarks for a specific frame.
    
    Args:
        face_data: DataFrame to update
        frame_idx: Frame index
        face_landmarks: List of face landmark coordinates [x1, y1, x2, y2, ...]
    """
    if face_landmarks is None:
        return
        
    # Process landmark coordinates (x and y values)
    for i in range(0, len(face_landmarks), 2):
        if i//2 < FACE_LANDMARK_COUNT:
            # Update X coordinate
            face_data.iloc[frame_idx, i//2] = face_landmarks[i]
            # Update Y coordinate (offset by 468 columns)
            face_data.iloc[frame_idx, i//2 + FACE_LANDMARK_COUNT] = face_landmarks[i+1]

def draw_face_landmarks(
    image: np.ndarray, 
    face_data: pd.DataFrame, 
    frame_idx: int,
    selected_point: Optional[int] = None,
    hovered_point: Optional[int] = None,
    draw_connections: bool = False,
    point_size: int = 2
) -> np.ndarray:
    """
    Draws face landmarks on an image.
    
    Args:
        image: Image to draw on
        face_data: DataFrame containing face landmark data
        frame_idx: Current frame index
        selected_point: Index of selected landmark
        hovered_point: Index of hovered landmark
        draw_connections: Whether to draw connections between landmarks
        point_size: Size of the landmark points
        
    Returns:
        Image with drawn landmarks
    """
    try:
        # Get data for current frame
        face_x = face_data.iloc[frame_idx, :FACE_LANDMARK_COUNT].values
        face_y = face_data.iloc[frame_idx, FACE_LANDMARK_COUNT:].values
        
        # Default color for face landmarks
        default_color = (255, 255, 0)  # Yellow
        
        # Draw face mesh connections if requested
        if draw_connections:
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            # Create a temporary landmark proto structure for drawing
            landmark_list = []
            for i in range(FACE_LANDMARK_COUNT):
                x, y = face_x[i], face_y[i]
                if x > 0 or y > 0:  # Only include valid points
                    landmark = {"x": x, "y": y, "z": 0.0}
                    landmark_list.append(landmark)
                    
            # Draw the connections if we have enough valid landmarks
            if len(landmark_list) > 400:  # Threshold for a reasonably complete face
                # This requires more complex setup with MediaPipe drawing utilities
                # For simplicity, we'll skip connections in this implementation
                pass
        
        # Draw each landmark
        for i in range(FACE_LANDMARK_COUNT):
            x, y = int(face_x[i]), int(face_y[i])
            if x > 0 or y > 0:  # Only draw if point is valid
                radius = point_size + 3 if i == selected_point else point_size
                if i == selected_point:
                    color = (255, 0, 0)  # Blue (selected)
                elif i == hovered_point:
                    color = (0, 255, 255)  # Yellow (hovered)
                else:
                    color = default_color
                cv2.circle(image, (x, y), radius, color, -1)
                
                # Draw point ID for selected point
                if i == selected_point:
                    cv2.putText(image, f"ID: {i}", (x + 10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception as e:
        print(f"Error drawing face landmarks: {e}")
    
    return image

def save_face_data(face_data: Optional[pd.DataFrame], file_path: str) -> bool:
    """
    Saves face landmark data to a CSV file.
    
    Args:
        face_data: DataFrame for face landmarks
        file_path: Path to save the file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if face_data is None:
            return False
            
        # Save to CSV
        face_data.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving face data: {e}")
        return False

def load_face_data(file_path: str, expected_frame_count: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], bool, str]:
    """
    Loads face landmark data from a CSV file.
    
    Args:
        file_path: Path to the file
        expected_frame_count: Expected number of frames
        
    Returns:
        Tuple containing face data, success flag, and message
    """
    try:
        # Read CSV file
        data = pd.read_csv(file_path)
        
        # Check if the data contains face landmarks
        face_columns = [col for col in data.columns if col.startswith('Face_')]
        
        if not face_columns:
            return None, False, "No face landmark data found in the file"
            
        # Extract face data
        face_data = data[face_columns]
        
        # Check frame count if expected_frame_count is provided
        if expected_frame_count is not None:
            actual_frame_count = len(data)
            
            if actual_frame_count != expected_frame_count:
                return face_data, False, f"Frame count mismatch: expected {expected_frame_count}, got {actual_frame_count}"
        
        return face_data, True, "Face data loaded successfully"
        
    except Exception as e:
        return None, False, f"Error loading face data: {e}"

def process_video_with_mediapipe_face(
    video_path: str,
    progress_dialog=None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    max_num_faces: int = 1
) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Process a video with MediaPipe Face Mesh and extract face landmarks.
    
    Args:
        video_path: Path to the video file
        progress_dialog: PyQt progress dialog
        min_detection_confidence: MediaPipe detection confidence threshold
        min_tracking_confidence: MediaPipe tracking confidence threshold
        max_num_faces: Maximum number of faces to detect
        
    Returns:
        Tuple containing face data and success flag
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, False
            
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create empty DataFrame for storing face landmarks
        face_data = create_face_dataframe(frame_count)
        
        # Initialize MediaPipe face mesh solution
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Process each frame
        frame_idx = 0
        face_detected = False
        
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
                    f"Processing face landmarks: {frame_idx}/{frame_count} frames "
                    f"({progress}%, {fps:.1f} fps, est. time left: {time_left})"
                )
                
                # Check if canceled
                if progress_dialog.wasCanceled():
                    cap.release()
                    return None, False
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = face_mesh.process(frame_rgb)
            
            # Extract face landmarks if detected
            if results.multi_face_landmarks:
                face_landmarks = extract_face_landmarks(results, width, height)
                
                # Update DataFrame
                if face_landmarks:
                    update_face_dataframe(face_data, frame_idx, face_landmarks)
                    face_detected = True
            
            frame_idx += 1
        
        cap.release()
        
        # Return None if face was never detected
        if not face_detected:
            return None, True
        
        return face_data, True
        
    except Exception as e:
        print(f"Error processing video with MediaPipe Face Mesh: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def get_face_landmarks_from_frame(
    frame: np.ndarray,
    min_detection_confidence: float = 0.5,
    max_num_faces: int = 1
) -> Tuple[Optional[List[float]], np.ndarray]:
    """
    Extract face landmarks from a single frame.
    
    Args:
        frame: Input frame
        min_detection_confidence: MediaPipe detection confidence threshold
        max_num_faces: Maximum number of faces to detect
        
    Returns:
        Tuple containing face landmarks and annotated frame
    """
    try:
        # Initialize MediaPipe face mesh solution
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Get image dimensions
        height, width = frame.shape[:2]
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = face_mesh.process(frame_rgb)
        
        # Initialize results
        face_landmarks = None
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Extract face landmarks if detected
        if results.multi_face_landmarks:
            # Draw landmarks on annotated frame
            for detected_face in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=detected_face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw contours
                mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=detected_face,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            
            # Extract landmarks from the first detected face
            face_landmarks = extract_face_landmarks(results, width, height)
        
        return face_landmarks, annotated_frame
        
    except Exception as e:
        print(f"Error getting face landmarks from frame: {e}")
        return None, frame