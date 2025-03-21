import pandas as pd
import numpy as np
import os
import json
import cv2
from PyQt5.QtWidgets import QMessageBox

# Define the supported formats and their keypoint names
SUPPORTED_FORMATS = {
    "mediapipe33": [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
        'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
        'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ],
    "body25": [
        'NOSE', 'NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
        'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 'MID_HIP', 
        'RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'LEFT_HIP', 
        'LEFT_KNEE', 'LEFT_ANKLE', 'RIGHT_EYE', 'LEFT_EYE', 
        'RIGHT_EAR', 'LEFT_EAR', 'LEFT_BIG_TOE', 'LEFT_SMALL_TOE',
        'LEFT_HEEL', 'RIGHT_BIG_TOE', 'RIGHT_SMALL_TOE', 'RIGHT_HEEL'
    ],
    "rr21": [
        'NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT', 'RIGHT_FOOT'
    ]
}

# Define mapping from mediapipe33 to rr21
MEDIAPIPE33_TO_RR21 = {
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

# Define mapping from body25 to rr21
BODY25_TO_RR21 = {
    0: 0,   # NOSE -> NOSE
    15: 2,  # RIGHT_EYE -> RIGHT_EYE
    16: 1,  # LEFT_EYE -> LEFT_EYE
    17: 4,  # RIGHT_EAR -> RIGHT_EAR
    18: 3,  # LEFT_EAR -> LEFT_EAR
    2: 6,   # RIGHT_SHOULDER -> RIGHT_SHOULDER
    5: 5,   # LEFT_SHOULDER -> LEFT_SHOULDER
    3: 8,   # RIGHT_ELBOW -> RIGHT_ELBOW
    6: 7,   # LEFT_ELBOW -> LEFT_ELBOW
    4: 10,  # RIGHT_WRIST -> RIGHT_WRIST
    7: 9,   # LEFT_WRIST -> LEFT_WRIST
    9: 12,  # RIGHT_HIP -> RIGHT_HIP
    12: 11, # LEFT_HIP -> LEFT_HIP
    10: 14, # RIGHT_KNEE -> RIGHT_KNEE
    13: 13, # LEFT_KNEE -> LEFT_KNEE
    11: 16, # RIGHT_ANKLE -> RIGHT_ANKLE
    14: 15, # LEFT_ANKLE -> LEFT_ANKLE
    24: 18, # RIGHT_HEEL -> RIGHT_HEEL
    21: 17, # LEFT_HEEL -> LEFT_HEEL
    22: 20, # RIGHT_BIG_TOE -> RIGHT_FOOT
    19: 19  # LEFT_BIG_TOE -> LEFT_FOOT
}

def detect_pose_format(dataframe):
    """
    Detects the format of the pose data based on column count and names
    
    Args:
        dataframe: Pandas DataFrame containing pose data
    
    Returns:
        str: Name of the detected format ("mediapipe33", "body25", "rr21" or "unknown")
    """
    columns = dataframe.columns
    column_count = len(columns)
    
    # Check formats based on column count first
    # OpenPose format check (50 columns for x,y or 75 columns including confidence)
    if column_count == 50 or column_count == 75:
        return "body25"
    
    # MediaPipe format check (66 columns for x,y or 99 columns including confidence)
    if column_count == 66 or column_count == 99:
        return "mediapipe33"
    
    # RR21 format check (42 columns)
    if column_count == 42:
        # Additional check to verify it's our RR21 format
        if any("NOSE_X" in col for col in columns):
            rr21_kpts = set([f"{name}_X" for name in SUPPORTED_FORMATS["rr21"]])
            rr21_kpts.update([f"{name}_Y" for name in SUPPORTED_FORMATS["rr21"]])
            if all(col in rr21_kpts for col in columns):
                return "rr21"
    
    # Fallback to checking column names for additional format detection
    if any("pose_keypoints" in col for col in columns):
        return "body25"
        
    # If the format doesn't match known patterns but has even number of columns,
    # we assume it's a generic format
    if column_count % 2 == 0:
        return "generic"
        
    return "unknown"

def convert_to_rr21(pose_data, source_format):
    """
    Convert pose data from source format to RR21 format
    
    Args:
        pose_data: DataFrame containing pose data
        source_format: Source format name ("mediapipe33" or "body25")
    
    Returns:
        DataFrame: Converted pose data in RR21 format
    """  
    # If it's already RR21, no conversion needed
    if source_format == "rr21":
        return pose_data
    
    num_frames = len(pose_data)
    num_rr21_keypoints = len(SUPPORTED_FORMATS["rr21"])
    
    # Create empty DataFrame for RR21 format
    rr21_data = pd.DataFrame(
        np.zeros((num_frames, num_rr21_keypoints * 2)),
        columns=[f"{kp}_X" if i % 2 == 0 else f"{kp}_Y" 
                for kp in SUPPORTED_FORMATS["rr21"] for i in range(2)]
    )
    
    # Get column list once
    pose_columns = pose_data.columns.tolist()
    rr21_columns = rr21_data.columns.tolist()
    
    # Use the appropriate mapping
    mapping = MEDIAPIPE33_TO_RR21 if source_format == "mediapipe33" else BODY25_TO_RR21
    
    # Transfer coordinates using the mapping (more direct mathematical approach)
    for src_idx, dst_idx in mapping.items():
        # Calculate the actual column indices for source and destination
        src_x_idx = src_idx * 2
        src_y_idx = src_x_idx + 1
        dst_x_idx = dst_idx * 2
        dst_y_idx = dst_x_idx + 1
        
        # Map to dataframe column indices
        if src_x_idx < len(pose_columns) and src_y_idx < len(pose_columns):
            # Transfer X and Y coordinates
            rr21_data.iloc[:, dst_x_idx] = pose_data.iloc[:, src_x_idx]
            rr21_data.iloc[:, dst_y_idx] = pose_data.iloc[:, src_y_idx]
    
    return rr21_data

def load_pose_data(file_path, expected_frame_count=None, force_import=False):
    """
    Load pose data from file with format auto-detection and conversion to RR21
    
    Args:
        file_path: Path to the pose data file
        expected_frame_count: Expected number of frames (from video)
        force_import: Whether to force import despite frame count mismatch
    
    Returns:
        tuple: (DataFrame of pose data in RR21 format, original_format, keypoint_names, success, message)
    """
    try:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.csv':
            # Load as CSV
            original_pose_data = pd.read_csv(file_path)
            original_format = detect_pose_format(original_pose_data)
            
            # Validate column count
            column_count = len(original_pose_data.columns)
            if original_format == "unknown":
                return None, "unknown", [], False, "Invalid column count. Expected 50/75 (OpenPose), 66/99 (MediaPipe), or 42 (RR21)."
            
        elif ext == '.json':
            # Load as JSON (commonly used for OpenPose)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            # Convert JSON to DataFrame based on format
            if 'people' in json_data:
                # OpenPose format
                original_pose_data = convert_openpose_to_dataframe(json_data)
                original_format = "body25"
            else:
                # Generic JSON format
                original_pose_data = pd.DataFrame(json_data)
                original_format = detect_pose_format(original_pose_data)
                
                # Validate column count for JSON imports
                if original_format == "unknown":
                    return None, "unknown", [], False, "Invalid column count in JSON. Expected 50/75 (OpenPose), 66/99 (MediaPipe), or 42 (RR21)."
        else:
            return None, "unknown", [], False, f"Unsupported file format: {ext}"
        
        # Verify the data format
        if original_pose_data is None or len(original_pose_data.columns) % 2 != 0:
            return None, "unknown", [], False, "Invalid pose data format. Number of columns must be even."
        
        # Validate frame count if expected count is provided
        if expected_frame_count is not None:
            actual_frame_count = len(original_pose_data)
            
            if actual_frame_count != expected_frame_count:
                if not force_import:
                    # Return with a warning that can be overridden
                    return original_pose_data, original_format, [], False, f"Frame count mismatch. Pose data has {actual_frame_count} frames, but video has {expected_frame_count} frames."
                else:
                    # Force import by adjusting frame count
                    if actual_frame_count > expected_frame_count:
                        # Truncate extra frames
                        original_pose_data = original_pose_data.iloc[:expected_frame_count].reset_index(drop=True)
                    else:
                        # Pad with zeros
                        missing_frames = expected_frame_count - actual_frame_count
                        padding = pd.DataFrame(np.zeros((missing_frames, len(original_pose_data.columns))),
                                             columns=original_pose_data.columns)
                        original_pose_data = pd.concat([original_pose_data, padding], ignore_index=True)
            
        # Convert to RR21 format if needed
        if original_format in ["mediapipe33", "body25"]:
            pose_data = convert_to_rr21(original_pose_data, original_format)
        else:
            pose_data = original_pose_data
            
        # Always use RR21 keypoint names for consistency
        keypoint_names = SUPPORTED_FORMATS["rr21"]
                    
        return pose_data, original_format, keypoint_names, True, "Pose data loaded successfully."
        
    except Exception as e:
        print(f"Error loading pose data: {e}")
        return None, "unknown", [], False, f"Error loading pose data: {str(e)}"

def save_pose_data(pose_data, file_path, format_name="rr21"):
    """
    Save pose data to file in specified format
    
    Args:
        pose_data: DataFrame containing pose data (assumed to be in RR21 format)
        file_path: Path to save the file
        format_name: Format to save as (default: rr21)
    
    Returns:
        bool: Success or failure
    """
    try:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.csv':
            pose_data.to_csv(file_path, index=False)
            return True
        elif ext == '.json':
            # Convert to the appropriate JSON format
            if format_name == "body25":
                json_data = convert_dataframe_to_openpose(pose_data)
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                # Generic JSON format
                pose_data.to_json(file_path, orient="records")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error saving pose data: {e}")
        return False

def convert_openpose_to_dataframe(openpose_json):
    """
    Convert OpenPose JSON format to a DataFrame
    
    Args:
        openpose_json: OpenPose JSON data
    
    Returns:
        DataFrame: Converted pose data in Body25 format
    """
    all_frames = []
    
    # Handle single frame or multiple frames
    if 'people' in openpose_json:
        # Single frame
        frames = [openpose_json]
    else:
        # Multiple frames
        frames = openpose_json
        
    for frame in frames:
        if 'people' in frame and len(frame['people']) > 0:
            person = frame['people'][0]  # Get first person
            
            if 'pose_keypoints_2d' in person:
                # Extract keypoints
                keypoints = person['pose_keypoints_2d']
                
                # OpenPose keypoints are stored as [x1, y1, c1, x2, y2, c2, ...]
                # We need to convert to [x1, y1, x2, y2, ...]
                row = []
                for i in range(0, len(keypoints), 3):
                    if i+1 < len(keypoints):
                        row.append(keypoints[i])    # x
                        row.append(keypoints[i+1])  # y
                        
                all_frames.append(row)
                
    # Create DataFrame
    if all_frames:
        # Create column names for Body25 format
        columns = []
        for name in SUPPORTED_FORMATS["body25"]:
            columns.extend([f'{name}_X', f'{name}_Y'])
            
        return pd.DataFrame(all_frames, columns=columns)
    
    return None

# In convert_dataframe_to_openpose, the reverse mapping might be clearer with an explicit dict
def convert_dataframe_to_openpose(pose_data):
    # Create a reverse mapping dictionary for clearer code
    RR21_TO_BODY25 = {v: k for k, v in BODY25_TO_RR21.items()}
    
    frames = []
    
    # We need to expand RR21 to Body25 format with zeros for missing points
    for idx, row in pose_data.iterrows():
        # Initialize array with zeros for all OpenPose points (25 points * 3 values = 75)
        keypoints = [0.0] * (25 * 3)
        
        # Map from RR21 to Body25
        for rr21_idx, body25_idx in RR21_TO_BODY25.items():
            # Get RR21 column names
            src_x_col = f"{SUPPORTED_FORMATS['rr21'][body25_idx]}_X"
            src_y_col = f"{SUPPORTED_FORMATS['rr21'][body25_idx]}_Y"
            
            # Get values
            x_val = float(row[src_x_col])
            y_val = float(row[src_y_col])
            
            # Set in OpenPose array (using body25 index)
            keypoints[rr21_idx*3] = x_val      # x
            keypoints[rr21_idx*3+1] = y_val    # y
            keypoints[rr21_idx*3+2] = 1.0      # confidence
        
        frame = {
            "people": [{
                "pose_keypoints_2d": keypoints
            }]
        }
        frames.append(frame)
        
    return frames

def create_empty_pose_data(video_path, format_name="rr21"):
    """
    Create empty pose data for a video
    
    Args:
        video_path: Path to the video file
        format_name: Format to create (default: rr21)
    
    Returns:
        DataFrame: Empty pose data frame with correct dimensions
    """
    try:
        # Open video to get frame count
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Get the keypoint names for the format
        if format_name in SUPPORTED_FORMATS:
            keypoint_names = SUPPORTED_FORMATS[format_name]
        else:
            return None
            
        # Create column names
        column_names = []
        for name in keypoint_names:
            column_names.extend([f'{name}_X', f'{name}_Y'])
            
        # Create DataFrame with zeros
        pose_data = pd.DataFrame(np.zeros((num_frames, len(column_names))), 
                               columns=column_names)
        
        return pose_data
        
    except Exception as e:
        print(f"Error creating empty pose data: {e}")
        return None

def get_keypoint_connections(format_name="rr21"):
    """
    Get the keypoint connections for a pose format
    
    Args:
        format_name: Format name (rr21, mediapipe33, body25, etc.)
    
    Returns:
        list: List of tuples representing connected keypoints
    """
    if format_name == "mediapipe33":
        # MediaPipe POSE_CONNECTIONS
        return [
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5),
            (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
            (12, 14), (13, 15), (14, 16), (15, 17), (15, 19),
            (15, 21), (16, 18), (16, 20), (16, 22), (17, 19),
            (18, 20), (23, 24), (23, 25), (24, 26), (25, 27),
            (26, 28), (27, 29), (27, 31), (28, 30), (28, 32),
            (29, 31), (30, 32)
        ]
    elif format_name == "body25":
        # OpenPose Body25 connections
        return [
            (0, 1), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), 
            (5, 6), (6, 7), (8, 9), (8, 12), (9, 10), (10, 11),
            (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), 
            (16, 18), (14, 19), (19, 20), (14, 21), (11, 22),
            (22, 23), (11, 24)
        ]
    elif format_name == "rr21":
        # RR21 connections - matching the human skeleton structure
        return [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (15, 17), (15, 19),  # Left leg
            (12, 14), (14, 16), (16, 18), (16, 20)   # Right leg
        ]
    else:
        return []

def process_mediapipe_to_rr21(landmarks_list):
    """
    Process MediaPipe landmarks (33 keypoints) to RR21 format
    
    Args:
        landmarks_list: List of x,y coordinates from MediaPipe detection
        
    Returns:
        list: Coordinates in RR21 format
    """
    # If the input is not as expected, return empty list
    if not landmarks_list or len(landmarks_list) < 66:  # 33 keypoints * 2 coordinates
        return []
        
    # Create result list with 21 keypoints (x,y coordinates)
    rr21_landmarks = [0.0] * (21 * 2)
    
    # Map MediaPipe coordinates to RR21
    for mediapipe_idx, rr21_idx in MEDIAPIPE33_TO_RR21.items():
        # Calculate source and destination indices
        src_x_idx = mediapipe_idx * 2
        src_y_idx = src_x_idx + 1
        dst_x_idx = rr21_idx * 2
        dst_y_idx = dst_x_idx + 1
        
        # Copy coordinates if within bounds
        if src_x_idx < len(landmarks_list) and src_y_idx < len(landmarks_list):
            rr21_landmarks[dst_x_idx] = landmarks_list[src_x_idx]
            rr21_landmarks[dst_y_idx] = landmarks_list[src_y_idx]
    
    return rr21_landmarks