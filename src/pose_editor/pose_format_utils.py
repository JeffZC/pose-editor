import pandas as pd
import numpy as np
from .keypoint_formats import SUPPORTED_FORMATS, MEDIAPIPE33_TO_RR21_NAMES

def process_mediapipe_to_rr21(landmarks_list):
    """
    Convert MediaPipe pose landmarks to RR21 format.
    
    Args:
        landmarks_list: Either a flat list of landmarks or MediaPipe pose landmarks object
    
    Returns:
        A flat list of coordinates in RR21 format [x1, y1, x2, y2, ...] (no visibility)
    """
    # If input is already a list, make sure it's a NumPy array for easier processing
    if isinstance(landmarks_list, list):
        landmarks_array = np.array(landmarks_list)
    else:
        # If it's a MediaPipe pose landmarks object, extract the points
        try:
            landmarks_array = []
            for landmark in landmarks_list.landmark:
                landmarks_array.extend([landmark.x, landmark.y, landmark.visibility])
            landmarks_array = np.array(landmarks_array)
        except:
            # If extraction fails, return empty list
            return []
    
    # Check if we have enough data
    if len(landmarks_array) < 33:  # MediaPipe has 33 keypoints
        return []
    
    # Create output array for RR21 format (21 keypoints, x and y only)
    rr21_landmarks = np.zeros(21 * 2)
    
    # Define mapping from MediaPipe indices to RR21 indices
    # This matches the MEDIAPIPE33_TO_RR21 dictionary in keypoint_formats.py
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
    
    # Determine if we have a flat list with 3D or 2D coordinates
    landmarks_per_point = 3  # Default: (x, y, visibility)
    if len(landmarks_array) >= 132:  # Full 3D landmarks (x, y, z, visibility)
        landmarks_per_point = 4
    
    # Copy coordinates from MediaPipe to RR21
    for mp_idx, rr_idx in mapping.items():
        # Get source coordinates with proper stride
        src_x_idx = mp_idx * landmarks_per_point
        src_y_idx = mp_idx * landmarks_per_point + 1
        
        # Get destination indices in RR21 format (x, y only)
        dst_x_idx = rr_idx * 2
        dst_y_idx = rr_idx * 2 + 1
        
        # Copy coordinates if within bounds
        if src_x_idx + 1 < len(landmarks_array):
            # Note: MediaPipe returns normalized coordinates [0..1],
            # but we need pixel coordinates for our application
            rr21_landmarks[dst_x_idx] = landmarks_array[src_x_idx]
            rr21_landmarks[dst_y_idx] = landmarks_array[src_y_idx]
    
    return rr21_landmarks.tolist()

def load_pose_data(file_path, expected_frame_count=None, force_import=False):
    """
    Load pose data from a CSV or JSON file
    
    Args:
        file_path: Path to the file
        expected_frame_count: Expected number of frames (for validation)
        force_import: If True, will adjust data to match expected frame count
        
    Returns:
        tuple: (DataFrame, format_name, keypoint_names, success, message)
    """
    if not file_path or not os.path.exists(file_path):
        return None, "", [], False, "File not found"
    
    try:
        import os
        from .body_format import detect_pose_format, convert_format, SUPPORTED_FORMATS
        
        # Load data based on file extension
        if file_path.lower().endswith('.csv'):
            pose_data = pd.read_csv(file_path)
        elif file_path.lower().endswith('.json'):
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Convert JSON to DataFrame (simplified)
                pose_data = pd.DataFrame(data)
        else:
            return None, "", [], False, "Unsupported file format"
        
        # Detect format
        format_name = detect_pose_format(pose_data)
        if format_name == "unknown":
            return None, "", [], False, "Unknown pose data format"
        
        # Get keypoint names for the format
        keypoint_names = SUPPORTED_FORMATS[format_name]
        
        # Check frame count
        if expected_frame_count is not None and len(pose_data) != expected_frame_count and not force_import:
            return None, format_name, keypoint_names, False, f"Frame count mismatch: file has {len(pose_data)} frames but video has {expected_frame_count} frames"
        
        # Adjust frame count if forcing import
        if expected_frame_count is not None and len(pose_data) != expected_frame_count and force_import:
            if len(pose_data) < expected_frame_count:
                # Duplicate last frame to match expected count
                last_frame = pose_data.iloc[-1].copy()
                for _ in range(expected_frame_count - len(pose_data)):
                    pose_data = pd.concat([pose_data, pd.DataFrame([last_frame])], ignore_index=True)
            else:
                # Truncate to match expected count
                pose_data = pose_data.iloc[:expected_frame_count]
        
        # Convert to RR21 format if it's not already
        if format_name != "rr21":
            pose_data = convert_format(pose_data, format_name, "rr21")
            keypoint_names = SUPPORTED_FORMATS["rr21"]
            format_name = "rr21"  # Update format name after conversion
        
        return pose_data, format_name, keypoint_names, True, "Pose data loaded successfully"
        
    except Exception as e:
        import traceback
        return None, "", [], False, f"Error loading pose data: {str(e)}\n{traceback.format_exc()}"

def save_pose_data(pose_data, file_path, format_name="rr21"):
    """
    Save pose data to a file
    
    Args:
        pose_data: DataFrame containing pose data
        file_path: Path to save the file
        format_name: Format of the pose data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save based on file extension
        if file_path.lower().endswith('.csv'):
            pose_data.to_csv(file_path, index=False)
            return True
        elif file_path.lower().endswith('.json'):
            import json
            with open(file_path, 'w') as f:
                # Convert DataFrame to dict for JSON serialization
                json.dump(pose_data.to_dict(orient='records'), f, indent=2)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error saving pose data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False