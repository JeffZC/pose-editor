import pandas as pd
import numpy as np
import os
import json
import cv2
from PyQt5.QtWidgets import QMessageBox
from .keypoint_formats import (
    SUPPORTED_FORMATS, 
    MEDIAPIPE33_TO_RR21,
    BODY25_TO_RR21,
    MEDIAPIPE33_TO_RR21_NAMES,
    BODY25_TO_RR21_NAMES
)

def create_empty_pose_dataframe(num_frames, format_name):
    """
    Creates an empty DataFrame for pose data in the specified format.
    
    Args:
        num_frames: Number of frames
        format_name: Format name from SUPPORTED_FORMATS
        
    Returns:
        Empty DataFrame with appropriate columns
    """
    if format_name not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format_name}")
        
    keypoints = SUPPORTED_FORMATS[format_name]
    
    # Create columns for X, Y, visibility for each keypoint
    columns = []
    for kp in keypoints:
        columns.append(f"{kp}_X")
        columns.append(f"{kp}_Y")
        # Handle formats like rr21 that might not have visibility
        if format_name != "rr21": 
            columns.append(f"{kp}_V")
    
    # Create empty DataFrame
    df = pd.DataFrame(np.zeros((num_frames, len(columns))), columns=columns)
    
    return df

def convert_format(pose_data, source_format, target_format):
    """
    Converts pose data from one format to another using the defined mappings.
    
    Args:
        pose_data: DataFrame containing pose data
        source_format: Current format name
        target_format: Desired format name
        
    Returns:
        DataFrame in the target format
    """
    # Check if conversion is necessary
    if source_format == target_format:
        return pose_data.copy()
    
    # Convert to rr21 first if going from mediapipe33 to body25 or vice versa
    if source_format == "mediapipe33" and target_format == "body25":
        intermediate = convert_format(pose_data, "mediapipe33", "rr21")
        return convert_format(intermediate, "rr21", "body25")
    
    if source_format == "body25" and target_format == "mediapipe33":
        intermediate = convert_format(pose_data, "body25", "rr21")
        return convert_format(intermediate, "rr21", "mediapipe33")
        
    # Get the appropriate mapping
    if source_format == "mediapipe33" and target_format == "rr21":
        mapping = MEDIAPIPE33_TO_RR21_NAMES
    elif source_format == "body25" and target_format == "rr21":
        mapping = BODY25_TO_RR21_NAMES
    elif source_format == "rr21" and target_format == "mediapipe33":
        # Create reverse mapping
        mapping = {v: k for k, v in MEDIAPIPE33_TO_RR21_NAMES.items()}
    elif source_format == "rr21" and target_format == "body25":
        # Create reverse mapping
        mapping = {v: k for k, v in BODY25_TO_RR21_NAMES.items()}
    else:
        raise ValueError(f"No defined conversion path from {source_format} to {target_format}")

    # Create empty DataFrame for target format
    num_frames = len(pose_data)
    result = create_empty_pose_dataframe(num_frames, target_format)
    
    # Determine if source and target have visibility columns
    source_has_v = any(f"{kp}_V" in pose_data.columns for kp in SUPPORTED_FORMATS[source_format])
    target_has_v = any(f"{kp}_V" in result.columns for kp in SUPPORTED_FORMATS[target_format])

    # Copy data using the mapping
    for source_kp, target_kp in mapping.items():
        source_x_col = f"{source_kp}_X"
        source_y_col = f"{source_kp}_Y"
        source_v_col = f"{source_kp}_V"
        
        target_x_col = f"{target_kp}_X"
        target_y_col = f"{target_kp}_Y"
        target_v_col = f"{target_kp}_V"
        
        if source_x_col in pose_data.columns and target_x_col in result.columns:
            result[target_x_col] = pose_data[source_x_col].values
            result[target_y_col] = pose_data[source_y_col].values
            
            # Copy visibility if available in source and required in target
            if source_has_v and target_has_v and source_v_col in pose_data.columns:
                result[target_v_col] = pose_data[source_v_col].values
            elif target_has_v:
                # Set default visibility to 1.0 if not available in source but required in target
                result[target_v_col] = 1.0
    
    return result

def detect_pose_format(dataframe):
    """
    Detects the format of the pose data based on column count and names
    
    Args:
        dataframe: Pandas DataFrame containing pose data
    
    Returns:
        str: Name of the detected format ("mediapipe33", "body25", "rr21" or "unknown")
    """
    if not isinstance(dataframe, pd.DataFrame) or not hasattr(dataframe, 'columns'):
        return "unknown" 
        
    columns = dataframe.columns
    column_count = len(columns)
    col_set = set(columns)

    # Check rr21 (42 columns, specific names)
    if column_count == 42:
        rr21_cols = set()
        for name in SUPPORTED_FORMATS["rr21"]:
            rr21_cols.add(f"{name}_X")
            rr21_cols.add(f"{name}_Y")
        if col_set == rr21_cols:
            return "rr21"

    # Check body25 (75 columns, specific names)
    if column_count == 75:
        body25_cols = set()
        for name in SUPPORTED_FORMATS["body25"]:
            body25_cols.add(f"{name}_X")
            body25_cols.add(f"{name}_Y")
            body25_cols.add(f"{name}_V")
        if col_set == body25_cols:
            return "body25"

    # Check mediapipe33 (99 columns, specific names)
    if column_count == 99:
        mp33_cols = set()
        for name in SUPPORTED_FORMATS["mediapipe33"]:
            mp33_cols.add(f"{name}_X")
            mp33_cols.add(f"{name}_Y")
            mp33_cols.add(f"{name}_V")
        if col_set == mp33_cols:
            return "mediapipe33"

    # Fallback checks based on column count if exact name match fails
    if column_count == 75 or column_count == 50:
        if any("NECK_X" in col or "NOSE_X" in col for col in columns):
            return "body25"

    if column_count == 99 or column_count == 66:
        if any("LEFT_EYE_INNER_X" in col or "MOUTH_LEFT_X" in col for col in columns):
            return "mediapipe33"

    if any("pose_keypoints" in col for col in columns):
        return "body25"

    return "unknown"

def convert_list_to_dataframe(landmarks_list, format_name="mediapipe33"):
    """
    Converts a flat list of landmark coordinates [x1, y1, v1, x2, y2, v2, ...] 
    or [x1, y1, x2, y2, ...] to a single-row DataFrame.
    
    Args:
        landmarks_list: List of landmark coordinates.
        format_name: Format name for the landmarks (e.g., "mediapipe33", "rr21").
        
    Returns:
        DataFrame with landmark coordinates for a single frame.
    """
    if not landmarks_list:
        return create_empty_pose_dataframe(num_frames=0, format_name=format_name)

    keypoints = SUPPORTED_FORMATS.get(format_name)
    if not keypoints:
        raise ValueError(f"Unsupported format: {format_name}")

    expected_coords_per_kp = 3 if format_name != "rr21" else 2
    num_keypoints = len(keypoints)
    expected_length = num_keypoints * expected_coords_per_kp
    
    if len(landmarks_list) == num_keypoints * 2 and expected_coords_per_kp == 3:
        coords_per_kp = 2
        has_visibility = False
    elif len(landmarks_list) == num_keypoints * 3 and expected_coords_per_kp == 2:
        coords_per_kp = expected_coords_per_kp
        has_visibility = False
    elif len(landmarks_list) >= expected_length:
        coords_per_kp = expected_coords_per_kp
        has_visibility = (coords_per_kp == 3)
        landmarks_list = landmarks_list[:expected_length]
    else:
        raise ValueError(f"Landmark list length ({len(landmarks_list)}) does not match expected length "
                         f"({expected_length} or {num_keypoints * 2}) for format '{format_name}'.")

    columns = []
    data = []
    
    idx = 0
    for kp in keypoints:
        columns.append(f"{kp}_X")
        data.append(landmarks_list[idx])
        idx += 1
        
        columns.append(f"{kp}_Y")
        data.append(landmarks_list[idx])
        idx += 1
        
        if has_visibility:
            columns.append(f"{kp}_V")
            data.append(landmarks_list[idx])
            idx += 1
        elif format_name != "rr21":
            columns.append(f"{kp}_V")
            data.append(1.0)

    df = pd.DataFrame([data], columns=columns)
    return df

# --- ADDED PLACEHOLDER FUNCTIONS ---

def load_pose_data(file_path, expected_frame_count=None, force_import=False):
    """
    Loads pose data from a CSV or JSON file.
    Placeholder implementation. Needs actual logic.
    
    Args:
        file_path: Path to the file
        expected_frame_count: Expected number of frames
        force_import: Whether to force import despite frame count mismatch
        
    Returns:
        Tuple: (pose_data, format_name, keypoint_names, success_flag, message)
    """
    print(f"Placeholder: load_pose_data called for {file_path}")
    # This needs the actual implementation to read CSV/JSON, detect format,
    # handle frame count mismatch, and convert to internal format (e.g., rr21).
    # Returning dummy data to allow import.
    
    # Example dummy return (replace with real logic):
    dummy_df = create_empty_pose_dataframe(10, "rr21") # Create a small dummy dataframe
    return dummy_df, "rr21", SUPPORTED_FORMATS["rr21"], True, "Placeholder load successful"
    # raise NotImplementedError("load_pose_data function needs implementation.")

def save_pose_data(pose_data, file_path, format_name="rr21"):
    """
    Saves pose data to a CSV or JSON file.
    Placeholder implementation. Needs actual logic.
    
    Args:
        pose_data: DataFrame containing pose data (expected in rr21 format)
        file_path: Path to save the file
        format_name: Format to save in (currently only supports rr21)
        
    Returns:
        bool: True if save successful, False otherwise
    """
    print(f"Placeholder: save_pose_data called for {file_path} in format {format_name}")
    # This needs the actual implementation to save the DataFrame.
    # Currently only supports saving as CSV in rr21 format.
    if file_path.endswith('.csv') and format_name == "rr21":
        try:
            pose_data.to_csv(file_path, index=False)
            return True
        except Exception as e:
            print(f"Error in placeholder save_pose_data: {e}")
            return False
    elif file_path.endswith('.json'):
        # Add JSON saving logic if needed, potentially using export_to_openpose
        print("JSON saving not implemented in placeholder.")
        return False
    else:
        print("Unsupported file type or format in placeholder.")
        return False
    # raise NotImplementedError("save_pose_data function needs implementation.")

# --- END OF ADDED PLACEHOLDERS ---

def export_to_openpose(pose_data, output_dir, format_name="mediapipe33"):
    """
    Exports pose data to OpenPose JSON format (body25).
    Converts the input data to body25 format if necessary.
    
    Args:
        pose_data: DataFrame containing pose data
        output_dir: Directory to save the JSON files
        format_name: Format of the input pose_data
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        if format_name != "body25":
            print(f"Converting from {format_name} to body25 for OpenPose export...")
            body25_data = convert_format(pose_data, source_format=format_name, target_format="body25")
        else:
            body25_data = pose_data
            
        if body25_data is None or body25_data.empty:
            print("Error: Conversion to body25 format failed or resulted in empty data.")
            return False

        frames = convert_dataframe_to_openpose(body25_data)
        
        num_frames = len(body25_data)
        if len(frames) != num_frames:
            print(f"Warning: Number of generated OpenPose frames ({len(frames)}) does not match input frames ({num_frames}).")

        for i, frame in enumerate(frames):
            filename = os.path.join(output_dir, f"frame_{i:06d}_keypoints.json")
            with open(filename, 'w') as f:
                json.dump(frame, f, cls=NumpyEncoder)
                
        print(f"Successfully exported {len(frames)} frames to OpenPose format in {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error exporting to OpenPose format: {e}")
        import traceback
        traceback.print_exc()
        return False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def convert_dataframe_to_openpose(pose_data):
    """
    Converts a DataFrame (expected to be in body25 format) 
    to a list of OpenPose JSON-compatible dictionaries.
    
    Args:
        pose_data: DataFrame in body25 format.
        
    Returns:
        List of dictionaries, each representing an OpenPose frame.
    """
    body25_to_openpose = {
        'NOSE': 0, 'NECK': 1, 'RIGHT_SHOULDER': 2, 'RIGHT_ELBOW': 3, 'RIGHT_WRIST': 4,
        'LEFT_SHOULDER': 5, 'LEFT_ELBOW': 6, 'LEFT_WRIST': 7, 'MID_HIP': 8, 'RIGHT_HIP': 9,
        'RIGHT_KNEE': 10, 'RIGHT_ANKLE': 11, 'LEFT_HIP': 12, 'LEFT_KNEE': 13, 'LEFT_ANKLE': 14,
        'RIGHT_EYE': 15, 'LEFT_EYE': 16, 'RIGHT_EAR': 17, 'LEFT_EAR': 18, 'LEFT_BIG_TOE': 19,
        'LEFT_SMALL_TOE': 20, 'LEFT_HEEL': 21, 'RIGHT_BIG_TOE': 22, 'RIGHT_SMALL_TOE': 23,
        'RIGHT_HEEL': 24
    }
    
    keypoints = SUPPORTED_FORMATS["body25"]
    num_frames = len(pose_data)
    frames = []
    
    expected_cols = set()
    for kp in keypoints:
        expected_cols.add(f"{kp}_X")
        expected_cols.add(f"{kp}_Y")
        expected_cols.add(f"{kp}_V")
    
    if not expected_cols.issubset(pose_data.columns):
        missing_cols = expected_cols - set(pose_data.columns)
        print(f"Warning: Input DataFrame for OpenPose conversion is missing columns: {missing_cols}")

    for frame_idx in range(num_frames):
        keypoints_array = [0.0] * (25 * 3)
        
        for kp in keypoints:
            if kp in body25_to_openpose:
                openpose_idx = body25_to_openpose[kp]
                
                x_col = f"{kp}_X"
                y_col = f"{kp}_Y"
                v_col = f"{kp}_V"

                x_val, y_val, conf_val = 0.0, 0.0, 0.0

                if x_col in pose_data.columns and y_col in pose_data.columns:
                    x_val = float(pose_data.iloc[frame_idx][x_col])
                    y_val = float(pose_data.iloc[frame_idx][y_col])
                    
                    if v_col in pose_data.columns:
                        conf_val = float(pose_data.iloc[frame_idx][v_col])
                    else:
                        conf_val = 0.0
                
                keypoints_array[openpose_idx * 3] = x_val
                keypoints_array[openpose_idx * 3 + 1] = y_val
                keypoints_array[openpose_idx * 3 + 2] = conf_val
        
        frame = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": keypoints_array,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }
        frames.append(frame)
        
    return frames