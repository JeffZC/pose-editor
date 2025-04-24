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
        'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]
}

# Create mappings between formats
FORMAT_MAPPINGS = {
    "mediapipe33_to_rr21": {
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
        'LEFT_FOOT_INDEX': 'LEFT_FOOT_INDEX',
        'RIGHT_FOOT_INDEX': 'RIGHT_FOOT_INDEX',
    },
    "rr21_to_body25": {
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
    }
}

def create_empty_pose_dataframe(num_frames, format_name="mediapipe33"):
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
    num_keypoints = len(keypoints)
    
    # Create columns for X, Y, visibility for each keypoint
    columns = []
    for kp in keypoints:
        columns.append(f"{kp}_X")
        columns.append(f"{kp}_Y")
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
    mapping_key = f"{source_format}_to_{target_format}"
    if mapping_key not in FORMAT_MAPPINGS:
        raise ValueError(f"No mapping found for conversion: {mapping_key}")
    
    mapping = FORMAT_MAPPINGS[mapping_key]
    target_keypoints = SUPPORTED_FORMATS[target_format]
    
    # Create empty DataFrame for target format
    num_frames = len(pose_data)
    result = create_empty_pose_dataframe(num_frames, target_format)
    
    # Copy data using the mapping
    for source_kp, target_kp in mapping.items():
        if f"{source_kp}_X" in pose_data.columns and f"{target_kp}_X" in result.columns:
            result[f"{target_kp}_X"] = pose_data[f"{source_kp}_X"]
            result[f"{target_kp}_Y"] = pose_data[f"{source_kp}_Y"]
            if f"{source_kp}_V" in pose_data.columns:
                result[f"{target_kp}_V"] = pose_data[f"{source_kp}_V"]
            else:
                # Set default visibility to 1.0 if not available in source
                result[f"{target_kp}_V"] = 1.0
    
    return result

def draw_pose(image, pose_data, frame_idx, format_name="mediapipe33", 
              selected_point=None, hovered_point=None, point_size=5,
              thickness=2, draw_point_ids=False):
    """
    Draws pose landmarks and connections on an image.
    
    Args:
        image: Image to draw on
        pose_data: DataFrame containing pose data
        frame_idx: Frame index
        format_name: Format name from SUPPORTED_FORMATS
        selected_point: Index of selected point
        hovered_point: Index of hovered point
        point_size: Size of points
        thickness: Line thickness
        draw_point_ids: Whether to draw point IDs
        
    Returns:
        Image with pose drawn on it
    """
    if image is None or pose_data is None:
        return image
        
    if format_name not in SUPPORTED_FORMATS:
        print(f"Unsupported format: {format_name}")
        return image
        
    keypoints = SUPPORTED_FORMATS[format_name]
    connections = get_connections_for_format(format_name)
    
    # Define colors
    joint_color = (0, 255, 0)  # Green
    line_color = (255, 255, 255)  # White
    selected_color = (255, 0, 0)  # Blue
    hovered_color = (0, 255, 255)  # Yellow

    # Get coordinates for this frame
    try:
        coords = {}
        for i, kp in enumerate(keypoints):
            x_col = f"{kp}_X"
            y_col = f"{kp}_Y"
            v_col = f"{kp}_V"
            
            if x_col in pose_data.columns and y_col in pose_data.columns:
                x = int(pose_data.iloc[frame_idx][x_col])
                y = int(pose_data.iloc[frame_idx][y_col])
                
                # Check visibility if available
                visibility = 1.0
                if v_col in pose_data.columns:
                    visibility = pose_data.iloc[frame_idx][v_col]
                
                coords[kp] = (x, y, visibility)
        
        # Draw connections
        for connection in connections:
            kp1, kp2 = connection
            if kp1 in coords and kp2 in coords:
                x1, y1, v1 = coords[kp1]
                x2, y2, v2 = coords[kp2]
                
                # Only draw if both points are visible
                if v1 > 0.5 and v2 > 0.5:
                    cv2.line(image, (x1, y1), (x2, y2), line_color, thickness)
        
        # Draw joint points
        for i, kp in enumerate(keypoints):
            if kp in coords:
                x, y, v = coords[kp]
                
                # Only draw if point is visible
                if v > 0.5:
                    # Determine color and size based on selection
                    color = joint_color
                    size = point_size
                    
                    if i == selected_point:
                        color = selected_color
                        size = point_size + 2
                    elif i == hovered_point:
                        color = hovered_color
                        size = point_size + 1
                        
                    # Draw the joint point
                    cv2.circle(image, (x, y), size, color, -1)
                    
                    # Draw point ID if requested
                    if draw_point_ids or i == selected_point:
                        cv2.putText(image, f"{i}", (x + 5, y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
    except Exception as e:
        print(f"Error drawing pose: {e}")
        
    return image

def get_connections_for_format(format_name):
    """
    Returns connection pairs for the specified format.
    
    Args:
        format_name: Format name
        
    Returns:
        List of connection pairs
    """
    if format_name == "mediapipe33":
        return [
            ("NOSE", "LEFT_EYE_INNER"),
            ("LEFT_EYE_INNER", "LEFT_EYE"),
            ("LEFT_EYE", "LEFT_EYE_OUTER"),
            ("LEFT_EYE_OUTER", "LEFT_EAR"),
            
            ("NOSE", "RIGHT_EYE_INNER"),
            ("RIGHT_EYE_INNER", "RIGHT_EYE"),
            ("RIGHT_EYE", "RIGHT_EYE_OUTER"),
            ("RIGHT_EYE_OUTER", "RIGHT_EAR"),
            
            ("MOUTH_LEFT", "MOUTH_RIGHT"),
            
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            ("LEFT_WRIST", "LEFT_PINKY"),
            ("LEFT_WRIST", "LEFT_INDEX"),
            ("LEFT_WRIST", "LEFT_THUMB"),
            ("LEFT_PINKY", "LEFT_INDEX"),
            
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
            ("RIGHT_WRIST", "RIGHT_PINKY"),
            ("RIGHT_WRIST", "RIGHT_INDEX"),
            ("RIGHT_WRIST", "RIGHT_THUMB"),
            ("RIGHT_PINKY", "RIGHT_INDEX"),
            
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),
            
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE"),
            ("LEFT_ANKLE", "LEFT_HEEL"),
            ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
            ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
            
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),
            ("RIGHT_ANKLE", "RIGHT_HEEL"),
            ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
            ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
        ]
    elif format_name == "body25":
        return [
            ("NOSE", "NECK"),
            ("NECK", "RIGHT_SHOULDER"),
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
            
            ("NECK", "LEFT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            
            ("NECK", "MID_HIP"),
            
            ("MID_HIP", "RIGHT_HIP"),
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),
            ("RIGHT_ANKLE", "RIGHT_HEEL"),
            ("RIGHT_ANKLE", "RIGHT_BIG_TOE"),
            ("RIGHT_BIG_TOE", "RIGHT_SMALL_TOE"),
            
            ("MID_HIP", "LEFT_HIP"),
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE"),
            ("LEFT_ANKLE", "LEFT_HEEL"),
            ("LEFT_ANKLE", "LEFT_BIG_TOE"),
            ("LEFT_BIG_TOE", "LEFT_SMALL_TOE"),
            
            ("NOSE", "RIGHT_EYE"),
            ("RIGHT_EYE", "RIGHT_EAR"),
            
            ("NOSE", "LEFT_EYE"),
            ("LEFT_EYE", "LEFT_EAR"),
        ]
    elif format_name == "rr21":
        return [
            ("NOSE", "LEFT_EYE"),
            ("LEFT_EYE", "LEFT_EAR"),
            
            ("NOSE", "RIGHT_EYE"),
            ("RIGHT_EYE", "RIGHT_EAR"),
            
            ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            ("LEFT_SHOULDER", "LEFT_ELBOW"),
            ("LEFT_ELBOW", "LEFT_WRIST"),
            
            ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            ("RIGHT_ELBOW", "RIGHT_WRIST"),
            
            ("LEFT_SHOULDER", "LEFT_HIP"),
            ("RIGHT_SHOULDER", "RIGHT_HIP"),
            ("LEFT_HIP", "RIGHT_HIP"),
            
            ("LEFT_HIP", "LEFT_KNEE"),
            ("LEFT_KNEE", "LEFT_ANKLE"),
            ("LEFT_ANKLE", "LEFT_HEEL"),
            ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
            
            ("RIGHT_HIP", "RIGHT_KNEE"),
            ("RIGHT_KNEE", "RIGHT_ANKLE"),
            ("RIGHT_ANKLE", "RIGHT_HEEL"),
            ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
        ]
    else:
        return []

def save_pose_data(pose_data, file_path):
    """
    Saves pose data to a CSV file.
    
    Args:
        pose_data: DataFrame containing pose data
        file_path: Path to save the file
    """
    try:
        pose_data.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving pose data: {e}")
        return False

def load_pose_data(file_path, expected_frame_count=None):
    """
    Loads pose data from a CSV file.
    
    Args:
        file_path: Path to the file
        expected_frame_count: Optional expected number of frames
    
    Returns:
        Tuple of (pose_data, format_name, success, message)
    """
    try:
        # Read CSV
        pose_data = pd.read_csv(file_path)
        
        # Determine format based on columns
        format_name = None
        for fmt, keypoints in SUPPORTED_FORMATS.items():
            # Check if first keypoint exists
            if f"{keypoints[0]}_X" in pose_data.columns:
                format_name = fmt
                break
                
        if format_name is None:
            return None, None, False, "Unknown pose data format"
            
        # Check frame count if provided
        if expected_frame_count is not None:
            actual_frame_count = len(pose_data)
            if actual_frame_count != expected_frame_count:
                return pose_data, format_name, False, f"Frame count mismatch: expected {expected_frame_count}, got {actual_frame_count}"
                
        return pose_data, format_name, True, f"Loaded pose data in {format_name} format"
        
    except Exception as e:
        return None, None, False, f"Error loading pose data: {e}"

def get_keypoint_coordinates(pose_data, frame_idx, keypoint_name, format_name="mediapipe33"):
    """
    Get the coordinates of a specific keypoint in a specific frame.
    
    Args:
        pose_data: DataFrame containing pose data
        frame_idx: Frame index
        keypoint_name: Name of the keypoint
        format_name: Format name
        
    Returns:
        Tuple of (x, y, visibility)
    """
    if keypoint_name not in SUPPORTED_FORMATS[format_name]:
        return None
        
    try:
        x = pose_data.iloc[frame_idx][f"{keypoint_name}_X"]
        y = pose_data.iloc[frame_idx][f"{keypoint_name}_Y"]
        
        # Get visibility if available
        visibility = 1.0
        if f"{keypoint_name}_V" in pose_data.columns:
            visibility = pose_data.iloc[frame_idx][f"{keypoint_name}_V"]
            
        return (x, y, visibility)
    except:
        return None

def export_to_openpose(pose_data, output_dir, format_name="mediapipe33"):
    """
    Exports pose data to OpenPose JSON format.
    
    Args:
        pose_data: DataFrame containing pose data
        output_dir: Directory to save JSON files
        format_name: Source format name
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to OpenPose format (BODY_25 is closest to OpenPose)
        if format_name != "body25":
            # First convert to rr21, then to body25 if needed
            if format_name == "mediapipe33":
                pose_data = convert_format(pose_data, "mediapipe33", "rr21")
                
            if format_name == "mediapipe33" or format_name == "rr21":
                body25_data = convert_format(pose_data, "rr21", "body25")
            else:
                # Unknown format
                return False
        else:
            body25_data = pose_data
            
        # Convert to OpenPose format
        keypoints = SUPPORTED_FORMATS["body25"]
        num_frames = len(body25_data)
        
        frames = convert_dataframe_to_openpose(body25_data)
        
        # Write each frame to a JSON file
        for i, frame in enumerate(frames):
            filename = os.path.join(output_dir, f"frame_{i:06d}_keypoints.json")
            with open(filename, 'w') as f:
                json.dump(frame, f)
                
        return True
    
    except Exception as e:
        print(f"Error exporting to OpenPose format: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_dataframe_to_openpose(pose_data):
    """
    Converts DataFrame to OpenPose JSON format.
    
    Args:
        pose_data: DataFrame containing pose data in BODY_25 format
        
    Returns:
        List of frame dictionaries in OpenPose format
    """
    # Map from BODY_25 to OpenPose indices
    body25_to_openpose = {
        'NOSE': 0, 'NECK': 1, 'RIGHT_SHOULDER': 2, 'RIGHT_ELBOW': 3, 'RIGHT_WRIST': 4,
        'LEFT_SHOULDER': 5, 'LEFT_ELBOW': 6, 'LEFT_WRIST': 7, 'MID_HIP': 8, 
        'RIGHT_HIP': 9, 'RIGHT_KNEE': 10, 'RIGHT_ANKLE': 11, 'LEFT_HIP': 12, 
        'LEFT_KNEE': 13, 'LEFT_ANKLE': 14, 'RIGHT_EYE': 15, 'LEFT_EYE': 16, 
        'RIGHT_EAR': 17, 'LEFT_EAR': 18, 'LEFT_BIG_TOE': 19, 'LEFT_SMALL_TOE': 20,
        'LEFT_HEEL': 21, 'RIGHT_BIG_TOE': 22, 'RIGHT_SMALL_TOE': 23, 'RIGHT_HEEL': 24
    }
    
    keypoints = SUPPORTED_FORMATS["body25"]
    num_frames = len(pose_data)
    frames = []
    
    for frame_idx in range(num_frames):
        # Create array for 25 keypoints (x, y, confidence)
        keypoints_array = [0] * (25 * 3)
        
        # Fill array from DataFrame
        for kp in keypoints:
            if kp in body25_to_openpose:
                openpose_idx = body25_to_openpose[kp]
                
                # Get coordinates
                x_col = f"{kp}_X"
                y_col = f"{kp}_Y"
                v_col = f"{kp}_V"
                
                if x_col in pose_data.columns and y_col in pose_data.columns:
                    x_val = float(pose_data.iloc[frame_idx][x_col])
                    y_val = float(pose_data.iloc[frame_idx][y_col])
                    
                    # Get visibility/confidence
                    conf_val = 1.0
                    if v_col in pose_data.columns:
                        conf_val = float(pose_data.iloc[frame_idx][v_col])
                    
                    # Set in OpenPose array
                    keypoints_array[openpose_idx*3] = x_val      # x
                    keypoints_array[openpose_idx*3+1] = y_val    # y
                    keypoints_array[openpose_idx*3+2] = conf_val # confidence
        
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