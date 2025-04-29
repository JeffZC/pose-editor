import pandas as pd
import numpy as np
import os
import json

"""
Keypoint format definitions and mappings for pose data, including MediaPipe, Body25, and RR21 formats.
"""

from typing import Dict, List, Tuple, Optional, Union

# Define the MediaPipe 33-point skeleton format
MEDIAPIPE33_KEYPOINTS = [
    {'index': 0, 'name': 'NOSE', 'description': 'Tip of the nose'},
    {'index': 1, 'name': 'LEFT_EYE_INNER', 'description': 'Inner corner of the left eye'},
    {'index': 2, 'name': 'LEFT_EYE', 'description': 'Center of the left eye'},
    {'index': 3, 'name': 'LEFT_EYE_OUTER', 'description': 'Outer corner of the left eye'},
    {'index': 4, 'name': 'RIGHT_EYE_INNER', 'description': 'Inner corner of the right eye'},
    {'index': 5, 'name': 'RIGHT_EYE', 'description': 'Center of the right eye'},
    {'index': 6, 'name': 'RIGHT_EYE_OUTER', 'description': 'Outer corner of the right eye'},
    {'index': 7, 'name': 'LEFT_EAR', 'description': 'Tragion of the left ear'},
    {'index': 8, 'name': 'RIGHT_EAR', 'description': 'Tragion of the right ear'},
    {'index': 9, 'name': 'MOUTH_LEFT', 'description': 'Left corner of the mouth'},
    {'index': 10, 'name': 'MOUTH_RIGHT', 'description': 'Right corner of the mouth'},
    {'index': 11, 'name': 'LEFT_SHOULDER', 'description': 'Left shoulder joint'},
    {'index': 12, 'name': 'RIGHT_SHOULDER', 'description': 'Right shoulder joint'},
    {'index': 13, 'name': 'LEFT_ELBOW', 'description': 'Left elbow joint'},
    {'index': 14, 'name': 'RIGHT_ELBOW', 'description': 'Right elbow joint'},
    {'index': 15, 'name': 'LEFT_WRIST', 'description': 'Left wrist joint'},
    {'index': 16, 'name': 'RIGHT_WRIST', 'description': 'Right wrist joint'},
    {'index': 17, 'name': 'LEFT_PINKY', 'description': 'Knuckle of the left pinky finger'},
    {'index': 18, 'name': 'RIGHT_PINKY', 'description': 'Knuckle of the right pinky finger'},
    {'index': 19, 'name': 'LEFT_INDEX', 'description': 'Knuckle of the left index finger'},
    {'index': 20, 'name': 'RIGHT_INDEX', 'description': 'Knuckle of the right index finger'},
    {'index': 21, 'name': 'LEFT_THUMB', 'description': 'Tip of the left thumb'},
    {'index': 22, 'name': 'RIGHT_THUMB', 'description': 'Tip of the right thumb'},
    {'index': 23, 'name': 'LEFT_HIP', 'description': 'Left hip joint'},
    {'index': 24, 'name': 'RIGHT_HIP', 'description': 'Right hip joint'},
    {'index': 25, 'name': 'LEFT_KNEE', 'description': 'Left knee joint'},
    {'index': 26, 'name': 'RIGHT_KNEE', 'description': 'Right knee joint'},
    {'index': 27, 'name': 'LEFT_ANKLE', 'description': 'Left ankle joint'},
    {'index': 28, 'name': 'RIGHT_ANKLE', 'description': 'Right ankle joint'},
    {'index': 29, 'name': 'LEFT_HEEL', 'description': 'Left heel'},
    {'index': 30, 'name': 'RIGHT_HEEL', 'description': 'Right heel'},
    {'index': 31, 'name': 'LEFT_FOOT_INDEX', 'description': 'Tip of the left foot'},
    {'index': 32, 'name': 'RIGHT_FOOT_INDEX', 'description': 'Tip of the right foot'}
]

# Extract just the names for easy access
MEDIAPIPE33_KEYPOINT_NAMES = [kp['name'] for kp in MEDIAPIPE33_KEYPOINTS]

# Create mapping dictionaries
MEDIAPIPE33_INDEX_TO_KEYPOINT = {kp['index']: kp for kp in MEDIAPIPE33_KEYPOINTS}
MEDIAPIPE33_NAME_TO_INDEX = {kp['name']: kp['index'] for kp in MEDIAPIPE33_KEYPOINTS}

# Define body regions
BODY_REGIONS = {
    'face': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'torso': [11, 12, 23, 24],
    'left_arm': [11, 13, 15, 17, 19, 21],
    'right_arm': [12, 14, 16, 18, 20, 22],
    'left_leg': [23, 25, 27, 29, 31],
    'right_leg': [24, 26, 28, 30, 32]
}

# Pose connections
MEDIAPIPE_POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]

# Supported keypoint formats
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

# Format mappings
MEDIAPIPE33_TO_RR21 = {
    0: 0,   # NOSE -> NOSE
    2: 1,   # LEFTkorean air_EYE -> LEFT_EYE
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

MEDIAPIPE33_TO_RR21_NAMES = {
    SUPPORTED_FORMATS["mediapipe33"][idx1]: SUPPORTED_FORMATS["rr21"][idx2] 
    for idx1, idx2 in MEDIAPIPE33_TO_RR21.items()
}

BODY25_TO_RR21_NAMES = {
    SUPPORTED_FORMATS["body25"][idx1]: SUPPORTED_FORMATS["rr21"][idx2] 
    for idx1, idx2 in BODY25_TO_RR21.items()
}

def get_format_keypoints(format_name):
    """
    Returns the list of keypoint names for the specified format.
    
    Args:
        format_name: Name of the format (mediapipe33, body25, rr21)
        
    Returns:
        List of keypoint names or empty list if format not supported
    """
    return SUPPORTED_FORMATS.get(format_name, [])

def get_mapping(source_format, target_format):
    """
    Returns the appropriate mapping between formats.
    
    Args:
        source_format: Source format name
        target_format: Target format name
        
    Returns:
        Dictionary mapping between formats or None if mapping not supported
    """
    if source_format == "mediapipe33" and target_format == "rr21":
        return MEDIAPIPE33_TO_RR21_NAMES
    elif source_format == "body25" and target_format == "rr21":
        return BODY25_TO_RR21_NAMES
    return None

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
    if isinstance(input_data, list) or isinstance(input_data, np.ndarray):
        # Convert to numpy array for easier processing
        landmarks_array = np.array(input_data) if isinstance(input_data, list) else input_data
        
        # Check if we have enough data
        if len(landmarks_array) < 33:  # MediaPipe has 33 keypoints minimum
            return []
        
        # Create output array for RR21 format (21 keypoints, x, y and visibility)
        rr21_landmarks = []
        
        # Determine if we have a flat list with 3D or 2D coordinates
        landmarks_per_point = 3  # Default: (x, y, visibility)
        if len(landmarks_array) >= 132:  # Full 3D landmarks (x, y, z, visibility)
            landmarks_per_point = 4
            
        # Define mapping from MediaPipe indices to RR21 indices
        # This matches the MEDIAPIPE33_TO_RR21 dictionary
        mapping = MEDIAPIPE33_TO_RR21
        
        # Copy coordinates from MediaPipe to RR21
        for mp_idx, rr_idx in mapping.items():
            # Calculate source indices in the flat list
            src_x_idx = mp_idx * landmarks_per_point
            src_y_idx = mp_idx * landmarks_per_point + 1
            src_v_idx = mp_idx * landmarks_per_point + (landmarks_per_point - 1)  # Visibility is last
            
            # Ensure index is within bounds
            if src_x_idx < len(landmarks_array) and src_y_idx < len(landmarks_array) and src_v_idx < len(landmarks_array):
                x = landmarks_array[src_x_idx]
                y = landmarks_array[src_y_idx]
                v = landmarks_array[src_v_idx]
                
                # Scale coordinates if dimensions are provided
                if frame_width is not None:
                    x = x * frame_width
                
                if frame_height is not None:
                    y = y * frame_height
                
                # In the RR21 format, we need x, y and visibility for each keypoint
                rr21_landmarks.append(x)
                rr21_landmarks.append(y)
                rr21_landmarks.append(v)
            else:
                # If keypoint is missing, add zeros
                rr21_landmarks.append(0.0)
                rr21_landmarks.append(0.0)
                rr21_landmarks.append(0.0)
        
        return rr21_landmarks
        
    # Otherwise, the input is a DataFrame (from video processing)
    else:
        # Define mapping from MediaPipe to RR21 format
        mapping = MEDIAPIPE33_TO_RR21_NAMES
        
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
                rr21_data[x_col_rr] = input_data[x_col_mp]
                rr21_data[y_col_rr] = input_data[y_col_mp]
                rr21_data[v_col_rr] = input_data[v_col_mp]
        
        return rr21_data

class BodyKeypoints:
    """
    A class that stores and manages keypoint data for human pose.
    This replaces the previous approach of directly working with pandas DataFrames.
    """
    
    def __init__(self, data=None, format_name="rr21"):
        """
        Initialize the BodyKeypoints object
        
        Args:
            data: Either a pandas DataFrame, a list of keypoints, or None to create empty
            format_name: The keypoint format to use (e.g., "mediapipe33", "body25", "rr21")
        """
        self.format_name = format_name
        self.keypoints = {}  # Dictionary to store keypoints: {frame_idx: {keypoint_name: {"x": val, "y": val, "v": val}}}
        self.num_frames = 0
        
        # Initialize from provided data
        if data is not None:
            if isinstance(data, pd.DataFrame):
                self._init_from_dataframe(data)
            elif isinstance(data, list):
                if len(data) > 0 and isinstance(data[0], list):
                    # List of frames, each with a list of keypoints
                    for i, frame_data in enumerate(data):
                        self._add_frame_from_list(frame_data, i)
                    self.num_frames = len(data)
                else:
                    # Single frame as list
                    self._add_frame_from_list(data, 0)
                    self.num_frames = 1
    
    def _init_from_dataframe(self, df):
        """
        Initialize keypoints from a pandas DataFrame
        
        Args:
            df: Pandas DataFrame containing pose data
        """
        self.num_frames = len(df)
        if "format_name" not in dir(self) or self.format_name is None:
            self.format_name = self._detect_format(df)
        
        keypoints = SUPPORTED_FORMATS[self.format_name]
        
        for frame_idx in range(self.num_frames):
            self.keypoints[frame_idx] = {}
            frame_data = df.iloc[frame_idx]
            
            for kp in keypoints:
                x_col = f"{kp}_X"
                y_col = f"{kp}_Y"
                v_col = f"{kp}_V"
                
                if x_col in df.columns and y_col in df.columns:
                    self.keypoints[frame_idx][kp] = {
                        "x": float(frame_data[x_col]),
                        "y": float(frame_data[y_col])
                    }
                    
                    if v_col in df.columns:
                        self.keypoints[frame_idx][kp]["v"] = float(frame_data[v_col])
                    else:
                        self.keypoints[frame_idx][kp]["v"] = 1.0
    
    def _add_frame_from_list(self, landmarks_list, frame_idx=0):
        """
        Add a frame of keypoints from a flat list of values
        
        Args:
            landmarks_list: Flat list of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...]
            frame_idx: Frame index to store these keypoints
        """
        keypoints = SUPPORTED_FORMATS[self.format_name]
        num_keypoints = len(keypoints)
        expected_coords_per_kp = 3  # Default: x, y, visibility
        expected_length = num_keypoints * expected_coords_per_kp
        
        # Determine format and adjust if needed
        has_visibility = True
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
                         f"({expected_length} or {num_keypoints * 2}) for format '{self.format_name}'.")
        
        # Create new frame dict if needed
        if frame_idx not in self.keypoints:
            self.keypoints[frame_idx] = {}
        
        idx = 0
        for kp in keypoints:
            x = landmarks_list[idx]
            idx += 1
            
            y = landmarks_list[idx]
            idx += 1
            
            visibility = 1.0  # Default
            if has_visibility:
                visibility = landmarks_list[idx]
                idx += 1
                
            self.keypoints[frame_idx][kp] = {
                "x": x,
                "y": y,
                "v": visibility
            }
    
    def to_dataframe(self):
        """
        Convert the stored keypoints to a pandas DataFrame
        
        Returns:
            pandas.DataFrame: DataFrame containing the pose data
        """
        if not self.keypoints or self.num_frames == 0:
            # Create empty DataFrame with appropriate columns
            columns = []
            keypoints = SUPPORTED_FORMATS[self.format_name]
            for kp in keypoints:
                columns.extend([f"{kp}_X", f"{kp}_Y", f"{kp}_V"])
            return pd.DataFrame(columns=columns)
        
        rows = []
        for frame_idx in range(self.num_frames):
            if frame_idx not in self.keypoints:
                # Create empty row if frame doesn't exist
                row = {}
            else:
                row = {}
                frame_kps = self.keypoints[frame_idx]
                for kp, coords in frame_kps.items():
                    row[f"{kp}_X"] = coords.get("x", 0.0)
                    row[f"{kp}_Y"] = coords.get("y", 0.0) 
                    row[f"{kp}_V"] = coords.get("v", 1.0)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_keypoint(self, frame_idx, keypoint_name):
        """
        Get a specific keypoint's coordinates
        
        Args:
            frame_idx: Frame index
            keypoint_name: Name of the keypoint
        
        Returns:
            dict: Dictionary with x, y, v coordinates or None if not found
        """
        if frame_idx in self.keypoints and keypoint_name in self.keypoints[frame_idx]:
            return self.keypoints[frame_idx][keypoint_name]
        return None
    
    def set_keypoint(self, frame_idx, keypoint_name, x, y, visibility=1.0):
        """
        Set a specific keypoint's coordinates
        
        Args:
            frame_idx: Frame index
            keypoint_name: Name of the keypoint
            x: X coordinate
            y: Y coordinate
            visibility: Visibility value (default: 1.0)
        """
        if frame_idx not in self.keypoints:
            self.keypoints[frame_idx] = {}
        
        self.keypoints[frame_idx][keypoint_name] = {
            "x": x,
            "y": y,
            "v": visibility
        }
        
        self.num_frames = max(self.num_frames, frame_idx + 1)
    
    def convert_format(self, target_format):
        """
        Convert keypoints to a different format
        
        Args:
            target_format: Target format name (e.g., "mediapipe33", "body25", "rr21")
        
        Returns:
            BodyKeypoints: New BodyKeypoints object with converted data
        """
        if self.format_name == target_format:
            return self
        
        # Get the appropriate mapping
        if self.format_name == "mediapipe33" and target_format == "rr21":
            mapping = MEDIAPIPE33_TO_RR21_NAMES
        elif self.format_name == "body25" and target_format == "rr21":
            mapping = BODY25_TO_RR21_NAMES
        elif self.format_name == "rr21" and target_format == "mediapipe33":
            # Create reverse mapping
            mapping = {v: k for k, v in MEDIAPIPE33_TO_RR21_NAMES.items()}
        elif self.format_name == "rr21" and target_format == "body25":
            # Create reverse mapping
            mapping = {v: k for k, v in BODY25_TO_RR21_NAMES.items()}
        else:
            raise ValueError(f"No defined conversion path from {self.format_name} to {target_format}")
        
        # Create a new BodyKeypoints object with the target format
        result = BodyKeypoints(format_name=target_format)
        result.num_frames = self.num_frames
        
        # Convert each frame
        for frame_idx in range(self.num_frames):
            if frame_idx not in self.keypoints:
                continue
                
            result.keypoints[frame_idx] = {}
            source_frame = self.keypoints[frame_idx]
            
            for source_kp, target_kp in mapping.items():
                if source_kp in source_frame:
                    result.keypoints[frame_idx][target_kp] = {
                        "x": source_frame[source_kp]["x"],
                        "y": source_frame[source_kp]["y"],
                        "v": source_frame[source_kp].get("v", 1.0)
                    }
        
        return result
    
    def _detect_format(self, dataframe):
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

    @staticmethod
    def create_empty(num_frames, format_name):
        """
        Create an empty BodyKeypoints object with the specified number of frames
        
        Args:
            num_frames: Number of frames to create
            format_name: Format name (e.g., "mediapipe33", "body25", "rr21")
            
        Returns:
            BodyKeypoints: Empty BodyKeypoints object
        """
        keypoints = BodyKeypoints(format_name=format_name)
        keypoints.num_frames = num_frames
        return keypoints
    
    def save(self, file_path, format_name=None):
        """
        Save keypoints to file
        
        Args:
            file_path: Path to save the file
            format_name: Format to save as (defaults to current format)
            
        Returns:
            bool: Success or failure
        """
        if format_name is None:
            format_name = self.format_name
        
        # Convert format if needed
        keypoints = self
        if format_name != self.format_name:
            keypoints = self.convert_format(format_name)
        
        # Convert to DataFrame for saving
        df = keypoints.to_dataframe()
        
        # Save based on file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() == '.csv':
            df.to_csv(file_path, index=False)
            return True
        elif ext.lower() == '.pkl':
            df.to_pickle(file_path)
            return True
        elif ext.lower() == '.json':
            keypoints_dict = {
                "format": format_name,
                "frames": keypoints.keypoints
            }
            with open(file_path, 'w') as f:
                json.dump(keypoints_dict, f, cls=NumpyEncoder)
            return True
        else:
            return False
    
    @classmethod
    def load(cls, file_path, expected_frame_count=None, force_import=False):
        """
        Load keypoints from file
        
        Args:
            file_path: Path to load from
            expected_frame_count: Expected number of frames (for validation)
            force_import: Force import even if frame count doesn't match
            
        Returns:
            BodyKeypoints: Loaded keypoints
        """
        _, ext = os.path.splitext(file_path)
        
        if ext.lower() == '.csv':
            df = pd.read_csv(file_path)
            format_name = cls._detect_format_static(df)
            keypoints = cls(df, format_name)
            
        elif ext.lower() == '.pkl':
            df = pd.read_pickle(file_path)
            format_name = cls._detect_format_static(df)
            keypoints = cls(df, format_name)
            
        elif ext.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            format_name = data.get("format", "unknown")
            frames_data = data.get("frames", {})
            
            keypoints = cls(format_name=format_name)
            for frame_idx, frame_data in frames_data.items():
                frame_idx = int(frame_idx)  # Convert from string key
                keypoints.keypoints[frame_idx] = frame_data
                keypoints.num_frames = max(keypoints.num_frames, frame_idx + 1)
                
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Validate frame count if expected
        if expected_frame_count is not None and keypoints.num_frames != expected_frame_count:
            if not force_import:
                error_msg = f"File contains {keypoints.num_frames} frames, but expected {expected_frame_count}"
                raise ValueError(error_msg)
        
        return keypoints
    
    @staticmethod
    def _detect_format_static(dataframe):
        """
        Static method to detect format from a DataFrame
        
        Args:
            dataframe: Pandas DataFrame
            
        Returns:
            str: Format name
        """
        temp = BodyKeypoints()
        return temp._detect_format(dataframe)
    
    def to_openpose_json(self):
        """
        Convert keypoints to OpenPose JSON format
        
        Returns:
            list: List of OpenPose JSON frames
        """
        # First convert to body25 format if not already
        keypoints = self
        if self.format_name != "body25":
            keypoints = self.convert_format("body25")
        
        # Define mapping from body25 to OpenPose
        body25_to_openpose = {
            'NOSE': 0, 'NECK': 1, 'RIGHT_SHOULDER': 2, 'RIGHT_ELBOW': 3, 'RIGHT_WRIST': 4,
            'LEFT_SHOULDER': 5, 'LEFT_ELBOW': 6, 'LEFT_WRIST': 7, 'MIDHIP': 8, 'RIGHT_HIP': 9,
            'RIGHT_KNEE': 10, 'RIGHT_ANKLE': 11, 'LEFT_HIP': 12, 'LEFT_KNEE': 13, 'LEFT_ANKLE': 14,
            'RIGHT_EYE': 15, 'LEFT_EYE': 16, 'RIGHT_EAR': 17, 'LEFT_EAR': 18, 'LEFT_BIG_TOE': 19,
            'LEFT_SMALL_TOE': 20, 'LEFT_HEEL': 21, 'RIGHT_BIG_TOE': 22, 'RIGHT_SMALL_TOE': 23,
            'RIGHT_HEEL': 24
        }
        
        frames = []
        
        for frame_idx in range(keypoints.num_frames):
            if frame_idx not in keypoints.keypoints:
                continue
                
            keypoints_array = [0.0] * (25 * 3)
            frame_data = keypoints.keypoints[frame_idx]
            
            for kp, data in frame_data.items():
                if kp in body25_to_openpose:
                    openpose_idx = body25_to_openpose[kp]
                    
                    x_val = data.get("x", 0.0)
                    y_val = data.get("y", 0.0)
                    conf_val = data.get("v", 0.0)
                    
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
    
    def export_to_openpose(self, output_dir):
        """
        Export keypoints to OpenPose JSON files
        
        Args:
            output_dir: Output directory
            
        Returns:
            bool: Success or failure
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to OpenPose format
            frames = self.to_openpose_json()
            
            if len(frames) != self.num_frames:
                print(f"Warning: Number of generated OpenPose frames ({len(frames)}) does not match input frames ({self.num_frames}).")

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

# Helper functions that wrap BodyKeypoints methods for easier transition
def create_empty_pose_keypoints(num_frames, format_name):
    """Legacy wrapper for BodyKeypoints.create_empty()"""
    return BodyKeypoints.create_empty(num_frames, format_name)

def convert_format(keypoints, source_format, target_format):
    """Legacy wrapper for BodyKeypoints.convert_format()"""
    if isinstance(keypoints, pd.DataFrame):
        kp = BodyKeypoints(keypoints, source_format)
        return kp.convert_format(target_format).to_dataframe()
    elif isinstance(keypoints, BodyKeypoints):
        return keypoints.convert_format(target_format)
    else:
        raise ValueError("Input must be a DataFrame or BodyKeypoints object")

def detect_pose_format(dataframe):
    """Legacy wrapper for BodyKeypoints._detect_format_static()"""
    return BodyKeypoints._detect_format_static(dataframe)

def convert_list_to_keypoints(landmarks_list, format_name="mediapipe33"):
    """Legacy wrapper for BodyKeypoints._add_frame_from_list()"""
    kp = BodyKeypoints(format_name=format_name)
    kp._add_frame_from_list(landmarks_list, 0)
    return kp

def convert_list_to_dataframe(landmarks_list, format_name="mediapipe33"):
    """Legacy wrapper for compatibility"""
    kp = convert_list_to_keypoints(landmarks_list, format_name)
    return kp.to_dataframe()

def load_pose_data(file_path, expected_frame_count=None, force_import=False):
    """Legacy wrapper for BodyKeypoints.load()"""
    return BodyKeypoints.load(file_path, expected_frame_count, force_import)

def save_pose_data(pose_data, file_path, format_name="rr21"):
    """Legacy wrapper for BodyKeypoints.save()"""
    if isinstance(pose_data, pd.DataFrame):
        kp = BodyKeypoints(pose_data, format_name)
        return kp.save(file_path, format_name)
    elif isinstance(pose_data, BodyKeypoints):
        return pose_data.save(file_path, format_name)
    else:
        raise ValueError("Input must be a DataFrame or BodyKeypoints object")

def export_to_openpose(pose_data, output_dir, format_name="mediapipe33"):
    """Legacy wrapper for BodyKeypoints.export_to_openpose()"""
    if isinstance(pose_data, pd.DataFrame):
        kp = BodyKeypoints(pose_data, format_name)
        return kp.export_to_openpose(output_dir)
    elif isinstance(pose_data, BodyKeypoints):
        return pose_data.export_to_openpose(output_dir)
    else:
        raise ValueError("Input must be a DataFrame or BodyKeypoints object")