"""
Keypoint format definitions and mappings for pose data.
This module provides standardized definitions for different keypoint formats
and mappings between them for pose data conversion.
"""

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

# Create name-based mappings for easier reference
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