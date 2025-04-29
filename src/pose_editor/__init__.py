"""Pose-Editor package root."""
from .face_keypoints import *
from .hand_keypoints import *
from .body_keypoints import *
from .plot import *
from .pose_annotator import main as run_pose_editor
__all__ = ["run_pose_editor"]  # extend later as needed