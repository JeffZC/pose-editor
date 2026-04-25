import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import urllib.request
from pathlib import Path

from pose_format_utils import process_mediapipe_to_rr21, SUPPORTED_FORMATS


POSE_MODEL_VARIANTS = {
    "lite": {
        "filename": "pose_landmarker_lite.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    },
    "full": {
        "filename": "pose_landmarker_full.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    },
    "heavy": {
        "filename": "pose_landmarker_heavy.task",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    },
}


_image_landmarkers = {}


def normalize_pose_model_variant(model_variant: str) -> str:
    variant = (model_variant or "heavy").strip().lower()
    if variant not in POSE_MODEL_VARIANTS:
        return "heavy"
    return variant


def normalize_confidence_threshold(confidence_threshold: float) -> float:
    try:
        value = float(confidence_threshold)
    except (TypeError, ValueError):
        value = 0.9
    return max(0.0, min(1.0, value))


def download_pose_model(model_variant: str = "heavy") -> str:
    variant = normalize_pose_model_variant(model_variant)
    cfg = POSE_MODEL_VARIANTS[variant]

    model_dir = Path.home() / ".pose-editor" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / cfg["filename"]
    if not model_path.exists():
        req = urllib.request.Request(cfg["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(model_path, "wb") as out_file:
            out_file.write(response.read())

    return str(model_path)


def _get_image_landmarker(model_variant: str = "heavy", confidence_threshold: float = 0.9):
    variant = normalize_pose_model_variant(model_variant)
    threshold = normalize_confidence_threshold(confidence_threshold)
    cache_key = (variant, threshold)
    if cache_key not in _image_landmarkers:
        model_path = download_pose_model(variant)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=threshold,
            min_pose_presence_confidence=threshold,
            min_tracking_confidence=threshold,
        )
        _image_landmarkers[cache_key] = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    return _image_landmarkers[cache_key]


_video_landmarkers = {}


def _get_video_landmarker(model_variant: str = "heavy", confidence_threshold: float = 0.9):
    variant = normalize_pose_model_variant(model_variant)
    threshold = normalize_confidence_threshold(confidence_threshold)
    cache_key = (variant, threshold)
    if cache_key not in _video_landmarkers:
        model_path = download_pose_model(variant)
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=threshold,
            min_pose_presence_confidence=threshold,
            min_tracking_confidence=threshold,
        )
        _video_landmarkers[cache_key] = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    return _video_landmarkers[cache_key]


def _extract_landmarks_from_results(results, frame_shape):
    if not results or not results.pose_landmarks:
        return []

    height, width = frame_shape[:2]
    landmarks = []
    for landmark in results.pose_landmarks[0]:
        landmarks.append(landmark.x * width)
        landmarks.append(landmark.y * height)
    return landmarks


def _draw_landmarks(frame, results):
    if not results or not results.pose_landmarks:
        return frame

    annotated = frame.copy()
    height, width = annotated.shape[:2]

    landmarks = results.pose_landmarks[0]
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

    connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
    for connection in connections:
        start_idx = connection.start
        end_idx = connection.end
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            cv2.line(annotated, start_point, end_point, (0, 0, 255), 2)

    return annotated


def get_pose_landmarks_from_frame(frame, model_variant: str = "heavy", confidence_threshold: float = 0.9):
    """
    Detect pose landmarks for a single frame using MediaPipe Tasks API.

    Args:
        frame: OpenCV frame (BGR)

    Returns:
        tuple: (landmarks_list as flat [x1,y1,x2,y2...], annotated_frame)
    """
    if frame is None:
        return [], frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = _get_image_landmarker(model_variant, confidence_threshold).detect(mp_image)

    landmarks_list = _extract_landmarks_from_results(results, frame.shape)
    if not landmarks_list:
        return [], frame

    annotated_frame = _draw_landmarks(frame, results)
    return landmarks_list, annotated_frame


def process_video_with_mediapipe(video_path, progress_dialog=None, model_variant: str = "heavy", confidence_threshold: float = 0.9):
    """
    Process an entire video with MediaPipe pose detection.

    Args:
        video_path: Path to the video file
        progress_dialog: Optional progress object with setValue()/wasCanceled()

    Returns:
        tuple: (DataFrame with pose data in RR21 format, success flag)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None, False

        column_names = []
        for name in SUPPORTED_FORMATS["rr21"]:
            column_names.extend([f"{name}_X", f"{name}_Y"])

        pose_data = pd.DataFrame(np.zeros((frame_count, len(column_names))), columns=column_names)
        landmarker = _get_image_landmarker(model_variant, confidence_threshold)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if progress_dialog is not None:
                progress_percent = min(100, int((frame_idx / frame_count) * 100))
                try:
                    progress_dialog.setValue(progress_percent)
                    if hasattr(progress_dialog, "wasCanceled") and progress_dialog.wasCanceled():
                        break
                except Exception:
                    pass

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)

            landmarks_list = _extract_landmarks_from_results(results, frame.shape)
            if landmarks_list:
                rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
                for i in range(0, len(rr21_landmarks), 2):
                    if i + 1 < len(rr21_landmarks) and i // 2 < len(column_names) // 2:
                        pose_data.iloc[frame_idx, i] = rr21_landmarks[i]
                        pose_data.iloc[frame_idx, i + 1] = rr21_landmarks[i + 1]

            frame_idx += 1

        cap.release()
        return pose_data, True

    except Exception as e:
        print(f"Error processing video: {e}")
        if "cap" in locals() and cap is not None:
            cap.release()
        return None, False


def process_video_with_mediapipe_video_mode(video_path, progress_dialog=None, model_variant: str = "heavy", confidence_threshold: float = 0.9):
    """
    Process an entire video with MediaPipe pose detection using video mode.

    Args:
        video_path: Path to the video file
        progress_dialog: Optional progress object with setValue()/wasCanceled()

    Returns:
        tuple: (DataFrame with pose data in RR21 format, success flag)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return None, False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        column_names = []
        for name in SUPPORTED_FORMATS["rr21"]:
            column_names.extend([f"{name}_X", f"{name}_Y"])

        pose_data = pd.DataFrame(np.zeros((frame_count, len(column_names))), columns=column_names)
        landmarker = _get_video_landmarker(model_variant, confidence_threshold)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if progress_dialog is not None:
                progress_percent = min(100, int((frame_idx / frame_count) * 100))
                try:
                    progress_dialog.setValue(progress_percent)
                    if hasattr(progress_dialog, "wasCanceled") and progress_dialog.wasCanceled():
                        break
                except Exception:
                    pass

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            landmarks_list = _extract_landmarks_from_results(results, frame.shape)
            if landmarks_list:
                rr21_landmarks = process_mediapipe_to_rr21(landmarks_list)
                for i in range(0, len(rr21_landmarks), 2):
                    if i + 1 < len(rr21_landmarks) and i // 2 < len(column_names) // 2:
                        pose_data.iloc[frame_idx, i] = rr21_landmarks[i]
                        pose_data.iloc[frame_idx, i + 1] = rr21_landmarks[i + 1]

            frame_idx += 1

        cap.release()
        return pose_data, True

    except Exception as e:
        print(f"Error processing video in MediaPipe video mode: {e}")
        if "cap" in locals() and cap is not None:
            cap.release()
        return None, False
