import os
import cv2
from src.config import PROJECT_ROOT, DATA_DIR, IMAGES_DIR, VIDEOS_DIR

def resolve_media_path(path_or_name: str, media_type: str = 'auto') -> str:
    """
    Intelligently resolves the absolute path of an image or video file.
    Checks:
    1. Absolute path or relative to current working directory
    2. Path relative to data/images/ or data/videos/
    3. Path relative to data/
    4. Path relative to PROJECT_ROOT
    """
    if not path_or_name:
        return ""

    # Direct check
    if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
        return path_or_name

    if os.path.exists(path_or_name):
        return os.path.abspath(path_or_name)

    # Search candidates
    filename = os.path.basename(path_or_name)
    candidates = [
        os.path.join(IMAGES_DIR, path_or_name),
        os.path.join(IMAGES_DIR, filename),
        os.path.join(VIDEOS_DIR, path_or_name),
        os.path.join(VIDEOS_DIR, filename),
        os.path.join(DATA_DIR, path_or_name),
        os.path.join(PROJECT_ROOT, path_or_name),
        os.path.join(PROJECT_ROOT, filename)
    ]

    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)

    # If still not found, return the original absolute path representation
    return os.path.abspath(path_or_name)

def load_image(image_path: str):
    """
    Loads an image from path safely using OpenCV.
    Returns: (cv2_image_bgr, resolved_path) or raises FileNotFoundError.
    """
    resolved = resolve_media_path(image_path, media_type='image')
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Image not found at '{image_path}' (Resolved: '{resolved}')")

    img = cv2.imread(resolved)
    if img is None:
        raise ValueError(f"Failed to decode image at '{resolved}'")
    return img, resolved

def open_video(video_path: str):
    """
    Opens a video file or camera stream.
    Returns: (cv2.VideoCapture, metadata_dict, resolved_path)
    """
    # Camera index check (e.g., '0' or 0)
    if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
        cam_idx = int(video_path)
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera stream at index {cam_idx}")
        meta = {
            'is_camera': True,
            'fps': cap.get(cv2.CAP_PROP_FPS) or 30.0,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': -1
        }
        return cap, meta, f"Camera {cam_idx}"

    resolved = resolve_media_path(video_path, media_type='video')
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Video file not found at '{video_path}' (Resolved: '{resolved}')")

    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file at '{resolved}'")

    meta = {
        'is_camera': False,
        'fps': cap.get(cv2.CAP_PROP_FPS) or 30.0,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    return cap, meta, resolved
