import os

# Project Roots
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Core Folders
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
PUBLIC_DIR = os.path.join(DATA_DIR, "public")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUT_IMAGES_DIR = os.path.join(OUTPUTS_DIR, "images")
OUTPUT_VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")
SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")

# Ensure all output directories exist
for folder in [OUTPUT_IMAGES_DIR, OUTPUT_VIDEOS_DIR, SCREENSHOTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Model Weights Paths
CAR_DAMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "car_damage_yolo11m.pt")
FIRE_SEG_MODEL_PATH = os.path.join(MODELS_DIR, "fire_seg_yolo11n.pt")
MOBILE_MODEL_PATH = os.path.join(MODELS_DIR, "mobile_model.h5")

# Default Confidence Thresholds
DEFAULT_DAMAGE_CONF = 0.40  # 40% confidence for vehicle damage
DEFAULT_FIRE_CONF = 0.25    # 25% confidence for fire segmentation

# Color Mapping for Damage Classes (BGR Format)
DAMAGE_COLOR_MAP = {
    'crack': (220, 20, 220),         # Neon Purple
    'dent': (0, 220, 255),           # Bright Yellow
    'scratch': (50, 255, 50),        # Neon Green
    'broken_lamp': (255, 120, 0),    # Cyan / Sky Blue
    'shattered_glass': (0, 140, 255),# Vibrant Orange
    'flat_tire': (255, 0, 100)       # Pinkish Red
}

DEFAULT_DETECTION_COLOR = (0, 255, 255)  # Yellow default
FIRE_COLOR = (0, 69, 255)                # Orange-Red for fire

# Detectable Damage Classes
DAMAGE_CLASSES = [
    'crack',
    'dent',
    'scratch',
    'broken_lamp',
    'shattered_glass',
    'flat_tire'
]
