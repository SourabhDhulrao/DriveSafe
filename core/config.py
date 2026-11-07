"""
DriveSafe - Global Configuration and Constants
"""
import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
DB_PATH = os.path.join(BASE_DIR, "drivesafe.db")

# MediaPipe Landmark Indices for Eye Aspect Ratio (EAR)
# 6 landmarks per eye: [P1(outer), P2(upper1), P3(upper2), P4(inner), P5(lower2), P6(lower1)]
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Additional detailed eye contours for rendering
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Mouth Aspect Ratio (MAR) Indices
# Corners: 61 (left), 291 (right)
# Upper/Lower center: 13, 14
# Vertical pairs: (81, 178), (13, 14), (311, 402)
MOUTH_LANDMARKS = {
    "left_corner": 61,
    "right_corner": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "upper_left": 81,
    "lower_left": 178,
    "upper_right": 311,
    "lower_right": 402,
}
MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Standard 3D Facial Model Reference Points for Head Pose Estimation (solvePnP)
# Units in millimeters from face center
FACE_3D_MODEL_POINTS = [
    (0.0, 0.0, 0.0),          # Nose tip (index 1)
    (0.0, -330.0, -65.0),     # Chin (index 152 or 199)
    (-225.0, 170.0, -135.0),  # Left eye outer corner (index 33)
    (225.0, 170.0, -135.0),   # Right eye outer corner (index 263)
    (-150.0, -150.0, -125.0), # Left mouth corner (index 61)
    (150.0, -150.0, -125.0)   # Right mouth corner (index 291)
]

HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

# Default Baseline Thresholds (Adaptive when auto-calibration runs)
DEFAULT_EAR_THRESHOLD = 0.22         # Below this = eye closure
DEFAULT_EAR_REOPEN_THRESHOLD = 0.25  # Hysteresis reopen
DEFAULT_PERCLOS_THRESHOLD = 0.20     # >20% closure in window = high risk
DEFAULT_MAR_THRESHOLD = 0.58         # Above this = mouth open
DEFAULT_YAWN_MIN_DURATION_SEC = 1.4  # Sustained open mouth for >1.4s = yawn
DEFAULT_PITCH_NOD_DROP_DEG = 14.0    # Pitch dip below neutral
DEFAULT_MICROSLEEP_SEC = 1.1         # Eye closure > 1.1s = microsleep event

# Calibration Defaults
CALIBRATION_DURATION_SEC = 10.0      # Calibration window length
CALIBRATION_MIN_FRAMES = 60          # Minimum valid face frames required

# PERCLOS Rolling Window Length
PERCLOS_WINDOW_SEC = 30.0

# Drowsiness Risk State Classification Bounds (0-100)
RISK_STATES = {
    "SAFE": {"range": (0, 30), "color": "#10b981", "label": "Safe", "code": "safe"},
    "CAUTION": {"range": (31, 60), "color": "#f59e0b", "label": "Caution", "code": "caution"},
    "HIGH_RISK": {"range": (61, 80), "color": "#f97316", "label": "High Risk", "code": "high_risk"},
    "CRITICAL": {"range": (81, 100), "color": "#ef4444", "label": "Critical Alert", "code": "critical"}
}

# Smoothing Parameters
EMA_ALPHA_RISE = 0.25   # Fast reaction when risk increases
EMA_ALPHA_DECAY = 0.08  # Gradual decay when condition normalizes

# === Face Detection Accuracy Enhancements ===

# Night Driver & Low-Light Enhancement System
ENABLE_NIGHT_ENHANCEMENT = True       # Automatic night & low-light driver tracking
NIGHT_MODE_DEFAULT = "auto"           # "auto", "on", "off"
NIGHT_LUMINANCE_THRESHOLD = 85.0      # Below this average luma (0-255), night mode activates
NIGHT_GAMMA_BASE = 0.48               # Base non-linear gamma curve for shadow lifting
NIGHT_CLAHE_CLIP_LIMIT = 3.8          # Enhanced CLAHE contrast for low-light facial edge recovery
NIGHT_MIN_DETECTION_CONFIDENCE = 0.30 # Balanced confidence threshold to lock onto faces in dark cabins
NIGHT_MIN_TRACKING_CONFIDENCE = 0.30  # Reliable tracking through shadows and head turns

# Input Preprocessing
ENABLE_CLAHE_PREPROCESSING = True     # Adaptive histogram equalization for low-light
CLAHE_CLIP_LIMIT = 2.5                # Contrast limiting for standard CLAHE
CLAHE_TILE_GRID_SIZE = (8, 8)         # Grid size for CLAHE tiles

# Landmark Temporal Smoothing (reduces jitter between frames)
LANDMARK_SMOOTHING_WINDOW = 3         # Number of frames to average landmarks over
LANDMARK_SMOOTHING_ALPHA = 0.6        # EMA weight for latest frame landmarks (higher = less smoothing)

# EAR/MAR Signal Smoothing
EAR_SMOOTHING_WINDOW = 5             # Rolling median window for EAR noise filtering
MAR_SMOOTHING_WINDOW = 5             # Rolling median window for MAR noise filtering

# Head Pose Stability
HEAD_POSE_SMOOTHING_ALPHA = 0.5       # EMA alpha for head pose angles (reduce jitter)

# Extended 3D model points for improved solvePnP accuracy (8-point model)
FACE_3D_MODEL_POINTS_EXTENDED = [
    (0.0, 0.0, 0.0),            # Nose tip (index 1)
    (0.0, -330.0, -65.0),       # Chin (index 152)
    (-225.0, 170.0, -135.0),    # Left eye outer corner (index 33)
    (225.0, 170.0, -135.0),     # Right eye outer corner (index 263)
    (-150.0, -150.0, -125.0),   # Left mouth corner (index 61)
    (150.0, -150.0, -125.0),    # Right mouth corner (index 291)
    (-125.0, 170.0, -135.0),    # Left eye inner corner (index 133)
    (125.0, 170.0, -135.0),     # Right eye inner corner (index 362)
]
HEAD_POSE_LANDMARK_INDICES_EXTENDED = [1, 152, 33, 263, 61, 291, 133, 362]
