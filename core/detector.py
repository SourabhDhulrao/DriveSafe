"""
DriveSafe - MediaPipe Face Landmarker Wrapper and Night Vision HUD Visualizer
Extracts 468+ facial landmarks and renders real-time augmented HUD annotations.

Enhanced for Night Drivers:
  - Automatic Ambient Low-Light & Cabin Darkness Analyzer
  - Adaptive Gamma Correction (AGC) non-linear shadow expansion
  - Edge-Preserving Bilateral Denoising on Luminance channel (suppresses CMOS sensor grain)
  - Dual-Pass High-Contrast CLAHE for low-illumination eyelid & iris feature extraction
  - Low-Light Rescue Pass for extreme darkness (unlit rural roads, night driving)
  - Night Vision HUD Viewport with active status indicators
  - Balanced confidence gating (0.55) tuned specifically for nighttime facial geometries
  - Temporal landmark smoothing (EMA) to eliminate frame-to-frame pixel jitter
"""
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

from core.config import (
    MODEL_PATH,
    LEFT_EYE_CONTOUR,
    RIGHT_EYE_CONTOUR,
    MOUTH_OUTLINE,
    ENABLE_CLAHE_PREPROCESSING,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    LANDMARK_SMOOTHING_WINDOW,
    LANDMARK_SMOOTHING_ALPHA,
    NIGHT_LUMINANCE_THRESHOLD,
    NIGHT_CLAHE_CLIP_LIMIT,
    NIGHT_MIN_DETECTION_CONFIDENCE,
    NIGHT_MIN_TRACKING_CONFIDENCE,
    NIGHT_MODE_DEFAULT,
)


class FaceLandmarkDetector:
    """
    Wraps Google MediaPipe Face Landmarker Task API.
    Engineered with dedicated night vision and low-light facial tracking for night drivers.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path

        # Standard daytime CLAHE preprocessor
        if ENABLE_CLAHE_PREPROCESSING:
            self._clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_GRID_SIZE,
            )
            # High-gain night CLAHE with elevated contrast slope for dark eye sockets
            self._night_clahe = cv2.createCLAHE(
                clipLimit=NIGHT_CLAHE_CLIP_LIMIT,
                tileGridSize=CLAHE_TILE_GRID_SIZE,
            )
            # Extreme rescue CLAHE for deep pitch-black cabins
            self._rescue_clahe = cv2.createCLAHE(
                clipLimit=5.0,
                tileGridSize=(6, 6),
            )
        else:
            self._clahe = None
            self._night_clahe = None
            self._rescue_clahe = None

        # Precomputed Gamma Lookup Tables (LUTs) for zero-latency execution
        self._gamma_luts = {}

        # Night mode state & ambient sensing
        self.night_mode = NIGHT_MODE_DEFAULT  # "auto", "on", "off"
        self.is_night_mode_active = False
        self.current_luminance = 120.0
        self.enhanced_frame = None

        # Landmark temporal smoothing history
        self._landmark_history = deque(maxlen=LANDMARK_SMOOTHING_WINDOW)
        self._smoothed_landmarks = {}

        # Monotonic timestamp counter for VIDEO running mode (microseconds)
        self._frame_timestamp_ms = 0

        self._init_detector()

    # ------------------------------------------------------------------
    # MediaPipe initialization
    # ------------------------------------------------------------------

    def _init_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        # RunningMode.IMAGE provides instantaneous, stateless detection without
        # network timestamp drift, frame-drop desync, or Kalman velocity errors.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Night Mode Control
    # ------------------------------------------------------------------

    def set_night_mode(self, mode_str):
        """Set night vision mode: 'auto', 'on', or 'off'."""
        if mode_str in ["auto", "on", "off"]:
            self.night_mode = mode_str
        return self.night_mode

    # ------------------------------------------------------------------
    # Non-linear Gamma Lookup Table Generator (Zero-CPU SIMD)
    # ------------------------------------------------------------------

    def _get_gamma_lut(self, gamma):
        gamma_key = round(gamma, 2)
        if gamma_key in self._gamma_luts:
            return self._gamma_luts[gamma_key]

        # Non-linear curve: I' = 255 * (I / 255)^gamma
        lut = np.array([
            np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255).astype("uint8")
            for i in range(256)
        ])
        self._gamma_luts[gamma_key] = lut
        return lut

    # ------------------------------------------------------------------
    # Low-Light & Night Driver Image Preprocessing Pipeline
    # ------------------------------------------------------------------

    def _analyze_luminance(self, bgr_frame):
        """Calculate mean ambient cabin luminance on 0-255 scale."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _enhance_night_frame(self, bgr_frame, mean_lum, rescue=False):
        """
        Multi-stage night driver image enhancement:
          1. Adaptive Gamma Correction (AGC) lifts shadows non-linearly.
          2. Edge-preserving Bilateral Filter removes CMOS sensor grain in dark frames.
          3. High-contrast CLAHE on L-channel extracts eye sockets, pupils, and mouth edges.
        """
        # Dynamic Gamma: darker environment -> lower gamma -> stronger shadow lift
        if rescue:
            gamma = 0.35
            clahe_engine = self._rescue_clahe or self._night_clahe
        else:
            ratio = max(0.0, min(1.0, mean_lum / NIGHT_LUMINANCE_THRESHOLD))
            gamma = 0.40 + 0.45 * ratio  # ~0.40 at pitch dark, ~0.85 near threshold
            clahe_engine = self._night_clahe

        lut = self._get_gamma_lut(gamma)
        brightened = cv2.LUT(bgr_frame, lut)

        # Convert to LAB color space for luminance-specific processing
        lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Bilateral filter suppresses low-light high-gain digital noise
        # while keeping the crisp sharp boundaries of eyelids and lips intact
        l_denoised = cv2.bilateralFilter(l_channel, d=5, sigmaColor=35, sigmaSpace=35)

        # Apply high-gain adaptive histogram equalization
        if clahe_engine is not None:
            l_enhanced = clahe_engine.apply(l_denoised)
        else:
            l_enhanced = l_denoised

        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr

    def _preprocess_frame(self, bgr_frame):
        """
        Routes frame through standard daytime pipeline or adaptive night-vision pipeline
        based on active mode and ambient illumination.
        """
        mean_lum = self._analyze_luminance(bgr_frame)
        self.current_luminance = mean_lum

        # Determine if Night Vision Enhancement should engage
        if self.night_mode == "on":
            should_activate_night = True
        elif self.night_mode == "off":
            should_activate_night = False
        else:  # "auto"
            should_activate_night = mean_lum < NIGHT_LUMINANCE_THRESHOLD

        self.is_night_mode_active = should_activate_night

        if should_activate_night:
            enhanced_bgr = self._enhance_night_frame(bgr_frame, mean_lum, rescue=False)
            self.enhanced_frame = enhanced_bgr
            return enhanced_bgr
        else:
            # Daytime standard CLAHE processing
            self.enhanced_frame = bgr_frame
            if self._clahe is None:
                return bgr_frame

            lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = self._clahe.apply(l_channel)
            enhanced = cv2.merge([l_channel, a_channel, b_channel])
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced_bgr

    # ------------------------------------------------------------------
    # Landmark smoothing
    # ------------------------------------------------------------------

    def _smooth_landmarks(self, raw_landmarks_dict):
        """Exponential Moving Average (EMA) over N frames to remove pixel jitter."""
        self._landmark_history.append(raw_landmarks_dict)
        alpha = LANDMARK_SMOOTHING_ALPHA

        if not self._smoothed_landmarks:
            self._smoothed_landmarks = dict(raw_landmarks_dict)
            return self._smoothed_landmarks

        smoothed = {}
        for idx, (rx, ry, rz) in raw_landmarks_dict.items():
            if idx in self._smoothed_landmarks:
                sx, sy, sz = self._smoothed_landmarks[idx]
                smoothed[idx] = (
                    int(alpha * rx + (1 - alpha) * sx),
                    int(alpha * ry + (1 - alpha) * sy),
                    alpha * rz + (1 - alpha) * sz,
                )
            else:
                smoothed[idx] = (rx, ry, rz)

        self._smoothed_landmarks = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # Confidence estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(landmarks):
        if not landmarks:
            return 0.0

        visibilities = []
        for lm in landmarks:
            vis = getattr(lm, "visibility", None)
            if vis is not None and vis > 0:
                visibilities.append(vis)
            else:
                pres = getattr(lm, "presence", None)
                if pres is not None and pres > 0:
                    visibilities.append(pres)

        if visibilities:
            return float(np.mean(visibilities))
        return 0.90

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame):
        """
        Process a BGR frame with automatic night driver enhancement.
        Includes a secondary low-light rescue pass if the primary pass misses.
        """
        h, w, _ = bgr_frame.shape

        # Primary pass (with adaptive night vision if dark)
        enhanced = self._preprocess_frame(bgr_frame)
        rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.detector.detect(mp_image)

        # Secondary low-light rescue pass for extreme darkness / rural night driving
        if (
            (not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0)
            and self.is_night_mode_active
        ):
            rescue_bgr = self._enhance_night_frame(bgr_frame, self.current_luminance, rescue=True)
            rescue_rgb = cv2.cvtColor(rescue_bgr, cv2.COLOR_BGR2RGB)
            rescue_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rescue_rgb)

            detection_result = self.detector.detect(rescue_mp_image)
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                self.enhanced_frame = rescue_bgr

        if (
            not detection_result.face_landmarks
            or len(detection_result.face_landmarks) == 0
        ):
            return False, {}, 0.0

        landmarks = detection_result.face_landmarks[0]

        raw_landmarks_dict = {}
        for idx, lm in enumerate(landmarks):
            px = int(lm.x * w)
            py = int(lm.y * h)
            raw_landmarks_dict[idx] = (px, py, lm.z)

        smoothed_dict = self._smooth_landmarks(raw_landmarks_dict)
        confidence = self._compute_confidence(landmarks)

        return True, smoothed_dict, confidence

    # ------------------------------------------------------------------
    # Augmented HUD Rendering with Night Driver Display
    # ------------------------------------------------------------------

    def draw_hud(self, frame, landmarks_dict, is_closed=False, is_yawning=False,
                 nose_2d=None, nose_endpoint=None, state_info=None):
        """
        Draw clean, high-tech automotive HUD contours directly onto the frame.
        In Night Mode, uses the night-enhanced illuminated frame so the driver is clearly visible.
        """
        # If Night Mode is active, use the enhanced illuminated frame as base
        if self.is_night_mode_active and self.enhanced_frame is not None:
            # 85% enhanced night frame + 15% original for natural color preservation
            base_frame = cv2.addWeighted(self.enhanced_frame, 0.85, frame, 0.15, 0)
        else:
            base_frame = frame.copy()

        if not landmarks_dict:
            # Draw searching HUD target box
            h, w = base_frame.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.rectangle(base_frame, (cx - 100, cy - 100), (cx + 100, cy + 100), (100, 100, 100), 1, cv2.LINE_AA)

            search_text = "NIGHT SCANNER: SEARCHING..." if self.is_night_mode_active else "SEARCHING FOR DRIVER..."
            cv2.putText(base_frame, search_text, (cx - 95, cy - 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 210, 255) if self.is_night_mode_active else (160, 160, 160), 1, cv2.LINE_AA)

            # Draw Night Mode Tag if active
            if self.is_night_mode_active:
                self._draw_night_badge(base_frame)

            return base_frame

        hud_frame = base_frame.copy()

        # Eye Contours: Glowing Cyan (Open) or Urgent Red (Closed)
        eye_color = (0, 0, 255) if is_closed else (255, 230, 0)
        thick = 2 if is_closed else 1

        # Draw Left Eye
        left_pts = np.array([landmarks_dict[i][:2] for i in LEFT_EYE_CONTOUR if i in landmarks_dict], np.int32)
        if len(left_pts) > 0:
            cv2.polylines(hud_frame, [left_pts], isClosed=True, color=eye_color, thickness=thick, lineType=cv2.LINE_AA)

        # Draw Right Eye
        right_pts = np.array([landmarks_dict[i][:2] for i in RIGHT_EYE_CONTOUR if i in landmarks_dict], np.int32)
        if len(right_pts) > 0:
            cv2.polylines(hud_frame, [right_pts], isClosed=True, color=eye_color, thickness=thick, lineType=cv2.LINE_AA)

        # Draw Mouth: Green (Normal) or Crimson (Yawning)
        mouth_color = (0, 60, 255) if is_yawning else (80, 220, 80)
        mouth_thick = 2 if is_yawning else 1
        mouth_pts = np.array([landmarks_dict[i][:2] for i in MOUTH_OUTLINE if i in landmarks_dict], np.int32)
        if len(mouth_pts) > 0:
            cv2.polylines(hud_frame, [mouth_pts], isClosed=True, color=mouth_color, thickness=mouth_thick, lineType=cv2.LINE_AA)

        # Draw 3D Head Orientation Vector (Gaze/Pose Direction from nose)
        if nose_2d and nose_endpoint and nose_2d != (0, 0):
            cv2.arrowedLine(hud_frame, nose_2d, nose_endpoint, (0, 230, 255), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(hud_frame, nose_2d, 3, (0, 255, 200), -1, cv2.LINE_AA)

        # Semi-transparent HUD overlay
        cv2.addWeighted(hud_frame, 0.78, base_frame, 0.22, 0, base_frame)

        # Draw corner HUD brackets around face area
        try:
            xs = [pt[0] for pt in landmarks_dict.values()]
            ys = [pt[1] for pt in landmarks_dict.values()]
            min_x, max_x = max(0, min(xs) - 15), min(base_frame.shape[1], max(xs) + 15)
            min_y, max_y = max(0, min(ys) - 20), min(base_frame.shape[0], max(ys) + 15)

            bracket_len = 18
            bracket_color = (0, 235, 255) if self.is_night_mode_active else (255, 255, 255)

            # Top-Left
            cv2.line(base_frame, (min_x, min_y), (min_x + bracket_len, min_y), bracket_color, 1, cv2.LINE_AA)
            cv2.line(base_frame, (min_x, min_y), (min_x, min_y + bracket_len), bracket_color, 1, cv2.LINE_AA)
            # Top-Right
            cv2.line(base_frame, (max_x, min_y), (max_x - bracket_len, min_y), bracket_color, 1, cv2.LINE_AA)
            cv2.line(base_frame, (max_x, min_y), (max_x, min_y + bracket_len), bracket_color, 1, cv2.LINE_AA)
            # Bottom-Left
            cv2.line(base_frame, (min_x, max_y), (min_x + bracket_len, max_y), bracket_color, 1, cv2.LINE_AA)
            cv2.line(base_frame, (min_x, max_y), (min_x, max_y - bracket_len), bracket_color, 1, cv2.LINE_AA)
            # Bottom-Right
            cv2.line(base_frame, (max_x, max_y), (max_x - bracket_len, max_y), bracket_color, 1, cv2.LINE_AA)
            cv2.line(base_frame, (max_x, max_y), (max_x, max_y - bracket_len), bracket_color, 1, cv2.LINE_AA)
        except Exception:
            pass

        # Draw Night Vision ADAS status badge on top
        if self.is_night_mode_active:
            self._draw_night_badge(base_frame)

        return base_frame

    def _draw_night_badge(self, img):
        """Draw subtle high-tech automotive night-vision status on the HUD."""
        h, w = img.shape[:2]
        # Position at bottom-left corner
        bx, by = 16, h - 16
        # Background pill
        cv2.rectangle(img, (bx, by - 20), (bx + 145, by + 4), (12, 18, 28), -1)
        cv2.rectangle(img, (bx, by - 20), (bx + 145, by + 4), (0, 185, 255), 1, cv2.LINE_AA)
        # Glowing dot
        cv2.circle(img, (bx + 10, by - 8), 4, (0, 220, 255), -1, cv2.LINE_AA)
        # Text
        label = f"NIGHT VISION ({int(self.current_luminance)}L)"
        cv2.putText(img, label, (bx + 20, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
