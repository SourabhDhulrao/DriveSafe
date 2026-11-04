"""
DriveSafe - Feature Extraction Engine
Calculates EAR, MAR, Head Pose (solvePnP), Blink Dynamics, Yawn Detection, and PERCLOS.
"""
import time
import math
from collections import deque
import cv2
import numpy as np

from core.config import (
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    MOUTH_LANDMARKS,
    FACE_3D_MODEL_POINTS,
    HEAD_POSE_LANDMARK_INDICES,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_EAR_REOPEN_THRESHOLD,
    DEFAULT_MAR_THRESHOLD,
    DEFAULT_YAWN_MIN_DURATION_SEC,
    DEFAULT_PITCH_NOD_DROP_DEG,
    DEFAULT_MICROSLEEP_SEC,
    PERCLOS_WINDOW_SEC,
    EAR_SMOOTHING_WINDOW,
    MAR_SMOOTHING_WINDOW,
    HEAD_POSE_SMOOTHING_ALPHA,
    FACE_3D_MODEL_POINTS_EXTENDED,
    HEAD_POSE_LANDMARK_INDICES_EXTENDED,
)


def euclidean_dist(p1, p2):
    """Calculate Euclidean distance between two 2D points (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ---------------------------------------------------------------------------
# Rolling Median Signal Smoother  (shared by EAR / MAR / Head Pose)
# ---------------------------------------------------------------------------

class SignalSmoother:
    """
    Applies a rolling median filter to a scalar signal.
    Median filtering is preferred over simple averaging because it preserves
    sharp genuine transitions (e.g. eye closure onset) while eliminating
    single-frame outlier spikes caused by landmark jitter.
    """
    def __init__(self, window_size=5):
        self.window_size = max(1, window_size)
        self._buffer = deque(maxlen=self.window_size)

    def smooth(self, value):
        self._buffer.append(value)
        if len(self._buffer) < 3:
            return value
        sorted_buf = sorted(self._buffer)
        mid = len(sorted_buf) // 2
        return sorted_buf[mid]


# Global smoothers (instantiated once, reused across frames)
_ear_smoother_left = SignalSmoother(window_size=EAR_SMOOTHING_WINDOW)
_ear_smoother_right = SignalSmoother(window_size=EAR_SMOOTHING_WINDOW)
_mar_smoother = SignalSmoother(window_size=MAR_SMOOTHING_WINDOW)

# Head pose EMA state
_prev_pitch = 0.0
_prev_yaw = 0.0
_prev_roll = 0.0
_head_pose_initialized = False


def calculate_ear_single(eye_points):
    """
    Calculate Eye Aspect Ratio (EAR) for a single eye given 6 points.
    Points order: [P1, P2, P3, P4, P5, P6]
    EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = euclidean_dist(p2, p6)
    vertical_2 = euclidean_dist(p3, p5)
    horizontal = euclidean_dist(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_ear(landmarks_dict):
    """
    Compute EAR for both eyes from landmark coordinate dictionary {index: (x, y)}.
    Returns: (left_ear, right_ear, avg_ear)

    Enhanced: applies per-eye rolling median smoothing to eliminate single-frame
    jitter before the value reaches the blink tracker.
    """
    try:
        left_pts = [landmarks_dict[idx] for idx in LEFT_EYE_LANDMARKS]
        right_pts = [landmarks_dict[idx] for idx in RIGHT_EYE_LANDMARKS]

        left_ear_raw = calculate_ear_single(left_pts)
        right_ear_raw = calculate_ear_single(right_pts)

        # Rolling median smoothing
        left_ear = _ear_smoother_left.smooth(left_ear_raw)
        right_ear = _ear_smoother_right.smooth(right_ear_raw)
        avg_ear = (left_ear + right_ear) / 2.0

        return float(left_ear), float(right_ear), float(avg_ear)
    except KeyError:
        return 0.0, 0.0, 0.0


def calculate_mar(landmarks_dict):
    """
    Compute Mouth Aspect Ratio (MAR) using standard 6-point formulation:
    MAR = (|P_up_left - P_low_left| + |P_up_mid - P_low_mid| + |P_up_right - P_low_right|) / (2 * |P_left - P_right|)

    Enhanced: rolling median smoothing filters out lip landmark jitter that
    causes false yawn-start / yawn-end transitions.
    """
    try:
        p_left = landmarks_dict[MOUTH_LANDMARKS["left_corner"]]
        p_right = landmarks_dict[MOUTH_LANDMARKS["right_corner"]]
        p_top_mid = landmarks_dict[MOUTH_LANDMARKS["upper_lip"]]
        p_bot_mid = landmarks_dict[MOUTH_LANDMARKS["lower_lip"]]
        p_top_l = landmarks_dict[MOUTH_LANDMARKS["upper_left"]]
        p_bot_l = landmarks_dict[MOUTH_LANDMARKS["lower_left"]]
        p_top_r = landmarks_dict[MOUTH_LANDMARKS["upper_right"]]
        p_bot_r = landmarks_dict[MOUTH_LANDMARKS["lower_right"]]

        vert_1 = euclidean_dist(p_top_l, p_bot_l)
        vert_2 = euclidean_dist(p_top_mid, p_bot_mid)
        vert_3 = euclidean_dist(p_top_r, p_bot_r)
        horiz = euclidean_dist(p_left, p_right)

        if horiz < 1e-6:
            return 0.0

        mar_raw = (vert_1 + vert_2 + vert_3) / (2.0 * horiz)
        return float(_mar_smoother.smooth(mar_raw))
    except KeyError:
        return 0.0


def estimate_head_pose(landmarks_dict, img_w, img_h):
    """
    Estimate Head Pose (Pitch, Yaw, Roll) via cv2.solvePnP.

    Enhanced accuracy:
      1. Uses 8 anchor landmarks instead of 6 for a more constrained PnP
         solve that is less susceptible to individual landmark jitter.
      2. Uses SOLVEPNP_SQPNP solver which is algebraically optimal for
         small-N point sets and more robust than the iterative solver.
      3. Applies EMA temporal smoothing on the final Euler angles to
         eliminate frame-to-frame oscillation.

    Returns:
        pitch: degrees (positive = looking down, negative = looking up)
        yaw: degrees (positive = looking right, negative = looking left)
        roll: degrees (positive = tilting right, negative = tilting left)
        nose_2d: (x, y) coordinates
        nose_endpoint_2d: (x, y) for 3D orientation projection axis
    """
    global _prev_pitch, _prev_yaw, _prev_roll, _head_pose_initialized

    try:
        # Use the extended 8-point model for better accuracy
        indices = HEAD_POSE_LANDMARK_INDICES_EXTENDED
        model_points_list = FACE_3D_MODEL_POINTS_EXTENDED

        image_points = []
        for idx in indices:
            pt = landmarks_dict[idx]
            image_points.append([pt[0], pt[1]])

        image_points = np.array(image_points, dtype=np.float64)
        model_points = np.array(model_points_list, dtype=np.float64)

        # Approximate camera matrix assuming principal point at image center
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # SOLVEPNP_SQPNP: algebraically optimal for small point sets
        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP
        )

        if not success:
            return 0.0, 0.0, 0.0, (0, 0), (0, 0)

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)

        # Compute Euler angles from rotation matrix
        # Using decomposition: pitch around X, yaw around Y, roll around Z
        sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(rmat[2, 1], rmat[2, 2])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
            yaw = math.atan2(-rmat[2, 0], sy)
            roll = 0.0

        # Convert radians to degrees
        pitch_deg = float(np.degrees(pitch))
        yaw_deg = float(np.degrees(yaw))
        roll_deg = float(np.degrees(roll))

        # --- EMA temporal smoothing on Euler angles ---
        alpha = HEAD_POSE_SMOOTHING_ALPHA
        if _head_pose_initialized:
            pitch_deg = alpha * pitch_deg + (1 - alpha) * _prev_pitch
            yaw_deg = alpha * yaw_deg + (1 - alpha) * _prev_yaw
            roll_deg = alpha * roll_deg + (1 - alpha) * _prev_roll

        _prev_pitch = pitch_deg
        _prev_yaw = yaw_deg
        _prev_roll = roll_deg
        _head_pose_initialized = True

        # Project a 3D line from nose outward to visualize gaze/pose direction
        nose_3d_point = np.array([[0.0, 0.0, 1000.0]])
        nose_endpoint_2d, _ = cv2.projectPoints(nose_3d_point, rvec, tvec, camera_matrix, dist_coeffs)

        nose_2d = (int(image_points[0][0]), int(image_points[0][1]))
        endpoint = (int(nose_endpoint_2d[0][0][0]), int(nose_endpoint_2d[0][0][1]))

        return pitch_deg, yaw_deg, roll_deg, nose_2d, endpoint
    except Exception:
        return 0.0, 0.0, 0.0, (0, 0), (0, 0)


class BlinkTracker:
    """
    Stateful blink tracker with hysteresis.
    Calculates blink count, duration (ms), blink rate (blinks/min), and microsleep triggers.
    """
    def __init__(self, ear_threshold=DEFAULT_EAR_THRESHOLD, reopen_threshold=DEFAULT_EAR_REOPEN_THRESHOLD):
        self.ear_threshold = ear_threshold
        self.reopen_threshold = reopen_threshold
        self.is_closed = False
        self.close_start_time = None
        self.last_blink_duration_ms = 0.0
        self.total_blinks = 0
        self.recent_blinks = deque()  # (timestamp, duration_ms)
        self.current_closure_duration_sec = 0.0
        self.is_microsleep = False

    def update_thresholds(self, close_thresh, reopen_thresh):
        self.ear_threshold = close_thresh
        self.reopen_threshold = reopen_thresh

    def update(self, current_ear, current_time=None):
        if current_time is None:
            current_time = time.time()

        # State transition: OPEN -> CLOSED
        if not self.is_closed and current_ear < self.ear_threshold:
            self.is_closed = True
            self.close_start_time = current_time
            self.current_closure_duration_sec = 0.0
            self.is_microsleep = False

        # Ongoing closure: check for microsleep
        elif self.is_closed and current_ear <= self.reopen_threshold:
            self.current_closure_duration_sec = current_time - self.close_start_time
            if self.current_closure_duration_sec >= DEFAULT_MICROSLEEP_SEC:
                self.is_microsleep = True

        # State transition: CLOSED -> OPEN
        elif self.is_closed and current_ear > self.reopen_threshold:
            duration_sec = current_time - self.close_start_time
            duration_ms = duration_sec * 1000.0

            # Filter out spurious noise (< 50ms)
            if duration_ms >= 50.0:
                self.total_blinks += 1
                self.last_blink_duration_ms = round(duration_ms, 1)
                self.recent_blinks.append((current_time, self.last_blink_duration_ms))

            self.is_closed = False
            self.close_start_time = None
            self.current_closure_duration_sec = 0.0
            self.is_microsleep = False

        # Prune blinks older than 60 seconds
        cutoff_time = current_time - 60.0
        while self.recent_blinks and self.recent_blinks[0][0] < cutoff_time:
            self.recent_blinks.popleft()

        # Calculate rolling blinks per minute
        blink_rate_bpm = len(self.recent_blinks)

        # Calculate average duration over recent window
        if self.recent_blinks:
            avg_duration_ms = sum(d for _, d in self.recent_blinks) / len(self.recent_blinks)
        else:
            avg_duration_ms = self.last_blink_duration_ms

        return {
            "is_closed": self.is_closed,
            "blink_count": self.total_blinks,
            "blink_rate_bpm": blink_rate_bpm,
            "last_blink_duration_ms": self.last_blink_duration_ms,
            "avg_blink_duration_ms": round(avg_duration_ms, 1),
            "closure_duration_sec": round(self.current_closure_duration_sec, 2),
            "is_microsleep": self.is_microsleep
        }


class YawnTracker:
    """
    Tracks sustained mouth opening to reliably detect yawning vs. speaking/smiling.
    """
    def __init__(self, mar_threshold=DEFAULT_MAR_THRESHOLD, min_duration=DEFAULT_YAWN_MIN_DURATION_SEC):
        self.mar_threshold = mar_threshold
        self.min_duration = min_duration
        self.is_yawning = False
        self.yawn_start_time = None
        self.total_yawns = 0
        self.current_yawn_duration = 0.0
        self.recent_yawns = deque()  # timestamps

    def update_threshold(self, mar_threshold):
        self.mar_threshold = mar_threshold

    def update(self, current_mar, current_time=None):
        if current_time is None:
            current_time = time.time()

        just_completed_yawn = False

        if current_mar >= self.mar_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
            self.current_yawn_duration = current_time - self.yawn_start_time
            if self.current_yawn_duration >= self.min_duration:
                self.is_yawning = True
        else:
            if self.yawn_start_time is not None:
                duration = current_time - self.yawn_start_time
                if duration >= self.min_duration:
                    self.total_yawns += 1
                    self.recent_yawns.append(current_time)
                    just_completed_yawn = True
            self.is_yawning = False
            self.yawn_start_time = None
            self.current_yawn_duration = 0.0

        # Prune yawns older than 5 minutes
        cutoff = current_time - 300.0
        while self.recent_yawns and self.recent_yawns[0] < cutoff:
            self.recent_yawns.popleft()

        return {
            "is_yawning": self.is_yawning,
            "yawn_duration_sec": round(self.current_yawn_duration, 2),
            "total_yawns": self.total_yawns,
            "recent_yawns_5min": len(self.recent_yawns),
            "just_completed_yawn": just_completed_yawn
        }


class HeadNodTracker:
    """
    Detects downward head droop / nodding episodes characteristic of microsleeps.
    Compares current pitch to driver's calibrated neutral pitch.
    """
    def __init__(self, baseline_pitch=0.0, drop_threshold_deg=DEFAULT_PITCH_NOD_DROP_DEG):
        self.baseline_pitch = baseline_pitch
        self.drop_threshold_deg = drop_threshold_deg
        self.is_nodding = False
        self.nod_start_time = None
        self.total_nods = 0
        self.current_nod_duration = 0.0
        self.recent_nods = deque()

    def set_baseline(self, baseline_pitch):
        self.baseline_pitch = baseline_pitch

    def update(self, current_pitch, current_time=None):
        if current_time is None:
            current_time = time.time()

        just_nodded = False
        pitch_deviation = current_pitch - self.baseline_pitch

        # Standard pose: looking down increases positive pitch
        is_dipping = pitch_deviation >= self.drop_threshold_deg

        if is_dipping:
            if self.nod_start_time is None:
                self.nod_start_time = current_time
            self.current_nod_duration = current_time - self.nod_start_time
            if self.current_nod_duration >= 0.6:  # Sustained droop for at least 600ms
                self.is_nodding = True
        else:
            if self.is_nodding:
                # Driver recovered their head up after a sustained droop -> completed nod!
                self.total_nods += 1
                self.recent_nods.append(current_time)
                just_nodded = True
            self.is_nodding = False
            self.nod_start_time = None
            self.current_nod_duration = 0.0

        # Prune nods older than 5 minutes
        cutoff = current_time - 300.0
        while self.recent_nods and self.recent_nods[0] < cutoff:
            self.recent_nods.popleft()

        return {
            "is_nodding": self.is_nodding,
            "pitch_deviation": round(pitch_deviation, 1),
            "nod_duration_sec": round(self.current_nod_duration, 2),
            "total_nods": self.total_nods,
            "recent_nods_5min": len(self.recent_nods),
            "just_nodded": just_nodded
        }


class PERCLOSCalculator:
    """
    Percentage of Eye Closure (PERCLOS) over a sliding time window (default 30 seconds).
    Standard NHTSA metric: percentage of time driver's eyes are 80%+ closed.
    """
    def __init__(self, window_sec=PERCLOS_WINDOW_SEC):
        self.window_sec = window_sec
        self.history = deque()  # (timestamp, is_closed)

    def update(self, is_closed, current_time=None):
        if current_time is None:
            current_time = time.time()

        self.history.append((current_time, is_closed))

        # Evict samples outside the window
        cutoff = current_time - self.window_sec
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        if len(self.history) < 5:
            return 0.0

        closed_count = sum(1 for _, closed in self.history if closed)
        perclos = closed_count / float(len(self.history))
        return float(round(perclos, 4))
