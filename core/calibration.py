"""
DriveSafe - Personalized Auto-Calibration System
Learns driver-specific baselines for EAR, resting blink dynamics, and neutral head orientation.
Adapts detection thresholds to the individual driver.
"""
import time
import numpy as np
from core.config import (
    CALIBRATION_DURATION_SEC,
    CALIBRATION_MIN_FRAMES,
    DEFAULT_EAR_THRESHOLD,
    DEFAULT_EAR_REOPEN_THRESHOLD,
    DEFAULT_MAR_THRESHOLD,
    DEFAULT_PITCH_NOD_DROP_DEG
)


class CalibrationManager:
    """
    Manages personalized driver calibration sessions.
    Collects alert-state samples over a defined window (default 10s),
    derives customized baseline metrics, and computes tailored safety thresholds.
    """
    def __init__(self, duration_sec=CALIBRATION_DURATION_SEC):
        self.duration_sec = duration_sec
        self.state = "IDLE"  # IDLE, CALIBRATING, COMPLETED, FAILED
        self.start_time = None
        self.driver_name = "Default Driver"

        # Raw sample buffers
        self.ear_samples = []
        self.mar_samples = []
        self.pitch_samples = []
        self.yaw_samples = []
        self.roll_samples = []
        self.blink_count = 0

        # Calibrated Driver Baselines (defaults before calibration)
        self.baselines = {
            "is_calibrated": False,
            "calibrated_at": None,
            "driver_name": self.driver_name,
            "sample_count": 0,
            # Baselines
            "baseline_ear": 0.30,
            "baseline_mar": 0.25,
            "baseline_pitch": 0.0,
            "baseline_yaw": 0.0,
            "baseline_roll": 0.0,
            "baseline_blink_rate": 16.0,
            # Personalized Adaptive Thresholds
            "ear_threshold": DEFAULT_EAR_THRESHOLD,
            "ear_reopen_threshold": DEFAULT_EAR_REOPEN_THRESHOLD,
            "mar_threshold": DEFAULT_MAR_THRESHOLD,
            "pitch_nod_threshold": DEFAULT_PITCH_NOD_DROP_DEG,
            "calibration_quality": "Default (Uncalibrated)"
        }

    def start_calibration(self, driver_name=None, duration_sec=None):
        """Initiate an auto-calibration period."""
        if driver_name:
            self.driver_name = driver_name
        if duration_sec:
            self.duration_sec = duration_sec

        self.state = "CALIBRATING"
        self.start_time = time.time()
        self.ear_samples.clear()
        self.mar_samples.clear()
        self.pitch_samples.clear()
        self.yaw_samples.clear()
        self.roll_samples.clear()
        self.blink_count = 0

        return {
            "status": "started",
            "duration_sec": self.duration_sec,
            "message": "Auto-calibration initiated. Please look straight ahead at the road and maintain normal alert posture."
        }

    def process_frame(self, ear, mar, pitch, yaw, roll, blink_increment=0, current_time=None):
        """
        Process a single frame during calibration.
        Returns calibration progress dict (0-100%).
        """
        if self.state != "CALIBRATING":
            return self.get_status()

        if current_time is None:
            current_time = time.time()

        elapsed = current_time - self.start_time
        progress_pct = min(100.0, round((elapsed / self.duration_sec) * 100.0, 1))

        # Collect valid measurements
        if ear > 0.05:  # Valid face detection
            self.ear_samples.append(ear)
            self.mar_samples.append(mar)
            self.pitch_samples.append(pitch)
            self.yaw_samples.append(yaw)
            self.roll_samples.append(roll)
            self.blink_count += blink_increment

        # Check if calibration window has elapsed
        if elapsed >= self.duration_sec:
            self._finalize_calibration()

        return {
            "state": self.state,
            "progress_pct": progress_pct,
            "elapsed_sec": round(elapsed, 1),
            "remaining_sec": max(0.0, round(self.duration_sec - elapsed, 1)),
            "samples_collected": len(self.ear_samples),
            "is_calibrated": self.baselines["is_calibrated"]
        }

    def _finalize_calibration(self):
        """Compute statistical baselines and adapt safety thresholds."""
        if len(self.ear_samples) < CALIBRATION_MIN_FRAMES:
            self.state = "FAILED"
            self.baselines["calibration_quality"] = "Failed (Insufficient Face Detections)"
            return

        ear_arr = np.array(self.ear_samples)
        mar_arr = np.array(self.mar_samples)
        pitch_arr = np.array(self.pitch_samples)
        yaw_arr = np.array(self.yaw_samples)
        roll_arr = np.array(self.roll_samples)

        # To avoid skewing baseline EAR by normal blinks during calibration,
        # use the 75th percentile / upper quartile of open eye measurements
        # and discard blinks below 15th percentile
        open_eye_mask = ear_arr >= np.percentile(ear_arr, 20)
        if np.any(open_eye_mask):
            baseline_ear = float(np.mean(ear_arr[open_eye_mask]))
        else:
            baseline_ear = float(np.median(ear_arr))

        baseline_mar = float(np.median(mar_arr))
        baseline_pitch = float(np.median(pitch_arr))
        baseline_yaw = float(np.median(yaw_arr))
        baseline_roll = float(np.median(roll_arr))

        # Extrapolate baseline blink rate to BPM
        observed_bpm = (self.blink_count / max(1.0, self.duration_sec)) * 60.0
        # Clamp to realistic human range (10-25 bpm)
        baseline_blink_rate = max(10.0, min(30.0, observed_bpm if observed_bpm > 0 else 16.0))

        # Adapt detection thresholds to this specific driver:
        # Closure threshold is tailored to driver's open EAR (72% of open EAR)
        # Clamped within safe ergonomic bounds [0.18, 0.28]
        adapted_ear_threshold = round(max(0.18, min(0.28, baseline_ear * 0.72)), 3)
        adapted_ear_reopen = round(max(adapted_ear_threshold + 0.03, min(0.31, baseline_ear * 0.82)), 3)

        # Mouth opening threshold adapted to driver's neutral mouth shape
        # Minimum yawn MAR 0.52, maximum 0.70
        adapted_mar_threshold = round(max(0.52, min(0.70, baseline_mar * 2.2)), 3)

        # Head nod threshold: 14 degrees downward from driver's natural posture
        adapted_pitch_nod = 14.0

        # Assess calibration quality
        ear_variance = float(np.var(ear_arr[open_eye_mask])) if np.any(open_eye_mask) else 1.0
        if ear_variance < 0.003 and len(self.ear_samples) >= CALIBRATION_MIN_FRAMES:
            quality = "Optimal (High Confidence)"
        else:
            quality = "Good (Adaptive Calibrated)"

        self.baselines = {
            "is_calibrated": True,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "driver_name": self.driver_name,
            "sample_count": len(self.ear_samples),
            "baseline_ear": round(baseline_ear, 3),
            "baseline_mar": round(baseline_mar, 3),
            "baseline_pitch": round(baseline_pitch, 1),
            "baseline_yaw": round(baseline_yaw, 1),
            "baseline_roll": round(baseline_roll, 1),
            "baseline_blink_rate": round(baseline_blink_rate, 1),
            "ear_threshold": adapted_ear_threshold,
            "ear_reopen_threshold": adapted_ear_reopen,
            "mar_threshold": adapted_mar_threshold,
            "pitch_nod_threshold": adapted_pitch_nod,
            "calibration_quality": quality
        }

        self.state = "COMPLETED"

    def get_status(self):
        """Return current calibration status and active thresholds."""
        elapsed = 0.0
        progress_pct = 0.0
        if self.state == "CALIBRATING" and self.start_time:
            elapsed = time.time() - self.start_time
            progress_pct = min(100.0, round((elapsed / self.duration_sec) * 100.0, 1))
        elif self.state == "COMPLETED":
            progress_pct = 100.0

        return {
            "state": self.state,
            "progress_pct": progress_pct,
            "elapsed_sec": round(elapsed, 1),
            "remaining_sec": max(0.0, round(self.duration_sec - elapsed, 1)) if self.state == "CALIBRATING" else 0.0,
            "samples_collected": len(self.ear_samples),
            "baselines": self.baselines
        }

    def reset_to_defaults(self):
        """Reset calibration back to system defaults."""
        self.state = "IDLE"
        self.start_time = None
        self.ear_samples.clear()
        self.mar_samples.clear()
        self.pitch_samples.clear()
        self.yaw_samples.clear()
        self.roll_samples.clear()
        self.blink_count = 0
        self.baselines = {
            "is_calibrated": False,
            "calibrated_at": None,
            "driver_name": self.driver_name,
            "sample_count": 0,
            "baseline_ear": 0.30,
            "baseline_mar": 0.25,
            "baseline_pitch": 0.0,
            "baseline_yaw": 0.0,
            "baseline_roll": 0.0,
            "baseline_blink_rate": 16.0,
            "ear_threshold": DEFAULT_EAR_THRESHOLD,
            "ear_reopen_threshold": DEFAULT_EAR_REOPEN_THRESHOLD,
            "mar_threshold": DEFAULT_MAR_THRESHOLD,
            "pitch_nod_threshold": DEFAULT_PITCH_NOD_DROP_DEG,
            "calibration_quality": "Default (Uncalibrated)"
        }
