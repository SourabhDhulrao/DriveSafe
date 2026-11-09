"""
DriveSafe - Unified Pipeline Controller and Camera Streaming Engine
Coordinates detection, feature extraction, calibration, risk scoring, and streaming.
"""
import time
import base64
import threading
import cv2
import numpy as np

from core.detector import FaceLandmarkDetector
from core.features import (
    calculate_ear,
    calculate_mar,
    estimate_head_pose,
    BlinkTracker,
    YawnTracker,
    HeadNodTracker,
    PERCLOSCalculator
)
from core.calibration import CalibrationManager
from core.scoring import DrowsinessScorer, HeuristicDrowsinessScorer
from core.database import DatabaseManager


class DriveSafePipeline:
    """
    Central real-time perception pipeline.
    Processes individual frames, extracts biometric cues, manages calibration,
    updates risk scoring, and renders futuristic HUD overlays.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = FaceLandmarkDetector()
        self.calibration = CalibrationManager()
        self.scorer = DrowsinessScorer(HeuristicDrowsinessScorer())
        self.db = DatabaseManager()

        # Trackers
        self.blink_tracker = BlinkTracker()
        self.yawn_tracker = YawnTracker()
        self.nod_tracker = HeadNodTracker()
        self.perclos_calc = PERCLOSCalculator()

        # Session tracking
        self.session_id = None
        self.session_start_time = None
        self.driver_name = "Driver 1"
        self.max_risk_observed = 0.0
        self.risk_sum = 0.0
        self.frame_count = 0

        # Performance & FPS tracking
        self.fps = 0.0
        self._fps_last_time = time.time()
        self._fps_frame_count = 0

        # Incident throttle to avoid flooding DB
        self._last_incident_log_time = {}

        # Latest telemetry snapshot
        self.latest_telemetry = self._get_initial_telemetry()
        self.latest_landmarks_summary = None

        # Visual preferences
        self.show_hud_overlay = True
        self.force_sunglasses_mode = False

        # Start initial active session
        self.start_new_session(self.driver_name)

    def _get_initial_telemetry(self):
        return {
            "timestamp": time.time(),
            "face_detected": False,
            "confidence": 0.0,
            "sunglasses_detected": False,
            "sunglasses_ratio": 1.0,
            "sunglasses_mode": False,
            "ear": 0.30,
            "left_ear": 0.30,
            "right_ear": 0.30,
            "mar": 0.25,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "perclos": 0.0,
            "blink_count": 0,
            "blink_rate_bpm": 16,
            "last_blink_ms": 200,
            "avg_blink_ms": 200,
            "is_closed": False,
            "closure_duration_sec": 0.0,
            "yawn_count": 0,
            "is_yawning": False,
            "yawn_duration_sec": 0.0,
            "nod_count": 0,
            "is_nodding": False,
            "nod_duration_sec": 0.0,
            "risk_score": 0.0,
            "raw_risk_score": 0.0,
            "risk_state": "Safe",
            "risk_code": "safe",
            "risk_color": "#10b981",
            "risk_breakdown": {},
            "override_reason": None,
            "fps": 0.0,
            "night_mode": False,
            "ambient_luminance": 120.0,
            "calibration": self.calibration.get_status()
        }

    def start_new_session(self, driver_name="Driver 1"):
        """Begin a new session in SQLite."""
        with self.lock:
            if self.session_id:
                self.end_current_session()

            self.driver_name = driver_name
            self.session_start_time = time.time()
            self.max_risk_observed = 0.0
            self.risk_sum = 0.0
            self.frame_count = 0
            self.scorer.reset()

            self.session_id = self.db.start_session(
                driver_name=self.driver_name,
                baselines=self.calibration.baselines
            )

    def end_current_session(self):
        """End current session and commit statistics."""
        if not self.session_id:
            return

        duration = time.time() - (self.session_start_time or time.time())
        avg_risk = (self.risk_sum / max(1, self.frame_count)) if self.frame_count > 0 else 0.0

        summary = {
            "duration_sec": round(duration, 1),
            "avg_risk": round(avg_risk, 1),
            "max_risk": round(self.max_risk_observed, 1),
            "total_blinks": self.blink_tracker.total_blinks,
            "total_yawns": self.yawn_tracker.total_yawns,
            "total_nods": self.nod_tracker.total_nods
        }

        self.db.end_session(self.session_id, summary)
        self.session_id = None

    def start_calibration(self, duration_sec=10.0):
        """Trigger auto-calibration for current driver."""
        with self.lock:
            res = self.calibration.start_calibration(
                driver_name=self.driver_name,
                duration_sec=duration_sec
            )
            return res

    def process_frame(self, frame_bgr):
        """
        Process a single image frame (BGR numpy array).
        Returns:
            annotated_frame_bgr
            telemetry_dict
        """
        now = time.time()

        # Compute FPS
        self._fps_frame_count += 1
        if now - self._fps_last_time >= 1.0:
            self.fps = round(self._fps_frame_count / (now - self._fps_last_time), 1)
            self._fps_frame_count = 0
            self._fps_last_time = now

        h, w = frame_bgr.shape[:2]

        # 1. MediaPipe Face Landmark Detection
        face_detected, landmarks_dict, confidence = self.detector.process_frame(frame_bgr)

        # 2. Extract Biometric Features
        landmarks_summary = None
        if face_detected and len(landmarks_dict) > 0:
            left_ear, right_ear, avg_ear = calculate_ear(landmarks_dict)
            mar = calculate_mar(landmarks_dict)
            pitch, yaw, roll, nose_2d, nose_endpoint = estimate_head_pose(landmarks_dict, w, h)
            try:
                xs = [pt[0] for pt in landmarks_dict.values()]
                ys = [pt[1] for pt in landmarks_dict.values()]
                from core.config import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, MOUTH_OUTLINE
                landmarks_summary = {
                    "bbox": [round(min(xs) / w, 4), round(min(ys) / h, 4), round(max(xs) / w, 4), round(max(ys) / h, 4)],
                    "left_eye": [[round(landmarks_dict[i][0] / w, 4), round(landmarks_dict[i][1] / h, 4)] for i in LEFT_EYE_CONTOUR if i in landmarks_dict],
                    "right_eye": [[round(landmarks_dict[i][0] / w, 4), round(landmarks_dict[i][1] / h, 4)] for i in RIGHT_EYE_CONTOUR if i in landmarks_dict],
                    "mouth": [[round(landmarks_dict[i][0] / w, 4), round(landmarks_dict[i][1] / h, 4)] for i in MOUTH_OUTLINE if i in landmarks_dict],
                    "nose_vector": [round(nose_2d[0] / w, 4), round(nose_2d[1] / h, 4), round(nose_endpoint[0] / w, 4), round(nose_endpoint[1] / h, 4)] if (nose_2d and nose_endpoint and nose_2d != (0, 0)) else None
                }
            except Exception:
                landmarks_summary = None
        else:
            left_ear, right_ear, avg_ear = 0.0, 0.0, 0.0
            mar = 0.0
            pitch, yaw, roll, nose_2d, nose_endpoint = 0.0, 0.0, 0.0, (0, 0), (0, 0)
        self.latest_landmarks_summary = landmarks_summary

        # 3. Handle Auto-Calibration
        prev_blink_count = self.blink_tracker.total_blinks
        calib_status = self.calibration.process_frame(
            ear=avg_ear,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            blink_increment=0,
            current_time=now
        )

        # If calibration just completed, update active thresholds!
        if self.calibration.state == "COMPLETED" and self.calibration.baselines["is_calibrated"]:
            baselines = self.calibration.baselines
            self.blink_tracker.update_thresholds(
                baselines["ear_threshold"],
                baselines["ear_reopen_threshold"]
            )
            self.yawn_tracker.update_threshold(baselines["mar_threshold"])
            self.nod_tracker.set_baseline(baselines["baseline_pitch"])

        # 4. Update Stateful Trackers
        is_sunglasses = False
        contrast_ratio = 1.0

        if face_detected:
            # Sunglasses / Goggles detection (day & night)
            is_sunglasses, contrast_ratio = self.detector.detect_sunglasses(
                frame_bgr, landmarks_dict, current_ear=avg_ear
            )
            if self.force_sunglasses_mode:
                is_sunglasses = True

            blink_info = self.blink_tracker.update(avg_ear, current_time=now)
            # If sunglasses mode is engaged, suppress spurious eye-closure to avoid false alarms
            if is_sunglasses:
                blink_info["is_closed"] = False
                blink_info["closure_duration_sec"] = 0.0
                blink_info["is_microsleep"] = False

            yawn_info = self.yawn_tracker.update(mar, current_time=now)
            nod_info = self.nod_tracker.update(pitch, current_time=now)
            perclos = 0.0 if is_sunglasses else self.perclos_calc.update(blink_info["is_closed"], current_time=now)
        else:
            blink_info = {
                "is_closed": False,
                "blink_count": self.blink_tracker.total_blinks,
                "blink_rate_bpm": self.blink_tracker.update(0.3, now)["blink_rate_bpm"],
                "last_blink_duration_ms": self.blink_tracker.last_blink_duration_ms,
                "avg_blink_duration_ms": 200,
                "closure_duration_sec": 0.0,
                "is_microsleep": False
            }
            yawn_info = {
                "is_yawning": False,
                "yawn_duration_sec": 0.0,
                "total_yawns": self.yawn_tracker.total_yawns,
                "recent_yawns_5min": len(self.yawn_tracker.recent_yawns),
                "just_completed_yawn": False
            }
            nod_info = {
                "is_nodding": False,
                "pitch_deviation": 0.0,
                "nod_duration_sec": 0.0,
                "total_nods": self.nod_tracker.total_nods,
                "recent_nods_5min": len(self.nod_tracker.recent_nods),
                "just_nodded": False
            }
            perclos = self.perclos_calc.update(False, current_time=now)

        # 5. Multimodal Risk Scoring
        feature_bundle = {
            "face_detected": face_detected,
            "sunglasses_detected": is_sunglasses,
            "sunglasses_ratio": contrast_ratio,
            "ear": avg_ear,
            "baseline_ear": self.calibration.baselines.get("baseline_ear", 0.30),
            "ear_threshold": self.calibration.baselines.get("ear_threshold", 0.22),
            "perclos": perclos,
            "is_closed": blink_info["is_closed"],
            "closure_duration_sec": blink_info["closure_duration_sec"],
            "blink_rate_bpm": blink_info["blink_rate_bpm"],
            "avg_blink_duration_ms": blink_info["avg_blink_duration_ms"],
            "mar": mar,
            "is_yawning": yawn_info["is_yawning"],
            "yawn_duration_sec": yawn_info["yawn_duration_sec"],
            "recent_yawns_5min": yawn_info["recent_yawns_5min"],
            "pitch": pitch,
            "baseline_pitch": self.calibration.baselines.get("baseline_pitch", 0.0),
            "is_nodding": nod_info["is_nodding"],
            "nod_duration_sec": nod_info["nod_duration_sec"],
            "recent_nods_5min": nod_info["recent_nods_5min"]
        }

        risk_eval = self.scorer.score_frame(feature_bundle)

        # Update Session Rollups
        with self.lock:
            self.frame_count += 1
            self.risk_sum += risk_eval["score"]
            if risk_eval["score"] > self.max_risk_observed:
                self.max_risk_observed = risk_eval["score"]

        # 6. Safety Incident Logging to SQLite (Throttled)
        self._check_and_log_incident(now, risk_eval, feature_bundle, yawn_info, nod_info, blink_info)

        # 7. Render Augmented Visual Overlays
        annotated_frame = frame_bgr
        if self.show_hud_overlay:
            annotated_frame = self.detector.draw_hud(
                frame=frame_bgr,
                landmarks_dict=landmarks_dict,
                is_closed=blink_info["is_closed"],
                is_yawning=yawn_info["is_yawning"],
                nose_2d=nose_2d,
                nose_endpoint=nose_endpoint,
                state_info=risk_eval,
                is_sunglasses=is_sunglasses
            )

        # 8. Package Telemetry
        telemetry = {
            "timestamp": now,
            "session_id": self.session_id,
            "face_detected": face_detected,
            "confidence": round(confidence * 100, 1) if face_detected else 0.0,
            "sunglasses_detected": is_sunglasses,
            "sunglasses_ratio": contrast_ratio,
            "sunglasses_mode": is_sunglasses,
            "ear": round(avg_ear, 3),
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "mar": round(mar, 3),
            "pitch": round(pitch, 1),
            "yaw": round(yaw, 1),
            "roll": round(roll, 1),
            "perclos": round(perclos, 3),
            "blink_count": blink_info["blink_count"],
            "blink_rate_bpm": blink_info["blink_rate_bpm"],
            "last_blink_ms": blink_info["last_blink_duration_ms"],
            "avg_blink_ms": blink_info["avg_blink_duration_ms"],
            "is_closed": blink_info["is_closed"],
            "closure_duration_sec": blink_info["closure_duration_sec"],
            "is_microsleep": blink_info.get("is_microsleep", False),
            "yawn_count": yawn_info["total_yawns"],
            "is_yawning": yawn_info["is_yawning"],
            "yawn_duration_sec": yawn_info["yawn_duration_sec"],
            "nod_count": nod_info["total_nods"],
            "is_nodding": nod_info["is_nodding"],
            "nod_duration_sec": nod_info["nod_duration_sec"],
            "risk_score": risk_eval["score"],
            "raw_risk_score": risk_eval["raw_score"],
            "risk_state": risk_eval["state"],
            "risk_code": risk_eval["state_code"],
            "risk_color": risk_eval["state_color"],
            "risk_breakdown": risk_eval["breakdown"],
            "override_reason": risk_eval["override_reason"],
            "is_critical": risk_eval["is_critical"],
            "is_high_risk": risk_eval["is_high_risk"],
            "fps": self.fps,
            "night_mode": self.detector.is_night_mode_active,
            "ambient_luminance": round(self.detector.current_luminance, 1),
            "calibration": calib_status
        }

        self.latest_telemetry = telemetry
        return annotated_frame, telemetry

    def _check_and_log_incident(self, now, risk_eval, features, yawn_info, nod_info, blink_info):
        """Log critical events to database with sensible throttle."""
        if not self.session_id:
            return

        event_to_log = None
        desc = ""

        if blink_info.get("is_microsleep"):
            event_to_log = "MICROSLEEP"
            desc = f"Prolonged eye closure detected: {blink_info['closure_duration_sec']}s"
        elif risk_eval["is_critical"]:
            event_to_log = "CRITICAL_RISK"
            desc = f"Driver Drowsiness Critical Risk: {risk_eval['score']}/100 ({risk_eval.get('override_reason') or 'Multi-factor overload'})"
        elif yawn_info.get("just_completed_yawn"):
            event_to_log = "YAWN"
            desc = f"Driver yawn completed (Total: {yawn_info['total_yawns']})"
        elif nod_info.get("just_nodded"):
            event_to_log = "HEAD_NOD"
            desc = f"Head nodding episode detected (Total: {nod_info['total_nods']})"

        if event_to_log:
            last_time = self._last_incident_log_time.get(event_to_log, 0)
            # Throttle same event type to at least 4 seconds apart
            if now - last_time >= 4.0:
                self._last_incident_log_time[event_to_log] = now
                self.db.log_event(
                    session_id=self.session_id,
                    event_type=event_to_log,
                    risk_score=risk_eval["score"],
                    telemetry=features,
                    description=desc
                )


class CameraStream:
    """
    Thread-safe OpenCV Camera Stream Manager.
    Captures frames asynchronously in a dedicated worker thread.
    Gracefully manages camera open/close and fallback.
    """
    def __init__(self, camera_index=0, pipeline=None):
        self.camera_index = camera_index
        self.pipeline = pipeline or DriveSafePipeline()
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated = None
        self.thread = None
        self.camera_available = False

    def start(self):
        """Start background frame capture thread."""
        with self.lock:
            if self.is_running:
                return True

            try:
                # Open with directshow backend on Windows for fast initialization
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_index)

                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    # Auto-exposure and autofocus for best face detection
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                    self.camera_available = True
                else:
                    self.camera_available = False
            except Exception:
                self.camera_available = False

            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            return self.camera_available

    def _capture_loop(self):
        """Background continuous frame grabbing."""
        while self.is_running:
            if self.camera_available and self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Mirror horizontal frame for natural webcam mirror view
                    frame = cv2.flip(frame, 1)
                    annotated, _ = self.pipeline.process_frame(frame)
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_annotated = annotated
                else:
                    time.sleep(0.03)
            else:
                # If no hardware camera, sleep slightly
                time.sleep(0.05)

    def process_external_frame(self, frame_bgr):
        """
        Process a frame supplied from client-side webcam (via browser canvas / fetch).
        """
        annotated, telemetry = self.pipeline.process_frame(frame_bgr)
        summary = self.pipeline.latest_landmarks_summary
        with self.lock:
            self.latest_frame = frame_bgr
            self.latest_annotated = annotated
        return annotated, telemetry, summary

    def get_jpeg_frame(self):
        """Encode the latest annotated frame to JPEG for MJPEG stream."""
        with self.lock:
            if self.is_running and self.latest_annotated is not None:
                ret, jpeg = cv2.imencode('.jpg', self.latest_annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return jpeg.tobytes()

        # Return a standby placeholder frame if no frame is available yet or camera paused
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        if not self.is_running:
            cv2.putText(placeholder, "CAMERA PAUSED", (200, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(placeholder, "Click 'Camera ON' to resume driver monitoring", (135, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
        else:
            cv2.putText(placeholder, "WAITING FOR WEBCAM FEED...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(placeholder, "Click 'Start Camera' or grant browser camera permission", (110, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()

    def stop(self):
        """Stop capture loop and release camera resource."""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_available = False
            self.latest_frame = None
            self.latest_annotated = None
