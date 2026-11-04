"""
DriveSafe - Drowsiness Risk Scoring Engine
Combines EAR, PERCLOS, Blink Dynamics, MAR/Yawn, and Head Pose into a 0-100 Risk Score.
Features temporal exponential smoothing and a modular plug-and-play architecture for ML models.
"""
from abc import ABC, abstractmethod
import os
import joblib
import numpy as np

from core.config import (
    RISK_STATES,
    EMA_ALPHA_RISE,
    EMA_ALPHA_DECAY
)


class BaseDrowsinessModel(ABC):
    """Abstract base interface for drowsiness scoring models."""
    @abstractmethod
    def evaluate(self, features: dict) -> dict:
        """
        Evaluate extracted features and return raw risk score (0-100)
        and breakdown of contributing factors.
        """
        pass


class HeuristicDrowsinessScorer(BaseDrowsinessModel):
    """
    Biomarker-based heuristic fusion engine.
    Applies calibrated weights across PERCLOS, EAR depression, blink duration,
    sustained yawning, and head droop/nodding.
    """
    def __init__(self):
        # Contribution component weights (sum = 1.0)
        self.weights = {
            "perclos": 0.35,
            "blink_dynamics": 0.20,
            "head_posture": 0.25,
            "yawn": 0.20
        }

    def evaluate(self, features: dict) -> dict:
        """
        Compute multi-factor risk components.
        """
        ear = features.get("ear", 0.30)
        baseline_ear = features.get("baseline_ear", 0.30)
        ear_thresh = features.get("ear_threshold", 0.22)
        perclos = features.get("perclos", 0.0)

        is_closed = features.get("is_closed", False)
        closure_sec = features.get("closure_duration_sec", 0.0)
        blink_rate = features.get("blink_rate_bpm", 16)
        avg_blink_ms = features.get("avg_blink_duration_ms", 200)

        is_yawning = features.get("is_yawning", False)
        yawn_duration = features.get("yawn_duration_sec", 0.0)
        recent_yawns = features.get("recent_yawns_5min", 0)

        pitch = features.get("pitch", 0.0)
        baseline_pitch = features.get("baseline_pitch", 0.0)
        is_nodding = features.get("is_nodding", False)
        nod_duration = features.get("nod_duration_sec", 0.0)
        recent_nods = features.get("recent_nods_5min", 0)

        face_detected = features.get("face_detected", True)

        if not face_detected:
            # Absence of face or driver looking completely away from camera
            return {
                "raw_risk": 55.0,
                "breakdown": {
                    "perclos_score": 0.0,
                    "blink_score": 0.0,
                    "head_posture_score": 55.0,
                    "yawn_score": 0.0
                },
                "override_reason": "No face detected in camera frame"
            }

        # 1. PERCLOS & Eye Aspect Ratio Deficit Component (0 to 100)
        # Normal driving PERCLOS is < 0.08. Drowsy driving is 0.15 - 0.30+.
        perclos_risk = min(100.0, (perclos / 0.30) * 100.0)

        # Instantaneous eye closure penalty
        ear_deficit = max(0.0, (ear_thresh - ear) / max(0.05, ear_thresh))
        ear_instant_risk = min(100.0, ear_deficit * 100.0)
        eye_component = (perclos_risk * 0.65) + (ear_instant_risk * 0.35)

        # 2. Blink Dynamics Component (0 to 100)
        # Slow blinks (> 350ms) are classic indicators of microsleep onset.
        # Normal blink: 150-250ms.
        blink_duration_risk = 0.0
        if avg_blink_ms > 280:
            blink_duration_risk = min(100.0, ((avg_blink_ms - 280) / 300.0) * 100.0)

        # Abnormal blink rate: very high (>30 bpm) = struggle against sleep; very low (<8 bpm) = stare
        blink_rate_risk = 0.0
        if blink_rate > 28:
            blink_rate_risk = min(100.0, ((blink_rate - 28) / 20.0) * 70.0)
        elif blink_rate < 8 and eye_component > 20:
            blink_rate_risk = 40.0

        blink_component = max(blink_duration_risk, (blink_duration_risk * 0.6) + (blink_rate_risk * 0.4))

        # 3. Yawning & Oral Fatigue Component (0 to 100)
        yawn_component = 0.0
        if is_yawning:
            # Active yawn escalating by duration
            yawn_component = min(100.0, 50.0 + (yawn_duration / 3.0) * 50.0)
        else:
            # Cumulative fatigue from recent yawns (5-minute decay)
            yawn_component = min(75.0, recent_yawns * 22.0)

        # 4. Head Posture & Nodding Component (0 to 100)
        pitch_drop = pitch - baseline_pitch
        pitch_risk = 0.0
        if pitch_drop > 8.0:
            pitch_risk = min(100.0, ((pitch_drop - 8.0) / 20.0) * 100.0)

        nod_component = pitch_risk
        if is_nodding:
            nod_component = min(100.0, max(nod_component, 65.0 + nod_duration * 20.0))
        elif recent_nods > 0:
            nod_component = min(80.0, max(nod_component, recent_nods * 25.0))

        # Weighted Sum
        raw_risk = (
            (eye_component * self.weights["perclos"]) +
            (blink_component * self.weights["blink_dynamics"]) +
            (nod_component * self.weights["head_posture"]) +
            (yawn_component * self.weights["yawn"])
        )

        # Critical Overrides for Immediate Life-Safety Hazards
        override_reason = None

        # Continuous eye closure > 1.1s is an active microsleep
        if is_closed and closure_sec >= 1.0:
            critical_score = min(100.0, 85.0 + (closure_sec - 1.0) * 15.0)
            if critical_score > raw_risk:
                raw_risk = critical_score
                override_reason = f"Microsleep Detected: Eyes closed for {closure_sec:.1f}s"

        # Continuous downward head drop > 1.2s
        if is_nodding and nod_duration >= 1.2:
            critical_score = min(100.0, 85.0 + (nod_duration - 1.2) * 10.0)
            if critical_score > raw_risk:
                raw_risk = critical_score
                override_reason = f"Severe Head Nod: Driver head drooping for {nod_duration:.1f}s"

        return {
            "raw_risk": float(round(raw_risk, 2)),
            "breakdown": {
                "perclos_score": round(eye_component, 1),
                "blink_score": round(blink_component, 1),
                "head_posture_score": round(nod_component, 1),
                "yawn_score": round(yawn_component, 1)
            },
            "override_reason": override_reason
        }


class MLModelAdapter(BaseDrowsinessModel):
    """
    Machine Learning Model Wrapper.
    Can load an external trained model (e.g., Random Forest or XGBoost) from disk
    and run inference against standard feature vectors.
    """
    def __init__(self, model_file_path=None):
        self.model = None
        self.fallback = HeuristicDrowsinessScorer()
        self.is_loaded = False
        self.model_path = model_file_path

        if model_file_path and os.path.exists(model_file_path):
            self.load_model(model_file_path)

    def load_model(self, file_path):
        try:
            self.model = joblib.load(file_path)
            self.model_path = file_path
            self.is_loaded = True
            return True
        except Exception as e:
            self.is_loaded = False
            return False

    def extract_vector(self, features: dict) -> np.ndarray:
        """
        Converts feature dictionary into standardized feature vector:
        [ear, perclos, blink_rate, avg_blink_duration, mar, recent_yawns, pitch, recent_nods]
        """
        return np.array([[
            features.get("ear", 0.30),
            features.get("perclos", 0.0),
            features.get("blink_rate_bpm", 16.0),
            features.get("avg_blink_duration_ms", 200.0),
            features.get("mar", 0.25),
            features.get("recent_yawns_5min", 0),
            features.get("pitch", 0.0),
            features.get("recent_nods_5min", 0)
        ]], dtype=np.float32)

    def evaluate(self, features: dict) -> dict:
        if not self.is_loaded or self.model is None:
            # Gracefully fallback to heuristic scorer
            return self.fallback.evaluate(features)

        try:
            vector = self.extract_vector(features)
            # If classifier with predict_proba (e.g. classes [Alert, Drowsy])
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(vector)[0]
                drowsy_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
                score = drowsy_prob * 100.0
            else:
                score = float(self.model.predict(vector)[0])

            score = max(0.0, min(100.0, score))
            return {
                "raw_risk": score,
                "breakdown": {"ml_prediction": True, "score": score},
                "override_reason": None
            }
        except Exception:
            return self.fallback.evaluate(features)


class DrowsinessScorer:
    """
    Master Drowsiness Scoring Controller with Asymmetric Temporal Smoothing (EMA).
    Applies adaptive smoothing to eliminate high-frequency flutter while ensuring
    instantaneous escalation during true safety emergencies.
    """
    def __init__(self, model: BaseDrowsinessModel = None):
        self.model = model if model is not None else HeuristicDrowsinessScorer()
        self.smoothed_score = 0.0
        self.current_state = "SAFE"
        self.last_update_time = None

    def set_model(self, model: BaseDrowsinessModel):
        self.model = model

    def score_frame(self, features: dict) -> dict:
        """
        Score a single telemetry frame and update smoothed risk state.
        """
        eval_result = self.model.evaluate(features)
        raw_risk = eval_result["raw_risk"]
        override = eval_result.get("override_reason")

        # Asymmetric Temporal Smoothing (EMA)
        # Fast alpha when risk spikes (critical alert reaction)
        # Slow alpha when risk declines (prevents premature safety clearance)
        if raw_risk > self.smoothed_score:
            alpha = EMA_ALPHA_RISE
            # Accelerate even faster if high risk / critical
            if raw_risk > 70.0:
                alpha = 0.45
        else:
            alpha = EMA_ALPHA_DECAY

        self.smoothed_score = (alpha * raw_risk) + ((1.0 - alpha) * self.smoothed_score)
        final_score = round(max(0.0, min(100.0, self.smoothed_score)), 1)

        # Determine Risk State
        state_key = "SAFE"
        for key, conf in RISK_STATES.items():
            low, high = conf["range"]
            if low <= final_score <= high:
                state_key = key
                break

        state_info = RISK_STATES[state_key]
        self.current_state = state_key

        return {
            "score": final_score,
            "raw_score": raw_risk,
            "state": state_info["label"],
            "state_code": state_info["code"],
            "state_color": state_info["color"],
            "breakdown": eval_result.get("breakdown", {}),
            "override_reason": override,
            "is_critical": (state_info["code"] == "critical"),
            "is_high_risk": (state_info["code"] in ["high_risk", "critical"])
        }

    def reset(self):
        """Reset the smoothed score to zero."""
        self.smoothed_score = 0.0
        self.current_state = "SAFE"
