"""
DriveSafe - Flask Application Server
Exposes live video feeds, REST telemetry APIs, calibration controls, and SQLite session reporting.
"""
import base64
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

from core.camera import CameraStream, DriveSafePipeline
from core.database import init_db

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Initialize master pipeline and camera controller
pipeline = DriveSafePipeline()
camera_stream = CameraStream(camera_index=0, pipeline=pipeline)

# Start hardware camera capture in background (auto-handles availability)
camera_stream.start()


@app.route("/")
def index():
    """Render the master DriveSafe cockpit dashboard."""
    return render_template("index.html")


def generate_mjpeg_frames():
    """Generator for streaming live MJPEG frames with low latency."""
    while True:
        frame_bytes = camera_stream.get_jpeg_frame()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.033 if camera_stream.is_running else 0.1)  # Adaptive throttle


@app.route("/video_feed")
def video_feed():
    """MJPEG live webcam video stream."""
    return Response(
        generate_mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/process_frame", methods=["POST"])
def process_frame_api():
    """
    Client-side frame upload endpoint.
    Accepts base64 encoded frame from browser webcam, processes it,
    and returns live telemetry and annotated frame.
    """
    try:
        data = request.get_json(force=True)
        image_b64 = data.get("image", "")

        if not image_b64:
            return jsonify({"error": "No image data provided"}), 400

        # Strip base64 header if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        annotated_frame, telemetry, landmarks_summary = camera_stream.process_external_frame(frame_bgr)

        # Optionally return annotated image if requested
        return_annotated = data.get("return_annotated", False)
        annotated_b64 = None
        if return_annotated:
            ret, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "success": True,
            "telemetry": telemetry,
            "landmarks": landmarks_summary,
            "annotated_frame": annotated_b64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/telemetry")
def get_telemetry():
    """Return latest instantaneous telemetry snapshot."""
    return jsonify(pipeline.latest_telemetry)


@app.route("/api/calibration/start", methods=["POST"])
def start_calibration():
    """Trigger personalized auto-calibration."""
    req_data = request.get_json(silent=True) or {}
    driver_name = req_data.get("driver_name", pipeline.driver_name)
    duration_sec = float(req_data.get("duration_sec", 10.0))

    pipeline.driver_name = driver_name
    res = pipeline.start_calibration(duration_sec=duration_sec)
    return jsonify(res)


@app.route("/api/calibration/status")
def calibration_status():
    """Check calibration progress and derived baselines."""
    return jsonify(pipeline.calibration.get_status())


@app.route("/api/calibration/reset", methods=["POST"])
def reset_calibration():
    """Reset calibration back to system defaults."""
    pipeline.calibration.reset_to_defaults()
    return jsonify({"success": True, "message": "Calibration reset to factory defaults"})


@app.route("/api/session/start", methods=["POST"])
def start_session():
    """Start a new named driving session."""
    data = request.get_json(silent=True) or {}
    driver_name = data.get("driver_name", "Driver 1")
    pipeline.start_new_session(driver_name)
    return jsonify({
        "success": True,
        "session_id": pipeline.session_id,
        "driver_name": driver_name
    })


@app.route("/api/session/stop", methods=["POST"])
def stop_session():
    """Stop active driving session and summarize."""
    active_id = pipeline.session_id
    pipeline.end_current_session()
    return jsonify({
        "success": True,
        "message": f"Session {active_id} saved to database"
    })


@app.route("/api/events")
def get_events():
    """Fetch safety incident events log."""
    limit = int(request.args.get("limit", 20))
    session_id = request.args.get("session_id", None)
    events = pipeline.db.get_recent_events(limit=limit, session_id=session_id)
    return jsonify(events)


@app.route("/api/sessions")
def get_sessions():
    """Fetch past sessions history."""
    limit = int(request.args.get("limit", 10))
    sessions = pipeline.db.get_sessions(limit=limit)
    return jsonify(sessions)


@app.route("/api/settings", methods=["POST"])
def update_settings():
    """Update runtime visual and detection settings."""
    data = request.get_json(silent=True) or {}
    if "show_hud_overlay" in data:
        pipeline.show_hud_overlay = bool(data["show_hud_overlay"])
    if "night_mode" in data:
        pipeline.detector.set_night_mode(str(data["night_mode"]))
    return jsonify({
        "success": True,
        "show_hud_overlay": pipeline.show_hud_overlay,
        "night_mode": pipeline.detector.night_mode,
        "is_night_mode_active": pipeline.detector.is_night_mode_active
    })


@app.route("/api/camera/state")
def get_camera_state():
    """Return whether hardware camera stream is active."""
    return jsonify({
        "camera_enabled": camera_stream.is_running,
        "camera_available": camera_stream.camera_available
    })


@app.route("/api/camera/toggle", methods=["POST"])
def toggle_camera():
    """Toggle or explicitly set camera state."""
    data = request.get_json(silent=True) or {}
    current = camera_stream.is_running
    target = data.get("enabled", not current)

    if target:
        if not camera_stream.is_running:
            camera_stream.start()
    else:
        if camera_stream.is_running:
            camera_stream.stop()
        # Reset instantaneous telemetry when camera is turned off
        with pipeline.lock:
            pipeline.latest_telemetry["face_detected"] = False
            pipeline.latest_telemetry["confidence"] = 0.0
            pipeline.latest_telemetry["fps"] = 0.0

    return jsonify({
        "success": True,
        "camera_enabled": camera_stream.is_running,
        "camera_available": camera_stream.camera_available
    })


if __name__ == "__main__":
    init_db()
    print("DriveSafe Server starting on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
