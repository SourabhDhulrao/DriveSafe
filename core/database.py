"""
DriveSafe - SQLite Database Layer
Stores driving sessions, driver calibrations, telemetry metrics, and critical safety incidents.
"""
import sqlite3
import time
import uuid
from core.config import DB_PATH


def get_db_connection():
    """Create a thread-safe connection to SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables with indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT UNIQUE NOT NULL,
        driver_name TEXT DEFAULT 'Default Driver',
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_time TIMESTAMP,
        duration_sec REAL DEFAULT 0.0,
        calibrated_ear REAL DEFAULT 0.30,
        calibrated_pitch REAL DEFAULT 0.0,
        avg_risk_score REAL DEFAULT 0.0,
        max_risk_score REAL DEFAULT 0.0,
        total_blinks INTEGER DEFAULT 0,
        total_yawns INTEGER DEFAULT 0,
        total_nods INTEGER DEFAULT 0,
        status TEXT DEFAULT 'ACTIVE'
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS incident_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT NOT NULL,
        risk_score REAL NOT NULL,
        ear REAL,
        perclos REAL,
        mar REAL,
        pitch REAL,
        description TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON incident_events(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_time ON incident_events(timestamp)")

    conn.commit()
    conn.close()


class DatabaseManager:
    """Manages session lifecycle and event logging."""
    def __init__(self):
        init_db()

    def start_session(self, driver_name="Default Driver", baselines=None):
        """Create a new active driving session."""
        session_id = f"DRV-{uuid.uuid4().hex[:8].upper()}"
        calibrated_ear = baselines.get("baseline_ear", 0.30) if baselines else 0.30
        calibrated_pitch = baselines.get("baseline_pitch", 0.0) if baselines else 0.0

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                session_id, driver_name, start_time, calibrated_ear, calibrated_pitch, status
            ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, 'ACTIVE')
        """, (session_id, driver_name, calibrated_ear, calibrated_pitch))
        conn.commit()
        conn.close()
        return session_id

    def end_session(self, session_id, summary_stats):
        """Finalize and close an active session."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions SET
                end_time = CURRENT_TIMESTAMP,
                duration_sec = ?,
                avg_risk_score = ?,
                max_risk_score = ?,
                total_blinks = ?,
                total_yawns = ?,
                total_nods = ?,
                status = 'COMPLETED'
            WHERE session_id = ?
        """, (
            summary_stats.get("duration_sec", 0.0),
            summary_stats.get("avg_risk", 0.0),
            summary_stats.get("max_risk", 0.0),
            summary_stats.get("total_blinks", 0),
            summary_stats.get("total_yawns", 0),
            summary_stats.get("total_nods", 0),
            session_id
        ))
        conn.commit()
        conn.close()

    def log_event(self, session_id, event_type, risk_score, telemetry, description):
        """Log a critical safety incident or threshold breach."""
        if not session_id:
            return

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO incident_events (
                session_id, event_type, risk_score, ear, perclos, mar, pitch, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            event_type,
            risk_score,
            telemetry.get("ear", 0.0),
            telemetry.get("perclos", 0.0),
            telemetry.get("mar", 0.0),
            telemetry.get("pitch", 0.0),
            description
        ))
        conn.commit()
        conn.close()

    def get_recent_events(self, limit=20, session_id=None):
        """Retrieve recent incident events."""
        conn = get_db_connection()
        cursor = conn.cursor()
        if session_id:
            cursor.execute("""
                SELECT * FROM incident_events
                WHERE session_id = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM incident_events
                ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_sessions(self, limit=10):
        """Retrieve historical driving sessions."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions
            ORDER BY start_time DESC LIMIT ?
        """, (limit,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
