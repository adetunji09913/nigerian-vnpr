from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


VIOLATION_TYPES = ("RED_LIGHT", "BUS_LANE", "SPEEDING", "ILLEGAL_TURN")
VIOLATION_STATUSES = ("SUSPECTED", "PENDING_REVIEW", "CONFIRMED", "DISMISSED", "CANCELLED")
CITATION_STATUSES = ("PENDING_REVIEW", "APPROVED", "DISMISSED", "PAID", "OVERDUE", "CANCELLED")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_traffic_light_state(value: Any) -> str:
    normalized = str(value or "").strip().upper()
    aliases = {
        "AMBER": "YELLOW",
        "Y": "YELLOW",
        "YELLOW": "YELLOW",
        "RED": "RED",
        "GREEN": "GREEN",
        "UNKNOWN": "UNKNOWN",
        "UNAVAILABLE": "UNKNOWN",
        "N/A": "UNKNOWN",
        "NULL": "UNKNOWN",
        "NONE": "UNKNOWN",
    }
    if normalized in aliases:
        return aliases[normalized]
    return "UNKNOWN"


def _point(value: Iterable[float]) -> tuple[float, float]:
    values = list(value)
    if len(values) != 2:
        raise ValueError("A point must contain x and y")
    return float(values[0]), float(values[1])


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        intersects = (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
        if intersects:
            inside = not inside
        previous = current
    return inside


def line_side(point: tuple[float, float], line: list[tuple[float, float]]) -> float:
    if len(line) != 2:
        return 0.0
    (x1, y1), (x2, y2) = line
    x, y = point
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def estimate_speed_kmh(previous: tuple[float, float], current: tuple[float, float], elapsed_seconds: float, pixels_per_meter: float) -> float | None:
    if elapsed_seconds <= 0 or pixels_per_meter <= 0:
        return None
    distance_meters = math.dist(previous, current) / pixels_per_meter
    return distance_meters / elapsed_seconds * 3.6


@dataclass
class CameraRuleConfig:
    camera_id: str = "CAM-01"
    camera_name: str = "Camera 01"
    location_name: str = "Not configured"
    enabled: bool = True
    speed_limit: float | None = None
    speed_tolerance: float = 5.0
    traffic_light_enabled: bool = False
    traffic_light_mode: str = "MANUAL"
    traffic_light_state: str | None = None
    traffic_light_roi: dict[str, int] = field(default_factory=lambda: {"x1": 0, "y1": 0, "x2": 0, "y2": 0})
    traffic_light_min_confidence: float = 0.60
    traffic_light_stable_frames: int = 4
    traffic_light_debug_overlay: bool = True
    stop_line: list[tuple[float, float]] = field(default_factory=list)
    bus_lane_enabled: bool = False
    bus_lane_polygon: list[tuple[float, float]] = field(default_factory=list)
    bus_lane_min_duration: float = 3.0
    allowed_vehicle_types: list[str] = field(default_factory=lambda: ["bus"])
    speed_measurement_zone: list[tuple[float, float]] = field(default_factory=list)
    pixels_per_meter: float | None = None
    illegal_turn_enabled: bool = False
    entry_zone: list[tuple[float, float]] = field(default_factory=list)
    exit_zones: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    restricted_exit_zones: list[str] = field(default_factory=list)
    citation_amounts: dict[str, float] = field(default_factory=dict)
    evidence_video_enabled: bool = False
    evidence_video_seconds: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraRuleConfig":
        def polygon(value: Any) -> list[tuple[float, float]]:
            return [_point(item) for item in (value or [])]

        def normalized_roi(value: Any) -> dict[str, int]:
            if isinstance(value, dict):
                item = value.copy()
                return {
                    "x1": int(item.get("x1", 0)),
                    "y1": int(item.get("y1", 0)),
                    "x2": int(item.get("x2", 0)),
                    "y2": int(item.get("y2", 0)),
                }
            if isinstance(value, (list, tuple)) and len(value) >= 4:
                x1, y1, x2, y2 = value[:4]
                return {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        citation_amounts = {str(key): float(value) for key, value in (data.get("citation_amounts") or {}).items()}
        if "RED_LIGHT" not in citation_amounts:
            citation_amounts["RED_LIGHT"] = 50000.0
        return cls(
            camera_id=str(data.get("camera_id") or "CAM-01"),
            camera_name=str(data.get("camera_name") or "Camera 01"),
            location_name=str(data.get("location_name") or "Not configured"),
            enabled=bool(data.get("enabled", True)),
            speed_limit=float(data["speed_limit"]) if data.get("speed_limit") is not None else None,
            speed_tolerance=float(data.get("speed_tolerance", 5.0)),
            traffic_light_enabled=bool(data.get("traffic_light_enabled", False)),
            traffic_light_mode=str(data.get("traffic_light_mode", "MANUAL")).upper(),
            traffic_light_state=normalize_traffic_light_state(data.get("traffic_light_state")) if data.get("traffic_light_state") is not None else None,
            traffic_light_roi=normalized_roi(data.get("traffic_light_roi")),
            traffic_light_min_confidence=float(data.get("traffic_light_min_confidence", 0.60)),
            traffic_light_stable_frames=max(1, int(data.get("traffic_light_stable_frames", 4))),
            traffic_light_debug_overlay=bool(data.get("traffic_light_debug_overlay", True)),
            stop_line=polygon(data.get("stop_line")),
            bus_lane_enabled=bool(data.get("bus_lane_enabled", False)),
            bus_lane_polygon=polygon(data.get("bus_lane_polygon")),
            bus_lane_min_duration=float(data.get("bus_lane_min_duration", 3.0)),
            allowed_vehicle_types=[str(item).lower() for item in data.get("allowed_vehicle_types", ["bus"])],
            speed_measurement_zone=polygon(data.get("speed_measurement_zone")),
            pixels_per_meter=float(data["pixels_per_meter"]) if data.get("pixels_per_meter") else None,
            illegal_turn_enabled=bool(data.get("illegal_turn_enabled", False)),
            entry_zone=polygon(data.get("entry_zone")),
            exit_zones={str(key): polygon(value) for key, value in (data.get("exit_zones") or {}).items()},
            restricted_exit_zones=[str(item) for item in data.get("restricted_exit_zones", [])],
            citation_amounts=citation_amounts,
            evidence_video_enabled=bool(data.get("evidence_video_enabled", False)),
            evidence_video_seconds=float(data.get("evidence_video_seconds", 3.0)),
        )


class TrafficStore:
    def __init__(self, database_path: Path, evidence_path: Path | None = None) -> None:
        self.database_path = database_path
        self.evidence_path = evidence_path or database_path.parent / "traffic_evidence"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT NOT NULL UNIQUE, camera_name TEXT NOT NULL,
                location_name TEXT NOT NULL, camera_source TEXT, enabled INTEGER NOT NULL DEFAULT 1,
                speed_limit REAL, traffic_light_enabled INTEGER NOT NULL DEFAULT 0,
                bus_lane_enabled INTEGER NOT NULL DEFAULT 0, illegal_turn_enabled INTEGER NOT NULL DEFAULT 0,
                calibration_status TEXT NOT NULL DEFAULT 'NOT_CALIBRATED', config_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS traffic_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, plate_text TEXT, track_id INTEGER, vehicle_make TEXT,
                vehicle_model TEXT, vehicle_type TEXT, violation_type TEXT NOT NULL, violation_status TEXT NOT NULL,
                camera_id TEXT NOT NULL, timestamp TEXT NOT NULL, detection_confidence REAL,
                ocr_confidence REAL, estimated_speed REAL, speed_limit REAL, excess_speed REAL,
                traffic_light_state TEXT, zone_name TEXT, evidence_image TEXT, evidence_video TEXT,
                plate_image TEXT, fine_amount REAL, violation_date TEXT, violation_time TEXT,
                description TEXT, citation_id INTEGER, event_key TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL)""")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(traffic_violations)")}
            for column_name, column_type in {
                "plate_image": "TEXT",
                "fine_amount": "REAL",
                "violation_date": "TEXT",
                "violation_time": "TEXT",
            }.items():
                if column_name not in columns:
                    connection.execute(f"ALTER TABLE traffic_violations ADD COLUMN {column_name} {column_type}")
            connection.execute("""CREATE TABLE IF NOT EXISTS citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, citation_number TEXT NOT NULL UNIQUE, violation_id INTEGER NOT NULL,
                plate_text TEXT, violation_type TEXT NOT NULL, citation_status TEXT NOT NULL,
                amount REAL, issue_date TEXT NOT NULL, due_date TEXT, description TEXT, evidence_path TEXT,
                reviewed_by INTEGER, reviewed_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                FOREIGN KEY(violation_id) REFERENCES traffic_violations(id))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS traffic_review_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, violation_id INTEGER NOT NULL, admin_id INTEGER,
                action TEXT NOT NULL, reason TEXT, created_at TEXT NOT NULL)""")

    def save_camera(self, config: CameraRuleConfig, source: str | None = None) -> dict[str, Any]:
        now = utc_now()
        payload = {key: value for key, value in config.__dict__.items() if key not in {"camera_id", "camera_name", "location_name", "enabled", "speed_limit", "traffic_light_enabled", "bus_lane_enabled", "illegal_turn_enabled"}}
        with self._connect() as connection:
            connection.execute("""INSERT INTO cameras
                (camera_id,camera_name,location_name,camera_source,enabled,speed_limit,traffic_light_enabled,bus_lane_enabled,illegal_turn_enabled,calibration_status,config_json,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(camera_id) DO UPDATE SET camera_name=excluded.camera_name,
                location_name=excluded.location_name,camera_source=excluded.camera_source,enabled=excluded.enabled,
                speed_limit=excluded.speed_limit,traffic_light_enabled=excluded.traffic_light_enabled,
                bus_lane_enabled=excluded.bus_lane_enabled,illegal_turn_enabled=excluded.illegal_turn_enabled,
                calibration_status=excluded.calibration_status,config_json=excluded.config_json,updated_at=excluded.updated_at""",
                (config.camera_id, config.camera_name, config.location_name, source, int(config.enabled), config.speed_limit,
                 int(config.traffic_light_enabled), int(config.bus_lane_enabled), int(config.illegal_turn_enabled),
                 "CALIBRATED" if config.pixels_per_meter else "NOT_CALIBRATED", json.dumps(payload), now, now))
        return self.get_camera(config.camera_id) or {}

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        config = json.loads(result.pop("config_json") or "{}")
        config.update({key: result[key] for key in ("camera_id", "camera_name", "location_name", "enabled", "speed_limit", "traffic_light_enabled", "bus_lane_enabled", "illegal_turn_enabled")})
        return config

    def list_cameras(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            ids = [row["camera_id"] for row in connection.execute("SELECT camera_id FROM cameras ORDER BY camera_id")]
        cameras = []
        for camera_id in ids:
            camera = self.get_camera(camera_id)
            if camera:
                cameras.append(camera)
        return cameras

    def create_violation(self, candidate: dict[str, Any], config: CameraRuleConfig, frame: np.ndarray | None = None, video_frames: list[np.ndarray] | None = None) -> dict[str, Any] | None:
        event_key = str(candidate["event_key"])
        created = utc_now()
        candidate_timestamp = str(candidate.get("timestamp") or created)
        fine_amount = float(candidate.get("fine_amount") or config.citation_amounts.get(candidate.get("violation_type"), 50000.0) or 50000.0)
        evidence_image = None
        plate_image_path = candidate.get("plate_image")
        if frame is not None and frame.size:
            self.evidence_path.mkdir(parents=True, exist_ok=True)
            evidence_image = str(self.evidence_path / f"{event_key}-{uuid.uuid4().hex[:8]}.jpg")
            annotated = self._annotate_evidence(frame, candidate, config)
            cv2.imwrite(evidence_image, annotated)
        evidence_video = None
        if config.evidence_video_enabled and video_frames:
            self.evidence_path.mkdir(parents=True, exist_ok=True)
            first = next((item for item in video_frames if item is not None and item.size), None)
            if first is not None:
                evidence_video = str(self.evidence_path / f"{event_key}-{uuid.uuid4().hex[:8]}.mp4")
                height, width = first.shape[:2]
                writer = cv2.VideoWriter(evidence_video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
                if writer.isOpened():
                    for item in video_frames:
                        if item is not None and item.shape[:2] == (height, width):
                            writer.write(item)
                    writer.release()
                else:
                    evidence_video = None
        with self._connect() as connection:
            existing = connection.execute("SELECT * FROM traffic_violations WHERE event_key = ?", (event_key,)).fetchone()
            if existing:
                return dict(existing)
            violation_date = candidate_timestamp[:10] if candidate_timestamp else created[:10]
            violation_time = candidate_timestamp[11:19] if len(candidate_timestamp) >= 19 and "T" in candidate_timestamp else created[11:19]
            cursor = connection.execute("""INSERT INTO traffic_violations
                (plate_text,track_id,vehicle_make,vehicle_model,vehicle_type,violation_type,violation_status,camera_id,timestamp,
                detection_confidence,ocr_confidence,estimated_speed,speed_limit,excess_speed,traffic_light_state,zone_name,
                evidence_image,evidence_video,plate_image,fine_amount,violation_date,violation_time,description,event_key,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                candidate.get("plate_text") or "UNKNOWN", candidate.get("track_id"), candidate.get("vehicle_make"), candidate.get("vehicle_model"),
                candidate.get("vehicle_type"), candidate["violation_type"], "PENDING_REVIEW", config.camera_id, candidate_timestamp,
                candidate.get("detection_confidence"), candidate.get("ocr_confidence"), candidate.get("estimated_speed"),
                candidate.get("speed_limit"), candidate.get("excess_speed"), candidate.get("traffic_light_state"),
                candidate.get("zone_name"), evidence_image, evidence_video or candidate.get("evidence_video"), plate_image_path,
                fine_amount, violation_date, violation_time, candidate.get("description"), event_key, created, created))
            violation_id = cursor.lastrowid
            citation_number = self._next_citation_number(connection)
            citation_cursor = connection.execute("""INSERT INTO citations
                (citation_number,violation_id,plate_text,violation_type,citation_status,amount,issue_date,description,evidence_path,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (citation_number, violation_id, candidate.get("plate_text") or "UNKNOWN", candidate["violation_type"],
                "PENDING_REVIEW", fine_amount, candidate_timestamp,
                "System-generated / pending review. Application-configured amount; not a legal fine.", evidence_image, created, created))
            connection.execute("UPDATE traffic_violations SET citation_id = ? WHERE id = ?", (citation_cursor.lastrowid, violation_id))
        return self.get_violation(int(violation_id))

    @staticmethod
    def _annotate_evidence(frame: np.ndarray, candidate: dict[str, Any], config: CameraRuleConfig) -> np.ndarray:
        annotated = frame.copy()
        bbox = candidate.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (int(round(value)) for value in bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text_lines = [
            "RED LIGHT VIOLATION",
            f"Plate: {candidate.get('plate_text') or 'UNKNOWN'}",
            f"Track: {candidate.get('track_id', 'N/A')}",
            f"Time: {candidate.get('timestamp') or utc_now()}",
            f"Fine: ₦{float(candidate.get('fine_amount') or config.citation_amounts.get('RED_LIGHT', 50000.0)):,.0f}",
        ]
        for index, text in enumerate(text_lines):
            y = 24 + index * 22
            cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated

    @staticmethod
    def _next_citation_number(connection: sqlite3.Connection) -> str:
        year = datetime.now(timezone.utc).year
        row = connection.execute("SELECT COUNT(*) AS count FROM citations").fetchone()
        return f"VNPR-{year}-{int(row['count'] or 0) + 1:06d}"

    def get_violation(self, violation_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM traffic_violations WHERE id = ?", (violation_id,)).fetchone()
        return dict(row) if row else None

    def get_citation(self, citation_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM citations WHERE id = ?", (citation_id,)).fetchone()
        return dict(row) if row else None

    def list_citations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM citations ORDER BY id DESC LIMIT ?", (min(max(int(limit), 1), 500),)).fetchall()
        return [dict(row) for row in rows]

    def list_violations(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        clauses, values = [], []
        for key in ("plate_text", "violation_type", "camera_id", "violation_status"):
            if filters.get(key):
                clauses.append(f"{key} = ?")
                values.append(filters[key])
        query = "SELECT * FROM traffic_violations" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY id DESC LIMIT ?"
        values.append(min(max(int(filters.get("limit", 100)), 1), 500))
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, values).fetchall()]

    def review(self, violation_id: int, action: str, admin_id: int, reason: str | None = None) -> dict[str, Any]:
        actions = {"confirm": "CONFIRMED", "dismiss": "DISMISSED", "cancel": "CANCELLED"}
        if action not in actions:
            raise ValueError("Unsupported review action")
        now = utc_now()
        status = actions[action]
        citation_status = "APPROVED" if status == "CONFIRMED" else "DISMISSED" if status == "DISMISSED" else "CANCELLED"
        with self._connect() as connection:
            connection.execute("UPDATE traffic_violations SET violation_status = ?, updated_at = ? WHERE id = ?", (status, now, violation_id))
            connection.execute("UPDATE citations SET citation_status = ?, reviewed_by = ?, reviewed_at = ?, updated_at = ? WHERE violation_id = ?", (citation_status, admin_id, now, now, violation_id))
            connection.execute("INSERT INTO traffic_review_log (violation_id,admin_id,action,reason,created_at) VALUES (?,?,?,?,?)", (violation_id, admin_id, action.upper(), reason, now))
        return self.get_violation(violation_id) or {}

    def update_citation_status(self, citation_id: int, status: str, admin_id: int) -> dict[str, Any]:
        normalized = str(status).upper()
        if normalized not in CITATION_STATUSES:
            raise ValueError("Unsupported citation status")
        now = utc_now()
        with self._connect() as connection:
            connection.execute("UPDATE citations SET citation_status = ?, reviewed_by = ?, reviewed_at = ?, updated_at = ? WHERE id = ?", (normalized, admin_id, now, now, citation_id))
        return self.get_citation(citation_id) or {}

    def update_citation(self, citation_id: int, admin_id: int, amount: float | None = None, due_date: str | None = None, description: str | None = None) -> dict[str, Any]:
        now = utc_now()
        fields, values = [], []
        if amount is not None:
            if amount < 0:
                raise ValueError("Citation amount cannot be negative")
            fields.append("amount = ?")
            values.append(float(amount))
        if due_date is not None:
            fields.append("due_date = ?")
            values.append(due_date or None)
        if description is not None:
            fields.append("description = ?")
            values.append(description.strip() or None)
        if not fields:
            return self.get_citation(citation_id) or {}
        fields.extend(["reviewed_by = ?", "reviewed_at = ?", "updated_at = ?"])
        values.extend([admin_id, now, now, citation_id])
        with self._connect() as connection:
            cursor = connection.execute(f"UPDATE citations SET {', '.join(fields)} WHERE id = ?", values)
            if cursor.rowcount == 0:
                return {}
        return self.get_citation(citation_id) or {}

    def statistics(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT violation_type, violation_status, camera_id, timestamp, estimated_speed, speed_limit FROM traffic_violations").fetchall()
        by_type: dict[str, int] = defaultdict(int)
        by_status: dict[str, int] = defaultdict(int)
        by_camera: dict[str, int] = defaultdict(int)
        by_hour: dict[str, int] = defaultdict(int)
        now = datetime.now(timezone.utc)
        periods = {"today": 0, "this_week": 0, "this_month": 0}
        speeding = {"count": 0, "average_speed": 0.0, "average_excess": 0.0}
        speed_values, excess_values = [], []
        for row in rows:
            by_type[row["violation_type"]] += 1
            by_status[row["violation_status"]] += 1
            by_camera[row["camera_id"]] += 1
            try:
                event_time = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                age = now - event_time
                age_seconds = age.total_seconds()
                if 0 <= age_seconds < 86400:
                    periods["today"] += 1
                if 0 <= age_seconds < 7 * 86400:
                    periods["this_week"] += 1
                if 0 <= age_seconds < 31 * 86400:
                    periods["this_month"] += 1
                by_hour[f"{event_time.hour:02d}:00"] += 1
            except (TypeError, ValueError):
                pass
            if row["violation_type"] == "SPEEDING" and row["estimated_speed"] is not None:
                speeding["count"] += 1
                speed_values.append(float(row["estimated_speed"]))
                if row["speed_limit"] is not None:
                    excess_values.append(float(row["estimated_speed"]) - float(row["speed_limit"]))
        speeding["average_speed"] = round(sum(speed_values) / len(speed_values), 2) if speed_values else 0.0
        speeding["average_excess"] = round(sum(excess_values) / len(excess_values), 2) if excess_values else 0.0
        return {"total": sum(by_type.values()), "periods": periods, "by_type": dict(by_type), "by_status": dict(by_status), "by_camera": dict(by_camera), "by_hour": dict(by_hour), "speeding": speeding, "pending_reviews": by_status.get("PENDING_REVIEW", 0), "confirmed_citations": by_status.get("CONFIRMED", 0)}


@dataclass
class _TrackEvidence:
    points: deque[tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=60))
    frames: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    lane_entered_at: float | None = None
    lane_reported: bool = False
    red_reported: bool = False
    turn_reported: bool = False
    speed_reported: bool = False
    entry_seen: bool = False


class TrafficViolationMonitor:
    def __init__(self, store: TrafficStore, config: CameraRuleConfig, min_track_frames: int = 3, confidence_threshold: float = 0.55, ocr_threshold: float = 0.55, cooldown: float = 30.0) -> None:
        self.store = store
        self.config = config
        self.min_track_frames = max(2, min_track_frames)
        self.confidence_threshold = confidence_threshold
        self.ocr_threshold = ocr_threshold
        self.cooldown = cooldown
        self.tracks: dict[int, _TrackEvidence] = {}
        self.last_events: dict[tuple[int, str], float] = {}

    def observe(self, track_id: int, bbox: tuple[int, int, int, int], timestamp: float, plate_text: str | None = None,
                detection_confidence: float = 0.0, ocr_confidence: float = 0.0, frame: np.ndarray | None = None,
                vehicle_type: str | None = None, vehicle_make: str | None = None, vehicle_model: str | None = None) -> list[dict[str, Any]]:
        state = self.tracks.setdefault(track_id, _TrackEvidence())
        point = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
        state.points.append((point[0], point[1], timestamp))
        if self.config.evidence_video_enabled and frame is not None and frame.size:
            state.frames.append(frame.copy())

        if self.config.bus_lane_enabled and str(vehicle_type or "").lower() not in self.config.allowed_vehicle_types:
            point_is_in_lane = point_in_polygon(point, self.config.bus_lane_polygon)
            if point_is_in_lane and state.lane_entered_at is None:
                state.lane_entered_at = timestamp
            elif not point_is_in_lane and state.lane_entered_at is not None and not state.lane_reported:
                state.lane_entered_at = None

        if len(state.points) < self.min_track_frames or detection_confidence < self.confidence_threshold:
            return []
        candidates: list[dict[str, Any]] = []
        base = {"track_id": track_id, "plate_text": plate_text, "vehicle_type": vehicle_type, "vehicle_make": vehicle_make, "vehicle_model": vehicle_model, "detection_confidence": detection_confidence, "ocr_confidence": ocr_confidence, "timestamp": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(), "bbox": bbox}
        traffic_light_state = normalize_traffic_light_state(self.config.traffic_light_state)
        if self.config.traffic_light_enabled and traffic_light_state == "RED" and len(self.config.stop_line) == 2 and len(state.points) >= 2:
            before, current = state.points[-2], state.points[-1]
            crossed = line_side((before[0], before[1]), self.config.stop_line) * line_side((current[0], current[1]), self.config.stop_line) <= 0
            moving = math.dist(before[:2], current[:2]) > 2.0
            if crossed and moving and not state.red_reported and self._eligible(track_id, "RED_LIGHT", timestamp):
                state.red_reported = True
                candidates.append({**base, "violation_type": "RED_LIGHT", "traffic_light_state": "RED", "zone_name": "stop_line", "description": "Vehicle crossed the configured stop line while the traffic light was red.", "event_key": f"{self.config.camera_id}:{track_id}:RED_LIGHT", "fine_amount": self.config.citation_amounts.get("RED_LIGHT", 50000.0)})
        if self.config.bus_lane_enabled and str(vehicle_type or "").lower() not in self.config.allowed_vehicle_types:
            point_is_in_lane = point_in_polygon(point, self.config.bus_lane_polygon)
            if point_is_in_lane and state.lane_entered_at is not None and timestamp - state.lane_entered_at >= self.config.bus_lane_min_duration and not state.lane_reported and self._eligible(track_id, "BUS_LANE", timestamp):
                state.lane_reported = True
                candidates.append({**base, "violation_type": "BUS_LANE", "zone_name": "bus_lane", "description": f"Vehicle remained in the configured bus lane for {timestamp - state.lane_entered_at:.1f} seconds.", "event_key": f"{self.config.camera_id}:{track_id}:BUS_LANE"})
        if self.config.speed_measurement_zone and self.config.pixels_per_meter and self.config.speed_limit is not None and point_in_polygon(point, self.config.speed_measurement_zone) and len(state.points) >= 2:
            previous = state.points[-2]
            speed = estimate_speed_kmh(previous[:2], point, timestamp - previous[2], self.config.pixels_per_meter)
            if speed is not None and speed > self.config.speed_limit + self.config.speed_tolerance and not state.speed_reported and self._eligible(track_id, "SPEEDING", timestamp):
                state.speed_reported = True
                candidates.append({**base, "violation_type": "SPEEDING", "estimated_speed": round(speed, 2), "speed_limit": self.config.speed_limit, "excess_speed": round(speed - self.config.speed_limit, 2), "zone_name": "speed_measurement", "description": f"Estimated speed exceeded the configured limit by {speed - self.config.speed_limit:.1f} km/h.", "event_key": f"{self.config.camera_id}:{track_id}:SPEEDING"})
        if self.config.illegal_turn_enabled and self.config.entry_zone and point_in_polygon(point, self.config.entry_zone):
            state.entry_seen = True
        if self.config.illegal_turn_enabled and state.entry_seen and not state.turn_reported:
            for zone_name, polygon in self.config.exit_zones.items():
                if point_in_polygon(point, polygon) and zone_name in self.config.restricted_exit_zones and self._eligible(track_id, "ILLEGAL_TURN", timestamp):
                    state.turn_reported = True
                    candidates.append({**base, "violation_type": "ILLEGAL_TURN", "zone_name": zone_name, "description": f"Vehicle trajectory entered the configured entry zone and exited through restricted zone {zone_name}.", "event_key": f"{self.config.camera_id}:{track_id}:ILLEGAL_TURN:{zone_name}"})
        saved = []
        for candidate in candidates:
            if not plate_text or ocr_confidence < self.ocr_threshold:
                candidate["description"] += " Plate or OCR confidence is insufficient for automatic identification."
            record = self.store.create_violation(candidate, self.config, frame, list(state.frames))
            if record:
                saved.append(record)
        return saved

    def _eligible(self, track_id: int, violation_type: str, timestamp: float) -> bool:
        key = (track_id, violation_type)
        previous = self.last_events.get(key)
        if previous is not None and timestamp - previous < self.cooldown:
            return False
        self.last_events[key] = timestamp
        return True
