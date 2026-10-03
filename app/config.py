from dataclasses import dataclass, field
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(raw_path: str | None, fallback: Path) -> Path:
    if not raw_path:
        return fallback

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate if candidate.exists() else fallback


def _default_plate_model_path() -> Path:
    app_dir = os.path.dirname(__file__)
    candidates = [
        Path(os.path.join(app_dir, os.pardir, "runs", "detect", "runs", "detect", "nigerian_license_plate-4", "weights", "best.pt")),
        Path(os.path.join(app_dir, os.pardir, "runs", "detect", "nigerian_license_plate-4", "weights", "best.pt")),
        Path(os.path.join(app_dir, os.pardir, "models", "license_plate.pt")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _database_path() -> Path:
    raw_path = os.getenv("DATABASE_PATH")
    if not raw_path:
        return (PROJECT_ROOT / "data" / "vnpr.sqlite3").resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _vehicle_model_path() -> Path | None:
    configured_path = os.getenv("VEHICLE_MODEL_PATH")
    if configured_path and configured_path.strip().lower() == "none":
        return None
    if configured_path:
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        if candidate.exists():
            return candidate
    default_path = PROJECT_ROOT / "models" / "vbnpr_model.pt"
    return default_path if default_path.exists() else None


@dataclass(frozen=True)
class Settings:
    model_path: Path = _resolve_path(os.getenv("YOLO_MODEL_PATH"), _default_plate_model_path())
    database_path: Path = _database_path()
    vehicle_model_path: Path | None = _vehicle_model_path()
    detection_confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    detection_iou: float = float(os.getenv("YOLO_IOU", "0.45"))
    detection_imgsz: int = int(os.getenv("YOLO_IMGSZ", "640"))
    detection_multiscale: bool = os.getenv("YOLO_MULTISCALE", "false").lower() == "true"
    detection_tile_overlap: float = float(os.getenv("YOLO_TILE_OVERLAP", "0.20"))
    ocr_languages: tuple[str, ...] = tuple(os.getenv("OCR_LANGUAGES", "en").split(","))
    ocr_gpu: bool = os.getenv("OCR_GPU", "false").lower() == "true"
    ocr_padding_pct: float = float(os.getenv("OCR_PADDING_PCT", "0.10"))
    ocr_upscale_factor: float = float(os.getenv("OCR_UPSCALE_FACTOR", "5.0"))
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    tracker_type: str = os.getenv("TRACKER", "bytetrack.yaml")
    tracking_enabled: bool = os.getenv("TRACKING_ENABLED", "true").lower() == "true"
    ocr_interval: int = int(os.getenv("OCR_INTERVAL", "5"))
    max_frame_width: int = int(os.getenv("MAX_FRAME_WIDTH", "1280"))
    max_frame_height: int = int(os.getenv("MAX_FRAME_HEIGHT", "720"))
    track_timeout: float = float(os.getenv("TRACK_TIMEOUT", "10.0"))
    live_process_interval: float = float(os.getenv("LIVE_PROCESS_INTERVAL", "0.20"))
    camera_frame_fps: int = int(os.getenv("CAMERA_FPS", "20"))
    live_track_only_mode: bool = os.getenv("LIVE_TRACK_ONLY_MODE", "true").lower() == "true"
    live_track_only_ocr_interval: int = int(os.getenv("LIVE_TRACK_ONLY_OCR_INTERVAL", "8"))
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
    traffic_evidence_path: Path = _resolve_path(os.getenv("TRAFFIC_EVIDENCE_PATH"), PROJECT_ROOT / "data" / "traffic_evidence")
    traffic_min_track_frames: int = int(os.getenv("TRAFFIC_MIN_TRACK_FRAMES", "3"))
    traffic_confidence_threshold: float = float(os.getenv("TRAFFIC_CONFIDENCE_THRESHOLD", "0.55"))
    traffic_ocr_threshold: float = float(os.getenv("TRAFFIC_OCR_THRESHOLD", "0.55"))
    traffic_alert_cooldown: float = float(os.getenv("TRAFFIC_ALERT_COOLDOWN", "30"))
    red_light_fine: int = int(os.getenv("RED_LIGHT_FINE", "50000"))
    traffic_light_mode: str = os.getenv("TRAFFIC_LIGHT_MODE", "MANUAL").upper()
    traffic_light_roi: dict[str, int] = field(default_factory=lambda: {
        "x1": int(os.getenv("TRAFFIC_LIGHT_ROI_X1", "0")),
        "y1": int(os.getenv("TRAFFIC_LIGHT_ROI_Y1", "0")),
        "x2": int(os.getenv("TRAFFIC_LIGHT_ROI_X2", "0")),
        "y2": int(os.getenv("TRAFFIC_LIGHT_ROI_Y2", "0")),
    })
    traffic_light_min_confidence: float = float(os.getenv("TRAFFIC_LIGHT_MIN_CONFIDENCE", "0.60"))
    traffic_light_stable_frames: int = int(os.getenv("TRAFFIC_LIGHT_STABLE_FRAMES", "4"))
    traffic_light_debug_overlay: bool = os.getenv("TRAFFIC_LIGHT_DEBUG_OVERLAY", "true").lower() == "true"
    traffic_test_mode: bool = os.getenv("TRAFFIC_TEST_MODE", "false").lower() == "true"
    auth_secret: str = os.getenv("AUTH_SECRET", "change-this-development-secret")
    session_days: int = int(os.getenv("SESSION_DAYS", "7"))


settings = Settings()