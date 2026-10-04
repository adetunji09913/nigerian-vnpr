import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import base64
import logging
import sqlite3
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .auth import create_session, require_admin
from .config import settings
from .database import AdminStore, HistoryStore, PersistentVehicleStore, PlateRegistryStore, VehicleAlertStore, VehicleObligationStore, VehicleStatusStore
from .diagnostics import log_phase
from .detector import YOLOPlateDetector
from .live_tracker import LiveVehicleTracker
from .normalization import is_plausible_nigerian_plate, normalize_plate_text
from .ocr import OCRReader
from .pipeline import VNPRPipeline
from .segmentation.character_segmentation import segment_plate_characters
from .traffic import CameraRuleConfig, TrafficStore, TrafficViolationMonitor, normalize_traffic_light_state
from .traffic_light_detector import TrafficLightDetector
from .vehicle_classifier import VehicleClassifier

logger = logging.getLogger(__name__)
history = HistoryStore(settings.database_path)
persistent_vehicles = PersistentVehicleStore(settings.database_path)
admins = AdminStore(settings.database_path)
registry = PlateRegistryStore(settings.database_path)
vehicle_status = VehicleStatusStore(settings.database_path)
vehicle_obligations = VehicleObligationStore(settings.database_path)
vehicle_alerts = VehicleAlertStore(settings.database_path)
traffic_store = TrafficStore(settings.database_path, settings.traffic_evidence_path)
traffic_config = CameraRuleConfig.from_dict({
    **(traffic_store.get_camera("CAM-01") or {"camera_id": "CAM-01"}),
    "citation_amounts": {"RED_LIGHT": float(settings.red_light_fine)},
})
traffic_monitor = TrafficViolationMonitor(
    traffic_store,
    traffic_config,
    min_track_frames=settings.traffic_min_track_frames,
    confidence_threshold=settings.traffic_confidence_threshold,
    ocr_threshold=settings.traffic_ocr_threshold,
    cooldown=settings.traffic_alert_cooldown,
)
pipeline = VNPRPipeline(plate_registry=registry)


@asynccontextmanager
async def lifespan(application: FastAPI):
    model_path = settings.model_path
    model_exists = model_path.is_file()
    model_size = model_path.stat().st_size if model_exists else None
    logger.info("plate_model_path=%s exists=%s size_bytes=%s", model_path, model_exists, model_size)
    if model_size is not None and model_size < 1024:
        logger.warning("Plate model is under 1 KiB and may be a Git LFS pointer: %s", model_path)
    application.state.yolo_model = pipeline.detector._load()
    log_phase(logger, "plate_model_loaded", size_bytes=model_size)
    yield


app = FastAPI(title="Nigerian VNPR API", version="1.0.0", lifespan=lifespan)


def _status_payload(plate_text: str | None) -> dict[str, object]:
    plate = normalize_plate_text(str(plate_text or "")) if plate_text else ""
    log_phase(logger, "status_construction_before")
    if not plate:
        payload = {"plate_text": "", "status": "clear", "reason": None, "amount_owed": 0.0, "outstanding_amount": 0.0, "is_flagged": False}
        log_phase(logger, "status_construction_after")
        return payload
    data = vehicle_status.lookup(plate)
    data["amount_owed"] = float(data.get("amount_owed", 0.0) or 0.0)
    data["outstanding_amount"] = float(data.get("outstanding_amount", 0.0) or 0.0)
    log_phase(logger, "status_construction_after")
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


class LiveCameraManager:
    def __init__(self, monitor: TrafficViolationMonitor | None = None) -> None:
        self._lock = threading.Lock()
        self._tracker_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_event = threading.Event()
        self._capture: cv2.VideoCapture | None = None
        self._capture_thread: threading.Thread | None = None
        self._processing_thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._last_annotation_time = 0.0
        self._latest_frame: np.ndarray | None = None
        self._last_display_frame: np.ndarray | None = None
        self._actual_camera_width = 0
        self._actual_camera_height = 0
        self._actual_camera_fps = 0.0
        self._raw_frame_count = 0
        self._raw_started_at = 0.0
        self._stream_frame_count = 0
        self._stream_started_at = 0.0
        self._ai_frame_count = 0
        self._ai_started_at = 0.0
        self._yolo_time_total = 0.0
        self._ocr_time_total = 0.0
        self._ocr_call_count = 0
        self._total_processing_time = 0.0
        self._processing_in_progress = False
        self._traffic_monitor = monitor or traffic_monitor
        self._detector = YOLOPlateDetector(settings.model_path, settings.detection_confidence, settings.detection_iou, settings.detection_imgsz)
        self._ocr = OCRReader(settings.ocr_languages, settings.ocr_gpu)
        self._vehicle_classifier = VehicleClassifier(settings.vehicle_model_path)
        self._tracker = LiveVehicleTracker(ocr_interval=settings.ocr_interval, track_timeout=settings.track_timeout)
        self._traffic_light_detector = TrafficLightDetector(
            roi=self._traffic_monitor.config.traffic_light_roi,
            min_confidence=self._traffic_monitor.config.traffic_light_min_confidence,
            stable_frames=self._traffic_monitor.config.traffic_light_stable_frames,
        )

    def _refresh_traffic_light_detector(self) -> None:
        config = self._traffic_monitor.config
        self._traffic_light_detector.set_roi(config.traffic_light_roi)
        self._traffic_light_detector.min_confidence = max(0.0, min(1.0, float(config.traffic_light_min_confidence)))
        self._traffic_light_detector.stable_frames = max(1, int(config.traffic_light_stable_frames))

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> dict[str, object]:
        with self._lock:
            if self._running:
                return {"status": "already_started", "camera_index": settings.camera_index}
            if any(worker is not None and worker.is_alive() for worker in (self._capture_thread, self._processing_thread)):
                raise RuntimeError("Live camera workers are still stopping. Try again shortly.")
            if not settings.model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {settings.model_path}")
            capture = cv2.VideoCapture(settings.camera_index)
            if not capture.isOpened():
                raise RuntimeError("Unable to access webcam. Make sure a camera is available and not already in use.")
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.max_frame_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.max_frame_height)
            if settings.camera_frame_fps > 0:
                capture.set(cv2.CAP_PROP_FPS, settings.camera_frame_fps)
            self._capture = capture
            self._running = True
            self._error = None
            self._last_annotation_time = 0.0
            self._latest_frame = None
            self._last_display_frame = None
            self._actual_camera_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self._actual_camera_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            self._actual_camera_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self._raw_frame_count = 0
            self._raw_started_at = time.monotonic()
            self._stream_frame_count = 0
            self._stream_started_at = time.monotonic()
            self._ai_frame_count = 0
            self._ai_started_at = time.monotonic()
            self._yolo_time_total = 0.0
            self._ocr_time_total = 0.0
            self._ocr_call_count = 0
            self._processing_in_progress = False
            self._tracker = LiveVehicleTracker(ocr_interval=settings.ocr_interval, track_timeout=settings.track_timeout)
            self._stop_event.clear()
            self._frame_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_frames, name="vnpr-camera-capture", daemon=True)
            self._processing_thread = threading.Thread(target=self._process_latest_frame, name="vnpr-live-ai", daemon=True)
            self._capture_thread.start()
            self._processing_thread.start()
            return {"status": "started", "camera_index": settings.camera_index}

    def start_external(self) -> dict[str, object]:
        with self._lock:
            if self._running:
                return {"status": "already_started"}
            if not settings.model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {settings.model_path}")
            self._capture = None
            self._running = True
            self._error = None
            self._latest_frame = None
            self._last_display_frame = None
            self._actual_camera_width = 0
            self._actual_camera_height = 0
            self._actual_camera_fps = 0.0
            self._raw_frame_count = 0
            self._raw_started_at = time.monotonic()
            self._stream_frame_count = 0
            self._stream_started_at = time.monotonic()
            self._ai_frame_count = 0
            self._ai_started_at = time.monotonic()
            self._yolo_time_total = 0.0
            self._ocr_time_total = 0.0
            self._ocr_call_count = 0
            self._total_processing_time = 0.0
            self._processing_in_progress = False
            self._tracker = LiveVehicleTracker(ocr_interval=settings.ocr_interval, track_timeout=settings.track_timeout)
            self._stop_event.clear()
            self._frame_event.clear()
            self._capture_thread = None
            self._processing_thread = threading.Thread(target=self._process_latest_frame, name="vnpr-external-live-ai", daemon=True)
            self._processing_thread.start()
            return {"status": "started"}

    def submit_frame(self, frame: np.ndarray) -> dict[str, object]:
        if frame is None or frame.size == 0:
            raise ValueError("Invalid camera frame")
        with self._lock:
            if not self._running:
                raise RuntimeError("External camera is not running")
            self._latest_frame = frame
            self._actual_camera_height, self._actual_camera_width = frame.shape[:2]
            self._raw_frame_count += 1
        self._frame_event.set()
        return {"accepted": True, "width": frame.shape[1], "height": frame.shape[0]}

    def stop(self) -> dict[str, object]:
        with self._lock:
            capture = self._capture
            self._capture = None
            self._running = False
            self._stop_event.set()
            self._frame_event.set()
            capture_thread = self._capture_thread
            processing_thread = self._processing_thread
        if capture is not None:
            capture.release()
        for worker in (capture_thread, processing_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=2.0)
        with self._lock:
            self._capture_thread = capture_thread if capture_thread is not None and capture_thread.is_alive() else None
            self._processing_thread = processing_thread if processing_thread is not None and processing_thread.is_alive() else None
            with self._tracker_lock:
                self._tracker.tracked_vehicles.clear()
            self._error = None
        return {"status": "stopped"}

    def _capture_frames(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                capture = self._capture
                running = self._running
            if capture is None or not running:
                break

            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                with self._lock:
                    self._error = "Invalid camera frame"
                self._stop_event.wait(0.1)
                continue

            with self._lock:
                self._latest_frame = frame
                self._raw_frame_count += 1
            self._frame_event.set()

    def _process_latest_frame(self) -> None:
        next_process_at = 0.0
        while not self._stop_event.is_set():
            wait_for = max(0.0, next_process_at - time.monotonic())
            if self._stop_event.wait(wait_for):
                break
            if not self._frame_event.wait(0.1):
                continue

            with self._lock:
                self._frame_event.clear()
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
            if frame is None:
                continue

            next_process_at = time.monotonic() + max(0.0, settings.live_process_interval)
            with self._lock:
                self._processing_in_progress = True
            processing_started = time.perf_counter()
            try:
                annotated = self._annotate_frame(frame)
                if annotated is not None and annotated.size:
                    with self._lock:
                        self._last_display_frame = annotated.copy()
            except Exception as exc:
                logger.exception("Live frame processing failed")
                with self._lock:
                    self._error = f"Live processing failed: {exc}"
            finally:
                with self._lock:
                    self._ai_frame_count += 1
                    self._total_processing_time += time.perf_counter() - processing_started
                    self._processing_in_progress = False

    def _record_timing(self, operation: str, elapsed: float) -> None:
        with self._lock:
            if operation == "yolo":
                self._yolo_time_total += elapsed
            elif operation == "ocr":
                self._ocr_time_total += elapsed
                self._ocr_call_count += 1

    def status(self) -> dict[str, object]:
        now = time.time()
        monotonic_now = time.monotonic()
        with self._tracker_lock:
            tracked_vehicles = self._tracker.snapshot(now)
        payload = {
            "camera_running": self._running,
            "camera_index": settings.camera_index,
            "tracker": settings.tracker_type,
            "track_count": len(tracked_vehicles),
            "recognized_count": sum(1 for item in tracked_vehicles if item["plate"] and item["plate"] != "Reading..."),
            "tracking_enabled": settings.tracking_enabled,
            "ocr_interval": settings.ocr_interval,
            "track_timeout": settings.track_timeout,
            "actual_camera_resolution": {
                "width": self._actual_camera_width,
                "height": self._actual_camera_height,
            },
            "actual_camera_fps": self._actual_camera_fps,
            "raw_capture_fps": self._rate(self._raw_frame_count, self._raw_started_at, monotonic_now),
            "stream_fps": self._rate(self._stream_frame_count, self._stream_started_at, monotonic_now),
            "ai_processing_fps": self._rate(self._ai_frame_count, self._ai_started_at, monotonic_now),
            "average_yolo_ms": (self._yolo_time_total / self._ai_frame_count * 1000.0) if self._ai_frame_count else 0.0,
            "average_ocr_ms": (self._ocr_time_total / self._ocr_call_count * 1000.0) if self._ocr_call_count else 0.0,
            "average_total_processing_ms": (self._total_processing_time / self._ai_frame_count * 1000.0) if self._ai_frame_count else 0.0,
            "ocr_call_count": self._ocr_call_count,
            "processing_in_progress": self._processing_in_progress,
            "tracked_vehicles": tracked_vehicles,
            "error": self._error,
        }
        for item in payload["tracked_vehicles"]:
            plate = item.get("plate")
            status_data = _status_payload(plate) if plate and plate != "Reading..." else {"plate_text": "", "status": "clear", "reason": None, "amount_owed": 0.0, "outstanding_amount": 0.0, "is_flagged": False}
            item["confidence_bar"] = item.get("confidence_bar", "░░░░░░░░░")
            item["confidence_score"] = item.get("ocr_confidence_percent", 0.0)
            item["status"] = status_data.get("status", "clear") if status_data.get("status") != "clear" else item.get("plate_status", "Reading...")
            item["reason"] = status_data.get("reason")
            item["amount_owed"] = float(status_data.get("amount_owed", 0.0) or 0.0)
            item["outstanding_amount"] = float(status_data.get("outstanding_amount", 0.0) or 0.0)
            item["is_flagged"] = bool(status_data.get("is_flagged", False))
            if plate and plate != "Reading...":
                item["plate_status"] = item["status"]
            if item.get("registration_status") == "Registered":
                item["plate_status"] = "REGISTERED"
                item["status"] = "REGISTERED"
        return payload

    def _annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame

        self._refresh_traffic_light_detector()
        config = self._traffic_monitor.config
        effective_state = "UNKNOWN"
        if config.traffic_light_enabled:
            if str(config.traffic_light_mode or "MANUAL").upper() == "AUTO":
                detected_state, _, _ = self._traffic_light_detector.detect(frame)
                effective_state = normalize_traffic_light_state(detected_state)
                config.traffic_light_state = effective_state
                if config.traffic_light_debug_overlay:
                    frame = self._traffic_light_detector.render_overlay(frame, effective_state)
            else:
                effective_state = normalize_traffic_light_state(config.traffic_light_state) if config.traffic_light_state is not None else "UNKNOWN"
                config.traffic_light_state = effective_state
                if config.traffic_light_debug_overlay:
                    frame = self._traffic_light_detector.render_overlay(frame, effective_state)

        height, width = frame.shape[:2]
        max_dimension = max(width, height)
        scale = 1.0
        if max_dimension > max(settings.max_frame_width, settings.max_frame_height):
            scale = min(settings.max_frame_width / width, settings.max_frame_height / height, 1.0)
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        model = self._detector._load()
        yolo_started = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            tracker=settings.tracker_type,
            conf=settings.detection_confidence,
            iou=settings.detection_iou,
            imgsz=settings.detection_imgsz,
            verbose=False,
        )
        self._record_timing("yolo", time.perf_counter() - yolo_started)
        if not results:
            return frame

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return frame

        ids = getattr(boxes, "id", None)
        if ids is None:
            return frame

        ids_list = ids.int().cpu().tolist()
        conf_list = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        now = time.time()

        for index, coordinates in enumerate(xyxy):
            x1, y1, x2, y2 = (max(0, int(value)) for value in coordinates)
            track_id = int(ids_list[index]) if index < len(ids_list) else None
            detection_conf = float(conf_list[index]) if index < len(conf_list) else 0.0
            with self._tracker_lock:
                state = self._tracker.observe(track_id, (x1, y1, x2, y2), detection_conf, now)

            with self._tracker_lock:
                should_run_ocr = self._tracker.should_ocr(state.track_id, now)
            if settings.live_track_only_mode and state.plate and not should_run_ocr:
                should_run_ocr = False
            if settings.live_track_only_mode and state.plate and should_run_ocr and (now - state.last_ocr_time) < settings.live_track_only_ocr_interval:
                should_run_ocr = False

            if should_run_ocr:
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size:
                    ocr_started = time.perf_counter()
                    plate, score = self._ocr.read(crop)
                    self._record_timing("ocr", time.perf_counter() - ocr_started)
                    if plate:
                        with self._tracker_lock:
                            accepted = self._tracker.record_ocr_result(state.track_id, plate, score, detection_conf, now)
                        if accepted and not state.registration_recorded_plate:
                            make_prediction = self._vehicle_classifier.classify(frame)
                            registration = persistent_vehicles.record_recognition(
                                state.plate,
                                make_prediction.make if make_prediction else None,
                                "Live Camera",
                                detection_conf,
                                state.ocr_confidence,
                                vehicle_confidence=make_prediction.confidence if make_prediction else None,
                                ocr_threshold=settings.traffic_ocr_threshold,
                            )
                            if registration:
                                state.registration_recorded_plate = state.plate
                                state.vehicle_make = registration.get("vehicle_make")
                                state.registration_count = int(registration["recognition_count"])

            self._traffic_monitor.config.traffic_light_state = effective_state
            self._traffic_monitor.observe(
                state.track_id,
                (x1, y1, x2, y2),
                now,
                plate_text=state.plate or None,
                detection_confidence=detection_conf,
                ocr_confidence=state.ocr_confidence,
                frame=frame,
            )

            plate_label = state.plate or "Reading..."
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {state.track_id}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, plate_label, (x1, min(frame.shape[0] - 10, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        with self._tracker_lock:
            self._tracker.prune(now)
        return frame

    def generate(self):
        while not self._stop_event.is_set():
            with self._lock:
                if not self._running:
                    break
                display_frame = self._last_display_frame if self._last_display_frame is not None else self._latest_frame
                display_frame = display_frame.copy() if display_frame is not None else None
            if display_frame is None:
                self._frame_event.wait(0.05)
                continue
            success, encoded = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not success:
                self._stop_event.wait(0.05)
                continue
            with self._lock:
                self._stream_frame_count += 1
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )
            self._stop_event.wait(0.01)

    @staticmethod
    def _rate(count: int, started_at: float, now: float) -> float:
        elapsed = now - started_at
        return round(count / elapsed, 2) if started_at and elapsed > 0 else 0.0


live_camera = LiveCameraManager()
mobile_traffic_config = CameraRuleConfig.from_dict({**traffic_config.__dict__, "camera_id": "MOBILE-01", "camera_name": "Mobile browser", "citation_amounts": {"RED_LIGHT": float(settings.red_light_fine)}})
mobile_traffic_monitor = TrafficViolationMonitor(
    traffic_store,
    mobile_traffic_config,
    min_track_frames=settings.traffic_min_track_frames,
    confidence_threshold=settings.traffic_confidence_threshold,
    ocr_threshold=settings.traffic_ocr_threshold,
    cooldown=settings.traffic_alert_cooldown,
)
mobile_camera = LiveCameraManager(mobile_traffic_monitor)


def _decode_image(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    return image


def _encode_image(image: np.ndarray, fmt: str = ".png") -> str:
    success, encoded = cv2.imencode(fmt, image)
    if not success:
        raise ValueError("Unable to encode image for debug output")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _persist_upload_results(results: list, image_path: str | None = None) -> dict[str, dict[str, object]]:
    registrations: dict[str, dict[str, object]] = {}
    for result_index, result in enumerate(results):
        log_phase(logger, "vehicle_persistence_record_before", result_index=result_index)
        registration = persistent_vehicles.record_recognition(
            result.text,
            result.make,
            "Image Upload",
            result.detection_confidence,
            result.ocr_confidence,
            image_path=image_path,
            vehicle_confidence=result.vehicle_confidence,
            ocr_threshold=settings.traffic_ocr_threshold,
        )
        log_phase(logger, "vehicle_persistence_record_after", result_index=result_index)
        if registration:
            registrations[normalize_plate_text(result.text)] = registration
    return registrations


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_configured": settings.model_path.exists(),
        "model_path": str(settings.model_path),
        "confidence_threshold": settings.detection_confidence,
        "image_size": settings.detection_imgsz,
    }


@app.post("/api/v1/recognize")
async def recognize(request: Request, file: UploadFile = File(...), debug: bool = False) -> dict[str, object]:
    log_phase(logger, "recognition_start")
    require_admin(request, admins, settings.auth_secret)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")
    payload = await file.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds configured size limit")
    try:
        image = _decode_image(payload)
        log_phase(logger, "image_decoded", width=int(image.shape[1]), height=int(image.shape[0]))
        height, width = image.shape[:2]
        longest_side = max(width, height)
        coordinate_scale = 1.0
        if longest_side > 1280:
            scale = 1280 / longest_side
            coordinate_scale = 1.0 / scale
            resized_dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(image, resized_dimensions, interpolation=cv2.INTER_AREA)
        log_phase(logger, "recognition_pipeline_before", width=int(image.shape[1]), height=int(image.shape[0]))
        results = await run_in_threadpool(pipeline.recognize, image, debug=debug)
        log_phase(logger, "recognition_pipeline_after", result_count=len(results))
        valid_results = [
            result
            for result in results
            if is_plausible_nigerian_plate(result.text)
            and result.ocr_confidence >= settings.traffic_ocr_threshold
        ]
        log_phase(logger, "history_save_before", result_count=len(valid_results))
        history.save_results(valid_results)
        log_phase(logger, "history_save_after", result_count=len(valid_results))
        log_phase(logger, "vehicle_persistence_before", result_count=len(results))
        registrations = _persist_upload_results(results, file.filename)
        log_phase(logger, "vehicle_persistence_after", registration_count=len(registrations))
        log_phase(logger, "recognition_response_construction_before", result_count=len(results))
        response: dict[str, object] = {
            "count": len(results),
            "results": [
                {
                    **result.to_dict(),
                    "bbox": {
                        "x1": round(result.bbox.x1 * coordinate_scale),
                        "y1": round(result.bbox.y1 * coordinate_scale),
                        "x2": round(result.bbox.x2 * coordinate_scale),
                        "y2": round(result.bbox.y2 * coordinate_scale),
                    },
                    "xyxy": [
                        round(result.bbox.x1 * coordinate_scale),
                        round(result.bbox.y1 * coordinate_scale),
                        round(result.bbox.x2 * coordinate_scale),
                        round(result.bbox.y2 * coordinate_scale),
                    ],
                    "confidence": result.detection_confidence,
                    **_status_payload(result.text),
                    **(
                        {
                            "is_registered": True,
                            "registration_status": "Registered",
                            "vehicle_id": registration["vehicle_id"],
                            "recognition_count": registration["recognition_count"],
                            "first_seen": registration["first_seen"],
                            "last_seen": registration["last_seen"],
                            "vehicle_status": registration["status"],
                            "vehicle_make": registration["vehicle_make"],
                            "recognition_label": "NEW VEHICLE REGISTERED" if registration["is_new"] else "REGISTERED VEHICLE",
                        }
                        if (registration := registrations.get(normalize_plate_text(result.text)))
                        else {}
                    ),
                }
                for result in results
            ],
            "detection": pipeline.last_detection_diagnostics,
        }
        if debug:
            response["debug"] = {
                "model_path": str(settings.model_path),
                "confidence_threshold": settings.detection_confidence,
                "image_size": settings.detection_imgsz,
                "number_of_detections": len(results),
                "last_detection_confidence": results[0].detection_confidence if results else None,
                "bbox": results[0].bbox.to_dict() if results else None,
                "detection": pipeline.last_detection_diagnostics,
            }
        log_phase(logger, "recognition_response_construction_after", result_count=len(results))
        log_phase(logger, "recognition_complete")
        return response
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Recognition request failed")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}") from exc


@app.post("/api/v1/debug/detect")
async def debug_detect(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")
    payload = await file.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds configured size limit")

    try:
        image = _decode_image(payload)
        diagnostic = pipeline.detector.diagnose(image)
        for candidate in diagnostic["results"]:
            for detection in candidate["detections"]:
                x1, y1, x2, y2 = detection["bbox"]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{detection['confidence']:.2f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return {
            "model_path": str(settings.model_path),
            "model_exists": settings.model_path.exists(),
            "confidence_thresholds": [0.35, 0.25, 0.20, 0.15, 0.10],
            "image_sizes": [640, 832, 1024, 1280],
            "diagnostic": diagnostic,
            "original_image_base64": _encode_image(image),
        }
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/debug/segmentation")
async def debug_segmentation(request: Request, file: UploadFile = File(...), debug: bool = False) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")
    payload = await file.read()
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds configured size limit")

    try:
        image = _decode_image(payload)
        if hasattr(pipeline.detector, "detect_robust"):
            try:
                detections = pipeline.detector.detect_robust(image)
            except TypeError:
                detections = pipeline.detector.detect(image)
        else:
            detections = pipeline.detector.detect(image)

        fallback_crop = None
        fallback_segmentation = None
        debug_dir = tempfile.mkdtemp(prefix="vnpr_segdebug_") if debug else None
        if not detections:
            fallback_segmentation = segment_plate_characters(image, debug=bool(debug), debug_dir=debug_dir)
            if fallback_segmentation.get("success") or int(fallback_segmentation.get("character_count", 0)) > 0:
                fallback_crop = image

        response: dict[str, object] = {
            "plate_detected": bool(detections) or bool(fallback_segmentation and (fallback_segmentation.get("success") or int(fallback_segmentation.get("character_count", 0)) > 0)),
            "detection_count": len(detections) if detections else (1 if fallback_crop is not None else 0),
            "character_segmentation": {
                "enabled": bool(debug),
                "success": False,
                "character_count": 0,
                "boxes": [],
                "character_crops": [],
                "debug": {"reason": "no_plate_detected"},
            },
            "debug_images": {},
        }
        if not detections and fallback_crop is None:
            return response

        detection = detections[0] if detections else None
        crop = image[detection.y1:detection.y2, detection.x1:detection.x2] if detection is not None else fallback_crop
        if crop is None or crop.size == 0:
            return response

        if debug and debug_dir is None:
            debug_dir = tempfile.mkdtemp(prefix="vnpr_segdebug_")
        segmentation = fallback_segmentation or segment_plate_characters(crop, debug=bool(debug), debug_dir=debug_dir)
        response["character_segmentation"] = {
            "enabled": bool(debug),
            "success": bool(segmentation.get("success")),
            "character_count": int(segmentation.get("character_count", 0)),
            "boxes": _json_safe(segmentation.get("boxes", [])),
            "character_crops": [],
            "debug": _json_safe(segmentation.get("debug", {})),
        }

        if debug and debug_dir is not None:
            debug_images: dict[str, str] = {}
            for filename in sorted(Path(debug_dir).iterdir()):
                if filename.is_file():
                    encoded = base64.b64encode(filename.read_bytes()).decode("utf-8")
                    debug_images[filename.stem] = encoded
            response["debug_images"] = debug_images

        return response
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Segmentation debug endpoint failed")
        return {
            "plate_detected": False,
            "detection_count": 0,
            "character_segmentation": {"enabled": bool(debug), "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {"reason": str(exc)}},
            "debug_images": {},
        }


@app.get("/api/v1/history")
def get_history(request: Request, limit: int = 100) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    records = history.list_results(limit)
    return {"count": len(records), "results": records}


@app.delete("/api/v1/history")
def clear_history(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    history.clear()
    persistent_vehicles.clear_history()
    return {"ok": True, "count": 0}


@app.get("/api/v1/vehicle-history")
def get_vehicle_history(request: Request, plate_number: str | None = None, limit: int = 100) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    records = persistent_vehicles.list_history(limit, plate_number)
    return {"count": len(records), "results": records}


@app.get("/api/v1/vehicles")
def list_recognized_vehicles(request: Request, limit: int = 100) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 500")
    records = persistent_vehicles.list_vehicles(limit)
    return {"count": len(records), "results": records}

@app.get("/api/v1/violations")
def list_traffic_violations(request: Request, plate_text: str | None = None, violation_type: str | None = None, camera_id: str | None = None, violation_status: str | None = None, limit: int = 100) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    results = traffic_store.list_violations({"plate_text": plate_text, "violation_type": violation_type, "camera_id": camera_id, "violation_status": violation_status, "limit": limit})
    return {"count": len(results), "results": results}

@app.get("/api/v1/violations/{violation_id}")
def get_traffic_violation(violation_id: int, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    result = traffic_store.get_violation(violation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Traffic violation not found")
    return result


@app.get("/api/v1/violations/{violation_id}/evidence")
def get_traffic_evidence(violation_id: int, request: Request) -> FileResponse:
    require_admin(request, admins, settings.auth_secret)
    result = traffic_store.get_violation(violation_id)
    evidence = Path(str(result.get("evidence_image"))) if result and result.get("evidence_image") else None
    if evidence is None or not evidence.exists() or evidence.parent.resolve() != settings.traffic_evidence_path.resolve():
        raise HTTPException(status_code=404, detail="Evidence image not found")
    return FileResponse(evidence)


@app.get("/api/v1/violations/{violation_id}/evidence-video")
def get_traffic_evidence_video(violation_id: int, request: Request) -> FileResponse:
    require_admin(request, admins, settings.auth_secret)
    result = traffic_store.get_violation(violation_id)
    evidence = Path(str(result.get("evidence_video"))) if result and result.get("evidence_video") else None
    if evidence is None or not evidence.exists() or evidence.parent.resolve() != settings.traffic_evidence_path.resolve():
        raise HTTPException(status_code=404, detail="Evidence video not found")
    return FileResponse(evidence, media_type="video/mp4")

def _review_traffic_violation(violation_id: int, action: str, request: Request) -> dict[str, object]:
    admin = require_admin(request, admins, settings.auth_secret)
    result = traffic_store.get_violation(violation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Traffic violation not found")
    return traffic_store.review(violation_id, action, int(admin["id"]))

@app.post("/api/v1/violations/{violation_id}/confirm")
def confirm_traffic_violation(violation_id: int, request: Request) -> dict[str, object]:
    return _review_traffic_violation(violation_id, "confirm", request)

@app.post("/api/v1/violations/{violation_id}/dismiss")
def dismiss_traffic_violation(violation_id: int, request: Request) -> dict[str, object]:
    return _review_traffic_violation(violation_id, "dismiss", request)

@app.get("/api/v1/citations")
def list_traffic_citations(request: Request, limit: int = 100) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    results = traffic_store.list_citations(limit)
    return {"count": len(results), "results": results}

@app.get("/api/v1/citations/{citation_id}")
def get_traffic_citation(citation_id: int, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    result = traffic_store.get_citation(citation_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Citation not found")
    return result

@app.post("/api/v1/citations/{citation_id}/approve")
def approve_traffic_citation(citation_id: int, request: Request) -> dict[str, object]:
    admin = require_admin(request, admins, settings.auth_secret)
    try:
        return traffic_store.update_citation_status(citation_id, "APPROVED", int(admin["id"]))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/v1/citations/{citation_id}/cancel")
def cancel_traffic_citation(citation_id: int, request: Request) -> dict[str, object]:
    admin = require_admin(request, admins, settings.auth_secret)
    try:
        return traffic_store.update_citation_status(citation_id, "CANCELLED", int(admin["id"]))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/v1/citations/{citation_id}")
async def edit_traffic_citation(citation_id: int, request: Request) -> dict[str, object]:
    admin = require_admin(request, admins, settings.auth_secret)
    data = await request.json()
    try:
        return traffic_store.update_citation(
            citation_id,
            int(admin["id"]),
            float(data["amount"]) if data.get("amount") is not None else None,
            str(data["due_date"]) if data.get("due_date") is not None else None,
            str(data["description"]) if data.get("description") is not None else None,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/v1/traffic/statistics")
def traffic_statistics(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    return traffic_store.statistics()

@app.get("/api/v1/cameras")
def list_traffic_cameras(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    results = traffic_store.list_cameras()
    return {"count": len(results), "results": results}

@app.post("/api/v1/cameras")
async def create_traffic_camera(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    global traffic_config
    try:
        data = await request.json()
        config = CameraRuleConfig.from_dict(data)
        traffic_config = config
        traffic_monitor.config = config
        mobile_traffic_config = CameraRuleConfig.from_dict({**config.__dict__, "camera_id": "MOBILE-01", "camera_name": "Mobile browser", "citation_amounts": {"RED_LIGHT": float(settings.red_light_fine)}})
        mobile_camera._traffic_monitor.config = mobile_traffic_config
        return traffic_store.save_camera(config, str(data.get("camera_source") or ""))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.put("/api/v1/cameras/{camera_id}")
async def update_traffic_camera(camera_id: str, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    global traffic_config
    data = await request.json()
    data["camera_id"] = camera_id
    try:
        traffic_config = CameraRuleConfig.from_dict(data)
        traffic_monitor.config = traffic_config
        mobile_traffic_config = CameraRuleConfig.from_dict({**traffic_config.__dict__, "camera_id": "MOBILE-01", "camera_name": "Mobile browser", "citation_amounts": {"RED_LIGHT": float(settings.red_light_fine)}})
        mobile_camera._traffic_monitor.config = mobile_traffic_config
        return traffic_store.save_camera(traffic_config, str(data.get("camera_source") or ""))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/registry")
def get_registry(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    records = registry.list_all()
    return {"count": len(records), "results": records}


@app.post("/api/v1/registry")
async def register_plate(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    data = await request.json()
    plate_text = normalize_plate_text(str(data.get("plate_text", "")))
    if not plate_text or not is_plausible_nigerian_plate(plate_text):
        raise HTTPException(status_code=400, detail="Provide a valid Nigerian plate number")
    return registry.upsert(plate_text, bool(data.get("is_verified", False)))


@app.delete("/api/v1/registry/{plate_text}")
def unregister_plate(plate_text: str, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    normalized_plate = normalize_plate_text(plate_text)
    if not registry.remove(normalized_plate):
        raise HTTPException(status_code=404, detail="Plate is not registered")
    return {"plate_text": normalized_plate, "is_registered": False, "is_verified": False, "registration_status": "not_registered_unverified"}


@app.post("/api/v1/auth/signup")
async def signup(request: Request) -> dict[str, object]:
    data = await request.json()
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    if len(name) < 2 or "@" not in email or len(password) < 8:
        raise HTTPException(status_code=400, detail="Provide a name, valid email, and password of at least 8 characters")
    try:
        admin = admins.create(name, email, password)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An admin with this email already exists") from exc
    response = JSONResponse({"admin": admin})
    response.set_cookie("vnpr_session", create_session(admin["id"], settings.auth_secret, settings.session_days, admin), httponly=True, samesite="lax", secure=True, max_age=settings.session_days * 86400)
    return response


@app.post("/api/v1/auth/login")
async def login(request: Request) -> JSONResponse:
    data = await request.json()
    admin = admins.authenticate(str(data.get("email", "")).strip(), str(data.get("password", "")))
    if admin is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    response = JSONResponse({"admin": admin})
    response.set_cookie("vnpr_session", create_session(admin["id"], settings.auth_secret, settings.session_days, admin), httponly=True, samesite="lax", secure=True, max_age=settings.session_days * 86400)
    return response


@app.get("/api/v1/auth/me")
def current_admin(request: Request) -> dict[str, object]:
    return {"admin": require_admin(request, admins, settings.auth_secret)}


@app.post("/api/v1/auth/logout")
def logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie("vnpr_session")
    return response


@app.get("/api/v1/vehicles/{plate_text}")
def get_vehicle_profile(plate_text: str, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    normalized_plate = normalize_plate_text(plate_text)
    profile = vehicle_status.lookup(normalized_plate)
    profile["obligations"] = vehicle_obligations.list_for_plate(normalized_plate)
    return profile


@app.get("/api/v1/vehicles/{plate_text}/status")
def get_vehicle_status(plate_text: str, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    return vehicle_status.lookup(normalize_plate_text(plate_text))


@app.get("/api/v1/vehicles/alerts")
def get_vehicle_alerts(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    alerts = vehicle_alerts.list_alerts()
    return {"count": len(alerts), "results": alerts}


@app.get("/api/v1/vehicles/alerts/summary")
def get_vehicle_alert_summary(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    return vehicle_alerts.summary()


@app.post("/api/v1/vehicles/status")
async def set_vehicle_status(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    data = await request.json()
    plate_text = normalize_plate_text(str(data.get("plate_text", "")))
    if not plate_text:
        raise HTTPException(status_code=400, detail="Provide a valid plate number")
    return vehicle_status.set_status(plate_text, str(data.get("status", "clear")), str(data.get("reason") or ""))


@app.post("/api/v1/vehicles/obligations")
async def add_vehicle_obligation(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    data = await request.json()
    plate_text = normalize_plate_text(str(data.get("plate_text", "")))
    if not plate_text:
        raise HTTPException(status_code=400, detail="Provide a valid plate number")
    amount = float(data.get("amount", 0.0) or 0.0)
    return vehicle_obligations.add_obligation(plate_text, amount, str(data.get("reason") or ""), str(data.get("status", "OPEN")))


@app.get("/api/v1/vehicles/{plate_text}/obligations")
def list_vehicle_obligations(plate_text: str, request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    normalized_plate = normalize_plate_text(plate_text)
    obligations = vehicle_obligations.list_for_plate(normalized_plate)
    return {"plate_text": normalized_plate, "count": len(obligations), "results": obligations}


@app.post("/api/v1/live/start")
def start_live_camera() -> dict[str, object]:
    try:
        return live_camera.start()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/mobile/live/start")
def start_mobile_live_camera(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    try:
        return mobile_camera.start_external()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/v1/mobile/live/stop")
def stop_mobile_live_camera(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    return mobile_camera.stop()


@app.post("/api/v1/mobile/live/frame")
async def submit_mobile_live_frame(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Send a JPEG camera frame")
    payload = await file.read(settings.max_image_bytes + 1)
    if len(payload) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Camera frame exceeds configured size limit")
    try:
        return mobile_camera.submit_frame(_decode_image(payload))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/mobile/live/status")
def mobile_live_status(request: Request) -> dict[str, object]:
    require_admin(request, admins, settings.auth_secret)
    return mobile_camera.status()


@app.post("/api/v1/live/stop")
def stop_live_camera() -> dict[str, object]:
    return live_camera.stop()


@app.get("/api/v1/live/status")
def live_status() -> dict[str, object]:
    return live_camera.status()


@app.get("/api/v1/live")
def live_stream() -> StreamingResponse:
    if not live_camera.running:
        raise HTTPException(status_code=503, detail="Live camera is not running")
    try:
        return StreamingResponse(live_camera.generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")