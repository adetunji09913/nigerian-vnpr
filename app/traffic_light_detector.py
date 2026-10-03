from __future__ import annotations

from collections import Counter, deque
from typing import Any

import cv2
import numpy as np


VALID_TRAFFIC_LIGHT_STATES = ("RED", "YELLOW", "GREEN", "UNKNOWN")


class TrafficLightDetector:
    def __init__(self, roi: dict[str, int] | None = None, min_confidence: float = 0.60, stable_frames: int = 4) -> None:
        self.roi = self._normalize_roi(roi or {"x1": 0, "y1": 0, "x2": 0, "y2": 0})
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.stable_frames = max(1, int(stable_frames))
        self._history: deque[str] = deque(maxlen=max(1, self.stable_frames))
        self._last_state = "UNKNOWN"
        self._last_confidence = 0.0

    @staticmethod
    def _normalize_roi(roi: dict[str, Any] | None) -> dict[str, int]:
        if not isinstance(roi, dict):
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        try:
            return {
                "x1": int(float(roi.get("x1", 0) or 0)),
                "y1": int(float(roi.get("y1", 0) or 0)),
                "x2": int(float(roi.get("x2", 0) or 0)),
                "y2": int(float(roi.get("y2", 0) or 0)),
            }
        except (TypeError, ValueError):
            return {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    def set_roi(self, roi: dict[str, Any] | None) -> None:
        self.roi = self._normalize_roi(roi)

    def _resolve_roi(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
        height, width = frame_shape[:2]
        roi = self.roi.copy()
        for key in ("x1", "y1", "x2", "y2"):
            value = float(roi.get(key, 0) or 0)
            if 0.0 < abs(value) <= 1.0:
                if key.startswith("x"):
                    roi[key] = int(round(width * value))
                else:
                    roi[key] = int(round(height * value))
        x1, y1, x2, y2 = map(int, (roi["x1"], roi["y1"], roi["x2"], roi["y2"]))
        if x2 <= x1 or y2 <= y1:
            return None
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _state_for_ratio(red_ratio: float, yellow_ratio: float, green_ratio: float) -> tuple[str, float]:
        strongest = max(("RED", red_ratio), ("YELLOW", yellow_ratio), ("GREEN", green_ratio), key=lambda item: item[1])
        if strongest[1] <= 0.0:
            return "UNKNOWN", 0.0
        return strongest[0], strongest[1]

    def _analyze_frame(self, frame: np.ndarray) -> tuple[str, float, dict[str, float]]:
        if frame is None or frame.size == 0:
            return "UNKNOWN", 0.0, {"red_ratio": 0.0, "yellow_ratio": 0.0, "green_ratio": 0.0}

        roi_bounds = self._resolve_roi(frame.shape)
        if roi_bounds is None:
            return "UNKNOWN", 0.0, {"red_ratio": 0.0, "yellow_ratio": 0.0, "green_ratio": 0.0}

        x1, y1, x2, y2 = roi_bounds
        region = frame[y1:y2, x1:x2]
        if region.size == 0 or region.shape[0] == 0 or region.shape[1] == 0:
            return "UNKNOWN", 0.0, {"red_ratio": 0.0, "yellow_ratio": 0.0, "green_ratio": 0.0}

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        area = float(region.shape[0] * region.shape[1])
        if area <= 0.0:
            return "UNKNOWN", 0.0, {"red_ratio": 0.0, "yellow_ratio": 0.0, "green_ratio": 0.0}

        red_mask_low = cv2.inRange(hsv, (0, 80, 80), (15, 255, 255))
        red_mask_high = cv2.inRange(hsv, (165, 80, 80), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)
        yellow_mask = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
        green_mask = cv2.inRange(hsv, (35, 70, 70), (90, 255, 255))

        red_ratio = float(np.count_nonzero(red_mask)) / area
        yellow_ratio = float(np.count_nonzero(yellow_mask)) / area
        green_ratio = float(np.count_nonzero(green_mask)) / area

        red_saturation = float(s[red_mask > 0].mean()) if np.any(red_mask > 0) else 0.0
        yellow_saturation = float(s[yellow_mask > 0].mean()) if np.any(yellow_mask > 0) else 0.0
        green_saturation = float(s[green_mask > 0].mean()) if np.any(green_mask > 0) else 0.0
        red_brightness = float(v[red_mask > 0].mean()) if np.any(red_mask > 0) else 0.0
        yellow_brightness = float(v[yellow_mask > 0].mean()) if np.any(yellow_mask > 0) else 0.0
        green_brightness = float(v[green_mask > 0].mean()) if np.any(green_mask > 0) else 0.0

        candidate_state, candidate_confidence = self._state_for_ratio(red_ratio, yellow_ratio, green_ratio)

        if candidate_state == "RED" and (red_ratio < 0.04 or red_saturation < 40.0 or red_brightness < 55.0):
            candidate_state = "UNKNOWN"
            candidate_confidence = 0.0
        if candidate_state == "YELLOW" and (yellow_ratio < 0.04 or yellow_saturation < 40.0 or yellow_brightness < 55.0):
            candidate_state = "UNKNOWN"
            candidate_confidence = 0.0
        if candidate_state == "GREEN" and (green_ratio < 0.04 or green_saturation < 40.0 or green_brightness < 55.0):
            candidate_state = "UNKNOWN"
            candidate_confidence = 0.0

        if candidate_state == "UNKNOWN":
            candidate_confidence = 0.0
        else:
            candidate_confidence = min(1.0, max(candidate_confidence, min(1.0, (candidate_confidence + 0.2))))

        return candidate_state, candidate_confidence, {
            "red_ratio": round(red_ratio, 4),
            "yellow_ratio": round(yellow_ratio, 4),
            "green_ratio": round(green_ratio, 4),
            "red_saturation": round(red_saturation, 2),
            "yellow_saturation": round(yellow_saturation, 2),
            "green_saturation": round(green_saturation, 2),
            "red_brightness": round(red_brightness, 2),
            "yellow_brightness": round(yellow_brightness, 2),
            "green_brightness": round(green_brightness, 2),
        }

    def detect(self, frame: np.ndarray) -> tuple[str, float, dict[str, Any]]:
        state, confidence, metrics = self._analyze_frame(frame)
        self._history.append(state)
        if len(self._history) < self.stable_frames:
            self._last_state = "UNKNOWN"
            self._last_confidence = confidence
            return "UNKNOWN", confidence, {**metrics, "stable_frames_seen": len(self._history)}

        counts = Counter(self._history)
        dominant_state, dominant_count = counts.most_common(1)[0]
        if dominant_state == "UNKNOWN":
            self._last_state = "UNKNOWN"
            self._last_confidence = confidence
            return "UNKNOWN", confidence, {**metrics, "stable_frames_seen": len(self._history), "dominant_state": dominant_state}

        if dominant_count >= max(2, self.stable_frames - 1):
            self._last_state = dominant_state
            self._last_confidence = max(confidence, self.min_confidence)
            return dominant_state, self._last_confidence, {**metrics, "stable_frames_seen": len(self._history), "dominant_state": dominant_state}

        self._last_state = "UNKNOWN"
        self._last_confidence = confidence
        return "UNKNOWN", confidence, {**metrics, "stable_frames_seen": len(self._history), "dominant_state": dominant_state}

    def render_overlay(self, frame: np.ndarray, state: str) -> np.ndarray:
        if frame is None or frame.size == 0:
            return frame
        annotated = frame.copy()
        roi_bounds = self._resolve_roi(annotated.shape)
        if roi_bounds is not None:
            x1, y1, x2, y2 = roi_bounds
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(annotated, "TRAFFIC LIGHT", (max(0, x1), max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        padding = 20
        cv2.putText(annotated, f"TRAFFIC LIGHT: {state}", (padding, max(30, annotated.shape[0] - 48)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        roi_status = "ROI: ACTIVE" if roi_bounds is not None else "ROI: INACTIVE"
        cv2.putText(annotated, roi_status, (padding, max(30, annotated.shape[0] - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return annotated

    @property
    def last_state(self) -> str:
        return self._last_state

    @property
    def last_confidence(self) -> float:
        return self._last_confidence
