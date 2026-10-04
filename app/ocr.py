from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

from .diagnostics import log_phase

from .normalization import (
    compact_plate_text,
    correct_positional_plate,
    is_ocr_quality_plate,
    normalize_plate_text,
    validate_nigerian_plate,
)


logger = logging.getLogger(__name__)


class OCRReader:

    def __init__(
        self,
        languages: tuple[str, ...] = ("en",),
        gpu: bool = False,
        padding_pct: float = 0.10,
        upscale_factor: float = 5.0,
        allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ) -> None:

        self.languages = list(languages)
        self.gpu = gpu

        self.padding_pct = max(
            0.05,
            min(0.15, float(padding_pct))
        )

        self.upscale_factor = max(
            4.0,
            min(6.0, float(upscale_factor))
        )

        self.allowlist = allowlist

        self._reader: Any = None

        self.last_debug: dict[str, Any] = {}


    # =========================================================
    # LOAD EASYOCR
    # =========================================================

    def _load(self) -> Any:

        if self._reader is None:

            try:
                import easyocr

            except ImportError as exc:
                raise RuntimeError(
                    "Install easyocr to use OCR"
                ) from exc

            log_phase(logger, "before_easyocr_load")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu
            )
            log_phase(logger, "after_easyocr_load")

        return self._reader


    # =========================================================
    # CLEAN OCR TEXT
    # =========================================================

    def _clean(self, text: str) -> str:

        text = str(text or "").upper()

        text = text.replace("\n", "")
        text = text.replace(" ", "")
        text = text.replace("-", "")

        return re.sub(
            r"[^A-Z0-9]",
            "",
            text
        )


    # =========================================================
    # ADD PADDING AROUND PLATE
    # =========================================================

    def _pad_crop(self, image: Any) -> Any:

        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]

        pad_x = max(
            4,
            int(width * self.padding_pct)
        )

        pad_y = max(
            4,
            int(height * self.padding_pct)
        )

        padded = cv2.copyMakeBorder(
            image,
            top=pad_y,
            bottom=pad_y,
            left=pad_x,
            right=pad_x,
            borderType=cv2.BORDER_REPLICATE,
            value=(0, 0, 0),
        )

        return padded


    # =========================================================
    # RESIZE PLATE
    # =========================================================

    def resize_plate(
        self,
        image: Any,
        scale: float | None = None
    ) -> Any:

        if image is None or image.size == 0:
            return image

        target_scale = (
            self.upscale_factor
            if scale is None
            else float(scale)
        )

        new_width = max(
            1,
            int(image.shape[1] * target_scale)
        )

        new_height = max(
            1,
            int(image.shape[0] * target_scale)
        )

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )


    # =========================================================
    # CONTRAST ENHANCEMENT
    # =========================================================

    def enhance_contrast(
        self,
        gray: Any
    ) -> dict[str, Any]:

        if gray is None:
            return {}

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        clahe_img = clahe.apply(gray)

        denoised = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=8,
            templateWindowSize=7,
            searchWindowSize=21
        )

        blur = cv2.GaussianBlur(
            clahe_img,
            (3, 3),
            0
        )

        sharpened = cv2.addWeighted(
            clahe_img,
            1.25,
            blur,
            -0.25,
            0
        )

        return {
            "gray": gray,
            "clahe": clahe_img,
            "denoised": denoised,
            "sharpened": sharpened,
        }


    # =========================================================
    # PERSPECTIVE CORRECTION
    # =========================================================

    def correct_perspective(
        self,
        image: Any
    ) -> Any:

        if image is None or image.size == 0:
            return image

        if len(image.shape) == 3:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            gray = image

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        best_contour = None
        best_area = 0.0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 100:
                continue

            peri = cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                0.04 * peri,
                True
            )

            if len(approx) == 4:

                if area > best_area:

                    best_area = area
                    best_contour = approx

        if best_contour is None:
            return image

        pts = best_contour.reshape(
            4,
            2
        )

        rect = np.zeros(
            (4, 2),
            dtype=np.float32
        )

        s = pts.sum(axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(
            pts,
            axis=1
        )

        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        tl, tr, br, bl = rect

        width_top = np.linalg.norm(
            tr - tl
        )

        width_bottom = np.linalg.norm(
            br - bl
        )

        height_left = np.linalg.norm(
            bl - tl
        )

        height_right = np.linalg.norm(
            br - tr
        )

        max_width = max(
            int(max(
                width_top,
                width_bottom
            )),
            1
        )

        max_height = max(
            int(max(
                height_left,
                height_right
            )),
            1
        )

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32
        )

        transform = cv2.getPerspectiveTransform(
            rect,
            dst
        )

        rectified = cv2.warpPerspective(
            image,
            transform,
            (
                max_width,
                max_height
            )
        )

        if rectified.size == 0:
            return image

        return rectified


    # =========================================================
    # PREPROCESS PLATE
    # =========================================================

    def preprocess_plate(
        self,
        crop: Any
    ) -> dict[str, Any]:

        if crop is None or crop.size == 0:

            return {
                "padded": None,
                "rectified": None,
                "enlarged": None,
                "variants": [],
            }

        image = crop.copy()

        if len(image.shape) == 2:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        padded = self._pad_crop(
            image
        )

        rectified = self.correct_perspective(
            padded
        )

        original_enlarged = self.resize_plate(
            padded
        )

        enlarged = self.resize_plate(
            rectified
        )

        gray = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2GRAY
        )

        contrast = self.enhance_contrast(
            gray
        )

        rgb_original = cv2.cvtColor(
            original_enlarged,
            cv2.COLOR_BGR2RGB
        )

        variants = [
            (
                "original_rgb",
                rgb_original
            ),
            (
                "grayscale",
                gray
            ),
        ]

        return {
            "padded": padded,
            "rectified": rectified,
            "enlarged": enlarged,
            "variants": variants,
            "contrast": contrast,
        }


    # =========================================================
    # RUN EASYOCR
    # =========================================================

    def _run_variant_ocr(
        self,
        reader: Any,
        variant_name: str,
        image: Any
    ) -> dict[str, Any]:

        if image is None or image.size == 0:

            return {
                "text": "",
                "confidence": 0.0,
                "method": variant_name,
                "boxes": [],
                "raw_detections": [],
            }

        try:

            log_phase(
                logger,
                "ocr_readtext_before",
                variant=variant_name,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
            entries = reader.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=self.allowlist,
                mag_ratio=1.5,
                text_threshold=0.25,
                low_text=0.10,
                link_threshold=0.10,
                width_ths=0.5,
                height_ths=0.5,
                min_size=5,
                canvas_size=2560,
                decoder="greedy",
            )
            log_phase(
                logger,
                "ocr_readtext_after",
                variant=variant_name,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )

        except Exception as exc:
            log_phase(
                logger,
                "ocr_readtext_error",
                variant=variant_name,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
            raise

        if not entries:

            return {
                "text": "",
                "confidence": 0.0,
                "method": variant_name,
                "boxes": [],
                "raw_detections": [],
            }

        raw_detections: list[dict[str, Any]] = []

        # -----------------------------------------------------
        # SAVE ALL EASY-OCR DETECTIONS
        # -----------------------------------------------------

        for entry in entries:

            text = self._clean(
                str(entry[1])
            )

            if not text:
                continue

            if len(text) > 12:
                text = text[:12]

            confidence = float(
                entry[2]
            )

            points = np.asarray(
                entry[0],
                dtype=np.float32
            ).reshape(-1, 2)

            x_min, y_min = points.min(
                axis=0
            )

            x_max, y_max = points.max(
                axis=0
            )

            box = tuple(
                int(value)
                for value in points.reshape(-1).tolist()
            )

            raw_detections.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bounding_box": box,
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                }
            )

        if not raw_detections:

            return {
                "text": "",
                "confidence": 0.0,
                "method": variant_name,
                "boxes": [],
                "raw_detections": [],
            }


        # =====================================================
        # FIRST: CHECK IF EASYOCR FOUND THE WHOLE PLATE
        # =====================================================

        whole_plate_candidates = [
            item
            for item in raw_detections
            if len(item["text"]) >= 5
        ]

        if whole_plate_candidates:

            best = max(
                whole_plate_candidates,
                key=lambda item: item["confidence"]
            )

            raw_text = best["text"]

            corrected_text, correction_applied = (
                correct_positional_plate(
                    raw_text
                )
            )

            return {
                "text": corrected_text,
                "confidence": best["confidence"],
                "method": variant_name,
                "boxes": [
                    best["bounding_box"]
                ],
                "raw_detections": raw_detections,
                "corrected_text": corrected_text,
                "correction_applied": correction_applied,
            }


        # =====================================================
        # SECOND: COMBINE INDIVIDUAL CHARACTERS
        # =====================================================

        characters = []

        for item in raw_detections:

            if len(item["text"]) == 1:

                characters.append(
                    item
                )

        if len(characters) < 5:

            return {
                "text": "",
                "confidence": 0.0,
                "method": variant_name,
                "boxes": [],
                "raw_detections": raw_detections,
            }


        # -----------------------------------------------------
        # SORT CHARACTERS FROM LEFT TO RIGHT
        # -----------------------------------------------------

        characters.sort(
            key=lambda item: item["x_min"]
        )


        # -----------------------------------------------------
        # COMBINE CHARACTERS
        # -----------------------------------------------------

        combined_text = "".join(
            item["text"]
            for item in characters
        )

        combined_confidence = (
            sum(
                item["confidence"]
                for item in characters
            )
            / len(characters)
        )


        # =====================================================
        # CORRECT OCR CHARACTER CONFUSIONS
        # =====================================================

        corrected_text, correction_applied = (
            correct_positional_plate(
                combined_text
            )
        )


        # =====================================================
        # CREATE ONE BOX AROUND COMPLETE PLATE
        # =====================================================

        x_min = min(
            item["x_min"]
            for item in characters
        )

        y_min = min(
            item["y_min"]
            for item in characters
        )

        x_max = max(
            item["x_max"]
            for item in characters
        )

        y_max = max(
            item["y_max"]
            for item in characters
        )

        combined_box = (
            int(x_min),
            int(y_min),
            int(x_max),
            int(y_max)
        )


        return {
            "text": corrected_text,
            "confidence": combined_confidence,
            "method": variant_name,
            "boxes": [
                combined_box
            ],
            "raw_detections": raw_detections,
            "corrected_text": corrected_text,
            "correction_applied": correction_applied,
        }


    # =========================================================
    # SELECT BEST PLATE
    # =========================================================

    def select_best_plate(
        self,
        results: list[dict[str, Any]]
    ) -> tuple[
        str,
        float,
        str | None,
        list[dict[str, Any]]
    ]:

        if not results:

            return (
                "",
                0.0,
                None,
                []
            )

        candidates = []

        for item in results:

            raw_text = compact_plate_text(
                item.get("text", "")
            )

            confidence = float(
                item.get(
                    "confidence",
                    0.0
                )
            )

            corrected_text, correction_applied = (
                correct_positional_plate(
                    raw_text
                )
            )

            valid_format = is_ocr_quality_plate(
                corrected_text
            )

            candidates.append(
                {
                    "text": raw_text,
                    "confidence": confidence,
                    "method": item.get(
                        "method",
                        "unknown"
                    ),
                    "boxes": item.get(
                        "boxes",
                        []
                    ),
                    "corrected_text": corrected_text,
                    "correction_applied": correction_applied,
                    "valid_format": valid_format,
                    "accepted": (
                        confidence >= 0.20
                        and valid_format
                    ),
                    "raw_detections": item.get(
                        "raw_detections",
                        []
                    ),
                }
            )


        accepted_candidates = [
            item
            for item in candidates
            if item["accepted"]
        ]


        if not accepted_candidates:

            return (
                "",
                0.0,
                None,
                candidates
            )


        best = max(
            accepted_candidates,
            key=lambda item: (
                item["confidence"]
            )
        )


        return (
            best["corrected_text"],
            best["confidence"],
            best["method"],
            candidates,
        )


    # =========================================================
    # RUN OCR VARIANTS
    # =========================================================

    def run_ocr_variants(
        self,
        image: Any
    ) -> list[dict[str, Any]]:

        log_phase(
            logger,
            "ocr_preprocess_before",
            width=int(image.shape[1]) if image is not None and image.size else 0,
            height=int(image.shape[0]) if image is not None and image.size else 0,
        )
        pipeline = self.preprocess_plate(
            image
        )
        log_phase(
            logger,
            "ocr_preprocess_after",
            width=int(image.shape[1]) if image is not None and image.size else 0,
            height=int(image.shape[0]) if image is not None and image.size else 0,
        )

        variants = pipeline.get(
            "variants",
            []
        )

        if not variants:
            return []

        reader = self._load()

        results: list[dict[str, Any]] = []

        for variant_name, variant_image in variants[:2]:

            result = self._run_variant_ocr(
                reader,
                variant_name,
                variant_image
            )

            if result.get("text"):

                results.append(
                    result
                )

        return results


    # =========================================================
    # READ OCR
    # =========================================================

    def _read_with_variants(
        self,
        image: Any
    ) -> tuple[
        str,
        float,
        dict[str, Any]
    ]:

        variants = self.run_ocr_variants(
            image
        )

        if not variants:

            return (
                "",
                0.0,
                {
                    "results": [],
                    "final_method": None,
                    "final_confidence": 0.0,
                }
            )


        selected_text, selected_confidence, selected_method, fused = (
            self.select_best_plate(
                variants
            )
        )


        selected_fused = next(
            (
                item
                for item in fused
                if item.get(
                    "corrected_text"
                ) == selected_text
            ),
            {}
        )


        selected_raw_text = selected_fused.get(
            "text",
            ""
        )


        candidates = []

        for item in variants:

            corrected_item, _ = (
                correct_positional_plate(
                    item.get(
                        "text",
                        ""
                    )
                )
            )

            candidates.append(
                {
                    "text": item.get(
                        "text",
                        ""
                    ),
                    "corrected_text": corrected_item,
                    "confidence": float(
                        item.get(
                            "confidence",
                            0.0
                        )
                    ),
                    "valid_format": is_ocr_quality_plate(
                        corrected_item
                    ),
                    "method": item.get(
                        "method",
                        "unknown"
                    ),
                    "bounding_box": (
                        item.get(
                            "boxes",
                            [None]
                        )[0]
                    ),
                }
            )


        raw_detections = [
            detection
            for item in variants
            for detection in item.get(
                "raw_detections",
                []
            )
        ]


        debug_payload = {

            "results": [
                {
                    "text": item.get(
                        "text",
                        ""
                    ),
                    "confidence": float(
                        item.get(
                            "confidence",
                            0.0
                        )
                    ),
                    "method": item.get(
                        "method",
                        "unknown"
                    ),
                    "boxes": item.get(
                        "boxes",
                        []
                    ),
                }
                for item in variants
            ],

            "final_plate": selected_text,

            "final_confidence": selected_confidence,

            "final_method": selected_method,

            "raw_ocr_text": selected_raw_text,

            "corrected_text": selected_text,

            "correction_applied": (
                selected_raw_text != selected_text
            ),

            "correction_reason": (
                "positional_character_correction"
                if selected_raw_text != selected_text
                else None
            ),

            "raw_ocr_detections": raw_detections,

            "selected_ocr_candidate": {

                "text": selected_raw_text,

                "confidence": selected_confidence,

                "corrected_text": selected_text,

                "bounding_box": selected_fused.get(
                    "boxes",
                    [None]
                )[0],
            },

            "ocr_candidates_by_variant": fused,

            "ocr_attempts": len(
                variants
            ),

            "ocr_candidates": candidates,

            "ocr_confidence_before": max(
                (
                    item["confidence"]
                    for item in candidates
                ),
                default=0.0
            ),

            "ocr_confidence_after": selected_confidence,
        }


        return (
            selected_text,
            selected_confidence,
            debug_payload
        )


    # =========================================================
    # PUBLIC READ METHOD
    # =========================================================

    def read(
        self,
        crop: Any
    ) -> tuple[str, float]:

        log_phase(
            logger,
            "ocr_read_before",
            width=int(crop.shape[1]) if crop is not None and crop.size else 0,
            height=int(crop.shape[0]) if crop is not None and crop.size else 0,
        )
        selected_text, selected_confidence, debug_payload = (
            self._read_with_variants(
                crop
            )
        )
        log_phase(
            logger,
            "ocr_read_after",
            width=int(crop.shape[1]) if crop is not None and crop.size else 0,
            height=int(crop.shape[0]) if crop is not None and crop.size else 0,
        )

        self.last_debug = debug_payload

        return (
            selected_text,
            selected_confidence
        )


    # =========================================================
    # DEBUG METHOD
    # =========================================================

    def read_debug(
        self,
        crop: Any
    ) -> dict[str, Any]:

        pipeline = self.preprocess_plate(
            crop
        )

        if crop is None or crop.size == 0:

            return {
                "results": [],
                "final_plate": "",
                "final_confidence": 0.0,
                "preprocessed": {},
                "method": None,
            }


        if self.last_debug:

            debug_payload = self.last_debug

            variants = debug_payload.get(
                "results",
                []
            )

            selected_text = debug_payload.get(
                "final_plate",
                ""
            )

            selected_confidence = float(
                debug_payload.get(
                    "final_confidence",
                    0.0
                )
            )

        else:

            selected_text, selected_confidence, debug_payload = (
                self._read_with_variants(
                    crop
                )
            )

            variants = debug_payload.get(
                "results",
                []
            )


        debug_payload["preprocessed"] = {

            "padded": pipeline.get(
                "padded"
            ),

            "rectified": pipeline.get(
                "rectified"
            ),

            "enlarged": pipeline.get(
                "enlarged"
            ),

            "variants": [
                name
                for name, _
                in pipeline.get(
                    "variants",
                    []
                )
            ],
        }


        debug_payload["selected_plate"] = (
            selected_text
        )

        debug_payload["selected_confidence"] = (
            selected_confidence
        )

        debug_payload["raw_variants"] = (
            variants
        )


        return debug_payload