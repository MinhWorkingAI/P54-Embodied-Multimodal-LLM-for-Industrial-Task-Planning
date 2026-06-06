"""
simulation_backend/vision/detection_implementation/colour_detector.py
----------------------------------------------------------------------
HSV colour threshold detector for the simulation vision pipeline.

Config is read from environment variables (set in .env):
    COLOUR_MIN_AREA      : min contour area in pixels. Default: 30
    COLOUR_EXPECTED_AREA : expected pixel area of a full block face. Default: 100
    COLOUR_DEPTH_RADIUS  : depth sampling patch half-size. Default: 3

No model weights, no downloads, no GPU required.
Runs in under 5ms per frame on CPU using only OpenCV.

HSV ranges are calibrated for PyBullet flat-shaded rendering.
The exact HSV values are computed from scene_config.yaml RGBA colours:

    Colour      RGB (255 scale)       HSV (OpenCV H: 0-179)
    ----------  -------------------   ----------------------
    red         (255, 0,   0  )       H=0,   S=255, V=255
    blue        (0,   0,   255)       H=120, S=255, V=255
    green       (0,   204, 0  )       H=60,  S=255, V=204
    yellow      (255, 255, 0  )       H=30,  S=255, V=255
    grey        (140, 140, 140)       H=0,   S=0,   V=140
    dark grey   (89,  89,  89 )       H=0,   S=0,   V=89

Usage:
    Set VISION_DETECTOR=colour in .env.
"""

import logging
import os

import cv2
import numpy as np

from simulation_backend.simulation_environment.scene_builder import Detection
from simulation_backend.vision.camera import CameraFrame
from simulation_backend.vision.detection_base import DetectorBase
from simulation_backend.simulation_environment.object_registry import ObjectRegistry

logger = logging.getLogger(__name__)


# ── HSV colour ranges ──────────────────────────────────────────────────────────
#
# Calibrated for PyBullet flat-shaded rendering (scene_config.yaml RGBA values).
# Each entry: list of (lower_bound, upper_bound) for cv2.inRange().
# Multiple ranges are OR-combined. Red needs two because H wraps at 0/180.
#
# Tolerance applied:
#   H ± 10   (hue band)
#   S >= 225 (highly saturated — PyBullet renders pure solid colours)
#   V >= 170 (bright — flat shading, no shadows on blocks)
#   Grey / dark grey matched on V range only (S ≈ 0)

HSV_RANGES: dict[str, list[tuple]] = {
    "red": [
        # H=0 lower wrap
        (np.array([0,   225, 225]), np.array([8,   255, 255])),
        # H=180 upper wrap
        (np.array([172, 225, 225]), np.array([179, 255, 255])),
    ],
    "blue": [
        # H=120 centre ± 10
        (np.array([110, 225, 225]), np.array([130, 255, 255])),
    ],
    "green": [
        # H=60 centre ± 10, V from 174 (PyBullet green V=204)
        (np.array([50,  225, 174]), np.array([70,  255, 255])),
    ],
    "yellow": [
        # H=30 centre ± 10
        (np.array([20,  225, 225]), np.array([40,  255, 255])),
    ],
    "grey": [
        # S ≈ 0, V = 140 ± 40
        (np.array([0,   0,   100]), np.array([179, 30,  180])),
    ],
    "dark grey": [
        # S ≈ 0, V = 89 ± 40
        (np.array([0,   0,   49]),  np.array([179, 30,  129])),
    ],
}


class ColourDetector(DetectorBase):
    """
    HSV colour threshold detector.

    All config is read from environment variables at init time.
    No scene_config.yaml keys are used.

    Expected pixel areas are calibrated for the default camera position
    in scene_config.yaml (pos=[1.5, 0.0, 1.8], fov=60, 640x480).
    The blocks appear as ~100px² faces at that distance.
    Trays and the workstation are larger — confidence caps at 1.0.
    """

    name = "colour"

    def __init__(self, registry: ObjectRegistry, config: dict = None) -> None:
        super().__init__(registry, config)
        self._min_area      = int(os.getenv("COLOUR_MIN_AREA",      "30"))
        self._expected_area = int(os.getenv("COLOUR_EXPECTED_AREA", "100"))
        self._depth_radius  = int(os.getenv("COLOUR_DEPTH_RADIUS",  "3"))

        logger.info(
            f"[colour] min_area={self._min_area}  "
            f"expected_area={self._expected_area}  "
            f"depth_radius={self._depth_radius}"
        )

    # ── warmup ────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """No weights to load. Marks the detector ready immediately."""
        logger.info("[colour] Ready — no weights required.")
        self._warmed_up = True

    # ── detect ────────────────────────────────────────────────────────────────

    def detect(self, frame: CameraFrame) -> list[Detection]:
        """
        Run HSV colour detection on one camera frame.

        For each registered object whose colour is in HSV_RANGES:
            1. Build a binary mask for that colour in the HSV image.
            2. Find contours and take the largest one.
            3. Skip if area < COLOUR_MIN_AREA.
            4. Compute centroid pixel (cx, cy).
            5. Sample depth at centroid -> Z metres.
            6. Use registry (x, y) + depth Z as 3D position.
            7. confidence = min(area / COLOUR_EXPECTED_AREA, 1.0).

        Returns:
            list[Detection] — one per colour-matched object found.
        """
        self._ensure_warmed_up()

        hsv        = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2HSV)
        detections = []

        for entry in self._registry.all_entries():
            colour_name = entry.attributes.get("color", "").lower().strip()

            if colour_name not in HSV_RANGES:
                logger.debug(
                    f"[colour] '{entry.label}' colour='{colour_name}' "
                    f"not in HSV_RANGES — ground truth will cover it."
                )
                continue

            mask = self._colour_mask(hsv, colour_name)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                logger.debug(f"[colour] No contours for '{entry.label}'.")
                continue

            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)

            if area < self._min_area:
                logger.debug(
                    f"[colour] '{entry.label}' area {area:.0f} < "
                    f"{self._min_area} — too small, skipping."
                )
                continue

            M = cv2.moments(best)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            bx, by, bw, bh = cv2.boundingRect(best)
            bbox = {"x_min": bx, "y_min": by, "x_max": bx + bw, "y_max": by + bh}

            z          = self._depth_at_bbox_centre(frame, bbox, self._depth_radius)
            x          = float(entry.position[0])
            y          = float(entry.position[1])
            confidence = round(min(area / self._expected_area, 1.0), 3)

            detections.append(Detection(
                body_id=entry.body_id,
                label=entry.label,
                position_3d=(round(x, 4), round(y, 4), round(float(z), 4)),
                bounding_box_2d=bbox,
                confidence=confidence,
                source=self.name,
            ))

            logger.debug(
                f"[colour] '{entry.label}'  colour='{colour_name}'  "
                f"area={area:.0f}  conf={confidence:.2f}  "
                f"centroid=({cx},{cy})  z={z:.3f}m"
            )

        logger.info(
            f"[colour] {len(detections)} detected: "
            f"{[d.label for d in detections]}"
        )
        return detections

    # ── draw_detections ───────────────────────────────────────────────────────

    def draw_detections(
        self,
        frame:      np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """
        Cyan bounding boxes with [CLR] prefix.
        Distinct from yellow YOLO boxes and green ground truth.
        """
        canvas = frame.copy()
        for det in detections:
            bb = det.bounding_box_2d
            if bb is None:
                continue
            x1, y1 = int(bb["x_min"]), int(bb["y_min"])
            x2, y2 = int(bb["x_max"]), int(bb["y_max"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
            text = f"[CLR] {det.label}  {det.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                canvas, (x1, y1 - th - bl - 4), (x1 + tw + 4, y1),
                (255, 255, 0), -1,
            )
            cv2.putText(
                canvas, text, (x1 + 2, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return canvas

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _colour_mask(hsv: np.ndarray, colour_name: str) -> np.ndarray:
        """
        Build a binary mask for the given colour name by OR-combining
        all HSV ranges registered for that colour.

        Red needs two ranges because H wraps at 0/180 in OpenCV HSV.
        A morphological open removes isolated noise specks.
        """
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in HSV_RANGES[colour_name]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
