"""OpenCV blue-insole-bottom detector for the shoe state machine.

Detection point 2: detect when the blue underside of the insole appears in the
head camera. Traditional CV only — HSV threshold -> morphology -> contour area.

IMPORTANT: the head image forwarded from the robot laptop is **RGB** HWC uint8
(RealSense default is RGB, and the LeRobot replay yields RGB too), so we convert
with cv2.COLOR_RGB2HSV — NOT COLOR_BGR2HSV. Getting this wrong silently shifts
the hue band and the blue mask fails.

cv2 is imported lazily (in BlueInsoleDetector.__init__) so this module imports
fine on machines without OpenCV; only constructing the detector requires cv2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlueInsoleConfig:
    # OpenCV HSV space: H in 0..179, S/V in 0..255. Blue hue ≈ 100..130.
    # PLACEHOLDER — calibrate on real head frames under production lighting.
    h_lo: int = 100
    h_hi: int = 130
    s_lo: int = 80
    s_hi: int = 255
    v_lo: int = 40
    v_hi: int = 255
    # Min blue blob area as a fraction of the ROI area to count as "present".
    min_area_frac: float = 0.02
    # Region of interest as (x0, y0, x1, y1) fractions; None = whole frame.
    # Default = bottom 60% (the insole bottom is expected low in the head view).
    roi: tuple | None = (0.0, 0.4, 1.0, 1.0)
    open_ksize: int = 5   # morphology open kernel (despeckle); <=1 disables
    close_ksize: int = 7  # morphology close kernel (fill); <=1 disables


class BlueInsoleDetector:
    """Returns (present, debug) for an HWC uint8 RGB head image."""

    def __init__(self, cfg: BlueInsoleConfig | None = None) -> None:
        self.cfg = cfg or BlueInsoleConfig()
        import cv2  # lazy; raises a clear ImportError if OpenCV is missing

        self._cv2 = cv2

    def detect(self, head_rgb: np.ndarray) -> tuple[bool, dict]:
        cv2 = self._cv2
        cfg = self.cfg
        h, w = head_rgb.shape[:2]

        if cfg.roi is not None:
            x0, y0, x1, y1 = cfg.roi
            xs, ys, xe, ye = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
            roi = head_rgb[ys:ye, xs:xe]
        else:
            roi = head_rgb
        if roi.size == 0:
            return False, {"area_frac": 0.0, "n_contours": 0, "reason": "empty_roi"}

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)  # input is RGB
        mask = cv2.inRange(
            hsv,
            (int(cfg.h_lo), int(cfg.s_lo), int(cfg.v_lo)),
            (int(cfg.h_hi), int(cfg.s_hi), int(cfg.v_hi)),
        )
        if cfg.open_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if cfg.close_ksize > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_ksize, cfg.close_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = roi.shape[0] * roi.shape[1]
        max_area = max((cv2.contourArea(c) for c in cnts), default=0.0)
        area_frac = max_area / max(1, roi_area)
        present = area_frac >= cfg.min_area_frac
        return present, {"area_frac": float(area_frac), "n_contours": len(cnts)}
