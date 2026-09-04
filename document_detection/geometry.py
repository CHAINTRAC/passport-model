import cv2
import numpy as np
from typing import Dict, Any, Tuple


class GeometryAnalyzer:
    """
    Analyzes document aspect ratio, ISO/IEC 7810 ID-1 card dimensions (85.60mm x 53.98mm = ~1.586),
    and portrait vs. landscape layout structure.
    """

    @staticmethod
    def analyze(img: np.ndarray) -> Dict[str, Any]:
        if img is None or img.size == 0:
            return {
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "is_id1_geometry": False,
                "orientation": "unknown"
            }

        h, w, _ = img.shape
        aspect_ratio = round(float(w) / float(h), 2)
        inv_aspect_ratio = round(float(h) / float(w), 2)

        is_id1 = (1.35 <= aspect_ratio <= 1.75) or (1.35 <= inv_aspect_ratio <= 1.75)
        orientation = "landscape" if w >= h else "portrait"

        return {
            "width": w,
            "height": h,
            "aspect_ratio": aspect_ratio,
            "inv_aspect_ratio": inv_aspect_ratio,
            "is_id1_geometry": is_id1,
            "orientation": orientation
        }
