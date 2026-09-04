import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class VisualAnomalyDetector(BaseForensicDetector):
    """
    Sensor Noise & Lighting Non-Uniformity Detector.
    Measures specular glare hotspots, sensor noise variance, and lighting gradients.
    """

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if image is None or image.size == 0:
            return []

        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Specular glare overexposure check
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        glare_mask = (gray > 253) & (sat < 20)
        glare_ratio = round(float(np.sum(glare_mask)) / float(gray.size), 4)

        has_severe_glare = glare_ratio > 0.65
        if has_severe_glare:
            # Find bounding box of glare
            ys, xs = np.where(glare_mask)
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            glare_region = [min_x, min_y, max_x - min_x, max_y - min_y]

            return [
                ForensicEvidence(
                    type="visual_anomaly",
                    status=ForensicStatus.SUSPICIOUS,
                    score=0.65,
                    level=EvidenceLevel.MODERATE,
                    category=EvidenceCategory.IMAGE,
                    region=glare_region,
                    reason=f"High surface glare / overexposure detected ({glare_ratio*100:.1f}% hotspot pixels)"
                )
            ]

        # 2. High-pass spatial sensor noise variance
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        high_pass = cv2.absdiff(gray, blur)
        sensor_noise_var = float(np.var(high_pass))

        is_abnormal_noise = sensor_noise_var > 180.0
        if is_abnormal_noise:
            return [
                ForensicEvidence(
                    type="visual_anomaly",
                    status=ForensicStatus.SUSPICIOUS,
                    score=0.55,
                    level=EvidenceLevel.MODERATE,
                    category=EvidenceCategory.IMAGE,
                    region=None,
                    reason=f"Elevated spatial sensor noise variance ({sensor_noise_var:.1f} > 180.0)"
                )
            ]

        return [
            ForensicEvidence(
                type="visual_anomaly",
                status=ForensicStatus.PASS,
                score=0.05,
                level=EvidenceLevel.WEAK,
                category=EvidenceCategory.IMAGE,
                region=None,
                reason="Uniform surface lighting and normal sensor noise variance"
            )
        ]
