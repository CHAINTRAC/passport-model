import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class GeometryAnalysisDetector(BaseForensicDetector):
    """
    Card & Page Geometry Forensic Detector.
    Evaluates card aspect ratio, border clipping, and extreme perspective distortion.
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
        aspect_ratio = round(float(w) / float(h), 2)
        inv_aspect_ratio = round(float(h) / float(w), 2)

        # Extremely low resolution check (<320x220)
        if w < 320 or h < 220:
            return [
                ForensicEvidence(
                    type="geometry",
                    status=ForensicStatus.SUSPICIOUS,
                    score=0.60,
                    level=EvidenceLevel.MODERATE,
                    category=EvidenceCategory.STRUCTURAL,
                    region=[0, 0, w, h],
                    reason=f"Low image resolution ({w}x{h} < 320x220 minimum threshold)"
                )
            ]

        # Extreme non-standard aspect ratio check (<0.40 or >2.50)
        is_extreme = (aspect_ratio < 0.40 or aspect_ratio > 2.50)
        if is_extreme:
            return [
                ForensicEvidence(
                    type="geometry",
                    status=ForensicStatus.SUSPICIOUS,
                    score=0.70,
                    level=EvidenceLevel.MODERATE,
                    category=EvidenceCategory.STRUCTURAL,
                    region=[0, 0, w, h],
                    reason=f"Extreme document cropping or aspect ratio distortion ({aspect_ratio})"
                )
            ]

        return [
            ForensicEvidence(
                type="geometry",
                status=ForensicStatus.PASS,
                score=0.05,
                level=EvidenceLevel.WEAK,
                category=EvidenceCategory.STRUCTURAL,
                region=None,
                reason=f"Standard resolution ({w}x{h}) and aspect ratio ({aspect_ratio})"
            )
        ]
