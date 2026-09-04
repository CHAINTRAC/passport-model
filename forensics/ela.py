import os
import cv2
import numpy as np
from PIL import Image, ImageChops
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class ELADetector(BaseForensicDetector):
    """
    Error Level Analysis (ELA) Forensic Detector.
    Detects double-JPEG compression variance, digital text insertions, and template edits.
    Extracts bounding box region [x, y, w, h] for localized compression anomalies.
    """

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if not os.path.exists(img_path):
            return []

        temp_path = None
        try:
            original = Image.open(img_path).convert('RGB')
            temp_path = f"_temp_ela_{os.getpid()}.jpg"
            original.save(temp_path, 'JPEG', quality=90)
            recompressed = Image.open(temp_path)

            diff = ImageChops.difference(original, recompressed)
            diff_np = np.array(diff)
            ela_variance = float(np.var(diff_np))

            recompressed.close()
            original.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Localized anomaly contour region extraction
            gray_diff = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_region = None
            max_area = 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area > 400 and area > max_area:
                    max_area = area
                    max_region = [int(x), int(y), int(w), int(h)]

            is_suspicious = (ela_variance >= 350.0)
            score = round(min(ela_variance / 1000.0, 1.0), 2) if is_suspicious else 0.05

            status = ForensicStatus.SUSPICIOUS if is_suspicious else ForensicStatus.PASS
            level = EvidenceLevel.STRONG if (ela_variance >= 600.0) else (EvidenceLevel.MODERATE if is_suspicious else EvidenceLevel.WEAK)
            reason = f"Elevated ELA compression variance ({ela_variance:.1f} >= 350.0 - possible digital editing)" if is_suspicious else "Uniform JPEG compression profile detected"

            return [
                ForensicEvidence(
                    type="ela",
                    status=status,
                    score=score,
                    level=level,
                    category=EvidenceCategory.IMAGE,
                    region=max_region if is_suspicious else None,
                    reason=reason
                )
            ]
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return []
