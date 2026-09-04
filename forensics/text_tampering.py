import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class TextTamperingDetector(BaseForensicDetector):
    """
    Localized Text & Font Rendering Tampering Detector.
    Analyzes local gradient and Laplacian noise variance across individual text bounding boxes.
    """

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if image is None or image.size == 0:
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Find candidate text regions
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
            morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            region_variances = []
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > 40 and ch > 10 and (cw * ch) > 400:
                    patch = gray[y:y+ch, x:x+cw]
                    lap_var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                    region_variances.append(([int(x), int(y), int(cw), int(ch)], lap_var))

            if len(region_variances) < 3:
                return []

            vars_only = [v[1] for v in region_variances]
            median_var = np.median(vars_only)
            std_var = np.std(vars_only)

            anomaly_region = None
            max_z_score = 0.0

            for reg, v in region_variances:
                if std_var > 0:
                    z = (v - median_var) / std_var
                    if z > 3.0 and z > max_z_score:  # 3 std dev outlier in text sharpness/rendering
                        max_z_score = z
                        anomaly_region = reg

            is_tampered = anomaly_region is not None
            if is_tampered:
                return [
                    ForensicEvidence(
                        type="text_tampering",
                        status=ForensicStatus.SUSPICIOUS,
                        score=0.85,
                        level=EvidenceLevel.STRONG,
                        category=EvidenceCategory.CONTENT,
                        region=anomaly_region,
                        reason="Text region exhibits inconsistent rendering & local noise characteristics (possible digital insertion)"
                    )
                ]
        except Exception:
            pass

        return []
