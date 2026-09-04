import os
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class JPEGAnalysisDetector(BaseForensicDetector):
    """
    JPEG Quantization & Quality Matrix Detector.
    Analyzes JPEG quantization tables for double compression or re-saving artifacts.
    """

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if not os.path.exists(img_path) or not img_path.lower().endswith(('.jpg', '.jpeg')):
            return []

        try:
            with Image.open(img_path) as img:
                qtables = getattr(img, 'quantization', None)
                if not qtables:
                    return []

                # Analyze luminance quantization matrix variance
                q_matrix = qtables.get(0, [])
                if not q_matrix:
                    return []

                avg_q = sum(q_matrix) / len(q_matrix) if len(q_matrix) > 0 else 0
                is_low_quality = avg_q > 35.0  # High quantization values mean heavy compression

                status = ForensicStatus.SUSPICIOUS if is_low_quality else ForensicStatus.PASS
                level = EvidenceLevel.WEAK if is_low_quality else EvidenceLevel.WEAK
                reason = f"High JPEG quantization loss (avg q-step {avg_q:.1f})" if is_low_quality else "Standard JPEG quantization table"

                return [
                    ForensicEvidence(
                        type="jpeg",
                        status=status,
                        score=0.40 if is_low_quality else 0.05,
                        level=level,
                        category=EvidenceCategory.IMAGE,
                        region=None,
                        reason=reason
                    )
                ]
        except Exception:
            return []
