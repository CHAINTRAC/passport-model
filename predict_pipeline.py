import os
from typing import Dict, Any, Optional
from aggregation.evidence_aggregator import MasterEvidenceAggregator


class DocumentAuthenticityPipeline:
    """
    Unified Facade & Production Pipeline Wrapper for Document Verification & Fraud Detection.
    Delegates to the modular evidence aggregator engine while maintaining full backwards compatibility.
    """

    def __init__(self, model_path="model/fine_tuned_model_20.keras"):
        self.aggregator = MasterEvidenceAggregator()
        self.model = getattr(self.aggregator, "model", None)

    def assess_image_quality(self, img_path: str) -> Dict[str, Any]:
        """Pre-Flight Quality Assessor wrapper for legacy callers."""
        import cv2, numpy as np
        if not os.path.exists(img_path):
            return {'is_sufficient': False, 'reasons': ['File does not exist']}

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {'is_sufficient': False, 'reasons': ['Unreadable image file']}

        h, w, c = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < 25.0
        is_low_res = (w < 320 or h < 220)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        glare_ratio = float(np.sum((gray > 253) & (sat < 20))) / float(gray.size)
        has_severe_glare = glare_ratio > 0.65

        reasons = []
        if is_blurry:
            reasons.append(f"Severe blur detected (Blur Score: {blur_score:.1f} < 25.0)")
        if is_low_res:
            reasons.append(f"Low resolution ({w}x{h} < 320x220 minimum threshold)")
        if has_severe_glare:
            reasons.append(f"High surface glare / overexposure ({glare_ratio*100:.1f}%)")

        return {
            'is_sufficient': not (is_blurry or is_low_res or has_severe_glare),
            'blur_score': round(blur_score, 2),
            'resolution': f"{w}x{h}",
            'glare_ratio': round(glare_ratio, 3),
            'reasons': reasons
        }

    def predict_document_authenticity(
        self,
        img_path: str,
        doc_type: str = "auto",
        doc_number: Optional[str] = None,
        mrz_line1: Optional[str] = None,
        mrz_line2: Optional[str] = None,
        visual_fields: Optional[Dict[str, Any]] = None,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Master Pipeline Method.
        Delegates to MasterEvidenceAggregator and returns serialized dictionary.
        """
        response = self.aggregator.process(
            img_path=img_path,
            explicit_doc_type=doc_type,
            doc_number=doc_number,
            mrz_line1=mrz_line1,
            mrz_line2=mrz_line2,
            original_filename=original_filename
        )
        return response.model_dump()
