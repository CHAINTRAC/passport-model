from typing import Dict, List
from schemas.detection import DocumentTypeDetectionResult, DetectionSignal


class DetectionResultBuilder:
    """
    Evaluates accumulated evidence scores and builds DocumentTypeDetectionResult.
    Threshold logic:
    If highest score >= 0.45 -> doc_type = top candidate
    If highest score < 0.45 -> doc_type = "unknown"
    Confidence is normalized relative to 1.0 max score cap.
    """

    CONFIDENCE_THRESHOLD = 0.45

    @classmethod
    def build(
        cls,
        scores: Dict[str, float],
        signals: List[DetectionSignal],
        explicit_doc_type: str = "auto"
    ) -> DocumentTypeDetectionResult:
        # If explicit doc_type was requested (not auto)
        norm_explicit = explicit_doc_type.lower().strip()
        if norm_explicit in ("passport", "aadhaar", "aadhar", "dl", "driving_licence", "driving_license"):
            mapped = "aadhaar" if norm_explicit in ("aadhaar", "aadhar") else ("dl" if norm_explicit in ("dl", "driving_licence", "driving_license") else "passport")
            return DocumentTypeDetectionResult(
                doc_type=mapped,
                confidence=1.0,
                signals=[DetectionSignal(name="EXPLICIT_USER_DOC_TYPE", target_doc_type=mapped, weight=1.0, details=f"User specified '{explicit_doc_type}'")],
                scores=scores,
                is_confident=True
            )

        # Find top candidate
        top_doc_type = "unknown"
        top_score = 0.0

        for dtype, score in scores.items():
            if score > top_score:
                top_score = score
                top_doc_type = dtype

        is_confident = top_score >= cls.CONFIDENCE_THRESHOLD
        final_doc_type = top_doc_type if is_confident else "unknown"
        confidence = round(min(top_score, 1.0), 2)

        return DocumentTypeDetectionResult(
            doc_type=final_doc_type,
            confidence=confidence,
            signals=signals,
            scores={k: round(v, 2) for k, v in scores.items()},
            is_confident=is_confident
        )
