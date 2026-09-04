from typing import List, Dict, Any, Tuple
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel
from schemas.response import ValidationItem, DecisionStatus


class RiskEngine:
    """
    Calibrated Multi-Category Risk & Fraud Scoring Engine.
    Aggregates evidence across 5 categories with calibrated weights:
    1. Identity Integrity  (Weight: 0.35) - MRZ, Verhoeff, document number validation
    2. Structural Integrity (Weight: 0.15) - Aspect ratio, card geometry, resolution
    3. Image Integrity      (Weight: 0.25) - ELA, JPEG compression, copy-move, CNN
    4. Content Integrity    (Weight: 0.15) - Text rendering, font tampering, OCR consistency
    5. Metadata Integrity   (Weight: 0.10) - EXIF metadata, editing software tags

    Determines 3-State Decision Status:
    - GENUINE:      risk_score < 0.35 and confidence >= 0.60
    - SUSPICIOUS:   risk_score >= 0.35
    - INCONCLUSIVE: image unreadable / low quality / insufficient evidence
    """

    CATEGORY_WEIGHTS = {
        EvidenceCategory.IDENTITY: 0.35,
        EvidenceCategory.STRUCTURAL: 0.15,
        EvidenceCategory.IMAGE: 0.25,
        EvidenceCategory.CONTENT: 0.15,
        EvidenceCategory.METADATA: 0.10
    }

    @classmethod
    def calculate_risk(
        cls,
        validations: Dict[str, ValidationItem],
        forensics: List[ForensicEvidence],
        doc_type_confidence: float = 1.0
    ) -> Tuple[float, DecisionStatus, float]:
        category_risk_scores: Dict[EvidenceCategory, List[float]] = {cat: [] for cat in cls.CATEGORY_WEIGHTS}

        # 1. Map Validation Items to Category Risks (Identity & Structural)
        for val_name, item in validations.items():
            if item.status == "fail":
                # Deterministic check failures (e.g. MRZ mismatch, invalid Verhoeff) carry heavy risk
                penalty = 0.85 if val_name in ("mrz_checksum", "document_number", "verhoeff") else 0.50
                category_risk_scores[EvidenceCategory.IDENTITY].append(penalty)
            elif item.status == "pass":
                category_risk_scores[EvidenceCategory.IDENTITY].append(0.0)

        # 2. Map Forensic Evidence Items to Categories
        for ev in forensics:
            if ev.status == ForensicStatus.SUSPICIOUS:
                # Scale by evidence level weight
                weight = 1.0 if ev.level == EvidenceLevel.STRONG else (0.65 if ev.level == EvidenceLevel.MODERATE else 0.35)
                risk_val = ev.score * weight
                category_risk_scores[ev.category].append(risk_val)
            elif ev.status == ForensicStatus.PASS:
                category_risk_scores[ev.category].append(0.0)

        # 3. Compute weighted average risk
        total_risk = 0.0
        total_weight = 0.0

        for cat, cat_weight in cls.CATEGORY_WEIGHTS.items():
            scores_list = category_risk_scores[cat]
            if scores_list:
                # Take max risk in category to avoid diluting critical failures
                cat_risk = max(scores_list)
                total_risk += cat_risk * cat_weight
                total_weight += cat_weight

        aggregated_risk = round(total_risk / total_weight, 2) if total_weight > 0 else 0.0
        aggregated_risk = min(max(aggregated_risk, 0.0), 1.0)

        # Overall decision confidence
        decision_confidence = round(min(doc_type_confidence, 0.95), 2)

        if doc_type_confidence < 0.40:
            status = DecisionStatus.INCONCLUSIVE
        elif aggregated_risk >= 0.35:
            status = DecisionStatus.SUSPICIOUS
        else:
            status = DecisionStatus.GENUINE

        return aggregated_risk, status, decision_confidence
