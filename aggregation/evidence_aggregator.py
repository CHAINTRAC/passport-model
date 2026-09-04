import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from schemas.detection import DocumentTypeDetectionResult
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceLevel
from schemas.response import (
    VerificationResponse,
    DocumentInfo,
    DecisionInfo,
    DecisionStatus,
    ValidationItem,
    EvidenceSummary
)
from document_detection.detector import DocumentTypeDetector
from document_detection.ocr_features import OCRFeatureExtractor
from verifiers.passport import PassportVerifier
from verifiers.driving_license import DrivingLicenseVerifier
from verifiers.aadhaar import AadhaarVerifier
from forensics.ela import ELADetector
from forensics.cnn import CNNClassifierDetector
from forensics.jpeg import JPEGAnalysisDetector
from forensics.metadata import MetadataAnalysisDetector
from forensics.copy_move import CopyMoveDetector
from forensics.text_tampering import TextTamperingDetector
from forensics.geometry_analysis import GeometryAnalysisDetector
from forensics.visual_anomaly import VisualAnomalyDetector
from aggregation.risk_engine import RiskEngine


class MasterEvidenceAggregator:
    """
    Production Master Evidence Aggregator & Pipeline Orchestrator.
    Executes DocumentTypeDetector, executes document-specific verifier (or generic verifier if unknown),
    runs all modular forensic detectors, aggregates evidence into risk categories,
    and returns standardized VerificationResponse.
    """

    def __init__(self):
        self.doc_detector = DocumentTypeDetector()
        self.passport_verifier = PassportVerifier()
        self.dl_verifier = DrivingLicenseVerifier()
        self.aadhaar_verifier = AadhaarVerifier()
        self.cnn_detector = CNNClassifierDetector()

        # Modular Forensic Detectors
        self.forensic_detectors = [
            ELADetector(),
            self.cnn_detector,
            JPEGAnalysisDetector(),
            MetadataAnalysisDetector(),
            CopyMoveDetector(),
            TextTamperingDetector(),
            GeometryAnalysisDetector(),
            VisualAnomalyDetector()
        ]
        self.model = getattr(self.cnn_detector, "model", None)

    def process(
        self,
        img_path: str,
        explicit_doc_type: str = "auto",
        doc_number: Optional[str] = None,
        mrz_line1: Optional[str] = None,
        mrz_line2: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> VerificationResponse:
        filename = original_filename or os.path.basename(img_path)

        if not os.path.exists(img_path):
            return VerificationResponse(
                success=False,
                filename=filename,
                document=DocumentInfo(type="unknown", confidence=0.0),
                decision=DecisionInfo(status=DecisionStatus.INVALID_INPUT, risk_score=1.0, confidence=0.0),
                reasons=["Image file does not exist"]
            )

        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            return VerificationResponse(
                success=False,
                filename=filename,
                document=DocumentInfo(type="unknown", confidence=0.0),
                decision=DecisionInfo(status=DecisionStatus.INVALID_INPUT, risk_score=1.0, confidence=0.0),
                reasons=["Unreadable image file"]
            )

        # 1. Document Detection & Classification
        detection_res: DocumentTypeDetectionResult = self.doc_detector.detect(
            img_path=img_path,
            explicit_doc_type=explicit_doc_type,
            doc_number=doc_number,
            mrz_line1=mrz_line1,
            mrz_line2=mrz_line2,
            original_filename=filename
        )

        detected_type = detection_res.doc_type
        doc_confidence = detection_res.confidence

        # OCR Feature Extraction
        ocr_feats = OCRFeatureExtractor.extract_features(img)
        metadata_payload = {
            "doc_number": doc_number,
            "mrz_line1": mrz_line1,
            "mrz_line2": mrz_line2,
            "original_filename": filename
        }

        # 2. Document-Specific Verifier Execution
        validations: Dict[str, ValidationItem] = {}
        if detected_type == "passport":
            validations = self.passport_verifier.verify(img, img_path, ocr_feats, metadata_payload)
        elif detected_type == "dl":
            validations = self.dl_verifier.verify(img, img_path, ocr_feats, metadata_payload)
        elif detected_type == "aadhaar":
            validations = self.aadhaar_verifier.verify(img, img_path, ocr_feats, metadata_payload)
        else:
            # Document type unknown: do NOT force a verifier; run generic checks
            validations["document_type"] = ValidationItem(
                status="unverified",
                confidence=doc_confidence,
                message="Document type could not be confidently identified; generic forensics executed"
            )

        # 3. Independent Modular Forensic Analysis
        forensics: List[ForensicEvidence] = []
        for detector in self.forensic_detectors:
            try:
                evidence_list = detector.analyze(img, img_path, metadata_payload)
                forensics.extend(evidence_list)
            except Exception:
                pass

        # 4. Risk Engine Aggregation & Decision Determination
        risk_score, decision_status, decision_confidence = RiskEngine.calculate_risk(
            validations=validations,
            forensics=forensics,
            doc_type_confidence=doc_confidence
        )

        # 5. Build Evidence Summary Counts
        strong_count = sum(1 for f in forensics if f.status == ForensicStatus.SUSPICIOUS and f.level == EvidenceLevel.STRONG)
        moderate_count = sum(1 for f in forensics if f.status == ForensicStatus.SUSPICIOUS and f.level == EvidenceLevel.MODERATE)
        weak_count = sum(1 for f in forensics if f.status == ForensicStatus.SUSPICIOUS and f.level == EvidenceLevel.WEAK)
        summary = EvidenceSummary(strong=strong_count, moderate=moderate_count, weak=weak_count)

        # 6. Legacy Compatibility Mapping
        legacy_reasons = []
        for k, v in validations.items():
            if v.status == "fail":
                legacy_reasons.append(v.message or f"Validation check '{k}' failed")
        for f in forensics:
            if f.status == ForensicStatus.SUSPICIOUS:
                legacy_reasons.append(f.reason)

        legacy_verdict = "GENUINE" if decision_status == DecisionStatus.GENUINE else ("SUSPICIOUS" if decision_status == DecisionStatus.SUSPICIOUS else "INCONCLUSIVE")

        legacy_evidence_table = {
            "quality_assessment": {
                "is_sufficient": True,
                "resolution": f"{img.shape[1]}x{img.shape[0]}"
            },
            "detection_scores": detection_res.scores,
            "detection_signals": [s.model_dump() for s in detection_res.signals]
        }

        for f in forensics:
            legacy_evidence_table[f.type] = {
                "score": f.score,
                "status": f.status,
                "reason": f.reason,
                "region": f.region
            }

        return VerificationResponse(
            success=True,
            filename=filename,
            document=DocumentInfo(type=detected_type, confidence=doc_confidence),
            decision=DecisionInfo(status=decision_status, risk_score=risk_score, confidence=decision_confidence),
            validation=validations,
            forensics=forensics,
            evidence_summary=summary,
            doc_type=detected_type,
            verdict=legacy_verdict,
            risk_score=risk_score,
            reasons=legacy_reasons,
            evidence_table=legacy_evidence_table
        )
