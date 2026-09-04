import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

from schemas.detection import DocumentTypeDetectionResult, DetectionSignal
from document_detection.ocr_features import OCRFeatureExtractor
from document_detection.geometry import GeometryAnalyzer
from document_detection.result import DetectionResultBuilder
from validation.mrz import MRZValidator
from validation.verhoeff import VerhoeffValidator
from validation.document_number import DocumentNumberValidator


class DocumentTypeDetector:
    """
    Production Document-Type Detector.
    Accumulates evidence scores across 7 hierarchical tiers:
    1. Explicit API request
    2. MRZ validation / structure
    3. Document number validation & checksums
    4. OCR semantic features & keywords
    5. QR code analysis
    6. Layout / card geometry
    7. Filename / path hints (lowest weight)
    """

    def detect(
        self,
        img_path: str,
        explicit_doc_type: str = "auto",
        doc_number: Optional[str] = None,
        mrz_line1: Optional[str] = None,
        mrz_line2: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> DocumentTypeDetectionResult:
        # Check explicit doc type first
        if explicit_doc_type.lower().strip() not in ("auto", ""):
            return DetectionResultBuilder.build({}, [], explicit_doc_type=explicit_doc_type)

        scores = {"passport": 0.0, "aadhaar": 0.0, "dl": 0.0}
        signals: List[DetectionSignal] = []

        # 1. Image load & visual feature extraction
        img = cv2.imread(img_path) if os.path.exists(img_path) else None
        ocr_feats = OCRFeatureExtractor.extract_features(img) if img is not None else {}
        geom_feats = GeometryAnalyzer.analyze(img) if img is not None else {}

        # 2. Tier 2: MRZ Line Signals
        if mrz_line1 and mrz_line2:
            mrz_res = MRZValidator.validate_mrz_lines(mrz_line1, mrz_line2)
            if mrz_res["valid_mrz"]:
                scores["passport"] += 0.55
                signals.append(DetectionSignal(name="VALID_MRZ_CHECKSUM", target_doc_type="passport", weight=0.55, details="Valid ICAO Doc 9303 MRZ lines provided"))
            else:
                scores["passport"] += 0.30
                signals.append(DetectionSignal(name="MRZ_LINES_PROVIDED", target_doc_type="passport", weight=0.30, details="MRZ lines provided"))
        elif ocr_feats.get("has_mrz", False):
            scores["passport"] += 0.45
            signals.append(DetectionSignal(name="MRZ_ZONE_DETECTED", target_doc_type="passport", weight=0.45, details="MRZ bottom character line structure detected"))

        # 3. Tier 3: Document Number Validation Signals
        dl_cand = doc_number or DocumentNumberValidator.extract_dl_number(ocr_feats.get("extracted_text", ""))
        aadh_cand = doc_number or DocumentNumberValidator.extract_aadhaar_number(ocr_feats.get("extracted_text", ""))
        pass_cand = doc_number or DocumentNumberValidator.extract_passport_number(ocr_feats.get("extracted_text", ""))

        if pass_cand:
            p_res = DocumentNumberValidator.validate_passport_number(pass_cand)
            if p_res["valid_format"]:
                scores["passport"] += 0.40
                signals.append(DetectionSignal(name="PASSPORT_NUM_PATTERN", target_doc_type="passport", weight=0.40, details=f"Valid Passport format '{p_res['cleaned_number']}'"))

        if aadh_cand:
            a_res = DocumentNumberValidator.validate_aadhaar_number(aadh_cand)
            if a_res["valid_format"]:
                scores["aadhaar"] += 0.50
                signals.append(DetectionSignal(name="VERHOEFF_AADHAAR_VALID", target_doc_type="aadhaar", weight=0.50, details=f"Valid 12-digit Aadhaar & Verhoeff check ('{a_res['cleaned_number']}')"))

        if dl_cand:
            dl_res = DocumentNumberValidator.validate_dl_number(dl_cand)
            if dl_res["valid_format"]:
                scores["dl"] += 0.45
                signals.append(DetectionSignal(name="DL_NUM_PATTERN", target_doc_type="dl", weight=0.45, details=f"Valid DL format '{dl_res['cleaned_number']}'"))

        # 4. Tier 4: OCR Semantic Features & Keywords
        p_kw = ocr_feats.get("passport_keyword_matches", 0)
        dl_kw = ocr_feats.get("dl_keyword_matches", 0)
        a_kw = ocr_feats.get("aadhaar_keyword_matches", 0)

        if p_kw > 0:
            w = min(0.60, 0.45 + 0.10 * (p_kw - 1))
            scores["passport"] += w
            signals.append(DetectionSignal(name="PASSPORT_OCR_KEYWORDS", target_doc_type="passport", weight=round(w, 2), details=f"{p_kw} Passport text keywords matched"))

        if dl_kw > 0:
            w = min(0.60, 0.45 + 0.10 * (dl_kw - 1))
            scores["dl"] += w
            signals.append(DetectionSignal(name="DL_OCR_KEYWORDS", target_doc_type="dl", weight=round(w, 2), details=f"{dl_kw} Driving Licence text keywords matched"))

        if a_kw > 0:
            w = min(0.60, 0.45 + 0.10 * (a_kw - 1))
            scores["aadhaar"] += w
            signals.append(DetectionSignal(name="AADHAAR_OCR_KEYWORDS", target_doc_type="aadhaar", weight=round(w, 2), details=f"{a_kw} Aadhaar text keywords matched"))

        # 5. Tier 5: QR Code Analysis
        if ocr_feats.get("has_qr", False):
            scores["aadhaar"] += 0.45
            signals.append(DetectionSignal(name="UIDAI_QR_DETECTED", target_doc_type="aadhaar", weight=0.45, details="QR code detected on document"))

        # 6. Tier 6: Card Geometry (Supporting signal ONLY)
        if geom_feats.get("is_id1_geometry", False):
            if not ocr_feats.get("has_mrz", False):
                scores["dl"] += 0.15
                scores["aadhaar"] += 0.10
                signals.append(DetectionSignal(name="ID1_CARD_GEOMETRY", target_doc_type="dl", weight=0.15, details=f"ISO/IEC 7810 ID-1 card aspect ratio ({geom_feats['aspect_ratio']})"))

        # 7. Tier 7: Filename & Path Hints (Lowest weight tier)
        name_to_check = (original_filename or "") + " " + (img_path or "")
        name_lower = name_to_check.lower()

        if any(k in name_lower for k in ["passport", "mrz"]):
            scores["passport"] += 0.10
            signals.append(DetectionSignal(name="FILENAME_PASSPORT_HINT", target_doc_type="passport", weight=0.10, details="Filename contains 'passport'"))
        elif any(k in name_lower for k in ["aadhar", "aadhaar", "uidai"]):
            scores["aadhaar"] += 0.10
            signals.append(DetectionSignal(name="FILENAME_AADHAAR_HINT", target_doc_type="aadhaar", weight=0.10, details="Filename contains 'aadhaar'"))
        elif any(k in name_lower for k in ["dl", "driver", "licence", "license"]):
            scores["dl"] += 0.10
            signals.append(DetectionSignal(name="FILENAME_DL_HINT", target_doc_type="dl", weight=0.10, details="Filename contains 'dl'"))

        return DetectionResultBuilder.build(scores, signals, explicit_doc_type=explicit_doc_type)
