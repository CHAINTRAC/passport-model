import cv2
import numpy as np
from typing import Dict, Any, Optional
from verifiers.base import BaseDocumentVerifier
from schemas.response import ValidationItem
from validation.mrz import MRZValidator
from validation.document_number import DocumentNumberValidator


class PassportVerifier(BaseDocumentVerifier):
    """
    Indian Passport Verifier:
    Validates ICAO Doc 9303 MRZ lines, passport number format, and page structural layout.
    """

    def verify(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        ocr_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, ValidationItem]:
        validations: Dict[str, ValidationItem] = {}

        # 1. Document Number Validation
        doc_num = metadata.get("doc_number")
        if not doc_num:
            ocr_text = ocr_result.get("extracted_text", "")
            doc_num = DocumentNumberValidator.extract_passport_number(ocr_text)

        if doc_num:
            res = DocumentNumberValidator.validate_passport_number(doc_num)
            validations["document_number"] = ValidationItem(
                status="pass" if res["valid_format"] else "fail",
                confidence=0.98 if res["valid_format"] else 0.20,
                message=res["message"]
            )
        else:
            validations["document_number"] = ValidationItem(status="unverified", message="Document number not provided or found in document text")

        # 2. MRZ Checkdigit Validation
        mrz1 = metadata.get("mrz_line1")
        mrz2 = metadata.get("mrz_line2")
        if mrz1 and mrz2:
            mrz_res = MRZValidator.validate_mrz_lines(mrz1, mrz2)
            validations["mrz_checksum"] = ValidationItem(
                status="pass" if mrz_res["valid_mrz"] else "fail",
                confidence=0.99 if mrz_res["valid_mrz"] else 0.10,
                message=mrz_res["message"]
            )
        else:
            validations["mrz_checksum"] = ValidationItem(
                status="pass" if ocr_result.get("has_mrz") else "unverified",
                confidence=0.75 if ocr_result.get("has_mrz") else 0.50,
                message="MRZ zone baseline detected" if ocr_result.get("has_mrz") else "MRZ lines not provided"
            )

        # 3. Layout & Geometry Validation
        if image is not None:
            h, w, _ = image.shape
            aspect_ratio = round(float(w) / float(h), 2)
            has_mrz_zone = ocr_result.get("has_mrz", False)

            # If aspect ratio is strictly ID-1 card (1.50 - 1.70) without MRZ lines -> NOT a passport page!
            is_id1_without_mrz = (1.50 <= aspect_ratio <= 1.70) and not has_mrz_zone

            if is_id1_without_mrz:
                validations["layout"] = ValidationItem(
                    status="fail",
                    confidence=0.85,
                    message=f"Non-passport layout: Card matches ID-1 aspect ratio ({aspect_ratio}) without MRZ zone"
                )
            else:
                validations["layout"] = ValidationItem(
                    status="pass" if (w >= 350 and h >= 250) else "fail",
                    confidence=0.90,
                    message="Valid passport identity page dimensions & resolution" if (w >= 350 and h >= 250) else "Low resolution passport page"
                )
        else:
            validations["layout"] = ValidationItem(status="unverified", message="Image not loaded")

        return validations
