import numpy as np
from typing import Dict, Any, Optional
from verifiers.base import BaseDocumentVerifier
from schemas.response import ValidationItem
from validation.document_number import DocumentNumberValidator
from indian_dl_verifier import IndianDLVerifier


class DrivingLicenseVerifier(BaseDocumentVerifier):
    """
    Driving Licence Verifier:
    Validates DL number pattern, Sarathi state format, Parivahan API status, and ID-1 card geometry.
    """

    def __init__(self):
        self.dl_verifier = IndianDLVerifier()

    def verify(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        ocr_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, ValidationItem]:
        validations: Dict[str, ValidationItem] = {}

        # 1. Document Number & Parivahan Validation
        doc_num = metadata.get("doc_number")
        if not doc_num:
            ocr_text = ocr_result.get("extracted_text", "")
            doc_num = DocumentNumberValidator.extract_dl_number(ocr_text)

        if doc_num:
            pat_res = DocumentNumberValidator.validate_dl_number(doc_num)
            parivahan_res = self.dl_verifier.verify(doc_num, perform_api_check=True)

            is_valid = parivahan_res.get("overall_valid", pat_res["valid_format"])
            validations["document_number"] = ValidationItem(
                status="pass" if is_valid else "fail",
                confidence=0.95 if is_valid else 0.30,
                message=parivahan_res.get("format_validation", {}).get("message", pat_res["message"])
            )
        else:
            validations["document_number"] = ValidationItem(status="unverified", message="DL number not provided or found in document text")

        # 2. Layout & Card Geometry Validation
        if image is not None:
            h, w, _ = image.shape
            aspect_ratio = round(float(w) / float(h), 2)
            inv_aspect_ratio = round(float(h) / float(w), 2)
            is_id1 = (1.35 <= aspect_ratio <= 1.75) or (1.35 <= inv_aspect_ratio <= 1.75)

            validations["layout"] = ValidationItem(
                status="pass" if is_id1 else "fail",
                confidence=0.90 if is_id1 else 0.40,
                message=f"ISO/IEC 7810 ID-1 standard card ratio ({aspect_ratio})" if is_id1 else f"Non-standard card aspect ratio ({aspect_ratio})"
            )
        else:
            validations["layout"] = ValidationItem(status="unverified", message="Image not loaded")

        return validations
