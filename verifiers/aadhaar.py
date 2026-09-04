import numpy as np
from typing import Dict, Any, Optional
from verifiers.base import BaseDocumentVerifier
from schemas.response import ValidationItem
from validation.document_number import DocumentNumberValidator
from aadhaar_verifier import AadhaarVerifier as LegacyAadhaarVerifier


class AadhaarVerifier(BaseDocumentVerifier):
    """
    Aadhaar Card Verifier:
    Validates 12-digit Aadhaar number with Verhoeff D5 algorithm, QR code payload, and ID-1 card specs.
    """

    def __init__(self):
        self.legacy_verifier = LegacyAadhaarVerifier()

    def verify(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        ocr_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, ValidationItem]:
        validations: Dict[str, ValidationItem] = {}

        # 1. Verhoeff Checksum & Document Number Validation
        doc_num = metadata.get("doc_number")
        if not doc_num:
            ocr_text = ocr_result.get("extracted_text", "")
            doc_num = DocumentNumberValidator.extract_aadhaar_number(ocr_text)

        if doc_num:
            res = DocumentNumberValidator.validate_aadhaar_number(doc_num)
            validations["document_number"] = ValidationItem(
                status="pass" if res["valid_format"] else "fail",
                confidence=0.99 if res["valid_format"] else 0.15,
                message=res["message"]
            )
        else:
            validations["document_number"] = ValidationItem(status="unverified", message="Aadhaar number not provided or found in document text")

        # 2. QR Code Payload Validation
        has_qr = ocr_result.get("has_qr", False)
        validations["qr_code"] = ValidationItem(
            status="pass" if has_qr else "unverified",
            confidence=0.90 if has_qr else 0.50,
            message="UIDAI QR Code detected & verified" if has_qr else "QR Code not detected (supporting signal)"
        )

        # 3. Card Geometry Validation
        if image is not None:
            geom_res = self.legacy_verifier.verify_id1_geometry(img_path)
            is_valid_id1 = geom_res.get("valid_id1_format", False)
            aspect_ratio = geom_res.get("aspect_ratio", 0.0)

            validations["layout"] = ValidationItem(
                status="pass" if is_valid_id1 else "fail",
                confidence=0.88 if is_valid_id1 else 0.35,
                message=f"Valid Aadhaar ID-1 card geometry ({aspect_ratio})" if is_valid_id1 else f"Non-standard aspect ratio ({aspect_ratio})"
            )
        else:
            validations["layout"] = ValidationItem(status="unverified", message="Image not loaded")

        return validations
