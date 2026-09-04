import os
import sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from document_detection.detector import DocumentTypeDetector
from validation.document_number import DocumentNumberValidator
from validation.mrz import MRZValidator
from validation.verhoeff import VerhoeffValidator


def test_mrz_checksum_validation():
    l1 = "P<INDSINGH<<GURPREET<<<<<<<<<<<<<<<<<<<<<<<<"
    l2 = "Z1234567<1IND8501019M3001019<<<<<<<<<<<<<<02"
    res = MRZValidator.validate_mrz_lines(l1, l2)
    assert res["valid_mrz"] is True
    assert res["details"]["passport_number"] == "Z1234567<"


def test_verhoeff_checksum_validation():
    # Valid Aadhaar 12-digit number passing Verhoeff checksum
    valid_num = "367598341258"
    assert VerhoeffValidator.validate_checksum(valid_num) is True

    invalid_num = "367598341257"
    assert VerhoeffValidator.validate_checksum(invalid_num) is False


def test_doc_number_validators():
    assert DocumentNumberValidator.validate_passport_number("Z1234567")["valid_format"] is True
    assert DocumentNumberValidator.validate_passport_number("12345678")["valid_format"] is False

    assert DocumentNumberValidator.validate_dl_number("DL0420110012345")["valid_format"] is True


def test_document_type_detector_scoring():
    detector = DocumentTypeDetector()
    res = detector.detect(
        img_path="sample/DL/3.jpg",
        explicit_doc_type="auto",
        original_filename="3.jpg"
    )
    # 3.jpg MUST NOT misclassify as passport!
    assert res.doc_type in ("dl", "unknown")
    assert res.doc_type != "passport"
