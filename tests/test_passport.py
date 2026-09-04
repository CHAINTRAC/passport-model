import os
import sys
import pytest
import cv2
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verifiers.passport import PassportVerifier


def test_passport_verifier():
    verifier = PassportVerifier()
    # Dummy passport identity page image
    img = np.zeros((600, 800, 3), dtype=np.uint8)

    metadata = {
        "doc_number": "Z1234567",
        "mrz_line1": "P<INDSINGH<<GURPREET<<<<<<<<<<<<<<<<<<<<<<<<" ,
        "mrz_line2": "Z1234567<1IND8501019M3001019<<<<<<<<<<<<<<02"
    }
    ocr_result = {"has_mrz": True}

    res = verifier.verify(img, "dummy.jpg", ocr_result, metadata)
    assert "document_number" in res
    assert res["document_number"].status == "pass"
    assert "mrz_checksum" in res
    assert res["mrz_checksum"].status == "pass"
