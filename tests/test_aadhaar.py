import os
import sys
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verifiers.aadhaar import AadhaarVerifier


def test_aadhaar_verifier():
    verifier = AadhaarVerifier()
    img = np.zeros((450, 720, 3), dtype=np.uint8)

    metadata = {"doc_number": "367598341258"}
    ocr_result = {"has_qr": True}

    res = verifier.verify(img, "dummy.jpg", ocr_result, metadata)
    assert "document_number" in res
    assert res["document_number"].status == "pass"
    assert "qr_code" in res
    assert res["qr_code"].status == "pass"
