import os
import sys
import pytest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verifiers.driving_license import DrivingLicenseVerifier


def test_driving_license_verifier():
    verifier = DrivingLicenseVerifier()
    # Dummy ID-1 landscape card image (735x454 -> 1.62 aspect ratio)
    img = np.zeros((454, 735, 3), dtype=np.uint8)

    metadata = {"doc_number": "DL0420110012345"}
    ocr_result = {"has_mrz": False}

    res = verifier.verify(img, "sample/DL/3.jpg", ocr_result, metadata)
    assert "document_number" in res
    assert "layout" in res
    assert res["layout"].status == "pass"
