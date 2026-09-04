import os
import sys
import pytest
import cv2
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forensics.ela import ELADetector
from forensics.copy_move import CopyMoveDetector
from forensics.geometry_analysis import GeometryAnalysisDetector


def test_ela_detector(tmp_path):
    detector = ELADetector()
    img_path = str(tmp_path / "test.jpg")
    img = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.imwrite(img_path, img)

    ev_list = detector.analyze(img, img_path)
    assert len(ev_list) == 1
    assert ev_list[0].type == "ela"
    assert ev_list[0].status.value in ("pass", "suspicious")


def test_geometry_detector():
    detector = GeometryAnalysisDetector()
    img = np.zeros((300, 450, 3), dtype=np.uint8)
    ev_list = detector.analyze(img, "dummy.jpg")
    assert len(ev_list) == 1
    assert ev_list[0].type == "geometry"
