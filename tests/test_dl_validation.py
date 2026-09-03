import os
import sys
import unittest
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

# Ensure root workspace directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indian_dl_verifier import IndianDLVerifier
from parivahan_api_client import ParivahanAPIClient


class TestIndianDLValidation(unittest.TestCase):

    def setUp(self):
        self.verifier = IndianDLVerifier()

    def test_normalize_dl_number(self):
        self.assertEqual(IndianDLVerifier.normalize_dl_number("DL-04-2011-0012345"), "DL0420110012345")
        self.assertEqual(IndianDLVerifier.normalize_dl_number("MH 12 2018 0054321"), "MH1220180054321")
        self.assertEqual(IndianDLVerifier.normalize_dl_number("ka.01.2020.0009876"), "KA0120200009876")

    def test_valid_dl_formats(self):
        valid_dls = [
            "DL0420110012345",
            "MH-12-2018-0054321",
            "KA 01 2020 0009876",
            "TN-05-2015-0001122"
        ]
        for dl in valid_dls:
            res = self.verifier.validate_format(dl)
            self.assertTrue(res["is_valid"], f"Failed for valid DL: {dl}")
            self.assertIsNone(res["error_code"])

    def test_invalid_state_code(self):
        res = self.verifier.validate_format("XX0420110012345")
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["error_code"], "INVALID_STATE_CODE")

    def test_invalid_issue_year(self):
        res = self.verifier.validate_format("DL0418400012345")
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["error_code"], "INVALID_ISSUE_YEAR")

    def test_invalid_dl_length(self):
        res = self.verifier.validate_format("DL042011")
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["error_code"], "INVALID_DL_FORMAT")

    def test_parivahan_api_client_unconfigured_fallback(self):
        client = ParivahanAPIClient(api_url="", api_key="")
        res = client.verify_dl_live("DL0420110012345")
        self.assertEqual(res["status"], "SKIPPED")
        self.assertFalse(res["api_configured"])
        self.assertIsNone(res["verified_live"])

    def test_sample_dl_folder_images(self):
        sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample", "DL"))
        self.assertTrue(os.path.exists(sample_dir), f"Sample DL directory not found: {sample_dir}")
        
        files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        self.assertGreater(len(files), 0, "No image files found in sample/DL folder")
        
        print("\n" + "=" * 70)
        print("          SAMPLE / DL FOLDER IMAGE VALIDATION REPORT")
        print("=" * 70)
        for idx, filename in enumerate(files, 1):
            file_path = os.path.join(sample_dir, filename)
            self.assertGreater(os.path.getsize(file_path), 0, f"Sample file is empty: {filename}")
            
            img = cv2.imread(file_path)
            self.assertIsNotNone(img, f"Failed to decode image: {filename}")
            
            h, w, c = img.shape
            aspect_ratio = round(w / float(h), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
            is_id1 = (1.35 <= aspect_ratio <= 1.75) or (1.35 <= round(h / float(w), 2) <= 1.75)
            
            print(f"[{idx}] {filename}")
            print(f"    - Dimensions  : {w}x{h} px (Aspect Ratio: {aspect_ratio})")
            print(f"    - ID-1 Format : {'PASS' if is_id1 else 'PORTRAIT/CROPPED'}")
            print(f"    - Image Quality: Blur Score {blur_score} ({'PASS' if blur_score >= 25.0 else 'BLURRY'})")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()
