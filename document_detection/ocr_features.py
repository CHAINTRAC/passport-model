import cv2
import re
import numpy as np
from typing import Dict, Any, List, Tuple

_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception:
            _easyocr_reader = False
    return _easyocr_reader if _easyocr_reader is not False else None


class OCRFeatureExtractor:
    """
    Extracts visual text signals, OCR semantic keywords, MRZ lines, QR codes, and region metadata.
    Uses EasyOCR (with max 800px image scaling for ultra-fast performance) and OpenCV text segmenters.
    """

    PASSPORT_KEYWORDS = ["PASSPORT", "INDIAN PASSPORT", "REPUBLIC OF INDIA", "P<IND", "PASSPORT NO", "SURNAME", "GIVEN NAME", "NATIONALITY", "PLACE OF BIRTH", "DATE OF EXPIRY"]
    DL_KEYWORDS = ["DRIVING LICENCE", "DRIVING LICENSE", "UNION OF INDIA", "INDIAN UNION", "FORM 7", "TRANSPORT", "LICENCE NO", "AUTHORISATION", "DL NO", "NON-TRANSPORT", "ORGAN DONOR", "BLOOD GROUP", "VALIDITY", "HOLDER'S SIGNATURE", "SON/DAUGHTER", "ISSUED BY"]
    AADHAAR_KEYWORDS = ["GOVERNMENT OF INDIA", "UNIQUE IDENTIFICATION", "AUTHORITY OF INDIA", "AADHAAR", "UIDAI", "ENROLMENT", "MALE", "FEMALE", "DOB:", "YEAR OF BIRTH", "MY AADHAAR", "HELP@UIDAI.GOV.IN", "WWW.UIDAI.GOV.IN"]

    @classmethod
    def extract_features(cls, img: np.ndarray) -> Dict[str, Any]:
        if img is None or img.size == 0:
            return {
                "has_mrz": False,
                "has_qr": False,
                "bottom_edge_density": 0.0,
                "wide_line_count": 0,
                "passport_keyword_matches": 0,
                "dl_keyword_matches": 0,
                "aadhaar_keyword_matches": 0,
                "extracted_text": "",
                "text_regions": []
            }

        h, w, _ = img.shape
        aspect_ratio = round(float(w) / float(h), 2)
        inv_aspect_ratio = round(float(h) / float(w), 2)
        is_id1 = (1.48 <= aspect_ratio <= 1.72) or (1.48 <= inv_aspect_ratio <= 1.72)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. QR Code Detection
        has_qr = False
        qr_detector = cv2.QRCodeDetector()
        try:
            retval, points = qr_detector.detect(img)
            has_qr = bool(retval)
            if not has_qr:
                retval, points = qr_detector.detect(gray)
                has_qr = bool(retval)
        except Exception:
            has_qr = False

        # 2. EasyOCR Text Keyword Extraction (scaled to max 800px width for fast execution)
        extracted_text = ""
        passport_kw_count = 0
        dl_kw_count = 0
        aadhaar_kw_count = 0

        reader = get_easyocr_reader()
        if reader is not None:
            try:
                # Resize for fast CPU OCR
                scale = 800.0 / max(w, h) if max(w, h) > 800 else 1.0
                if scale < 1.0:
                    ocr_img = cv2.resize(img, (int(w * scale), int(h * scale)))
                else:
                    ocr_img = img

                ocr_results = reader.readtext(ocr_img, detail=0)
                extracted_text = " ".join(ocr_results).upper()

                for kw in cls.PASSPORT_KEYWORDS:
                    if kw in extracted_text:
                        passport_kw_count += 1
                for kw in cls.DL_KEYWORDS:
                    if kw in extracted_text:
                        dl_kw_count += 1
                for kw in cls.AADHAAR_KEYWORDS:
                    if kw in extracted_text:
                        aadhaar_kw_count += 1
            except Exception:
                pass

        # 3. MRZ Line Pattern Detection in Bottom 25%
        bottom_region = gray[int(h * 0.75):, :]
        sobelx = cv2.Sobel(bottom_region, cv2.CV_64F, 1, 0, ksize=3)
        edge_density = float(np.mean(np.abs(sobelx)))

        _, thresh = cv2.threshold(bottom_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.12), 1))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        wide_lines = 0
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > w * 0.65 and ch < h * 0.10:
                wide_lines += 1

        # MRZ line check: requires explicit MRZ text keywords or (2 wide lines AND not ID-1 card)
        has_mrz = ("P<IND" in extracted_text) or ((wide_lines >= 2) and not is_id1)

        # 4. Text region bounding boxes
        text_regions = []
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, text_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        t_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        t_morphed = cv2.morphologyEx(text_thresh, cv2.MORPH_CLOSE, t_kernel)
        t_cnts, _ = cv2.findContours(t_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in t_cnts:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 30 and ch > 8 and (cw * ch) > 300:
                text_regions.append([int(x), int(y), int(cw), int(ch)])

        return {
            "has_mrz": has_mrz,
            "has_qr": has_qr,
            "bottom_edge_density": round(edge_density, 2),
            "wide_line_count": wide_lines,
            "passport_keyword_matches": passport_kw_count,
            "dl_keyword_matches": dl_kw_count,
            "aadhaar_keyword_matches": aadhaar_kw_count,
            "extracted_text": extracted_text,
            "text_regions": text_regions[:20]
        }
