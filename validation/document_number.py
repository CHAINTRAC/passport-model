import re
from typing import Dict, Any, Optional
from validation.verhoeff import VerhoeffValidator


class DocumentNumberValidator:
    """
    Format & Pattern Validators for Indian Government ID Documents:
    1. Indian Passport: 1 Letter followed by 7 Digits (e.g. Z1234567)
    2. Aadhaar: 12 Digits starting with 2-9, passing Verhoeff checksum
    3. Driving Licence: 15-char Sarathi pattern (State code + 2 RTO digits + 4 Year digits + 7 Serial digits)
    """

    @staticmethod
    def normalize_ocr_digits(text: str) -> str:
        """Corrects common OCR letter-to-digit misreadings in numeric contexts."""
        mapping = {'O': '0', 'o': '0', 'Q': '0', 'S': '5', 's': '5', 'Z': '2', 'z': '2', 'I': '1', 'l': '1', '|': '1', 'B': '8'}
        return "".join(mapping.get(c, c) for c in text)

    @staticmethod
    def validate_passport_number(passport_num: str) -> Dict[str, Any]:
        clean_num = str(passport_num).strip().upper()
        clean_num = re.sub(r'[\s\-\/\.]+', '', clean_num)
        pattern = r'^[A-Z][0-9]{7}$'
        is_valid = bool(re.match(pattern, clean_num))
        return {
            "valid_format": is_valid,
            "cleaned_number": clean_num,
            "message": f"Valid Passport number pattern ('{clean_num}')" if is_valid else f"Invalid Passport number format ('{clean_num}')"
        }

    @staticmethod
    def validate_aadhaar_number(aadhaar_num: str) -> Dict[str, Any]:
        raw_num = str(aadhaar_num).strip()
        if raw_num.upper().startswith("VID"):
            raw_num = raw_num[3:].strip()
        
        clean_num = re.sub(r'[\s\-\/\.]+', '', raw_num)
        clean_num = DocumentNumberValidator.normalize_ocr_digits(clean_num)

        pattern = r'^[2-9][0-9]{11}$'
        has_pattern = bool(re.match(pattern, clean_num))
        passes_verhoeff = VerhoeffValidator.validate_checksum(clean_num) if has_pattern else False
        is_valid = has_pattern and passes_verhoeff
        return {
            "valid_format": is_valid,
            "has_valid_pattern": has_pattern,
            "passes_verhoeff": passes_verhoeff,
            "cleaned_number": clean_num,
            "message": f"Valid Aadhaar number '{clean_num}' & Verhoeff check" if is_valid else f"Invalid Aadhaar number format ('{clean_num}') or Verhoeff failure"
        }

    INDIAN_STATE_CODES = {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
        "GA", "GJ", "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD",
        "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "OR", "PB", "PY",
        "RJ", "SK", "TN", "TS", "TR", "UP", "UK", "UA", "WB"
    }

    @staticmethod
    def validate_dl_number(dl_num: str) -> Dict[str, Any]:
        raw_num = str(dl_num).upper().strip()
        clean_num = re.sub(r'[\s\-\/\.]+', '', raw_num)

        # Fix OCR digit confusions in state RTO and year/serial portions
        if len(clean_num) >= 4 and clean_num[:2].isalpha():
            prefix = clean_num[:2]
            rest = DocumentNumberValidator.normalize_ocr_digits(clean_num[2:])
            clean_num = prefix + rest

        has_state_code = clean_num[:2] in DocumentNumberValidator.INDIAN_STATE_CODES
        sarathi_pattern = r'^[A-Z]{2}[0-9]{2}[0-9]{4}[0-9]{7}$'
        general_pattern = r'^[A-Z]{2}[0-9]{2,13}$'

        is_strict_valid = has_state_code and bool(re.match(sarathi_pattern, clean_num))
        is_general_valid = has_state_code and bool(re.match(general_pattern, clean_num))

        return {
            "valid_format": is_general_valid,
            "is_strict_sarathi": is_strict_valid,
            "cleaned_number": clean_num,
            "message": f"Valid Driving Licence pattern ('{clean_num}')" if is_general_valid else f"Invalid Driving Licence number format ('{clean_num}')"
        }

    @staticmethod
    def extract_dl_number(text: str) -> Optional[str]:
        if not text:
            return None
        patterns = [
            r'[A-Z]{2}[\s\-\/\.]?[0-9OQSZIB]{2}[\s\-\/\.]?[0-9]{4}[\s\-\/\.]?[0-9]{7}',
            r'[A-Z]{2}[\s\-\/\.]?[0-9]{2,13}'
        ]
        for pat in patterns:
            matches = re.findall(pat, text.upper())
            for m in matches:
                res = DocumentNumberValidator.validate_dl_number(m)
                if res["valid_format"]:
                    return res["cleaned_number"]
        return None

    @staticmethod
    def extract_aadhaar_number(text: str) -> Optional[str]:
        if not text:
            return None
        matches = re.findall(r'(?<!VID\s)(?<!\d)[2-9][0-9OQSZIB]{3}[\s\-]?[0-9OQSZIB]{4}[\s\-]?[0-9OQSZIB]{4}(?!\d)', text.upper())
        for m in matches:
            res = DocumentNumberValidator.validate_aadhaar_number(m)
            if res["valid_format"]:
                return res["cleaned_number"]
        return None

    @staticmethod
    def extract_passport_number(text: str) -> Optional[str]:
        if not text:
            return None
        matches = re.findall(r'\b[A-Z][0-9]{7}\b', text.upper())
        for m in matches:
            res = DocumentNumberValidator.validate_passport_number(m)
            if res["valid_format"]:
                return res["cleaned_number"]
        return None
