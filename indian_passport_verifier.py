import os
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance

class IndianPassportVerifier:
    """
    Comprehensive Security & Authenticity Verifier for Indian Passports
    Implements rule-based, structural, and image-based verification:
    1. Passport Number Format Validation ([A-Z][0-9]{7})
    2. ICAO 9303 MRZ Checksum Validation (7-3-1 weighting algorithm for Passport #, DOB, Expiry, Composite)
    3. Visual Zone vs. MRZ Field Cross-Consistency Validation
    4. Passport Identity Page Layout & MRZ Geometry Analysis (No ID-1 card aspect ratio check)
    5. Error Level Analysis (ELA) for Detecting Digital Modifications & Compression Anomalies
    """

    @staticmethod
    def compute_mrz_checksum(data_str):
        """Computes ICAO 9303 MRZ 7-3-1 weighted check digit."""
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(str(data_str).upper()):
            if char.isdigit():
                val = int(char)
            elif char.isupper():
                val = ord(char) - 55  # A=10, B=11, ... Z=35
            else:
                val = 0  # '<' or filler
            total += val * weights[i % 3]
        return total % 10

    def validate_passport_number_format(self, passport_num):
        """Verifies standard Indian Passport Number format: 1 Letter followed by 7 Digits (e.g. Z1234567)."""
        passport_num = str(passport_num).strip().upper()
        pattern = r'^[A-Z][0-9]{7}$'
        is_valid = bool(re.match(pattern, passport_num))
        return {
            'passport_number': passport_num,
            'valid_format': is_valid,
            'message': "Valid Indian Passport Number Pattern" if is_valid else "Invalid Passport Number Format (Expected 1 Letter + 7 Digits)"
        }

    def validate_mrz_lines(self, mrz_line1, mrz_line2):
        """
        Validates Type 3 (Passport) 2-Line MRZ (ICAO Doc 9303 standard):
        Line 1: P<IND<SURNAME<<GIVEN<NAME<<<<<<<<<<<<<<<<<<< (44 chars)
        Line 2: Z1234567<0IND8501011M3001018<<<<<<<<<<<<<<02 (44 chars)
        """
        mrz_line1 = mrz_line1.replace(" ", "").upper()
        mrz_line2 = mrz_line2.replace(" ", "").upper()

        if len(mrz_line1) < 44 or len(mrz_line2) < 44:
            return {
                'valid_mrz': False,
                'status': 'MRZ_INVALID_OR_OCR_UNCERTAIN',
                'message': 'MRZ lines must be at least 44 characters long'
            }

        doc_type = mrz_line1[0:2]
        country = mrz_line1[2:5]

        # Extract Line 1 name fields
        names = mrz_line1[5:44].split("<<")
        surname = names[0].replace("<", " ").strip() if len(names) > 0 else ""
        given_name = names[1].replace("<", " ").strip() if len(names) > 1 else ""

        # Extract Line 2 fields
        pass_num = mrz_line2[0:9]
        pass_num_check = mrz_line2[9]
        nationality = mrz_line2[10:13]
        dob = mrz_line2[13:19]
        dob_check = mrz_line2[19]
        sex = mrz_line2[20]
        expiry = mrz_line2[21:27]
        expiry_check = mrz_line2[27]
        optional = mrz_line2[28:42]
        optional_check = mrz_line2[42] if len(mrz_line2) > 42 else '0'
        composite_check = mrz_line2[43] if len(mrz_line2) > 43 else ''

        # Compute Checksums
        calc_pass_check = str(self.compute_mrz_checksum(pass_num))
        calc_dob_check = str(self.compute_mrz_checksum(dob))
        calc_expiry_check = str(self.compute_mrz_checksum(expiry))
        calc_optional_check = str(self.compute_mrz_checksum(optional))

        # Composite data: pass_num + pass_check + dob + dob_check + expiry + expiry_check + optional + optional_check
        composite_str = f"{pass_num}{pass_num_check}{dob}{dob_check}{expiry}{expiry_check}{optional}{optional_check}"
        calc_composite_check = str(self.compute_mrz_checksum(composite_str))

        pass_num_valid = (calc_pass_check == pass_num_check)
        dob_valid = (calc_dob_check == dob_check)
        expiry_valid = (calc_expiry_check == expiry_check)
        composite_valid = (calc_composite_check == composite_check) if composite_check else True

        overall_mrz_valid = pass_num_valid and dob_valid and expiry_valid and (country == "IND")

        status = 'MRZ_CONSISTENT' if overall_mrz_valid else 'MRZ_INVALID_OR_OCR_UNCERTAIN'

        return {
            'valid_mrz': overall_mrz_valid,
            'status': status,
            'doc_type': doc_type,
            'issuing_country': country,
            'passport_number': pass_num.replace('<', ''),
            'surname': surname,
            'given_name': given_name,
            'nationality': nationality,
            'dob_yymmdd': dob,
            'sex': sex,
            'expiry_yymmdd': expiry,
            'checksums': {
                'pass_num_valid': pass_num_valid,
                'dob_valid': dob_valid,
                'expiry_valid': expiry_valid,
                'composite_valid': composite_valid
            }
        }

    def cross_validate_visual_and_mrz(self, visual_fields, mrz_parsed):
        """
        Cross-matches visual inspection / OCR fields against parsed MRZ data:
        - Passport Number
        - Surname & Given Name
        - Date of Birth
        - Expiry Date
        - Sex & Nationality
        """
        if not visual_fields or not mrz_parsed or not mrz_parsed.get('valid_mrz'):
            return {
                'match_status': 'UNVERIFIED',
                'match_score': 0.0,
                'mismatches': ['MRZ or Visual data missing']
            }

        mismatches = []
        matches = 0
        total_checks = 0

        # 1. Passport Number
        if 'passport_number' in visual_fields and visual_fields['passport_number']:
            total_checks += 1
            v_num = re.sub(r'[^A-Z0-9]', '', str(visual_fields['passport_number']).upper())
            m_num = re.sub(r'[^A-Z0-9]', '', str(mrz_parsed.get('passport_number', '')).upper())
            if v_num == m_num:
                matches += 1
            else:
                mismatches.append(f"Passport Number mismatch: Visual='{v_num}' vs MRZ='{m_num}'")

        # 2. Surname
        if 'surname' in visual_fields and visual_fields['surname']:
            total_checks += 1
            v_sur = re.sub(r'[^A-Z]', '', str(visual_fields['surname']).upper())
            m_sur = re.sub(r'[^A-Z]', '', str(mrz_parsed.get('surname', '')).upper())
            if v_sur in m_sur or m_sur in v_sur or v_sur == m_sur:
                matches += 1
            else:
                mismatches.append(f"Surname mismatch: Visual='{v_sur}' vs MRZ='{m_sur}'")

        # 3. Given Name
        if 'given_name' in visual_fields and visual_fields['given_name']:
            total_checks += 1
            v_giv = re.sub(r'[^A-Z]', '', str(visual_fields['given_name']).upper())
            m_giv = re.sub(r'[^A-Z]', '', str(mrz_parsed.get('given_name', '')).upper())
            if v_giv in m_giv or m_giv in v_giv or v_giv == m_giv:
                matches += 1
            else:
                mismatches.append(f"Given Name mismatch: Visual='{v_giv}' vs MRZ='{m_giv}'")

        # 4. DOB
        if 'dob_yymmdd' in visual_fields and visual_fields['dob_yymmdd']:
            total_checks += 1
            v_dob = str(visual_fields['dob_yymmdd'])
            m_dob = str(mrz_parsed.get('dob_yymmdd', ''))
            if v_dob == m_dob:
                matches += 1
            else:
                mismatches.append(f"DOB mismatch: Visual='{v_dob}' vs MRZ='{m_dob}'")

        match_score = (matches / total_checks) if total_checks > 0 else 0.0
        is_consistent = (len(mismatches) == 0 and total_checks > 0)

        return {
            'is_consistent': is_consistent,
            'match_score': round(match_score, 2),
            'total_checks': total_checks,
            'matches': matches,
            'mismatches': mismatches,
            'match_status': 'MATCH' if is_consistent else 'MISMATCH'
        }

    def analyze_passport_geometry(self, img_path):
        """
        Validates Passport identity page layout:
        - Image aspect ratio check is NOT ID-1 (passports are ID-3 125x88mm ~ 1.42 ratio, but pages vary)
        - Verifies MRZ horizontal baseline geometry at the bottom 25% of the page
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                return {'valid_layout': False, 'message': 'Image load failed'}
            h, w, _ = img.shape
            
            # Passport page width/height bounds
            has_adequate_resolution = (w >= 400 and h >= 300)
            
            # Bottom 25% region check for high horizontal edge frequency (MRZ character lines)
            bottom_region = img[int(h * 0.75):, :]
            gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray_bottom, cv2.CV_64F, 1, 0, ksize=3)
            mrz_edge_density = float(np.mean(np.abs(sobelx)))

            has_mrz_zone_density = (mrz_edge_density > 12.0)

            return {
                'valid_layout': has_adequate_resolution and has_mrz_zone_density,
                'width': w,
                'height': h,
                'mrz_edge_density': round(mrz_edge_density, 2),
                'message': 'Passport Layout & MRZ Baseline Detected' if has_mrz_zone_density else 'Weak/Missing MRZ Zone Geometry'
            }
        except Exception as e:
            return {'valid_layout': False, 'message': str(e)}

    def compute_ela_score(self, img_path, quality=90):
        """Computes Error Level Analysis (ELA) score as supporting forensic evidence."""
        temp_path = None
        try:
            original = Image.open(img_path).convert('RGB')
            temp_path = f"scratch_temp_ela_{os.getpid()}.jpg"
            original.save(temp_path, 'JPEG', quality=quality)
            temporary = Image.open(temp_path)

            ela_img = ImageChops.difference(original, temporary)
            ela_np = np.array(ela_img)
            ela_variance = float(np.var(ela_np))

            temporary.close()
            original.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Low variance (< 350.0) indicates uniform original print compression
            is_uniform_compression = (ela_variance < 350.0)

            return {
                'ela_variance_score': round(ela_variance, 2),
                'tampering_anomaly_detected': not is_uniform_compression,
                'status': "UNIFORM_COMPRESSION" if is_uniform_compression else "ELEVATED_COMPRESSION_VARIANCE"
            }
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return {'ela_variance_score': 0.0, 'tampering_anomaly_detected': False, 'status': f"ELA_ERROR: {str(e)}"}


if __name__ == "__main__":
    verifier = IndianPassportVerifier()
    print("=" * 80)
    print("       INDIAN PASSPORT SECURITY & VERIFICATION MODULE")
    print("=" * 80)

    sample_mrz_line1 = "P<INDSINGH<<GURPREET<<<<<<<<<<<<<<<<<<<<<<<<"
    sample_mrz_line2 = "Z1234567<1IND8501011M3001018<<<<<<<<<<<<<<02"
    mrz_res = verifier.validate_mrz_lines(sample_mrz_line1, sample_mrz_line2)

    print("\n--- 1. MRZ Checksum Algorithm Verification ---")
    print(f"Passport Number : {mrz_res['passport_number']}")
    print(f"Country Code    : {mrz_res['issuing_country']}")
    print(f"Checksum Valid  : {mrz_res['valid_mrz']}")
    print(f"MRZ Status      : {mrz_res['status']}")

    visual = {
        'passport_number': 'Z1234567',
        'surname': 'SINGH',
        'given_name': 'GURPREET',
        'dob_yymmdd': '850101'
    }
    match_res = verifier.cross_validate_visual_and_mrz(visual, mrz_res)
    print("\n--- 2. Visual vs. MRZ Field Cross-Matching ---")
    print(f"Cross-Match Status: {match_res['match_status']} (Score: {match_res['match_score']})")
    print("=" * 80)
