from typing import Dict, Any, Tuple


class MRZValidator:
    """
    ICAO Doc 9303 Standard 2-Line Type 3 MRZ Validator & Parser.
    Implements 7-3-1 weighted check digit algorithm.
    """

    @staticmethod
    def compute_checkdigit(data_str: str) -> int:
        """Computes ICAO 9303 MRZ 7-3-1 weighted check digit."""
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(str(data_str).upper()):
            if char.isdigit():
                val = int(char)
            elif char.isupper():
                val = ord(char) - 55  # A=10, B=11, ..., Z=35
            else:
                val = 0  # '<' or filler
            total += val * weights[i % 3]
        return total % 10

    @classmethod
    def validate_mrz_lines(cls, line1: str, line2: str) -> Dict[str, Any]:
        """
        Parses and validates ICAO Doc 9303 2-line Passport MRZ.
        """
        l1 = line1.replace(" ", "").upper()
        l2 = line2.replace(" ", "").upper()

        if len(l1) < 44 or len(l2) < 44:
            return {
                "valid_mrz": False,
                "status": "INVALID_LENGTH",
                "message": "MRZ lines must be at least 44 characters long",
                "details": {}
            }

        doc_type = l1[0:2]
        country = l1[2:5]

        names = l1[5:44].split("<<")
        surname = names[0].replace("<", " ").strip() if len(names) > 0 else ""
        given_name = names[1].replace("<", " ").strip() if len(names) > 1 else ""

        pass_num = l2[0:9]
        pass_num_check = l2[9]
        nationality = l2[10:13]
        dob = l2[13:19]
        dob_check = l2[19]
        sex = l2[20]
        expiry = l2[21:27]
        expiry_check = l2[27]
        optional = l2[28:42]
        optional_check = l2[42] if len(l2) > 42 else '0'
        composite_check = l2[43] if len(l2) > 43 else ''

        calc_pass_check = str(cls.compute_checkdigit(pass_num))
        calc_dob_check = str(cls.compute_checkdigit(dob))
        calc_expiry_check = str(cls.compute_checkdigit(expiry))

        composite_str = f"{pass_num}{pass_num_check}{dob}{dob_check}{expiry}{expiry_check}{optional}{optional_check}"
        calc_composite_check = str(cls.compute_checkdigit(composite_str))

        pass_valid = (calc_pass_check == pass_num_check)
        dob_valid = (calc_dob_check == dob_check)
        expiry_valid = (calc_expiry_check == expiry_check)
        composite_valid = (calc_composite_check == composite_check) if composite_check else True

        overall_valid = pass_valid and dob_valid and expiry_valid and (country in ("IND", "P<"))

        return {
            "valid_mrz": overall_valid,
            "status": "MRZ_CONSISTENT" if overall_valid else "MRZ_CHECKSUM_FAILURE",
            "message": "MRZ checkdigits mathematically valid" if overall_valid else "MRZ checkdigit mismatch",
            "details": {
                "passport_number": pass_num,
                "pass_num_valid": pass_valid,
                "country": country,
                "nationality": nationality,
                "surname": surname,
                "given_name": given_name,
                "dob_yymmdd": dob,
                "dob_valid": dob_valid,
                "expiry_yymmdd": expiry,
                "expiry_valid": expiry_valid,
                "sex": sex,
                "composite_valid": composite_valid
            }
        }
