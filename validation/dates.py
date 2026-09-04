import re
from datetime import datetime
from typing import Dict, Any, Optional


class DateValidator:
    """
    Validates logical ranges of Date of Birth (DOB) and Expiry dates.
    """

    @staticmethod
    def validate_dob(dob_str: str) -> Dict[str, Any]:
        """Validates DOB is in past and age is between 0 and 120."""
        dob_str = str(dob_str).strip()
        try:
            # Try formats YYMMDD or DD/MM/YYYY or YYYY-MM-DD
            if len(dob_str) == 6 and dob_str.isdigit():
                dt = datetime.strptime(dob_str, "%y%m%d")
            elif "/" in dob_str:
                dt = datetime.strptime(dob_str, "%d/%m/%Y")
            elif "-" in dob_str:
                dt = datetime.strptime(dob_str, "%Y-%m-%d")
            else:
                return {"valid_dob": False, "message": "Unknown DOB date format"}

            now = datetime.now()
            if dt > now:
                return {"valid_dob": False, "message": "Date of Birth cannot be in the future"}
            age = (now - dt).days // 365
            if age > 120:
                return {"valid_dob": False, "message": "Unlikely DOB age > 120 years"}

            return {"valid_dob": True, "dob": dt.strftime("%Y-%m-%d"), "age": age, "message": "Valid DOB"}
        except Exception as e:
            return {"valid_dob": False, "message": f"DOB parse error: {str(e)}"}

    @staticmethod
    def validate_expiry(expiry_str: str) -> Dict[str, Any]:
        """Checks expiry date format."""
        expiry_str = str(expiry_str).strip()
        try:
            if len(expiry_str) == 6 and expiry_str.isdigit():
                dt = datetime.strptime(expiry_str, "%y%m%d")
            elif "/" in expiry_str:
                dt = datetime.strptime(expiry_str, "%d/%m/%Y")
            elif "-" in expiry_str:
                dt = datetime.strptime(expiry_str, "%Y-%m-%d")
            else:
                return {"valid_expiry": False, "message": "Unknown expiry date format"}

            now = datetime.now()
            is_expired = dt < now
            return {
                "valid_expiry": True,
                "expiry": dt.strftime("%Y-%m-%d"),
                "is_expired": is_expired,
                "message": "Document expired" if is_expired else "Document active/valid"
            }
        except Exception as e:
            return {"valid_expiry": False, "message": f"Expiry parse error: {str(e)}"}
