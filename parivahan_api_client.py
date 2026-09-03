import os
import json
import logging
from typing import Dict, Any, Optional
import urllib.request
import urllib.error

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("parivahan_api_client")


class ParivahanAPIClient:
    """
    Client for real-time Indian Driving Licence (DL) verification via Parivahan or 
    3rd-party Gateway APIs (e.g. Surepass, Cashfree, Setu, Sandbox).

    If API credentials (PARIVAHAN_API_KEY / PARIVAHAN_API_URL) are not set in .env,
    the client safely skips real-time verification without raising any runtime errors.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        timeout: int = 5
    ):
        self.api_url = api_url or os.getenv("PARIVAHAN_API_URL", "").strip()
        self.api_key = api_key or os.getenv("PARIVAHAN_API_KEY", "").strip()
        self.client_id = client_id or os.getenv("PARIVAHAN_CLIENT_ID", "").strip()
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Returns True if minimum required API credentials are present in environment/config."""
        return bool(self.api_url and self.api_key)

    def verify_dl_live(self, normalized_dl: str, dob: Optional[str] = None) -> Dict[str, Any]:
        """
        Queries the configured Parivahan/Verification API for live DL status.

        Args:
            normalized_dl (str): Cleaned 15-character DL number (e.g., 'DL0420110012345')
            dob (str, optional): Date of Birth in YYYY-MM-DD or DD/MM/YYYY format if required by API

        Returns:
            dict: Standardized API response summary.
        """
        # If API credentials are not provided in .env, safely skip without error
        if not self.is_configured():
            logger.info("Parivahan API credentials not found in .env. Skipping real-time verification.")
            return {
                "status": "SKIPPED",
                "api_configured": False,
                "verified_live": None,
                "message": "Parivahan API credentials not configured in .env. Real-time verification skipped.",
                "details": None
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "User-Agent": "MIDV2020-DLVerifier/1.0"
        }
        if self.client_id:
            headers["x-client-id"] = self.client_id

        payload = {
            "dl_number": normalized_dl,
            "dob": dob or ""
        }

        try:
            req = urllib.request.Request(
                url=self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_bytes = response.read()
                data = json.loads(response_bytes.decode("utf-8"))

                return {
                    "status": "SUCCESS",
                    "api_configured": True,
                    "verified_live": data.get("valid", True),
                    "message": "Real-time Parivahan API verification successful.",
                    "details": data
                }

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            logger.error(f"Parivahan API HTTP Error {e.code}: {error_body}")
            return {
                "status": "HTTP_ERROR",
                "api_configured": True,
                "verified_live": False,
                "http_code": e.code,
                "message": f"Parivahan API responded with HTTP {e.code}.",
                "details": None
            }

        except urllib.error.URLError as e:
            logger.error(f"Parivahan API Connection Error: {e.reason}")
            return {
                "status": "CONNECTION_ERROR",
                "api_configured": True,
                "verified_live": None,
                "message": f"Failed to connect to Parivahan API: {e.reason}",
                "details": None
            }

        except Exception as e:
            logger.error(f"Unexpected error during Parivahan API verification: {str(e)}")
            return {
                "status": "ERROR",
                "api_configured": True,
                "verified_live": None,
                "message": f"Unexpected error during API call: {str(e)}",
                "details": None
            }
