"""
Manual API Client Test Script for Document Authenticity Server.
Usage:
    1. Start the server in one terminal:
       python server.py  (or uvicorn server:app --reload)
    2. Run this script in another terminal:
       python test_api.py
"""

import sys
import os
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


PORT = os.getenv("PORT", "8000")
SERVER_URL = os.getenv("SERVER_URL", f"http://localhost:{PORT}")
API_KEY = os.getenv("API_KEY", "midv2020-secret-api-key-2026")



def test_server():
    print("=" * 60)
    print("      DOCUMENT AUTHENTICITY FASTAPI MANUAL TEST SUITE")
    print("=" * 60)

    # 1. Health check
    print("\n1. Testing GET /health...")
    try:
        r = requests.get(f"{SERVER_URL}/health")
        print(f"   Status Code: {r.status_code}")
        print(f"   Response   : {r.json()}")
    except Exception as e:
        print(f"   [ERROR] Could not connect to server at {SERVER_URL}: {e}")
        sys.exit(1)

    # 2. Ready check
    print("\n2. Testing GET /ready...")
    r = requests.get(f"{SERVER_URL}/ready")
    print(f"   Status Code: {r.status_code}")
    print(f"   Response   : {r.json()}")

    # 3. Test verification endpoint with sample image if available
    sample_img = None
    for candidate in [
        "sample/passport/sample_passport.jpg",
        "sample/aadhar/sample_aadhar.jpg",
    ]:
        if os.path.exists(candidate):
            sample_img = candidate
            break

    if not sample_img and os.path.exists("sample"):
        for root, _, files in os.walk("sample"):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    sample_img = os.path.join(root, f)
                    break
            if sample_img:
                break

    if sample_img:
        print(f"\n3. Testing POST /api/v1/verify with image '{sample_img}'...")
        headers = {"X-API-Key": API_KEY}
        with open(sample_img, "rb") as f:
            files = {"image": (os.path.basename(sample_img), f, "image/jpeg")}
            data = {"doc_type": "auto"}
            r = requests.post(f"{SERVER_URL}/api/v1/verify", headers=headers, files=files, data=data)

        print(f"   Status Code: {r.status_code}")
        print(f"   Response   : {r.json()}")
    else:
        print("\n3. [INFO] No sample image found in 'sample/' directory to test upload.")

    print("\n" + "=" * 60)
    print("                     TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    test_server()
