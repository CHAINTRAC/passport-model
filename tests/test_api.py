import os
import sys
import io
import pytest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app, DEFAULT_API_KEY
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_dummy_image_bytes(format="JPEG", width=350, height=250, color=(200, 200, 200)):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


# 1. Health & Readiness Tests

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Document Authenticity API"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


# 2. Authentication Tests

def test_missing_api_key(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/verify",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 401


def test_invalid_api_key(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": "wrong-key"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 403


# 3. Verification Endpoint Integration Tests

def test_verify_dl_3_jpg_auto_doc_type(client):
    # Verify sample/DL/3.jpg does NOT misclassify as passport
    img_path = "sample/DL/3.jpg"
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            response = client.post(
                "/api/v1/verify",
                headers={"X-API-Key": DEFAULT_API_KEY},
                files={"image": ("3.jpg", f.read(), "image/jpeg")},
                data={"doc_type": "auto"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["document"]["type"] in ("dl", "unknown")
        assert data["document"]["type"] != "passport"
        assert "decision" in data
        assert "forensics" in data
        assert "evidence_summary" in data


def test_verify_passport_document(client):
    img_bytes = create_dummy_image_bytes("JPEG", 400, 300)
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("passport_sample.jpg", img_bytes, "image/jpeg")},
        data={
            "doc_type": "passport",
            "doc_number": "Z1234567",
            "mrz_line1": "P<INDSINGH<<GURPREET<<<<<<<<<<<<<<<<<<<<<<<<",
            "mrz_line2": "Z1234567<1IND8501019M3001019<<<<<<<<<<<<<<02"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document"]["type"] == "passport"
    assert data["validation"]["mrz_checksum"]["status"] == "pass"


def test_verify_aadhaar_document(client):
    img_bytes = create_dummy_image_bytes("PNG", 400, 250)
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("aadhaar_sample.png", img_bytes, "image/png")},
        data={
            "doc_type": "aadhaar",
            "doc_number": "367598341258"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document"]["type"] == "aadhaar"
    assert data["validation"]["document_number"]["status"] == "pass"
