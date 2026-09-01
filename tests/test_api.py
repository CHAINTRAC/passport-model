import os
import sys
import io
import pytest
import numpy as np
import cv2
from PIL import Image

# Ensure root workspace directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app, DEFAULT_API_KEY
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def create_dummy_image_bytes(format="JPEG", width=350, height=250, color=(200, 200, 200)):
    """Creates an in-memory test image byte buffer."""
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
    assert data["verify"] == "/api/v1/verify"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "model_loaded" in data
    assert "pipeline_ready" in data


# 2. Authentication Tests

def test_missing_api_key(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/verify",
        files={"image": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 401
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "MISSING_API_KEY"


def test_invalid_api_key(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": "wrong-api-key-value"},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 403
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_API_KEY"


# 3. Validation Tests

def test_invalid_doc_type(client):
    img_bytes = create_dummy_image_bytes()
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        data={"doc_type": "invalid_type_name"}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_DOC_TYPE"


def test_unsupported_file_mime_type(client):
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("test.txt", b"Hello text content", "text/plain")}
    )
    assert response.status_code == 415
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_corrupt_undecodable_image(client):
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("test.jpg", b"NOT_A_REAL_IMAGE_DATA_CORRUPT", "image/jpeg")}
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_IMAGE"


# 4. Verification Endpoint Integration Tests

def test_verify_passport_document(client):
    img_bytes = create_dummy_image_bytes("JPEG", 400, 300)
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("passport_sample.jpg", img_bytes, "image/jpeg")},
        data={
            "doc_type": "passport",
            "doc_number": "Z1234567",
            "mrz_line1": "P<INDTEST<<SAMPLE<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "mrz_line2": "Z1234567<4IND9001011M3001017<<<<<<<<<<<<<<04"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "passport_sample.jpg"
    assert data["doc_type"] == "passport"
    assert "verdict" in data
    assert "risk_score" in data
    assert "evidence_table" in data


def test_verify_aadhaar_document(client):
    img_bytes = create_dummy_image_bytes("PNG", 400, 250)
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("aadhaar_sample.png", img_bytes, "image/png")},
        data={
            "doc_type": "aadhaar",
            "doc_number": "367598341257"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "aadhaar_sample.png"
    assert data["doc_type"] == "aadhaar"
    assert "verdict" in data
    assert "risk_score" in data
    assert "evidence_table" in data


def test_verify_auto_doc_type(client):
    img_bytes = create_dummy_image_bytes("JPEG", 400, 300)
    response = client.post(
        "/api/v1/verify",
        headers={"X-API-Key": DEFAULT_API_KEY},
        files={"image": ("auto_sample.jpg", img_bytes, "image/jpeg")},
        data={"doc_type": "auto"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "auto_sample.jpg"
    assert "verdict" in data
