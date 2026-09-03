import os
import sys
import tempfile
import secrets
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict_pipeline import DocumentAuthenticityPipeline

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Configure Logging (DO NOT log API keys, document numbers, MRZ data, or file contents)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("document_authenticity_api")

# Environment & Configuration Defaults
DEFAULT_API_KEY = "midv2020-secret-api-key-2026"
API_KEY = os.getenv("API_KEY", DEFAULT_API_KEY)
MAX_UPLOAD_SIZE_MB = float(os.getenv("MAX_UPLOAD_SIZE_MB", "10.0"))
MAX_UPLOAD_SIZE_BYTES = int(MAX_UPLOAD_SIZE_MB * 1024 * 1024)
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "").strip()

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_DOC_TYPES = {"auto", "passport", "aadhaar", "dl", "driving_licence", "driving_license"}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# --- Pydantic Schema Contracts ---

class ErrorDetail(BaseModel):
    code: str = Field(..., json_schema_extra={"example": "INVALID_IMAGE"})
    message: str = Field(..., json_schema_extra={"example": "Uploaded file could not be decoded as an image."})


class ErrorResponse(BaseModel):
    success: bool = Field(False, json_schema_extra={"example": False})
    error: ErrorDetail


class VerificationResponse(BaseModel):
    success: bool = Field(True, json_schema_extra={"example": True})
    filename: str = Field(..., json_schema_extra={"example": "passport_sample.jpg"})
    doc_type: str = Field(..., json_schema_extra={"example": "passport"})
    verdict: str = Field(..., json_schema_extra={"example": "GENUINE"})
    risk_score: float = Field(..., json_schema_extra={"example": 0.15})
    reasons: List[str] = Field(default_factory=list)
    evidence_table: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = Field("healthy", json_schema_extra={"example": "healthy"})


class ReadyResponse(BaseModel):
    status: str = Field("ready", json_schema_extra={"example": "ready"})
    model_loaded: bool = Field(True, json_schema_extra={"example": True})
    pipeline_ready: bool = Field(True, json_schema_extra={"example": True})


class RootResponse(BaseModel):
    service: str = Field("Document Authenticity API", json_schema_extra={"example": "Document Authenticity API"})
    version: str = Field("1.0.0", json_schema_extra={"example": "1.0.0"})
    docs: str = Field("/docs", json_schema_extra={"example": "/docs"})
    health: str = Field("/health", json_schema_extra={"example": "/health"})
    ready: str = Field("/ready", json_schema_extra={"example": "/ready"})
    verify: str = Field("/api/v1/verify", json_schema_extra={"example": "/api/v1/verify"})


# --- Lifespan Handler ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing DocumentAuthenticityPipeline on startup...")
    try:
        pipeline = DocumentAuthenticityPipeline()
        app.state.pipeline = pipeline
        app.state.pipeline_ready = True
        logger.info("DocumentAuthenticityPipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DocumentAuthenticityPipeline: {str(e)}")
        app.state.pipeline = None
        app.state.pipeline_ready = False
        raise RuntimeError(f"Pipeline startup failed: {str(e)}")
    yield
    logger.info("Shutting down API server...")


app = FastAPI(
    title="Document Authenticity & Fraud Detection API",
    description="Production REST API for verifying document authenticity (Indian Passport & Aadhaar Card).",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
if ALLOWED_ORIGINS_ENV:
    origins = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Helper functions
def verify_api_key(api_key: Optional[str]) -> Optional[JSONResponse]:
    """Validates X-API-Key with constant-time comparison."""
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"success": False, "error": {"code": "MISSING_API_KEY", "message": "API key header 'X-API-Key' is missing"}}
        )
    
    if not secrets.compare_digest(api_key, API_KEY):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"success": False, "error": {"code": "INVALID_API_KEY", "message": "Invalid API key"}}
        )
    
    return None


def validate_image_content(content: bytes, filename: str, content_type: Optional[str]) -> Optional[JSONResponse]:
    """Validates file size, MIME type, and image decodability."""
    # 1. File size check
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"success": False, "error": {"code": "FILE_TOO_LARGE", "message": f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE_MB} MB"}}
        )

    # 2. Extension & MIME check
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    is_valid_mime = content_type and content_type.lower() in ALLOWED_MIME_TYPES
    is_valid_ext = ext in ALLOWED_EXTENSIONS

    if not (is_valid_mime or is_valid_ext):
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={"success": False, "error": {"code": "UNSUPPORTED_FILE_TYPE", "message": f"Unsupported file type. Allowed formats: JPEG, PNG, WEBP"}}
        )

    # 3. Actual Image Decodability
    try:
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Decoded image is empty or invalid")
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": {"code": "INVALID_IMAGE", "message": "Uploaded file could not be decoded as a valid image"}}
        )

    return None


# --- Endpoints ---

@app.get("/", response_model=RootResponse)
def root():
    """Returns API metadata and endpoint links."""
    return RootResponse()


@app.get("/health", response_model=HealthResponse)
def health():
    """Returns service vitality status."""
    return HealthResponse(status="healthy")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request):
    """Returns pipeline readiness status."""
    pipeline_ready = getattr(request.app.state, "pipeline_ready", False)
    pipeline = getattr(request.app.state, "pipeline", None)
    model_loaded = pipeline is not None and getattr(pipeline, "model", None) is not None

    if not pipeline_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "model_loaded": model_loaded, "pipeline_ready": False}
        )
    return ReadyResponse(status="ready", model_loaded=model_loaded, pipeline_ready=True)


@app.post(
    "/api/v1/verify",
    response_model=VerificationResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    }
)
def verify_document(
    request: Request,
    image: UploadFile = File(...),
    doc_type: str = Form("auto"),
    doc_number: Optional[str] = Form(None),
    dob: Optional[str] = Form(None),
    mrz_line1: Optional[str] = Form(None),
    mrz_line2: Optional[str] = Form(None),
    api_key: Optional[str] = Security(api_key_header)
):
    """
    Main Verification Endpoint.
    Accepts multipart/form-data with image file and optional document verification parameters.
    """
    # 1. API Key Authentication
    auth_error = verify_api_key(api_key)
    if auth_error:
        return auth_error

    # 2. Pipeline Availability Check
    pipeline: DocumentAuthenticityPipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": {"code": "MODEL_UNAVAILABLE", "message": "Document verification pipeline is not initialized"}}
        )

    # 3. Document Type Validation
    norm_doc_type = doc_type.lower().strip() if doc_type else "auto"
    if norm_doc_type not in ALLOWED_DOC_TYPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": {"code": "INVALID_DOC_TYPE", "message": f"Unsupported document type '{doc_type}'. Allowed types: auto, passport, aadhaar, dl"}}
        )

    # Internal pipeline mapping
    if norm_doc_type in ("aadhaar", "aadhar"):
        internal_doc_type = "aadhar"
    elif norm_doc_type in ("dl", "driving_licence", "driving_license"):
        internal_doc_type = "dl"
    else:
        internal_doc_type = norm_doc_type

    # 4. Upload Content Validation
    try:
        content = image.file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": {"code": "INVALID_IMAGE", "message": "Failed to read uploaded image file"}}
        )

    validation_error = validate_image_content(content, image.filename or "file.jpg", image.content_type)
    if validation_error:
        return validation_error

    # 5. Save to Temporary File & Execute Pipeline
    ext = os.path.splitext(image.filename)[1].lower() if image.filename else ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".jpg"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = temp_file.name

    try:
        temp_file.write(content)
        temp_file.close()

        logger.info(f"Processing verification request for doc_type='{norm_doc_type}'")

        # Execute blocking pipeline inference (runs in FastAPI threadpool)
        raw_result = pipeline.predict_document_authenticity(
            img_path=temp_path,
            doc_type=internal_doc_type,
            doc_number=doc_number,
            mrz_line1=mrz_line1,
            mrz_line2=mrz_line2
        )

        # Normalize internal doc_type in output ("aadhar" -> "aadhaar")
        res_doc_type = raw_result.get("doc_type", norm_doc_type)
        if res_doc_type == "aadhar":
            res_doc_type = "aadhaar"

        response_payload = {
            "success": True,
            "filename": image.filename or "uploaded_image.jpg",
            "doc_type": res_doc_type,
            "verdict": raw_result.get("verdict", "INSUFFICIENT_IMAGE_QUALITY"),
            "risk_score": raw_result.get("risk_score", 0.0),
            "reasons": raw_result.get("reasons", []),
            "evidence_table": raw_result.get("evidence_table", {})
        }

        return response_payload

    except Exception as e:
        logger.error(f"Error during document prediction: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": {"code": "PIPELINE_ERROR", "message": "An error occurred while running prediction pipeline"}}
        )
    finally:
        # Mandatory temporary file deletion
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file '{temp_path}': {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
