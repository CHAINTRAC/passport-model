# Identity Document Authenticity & Fraud Detection Engine

A multi-layered production pipeline for predicting document authenticity and detecting counterfeit identity documents (Indian Passports and UIDAI Aadhaar Cards) using Deep Learning, Forensic Analysis, Algorithmic Checksums, and Visual-to-MRZ Cross-Matching.

---

## 🏛️ Repository Architecture

```
MIDV2020/
│
├── predict_pipeline.py           # Master production authenticity pipeline & CLI entry point
├── indian_passport_verifier.py   # Indian Passport verification engine (ICAO 9303, Visual ↔ MRZ cross-match)
├── aadhaar_verifier.py           # Aadhaar verification engine (Verhoeff checksum, ID-1 geometry, QR analysis)
│
├── model/                        # Saved CNN models & weights (fine_tuned_model_20.keras)
├── dataset/                      # Dataset files & image frames
├── sample/                       # Sample test image folders (passport/, aadhar/)
├── sample_output/                # Prediction CSV summaries & verdict classification directories
│
├── archive/                      # Historical research & experiment scripts
│   ├── experiments/              # Archived test scripts (test_sample.py, explain_gradcam.py, etc.)
│   └── notebooks/                # Archived research notebooks (SIDTD.ipynb)
│
└── README.md                     # Project documentation
```

---

## 🔄 End-to-End Pipeline Architecture

```
                          INPUT IMAGE
                               │
                               ▼
              Pre-Flight Image Quality Assessment
             (Blur / Resolution / Glare Hotspots)
                               │
                ┌──────────────┴──────────────┐
                │ Low Quality                 │ Sufficient Quality
                ▼                             ▼
   INSUFFICIENT_IMAGE_QUALITY       Document Type Router
                                     /            \
                                Passport         Aadhaar
                                   │                │
                         ┌─────────┼─────────┐      ├──────────────┐
                         │         │         │      │              │
                        CNN       ELA       MRZ    CNN            ELA
                         │         │         │      │              │
                         │         │    ICAO 9303   │       Verhoeff (D5)
                         │         │    Checkdigits │         + ID-1 Ratio
                         │         │         │      │              │
                         │         │    Visual ↔    │           Multi-Stage
                         │         │    MRZ Match   │           QR Check
                         └─────────┴────┬────┘      └──────┬───────┘
                                        │                  │
                                  Document-Specific Validation
                                        │                  │
                                        └─────────┬────────┘
                                                  ▼
                                         Evidence Aggregator
                                                  │
                                ┌─────────────────┼─────────────────┐
                                ▼                 ▼                 ▼
                             GENUINE          SUSPICIOUS          FAKE
```

---

## 🔑 Key Verification Modules & Rules

### 1. Pre-Flight Image Quality Assessment
- **Laplacian Blur Variance**: Flags images with blur score `< 25.0`.
- **Minimum Resolution**: Requires minimum `320x220` pixels.
- **Overexposure & Glare**: Identifies destructive flash glare hotspots.
- **Outcome**: Returns `INSUFFICIENT_IMAGE_QUALITY` if quality thresholds fail, preventing photograph defects from being misclassified as fraud.

### 2. Deep Learning Visual Feature Classifier (CNN)
- **Model Architecture**: Fine-tuned EfficientNetB0 (`model/fine_tuned_model_20.keras`).
- **Score Representation**: Outputs `cnn_score` (`0.0` - `1.0`) representing layout visual similarity.

### 3. Forensic Engine (ELA)
- **Error Level Analysis**: Evaluates compression variance (`ela_variance`) across re-saved JPEG layers.
- **Forensic Evidence**: High ELA variance (`> 350.0`) acts as supporting evidence for digital editing and photo swapping.

### 4. Indian Passport Branch
- **ICAO 9303 Checksum Engine**: Validates weighted 7-3-1 check digits for Passport Number, Date of Birth, Expiry Date, and Composite Check Digit.
- **Visual ↔ MRZ Cross-Matching**: Compares visual text fields against parsed MRZ data (Passport #, Surname, Given Name, Nationality, DOB, Sex, Expiry).
- **Non-Punitive Checksum Handling**: OCR or checksum failure outputs `MRZ_INVALID_OR_OCR_UNCERTAIN` signal rather than immediate binary fake.
- **Layout Geometry**: Evaluates bottom MRZ zone baseline density. *(ID-1 card ratio check is removed for passports)*.

### 5. Aadhaar Card Branch
- **ISO/IEC 7810 ID-1 Geometry**: Enforces standard ID-1 card aspect ratio (`1.50` - `1.70`).
- **Verhoeff Algorithm**: D5 Dihedral Group multiplication, permutation, and inverse tables for 12-digit Aadhaar number validation (`[2-9][0-9]{11}`).
- **Multi-Stage QR Inspection**: Evaluates `qr_detected`, `qr_readable`, `qr_format_valid`, and `qr_content_consistent`. Missing or blurry QR codes generate an `UNVERIFIED` flag, not an automatic fake verdict.

### 6. Evidence Aggregation & 4 Verdict States
- **Evidence Table**: Logs every individual signal for full explainability.
- **Final Verdicts**:
  1. `GENUINE`: Verified layout, valid checksums, and low risk score (`< 0.25`).
  2. `SUSPICIOUS`: Minor mismatches or unverified checksums (`0.25 <= risk < 0.55`).
  3. `FAKE`: Severe field mismatches, failed Verhoeff/MRZ checksums, or high risk (`>= 0.55`).
  4. `INSUFFICIENT_IMAGE_QUALITY`: Image unreadable, severely blurry, or cropped.

---

## 🚀 Usage Guide

### Command Line Interface (CLI)

#### Batch Directory Evaluation
```bash
# Evaluate passport directory
python predict_pipeline.py --dir sample/passport --doc-type passport --output sample_output/passport_results

# Evaluate Aadhaar directory
python predict_pipeline.py --dir sample/aadhar --doc-type aadhar --output sample_output/aadhar_results
```

#### Single Image Prediction
```bash
python predict_pipeline.py --image sample/passport/download.jpg --doc-type passport
```

### Python API Integration
```python
from predict_pipeline import DocumentAuthenticityPipeline

pipeline = DocumentAuthenticityPipeline()

# Predict authenticity of a single document
report = pipeline.predict_document_authenticity(
    img_path="sample/passport/download.jpg",
    doc_type="passport",
    doc_number="Z1234567",
    mrz_line1="P<INDSINGH<<GURPREET<<<<<<<<<<<<<<<<<<<<<<<<",
    mrz_line2="Z1234567<1IND8501011M3001018<<<<<<<<<<<<<<02"
)

print("Verdict    :", report['verdict'])
print("Risk Score :", report['risk_score'])
print("Reasons    :", report['reasons'])
print("Evidence   :", report['evidence_table'])
```

---

## ⚡ FastAPI Production Web Server

The repository includes a ready-to-run FastAPI REST API server (`server.py`) with header-based API key security (`X-API-Key`), multi-part image uploads, upload file validation, and automatic lifespan model initialization.

### 1. Running the API Server

```bash
# Start server with Uvicorn (loads model once on startup)
python server.py

# Or directly with Uvicorn CLI:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at: **`http://localhost:8000/docs`**

### 2. Environment Variables Configuration

| Variable | Description | Default |
| :--- | :--- | :--- |
| `API_KEY` | Secret key required in `X-API-Key` header | `midv2020-secret-api-key-2026` |
| `MAX_UPLOAD_SIZE_MB` | Maximum allowed file upload size in MB | `10.0` |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `""` (CORS disabled if empty) |

### 3. API Endpoints

- **`GET /`**: API Metadata and routing URLs.
- **`GET /health`**: Vitality health check (`{"status": "healthy"}`).
- **`GET /ready`**: Readiness health check verifying model is loaded.
- **`POST /api/v1/verify`**: Document Verification endpoint (`multipart/form-data`).

### 4. Integration Examples

#### cURL Command
```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "X-API-Key: midv2020-secret-api-key-2026" \
  -F "image=@sample/passport/sample_passport.jpg" \
  -F "doc_type=passport" \
  -F "doc_number=Z1234567" \
  -F "mrz_line1=P<INDTEST<<SAMPLE<<<<<<<<<<<<<<<<<<<<<<<<<<<" \
  -F "mrz_line2=Z1234567<4IND9001011M3001017<<<<<<<<<<<<<<04"
```

#### Python (`requests`) Backend Integration
```python
import requests

API_URL = "http://localhost:8000/api/v1/verify"
API_KEY = "midv2020-secret-api-key-2026"

headers = {"X-API-Key": API_KEY}
data = {
    "doc_type": "passport",
    "doc_number": "Z1234567"
}

with open("path/to/passport_image.jpg", "rb") as img_file:
    files = {"image": ("passport.jpg", img_file, "image/jpeg")}
    response = requests.post(API_URL, headers=headers, data=data, files=files)

res_json = response.json()
print("Success:", res_json["success"])
print("Verdict:", res_json["verdict"])
print("Risk Score:", res_json["risk_score"])
print("Evidence Table:", res_json["evidence_table"])
```

### 5. HTTP Error Codes Contract

| Status Code | Error Code | Cause |
| :--- | :--- | :--- |
| `401 Unauthorized` | `MISSING_API_KEY` | `X-API-Key` header missing |
| `403 Forbidden` | `INVALID_API_KEY` | Invalid API Key provided |
| `400 Bad Request` | `INVALID_DOC_TYPE` | `doc_type` not in `[auto, passport, aadhaar]` |
| `400 Bad Request` | `INVALID_IMAGE` | Corrupt or undecodable image file |
| `413 Payload Too Large` | `FILE_TOO_LARGE` | File exceeds `MAX_UPLOAD_SIZE_MB` |
| `415 Unsupported Media` | `UNSUPPORTED_FILE_TYPE` | Non-image file format uploaded |
| `503 Service Unavailable` | `MODEL_UNAVAILABLE` | Pipeline or TensorFlow model not loaded |
| `500 Internal Error` | `PIPELINE_ERROR` | Internal server prediction exception |
```
