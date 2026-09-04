
# 🛡️ Identity Document Authenticity & Fraud Detection Engine (v2.0)

A production-grade, multi-layered deep learning and forensic pipeline for verifying identity document authenticity and detecting counterfeits—specifically tailored for **Indian Passports**, **UIDAI Aadhaar Cards**, and **Indian Driving Licences (DL)**.

The system combines **Pre-Flight Quality Screening**, a **7-Tier Hierarchical Classification Engine**, **ICAO 9303 & Verhoeff Checksum Mathematics**, **8 Computer Vision Forensic Detectors**, a **Calibrated Multi-Category Risk Engine**, and a **Production FastAPI REST API** with strict **Pydantic v2 Schemas**.

---

## 🏛️ Repository Architecture

```
MIDV2020/
│
├── predict_pipeline.py           # Master production facade & CLI entry point
├── server.py                     # FastAPI REST server (Auth, CORS, Lifespan, Swagger docs)
├── main.py                       # CLI execution entry point
├── test_api.py                   # Automated API server integration tests
│
├── aggregation/                  # Risk aggregation & decision engine
│   ├── evidence_aggregator.py    # Master pipeline orchestrator & evidence table builder
│   └── risk_engine.py            # Calibrated 5-category risk scoring & 3-state decision engine
│
├── document_detection/           # Hierarchical Document Classification Engine
│   ├── detector.py               # 7-Tier hierarchical evidence accumulator & classifier
│   ├── ocr_features.py           # EasyOCR feature extraction (Text, MRZ, QR codes)
│   ├── geometry.py               # Card aspect ratio & contour boundary analysis
│   └── result.py                 # Classification result builder & confidence normalizer
│
├── verifiers/                    # Document-Specific Verification Engines
│   ├── base.py                   # Abstract base verifier interface
│   ├── passport.py               # Indian Passport verifier (ICAO 9303, Visual ↔ MRZ cross-match)
│   ├── aadhaar.py                # Aadhaar verifier (Verhoeff checksum, QR verification, ID-1 geometry)
│   └── driving_license.py        # Driving Licence verifier (Parivahan DL pattern & state codes)
│
├── forensics/                    # 8 Modular Computer Vision & Deep Learning Detectors
│   ├── base.py                   # Abstract forensic detector base class
│   ├── ela.py                    # Error Level Analysis (JPEG re-compression variance)
│   ├── cnn.py                    # Fine-tuned EfficientNetB0 visual feature classifier
│   ├── jpeg.py                    # Compression artifact & quantization table analyzer
│   ├── metadata.py               # EXIF & digital editing software metadata detector
│   ├── copy_move.py              # Copy-Move forgery & cloned region detector
│   ├── text_tampering.py         # Font inconsistency, size variance & baseline analyzer
│   ├── geometry_analysis.py      # Card edge rotation, shear & boundary alignment inspector
│   └── visual_anomaly.py         # Noise level, color distribution & visual artifact detector
│
├── validation/                   # Algorithmic Checksum & Rule Engine
│   ├── mrz.py                    # ICAO Doc 9303 7-3-1 weighted checkdigit validator
│   ├── verhoeff.py               # D5 Dihedral Group Verhoeff checksum validator
│   ├── document_number.py        # Regex validators (Passport, Aadhaar 12-digit, DL formats)
│   └── dates.py                  # Logical date validator (DOB, Expiry, Issue dates)
│
├── schemas/                      # Production Pydantic v2 Models
│   ├── response.py               # Response models (VerificationResponse, DecisionInfo, etc.)
│   ├── detection.py              # Document detection result & signal models
│   └── evidence.py               # Forensic evidence item & category models
│
├── model/                        # Deep Learning Model Artifacts
│   └── fine_tuned_model_20.keras # Fine-tuned Keras model weights
│
├── Dockerfile                    # Production Docker container definition
├── render.yaml                   # Cloud deployment manifest for Render
├── requirements.txt              # Production Python dependencies
├── .env.example                  # Environment configuration template
└── README.md                     # Project documentation
```

---

## 🔄 The 7-Layer Detection Workflow

The engine evaluates input document images through a sequential, 7-layer pipeline designed to maximize explainability and prevent false fraud verdicts caused by bad lighting or low resolution.

```
                     ┌──────────────────────────────────────────┐
                     │               INPUT IMAGE                │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Pre-Flight Image Quality Assessment                                      │
│ • Blur Check (Laplacian Variance >= 25.0)                                         │
│ • Minimum Resolution Check (>= 320x220 px)                                        │
│ • Overexposure & Glare Ratio Check (<= 65%)                                       │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │  [Quality Passed]
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: Hierarchical 7-Tier Document Type Classification Router                 │
│ • T1: Explicit Override │ T2: MRZ Lines │ T3: Doc Num Rules │ T4: OCR Keywords     │
│ • T5: UIDAI QR Code    │ T6: ID-1 Card Geometry            │ T7: Filename Hints   │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │  [Classified: Passport / Aadhaar / DL]
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: Deep OCR & Visual Feature Extraction Engine                              │
│ • GPU/CPU EasyOCR text extraction & bounding box localization                     │
│ • MRZ bottom zone detection & QR code decoding                                    │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: Specialized Rule-Based Document Verifiers                                │
│ • Indian Passport: ICAO 9303 7-3-1 checkdigits & Visual ↔ MRZ field cross-check  │
│ • Aadhaar Card: D5 Dihedral Group Verhoeff checksum & QR code payload verification│
│ • Driving Licence: Parivahan format regex, state code lookup & ID-1 geometry     │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: Modular Computer Vision & Deep Learning Forensic Suite                   │
│ 1. ELA (Error Level Analysis)          2. EfficientNetB0 CNN Classifier           │
│ 3. JPEG Quantization Analysis          4. EXIF & Software Metadata Inspection      │
│ 5. Copy-Move Forgery Detector          6. Text & Font Tampering Analyzer          │
│ 7. Card Geometry Anomaly Inspector     8. Visual Noise & Color Anomaly Detector   │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: Calibrated Multi-Category Risk & Fraud Engine                            │
│ Aggregates risk scores across 5 weighted categories:                              │
│ • Identity (35%) │ Image (25%) │ Structural (15%) │ Content (15%) │ Metadata (10%) │
│ Outputs Decision Status: GENUINE (<0.35) │ SUSPICIOUS (>=0.35) │ INCONCLUSIVE     │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 7: Pydantic v2 Schema Serialization & Production REST API / CLI Layer      │
│ • Serializes structured response with region bounding boxes & evidence summary    │
│ • OpenAPI / Swagger UI support, API Key security, Docker containerization        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detailed Layer Breakdown

### Layer 1: Pre-Flight Image Quality Screening
Before running heavy ML models, the image undergoes pre-flight assessment:
* **Laplacian Blur Variance**: Measures image focus. Blur scores `< 25.0` are flagged as severely blurry.
* **Minimum Resolution**: Enforces minimum dimensions of `320x220` pixels.
* **Surface Glare & Overexposure**: Detects camera flash hotspots where brightness exceeds `253` with low saturation. Images with glare exceeding `65%` are rejected.

### Layer 2: 7-Tier Hierarchical Document Type Router
Documents are dynamically routed without requiring manual labels via a 7-tier score accumulation model:
1. **Tier 1 (Explicit Request)**: Direct override if caller specifies `doc_type` (e.g. `passport`, `aadhaar`, `dl`).
2. **Tier 2 (MRZ Structure)**: Detects 2-line ICAO Doc 9303 Machine Readable Zone (`+0.55` weight for valid checksums, `+0.45` for MRZ zone match).
3. **Tier 3 (Document Number Validation)**: Validates number formats using pattern regex and Verhoeff math (`+0.50` weight for Aadhaar Verhoeff, `+0.40` for Passport pattern, `+0.45` for DL pattern).
4. **Tier 4 (OCR Semantic Keywords)**: Matches domain keywords (e.g. "REPUBLIC OF INDIA", "PASSPORT", "UNIQUE IDENTIFICATION AUTHORITY", "DRIVING LICENCE", "UNION OF INDIA") (`+0.45` to `+0.60` weight).
5. **Tier 5 (QR Code Detection)**: Identifies UIDAI QR code presence on Aadhaar cards (`+0.45` weight).
6. **Tier 6 (ISO/IEC 7810 ID-1 Geometry)**: Evaluates aspect ratio (1.35 to 1.75) for ID-1 card formats (`+0.10` to `+0.15` weight).
7. **Tier 7 (Filename & Path Hints)**: Lowest-priority fallback matching keywords in filename (`+0.10` weight).

### Layer 3: OCR & Feature Extraction Engine
Extracts full page text, key-value bounding boxes, MRZ character zones, and embedded QR codes using EasyOCR with GPU/CPU acceleration.

### Layer 4: Specialized Rule-Based Document Verifiers
* **Indian Passport Verifier**:
  - Validates ICAO 9303 7-3-1 checkdigits for Passport Number, Date of Birth, Date of Expiry, and Composite Check Digit.
  - Performs cross-matching between visual fields (Surname, Given Name, Passport Number, DOB, Sex, Expiry) and parsed MRZ strings.
* **Aadhaar Card Verifier**:
  - Validates 12-digit Aadhaar number using the **Verhoeff Checksum Algorithm** (D5 Dihedral Group mathematics), catching 100% of single-digit errors and >98% of adjacent transposition errors.
  - Performs multi-stage QR code inspection (detection, decodability, structure check).
* **Driving Licence Verifier**:
  - Validates Indian Parivahan Sarathi DL number patterns (e.g. `MH12 20150012345`) and state code lookup (MH, DL, KA, UP, TN, etc.).
  - Verifies ISO/IEC 7810 ID-1 standard card geometry aspect ratio.

### Layer 5: 8 Modular Computer Vision & Deep Learning Forensic Suite
Runs 8 independent forensic detectors, each generating structured `ForensicEvidence` items with bounding boxes:
1. **ELA (Error Level Analysis)**: Re-saves image at 95% JPEG quality and computes pixel-wise compression error variance (`> 350.0` signals digital manipulation/photo swapping).
2. **CNN Visual Classifier**: Fine-tuned EfficientNetB0 model (`model/fine_tuned_model_20.keras`) evaluating visual layout authenticity.
3. **JPEG Compression Analyzer**: Examines quantization tables and detects double-compression artifacts.
4. **Metadata & EXIF Inspector**: Scans file metadata for editing software signatures (e.g. Photoshop, GIMP, Canva, PicsArt).
5. **Copy-Move Forgery Detector**: Uses block-matching to detect cloned, duplicated regions within the image.
6. **Text Tampering Analyzer**: Scans for font inconsistencies, size variances, and text baseline misalignments.
7. **Geometry Analysis**: Detects card boundary rotation, perspective warping, shear, and irregular aspect ratios.
8. **Visual Anomaly Detector**: Identifies unnatural noise distributions, edge discontinuities, and color space anomalies.

### Layer 6: Calibrated Multi-Category Risk Engine
Aggregates findings into 5 weighted risk categories to calculate an overall `risk_score` (`0.0` to `1.0`):

| Risk Category | Weight | Primary Components Evaluated |
| :--- | :---: | :--- |
| **Identity Integrity** | **35%** | MRZ checkdigit math, Verhoeff algorithm, Visual ↔ MRZ cross-check, DL pattern |
| **Image Integrity** | **25%** | ELA variance score, CNN visual classifier, JPEG compression artifacts |
| **Structural Integrity**| **15%** | ISO/IEC 7810 ID-1 aspect ratio, edge shear, contour boundaries |
| **Content Integrity** | **15%** | Text font consistency, OCR alignment, text tampering indicators |
| **Metadata Integrity** | **10%** | EXIF tags, software modification history, metadata flags |

#### Decision Matrix:
* **`GENUINE`**: `risk_score < 0.35` and `confidence >= 0.60`
* **`SUSPICIOUS`**: `risk_score >= 0.35`
* **`INCONCLUSIVE`**: Image unreadable, classification confidence `< 0.40`, or insufficient evidence.

### Layer 7: Pydantic v2 Schemas & REST API
Presents findings via strict Pydantic v2 models supporting both modern structured data and backward-compatible legacy fields.

---

## 📊 Pydantic v2 Response Schema Contract

Every API call returns a standardized `VerificationResponse` object:

```json
{
  "success": true,
  "filename": "passport_sample.jpg",
  "document": {
    "type": "passport",
    "confidence": 0.95
  },
  "decision": {
    "status": "genuine",
    "risk_score": 0.12,
    "confidence": 0.95
  },
  "validation": {
    "mrz_checksum": {
      "status": "pass",
      "confidence": 0.99,
      "message": "MRZ checkdigits mathematically valid"
    },
    "layout": {
      "status": "pass",
      "confidence": 0.90,
      "message": "Document layout geometry consistent"
    }
  },
  "forensics": [
    {
      "type": "ela",
      "status": "pass",
      "score": 0.08,
      "level": "weak",
      "category": "image",
      "region": null,
      "reason": "Error Level Analysis variance (142.3) within normal limits."
    },
    {
      "type": "metadata",
      "status": "pass",
      "score": 0.0,
      "level": "weak",
      "category": "metadata",
      "region": null,
      "reason": "No digital editing software signatures detected."
    }
  ],
  "evidence_summary": {
    "strong": 0,
    "moderate": 0,
    "weak": 0
  },
  "doc_type": "passport",
  "verdict": "GENUINE",
  "risk_score": 0.12,
  "reasons": [],
  "evidence_table": {
    "quality_assessment": {
      "is_sufficient": true,
      "resolution": "1280x720"
    }
  }
}
```

---

## ⚡ Quick Start & Setup Guide

### 1. Installation

Clone the repository and install dependencies in a Python 3.10 environment:

```bash
# Clone repository
git clone https://github.com/your-username/MIDV2020.git
cd MIDV2020

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and set your configuration parameters:

```bash
cp .env.example .env
```

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `API_KEY` | Secret key required in `X-API-Key` header | `midv2020-secret-api-key-2026` |
| `MAX_UPLOAD_SIZE_MB` | Maximum allowed upload size (MB) | `10.0` |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | `""` (All origins allowed if empty) |
| `PORT` | Web server execution port | `8000` |

---

## 🖥️ Usage Guide

### A. Python API Usage

```python
from predict_pipeline import DocumentAuthenticityPipeline

# Initialize master pipeline
pipeline = DocumentAuthenticityPipeline()

# Process document verification
result = pipeline.predict_document_authenticity(
    img_path="sample/passport/sample_passport.jpg",
    doc_type="auto",  # Options: "auto", "passport", "aadhaar", "dl"
    doc_number="Z1234567",
    mrz_line1="P<INDTEST<<SAMPLE<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    mrz_line2="Z1234567<4IND9001011M3001017<<<<<<<<<<<<<<04"
)

print("Document Type :", result['document']['type'])
print("Decision Status:", result['decision']['status'])
print("Risk Score    :", result['decision']['risk_score'])
```

### B. Command Line Interface (CLI)

```bash
# Evaluate a single document image
python predict_pipeline.py --image sample/passport/sample_passport.jpg --doc-type passport

# Process batch directory of Driving Licenses
python predict_pipeline.py --dir sample/dl --doc-type dl --output output/dl_results
```

### C. FastAPI REST API Server

Start the REST API server locally:

```bash
python server.py
# Or using uvicorn CLI:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger Documentation will be available at: **`http://localhost:8000/docs`**

#### API Endpoints Table

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/` | API Metadata & Route Sitemap | No |
| `GET` | `/health` | Vitality Health Check (`{"status": "healthy"}`) | No |
| `GET` | `/ready` | Readiness Check (Pipeline & ML Model loaded) | No |
| `POST` | `/api/v1/verify` | Document Authenticity & Fraud Verification | **Yes (`X-API-Key`)** |

#### cURL Request Example

```bash
curl -X POST "http://localhost:8000/api/v1/verify" \
  -H "X-API-Key: midv2020-secret-api-key-2026" \
  -F "image=@sample/passport/sample_passport.jpg" \
  -F "doc_type=auto" \
  -F "doc_number=Z1234567"
```

#### Python (`requests`) Integration Example

```python
import requests

API_URL = "http://localhost:8000/api/v1/verify"
HEADERS = {"X-API-Key": "midv2020-secret-api-key-2026"}

data = {
    "doc_type": "auto",
    "doc_number": "Z1234567"
}

with open("sample/passport/sample_passport.jpg", "rb") as f:
    files = {"image": ("passport.jpg", f, "image/jpeg")}
    response = requests.post(API_URL, headers=HEADERS, data=data, files=files)

res_json = response.json()
print("Success:", res_json["success"])
print("Document:", res_json["document"]["type"])
print("Decision:", res_json["decision"]["status"])
print("Risk Score:", res_json["decision"]["risk_score"])
```

---

## 🐳 Containerization & Cloud Deployment

### Docker Deployment

Build and run using Docker:

```bash
# Build Docker image
docker build -t document-authenticity-api .

# Run container
docker run -p 8000:8000 -e API_KEY="your-production-secret-key" document-authenticity-api
```

### Cloud Deployment (Render / Railway / AWS App Runner)

This repository includes a `render.yaml` blueprint configuration. To deploy on Render:
1. Connect this repository to your Render Dashboard.
2. Select **Web Service** (Docker Runtime).
3. Render automatically picks up `Dockerfile` and `render.yaml`.
4. Define your production `API_KEY` in environment variables.

---

## 🧪 Testing

Run automated API and verifier tests using `pytest`:

```bash
pytest tests/ -v
# Or run API integration tests directly:
python test_api.py
```

---

## 📜 License & Acknowledgments

* Designed for production identity document authenticity prediction and fraud detection.
* Built using OpenCV, EasyOCR, TensorFlow / Keras, FastAPI, and Pydantic v2.
