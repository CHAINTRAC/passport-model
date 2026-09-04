from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from schemas.evidence import ForensicEvidence


class DecisionStatus(str, Enum):
    GENUINE = "genuine"
    SUSPICIOUS = "suspicious"
    INCONCLUSIVE = "inconclusive"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_DOCUMENT = "unsupported_document"


class DocumentInfo(BaseModel):
    type: str = Field("unknown", description="Classified document type: passport, aadhaar, dl, or unknown")
    confidence: float = Field(0.0, description="Classification confidence (0.0 to 1.0)")


class DecisionInfo(BaseModel):
    status: DecisionStatus = Field(DecisionStatus.INCONCLUSIVE, description="Fraud decision status")
    risk_score: float = Field(0.0, description="Aggregated risk score (0.0 to 1.0)")
    confidence: float = Field(0.0, description="Decision confidence level")


class ValidationItem(BaseModel):
    status: str = Field("pass", description="pass, fail, unverified, or not_applicable")
    confidence: Optional[float] = Field(None, description="Optional validation confidence")
    message: Optional[str] = Field(None, description="Explanation message")


class EvidenceSummary(BaseModel):
    strong: int = Field(0, description="Count of strong forensic indicators")
    moderate: int = Field(0, description="Count of moderate forensic indicators")
    weak: int = Field(0, description="Count of weak forensic indicators")


class VerificationResponse(BaseModel):
    success: bool = Field(True, description="Request execution success status")
    filename: str = Field("uploaded_image.jpg", description="Original filename")
    document: DocumentInfo = Field(..., description="Document classification results")
    decision: DecisionInfo = Field(..., description="Fraud risk decision")
    validation: Dict[str, ValidationItem] = Field(default_factory=dict, description="Document-specific validation results")
    forensics: List[ForensicEvidence] = Field(default_factory=list, description="Forensic evidence items with region bounding boxes")
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary, description="Summary count of evidence signals")
    # Backward compatibility fields
    doc_type: str = Field("unknown", description="Legacy doc_type mapping")
    verdict: str = Field("INCONCLUSIVE", description="Legacy verdict mapping")
    risk_score: float = Field(0.0, description="Legacy risk_score mapping")
    reasons: List[str] = Field(default_factory=list, description="Legacy list of reason messages")
    evidence_table: Dict[str, Any] = Field(default_factory=dict, description="Legacy evidence table summary")


class ErrorDetail(BaseModel):
    code: str = Field(..., json_schema_extra={"example": "INVALID_IMAGE"})
    message: str = Field(..., json_schema_extra={"example": "Uploaded file could not be decoded as an image."})


class ErrorResponse(BaseModel):
    success: bool = Field(False, json_schema_extra={"example": False})
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str = Field("healthy", json_schema_extra={"example": "healthy"})


class ReadyResponse(BaseModel):
    status: str = Field("ready", json_schema_extra={"example": "ready"})
    model_loaded: bool = Field(True, json_schema_extra={"example": True})
    pipeline_ready: bool = Field(True, json_schema_extra={"example": True})


class RootResponse(BaseModel):
    service: str = Field("Document Authenticity API", json_schema_extra={"example": "Document Authenticity API"})
    version: str = Field("2.0.0", json_schema_extra={"example": "2.0.0"})
    docs: str = Field("/docs", json_schema_extra={"example": "/docs"})
    health: str = Field("/health", json_schema_extra={"example": "/health"})
    ready: str = Field("/ready", json_schema_extra={"example": "/ready"})
    verify: str = Field("/api/v1/verify", json_schema_extra={"example": "/api/v1/verify"})

