from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DetectionSignal(BaseModel):
    name: str = Field(..., description="Name of the signal, e.g. MRZ_VALID, VERHOEFF_VALID, QR_CODE_DETECTED")
    target_doc_type: str = Field(..., description="Document type targeted by signal, e.g. passport, aadhaar, dl")
    weight: float = Field(..., description="Score contribution weight")
    details: str = Field("", description="Detailed explanation of the signal")


class DocumentTypeDetectionResult(BaseModel):
    doc_type: str = Field("unknown", description="Classified document type: passport, aadhaar, dl, or unknown")
    confidence: float = Field(0.0, description="Classification confidence score (0.0 to 1.0)")
    signals: List[DetectionSignal] = Field(default_factory=list, description="Accumulated detection signals")
    scores: Dict[str, float] = Field(
        default_factory=lambda: {"passport": 0.0, "aadhaar": 0.0, "dl": 0.0},
        description="Accumulated evidence scores per document type"
    )
    is_confident: bool = Field(False, description="True if confidence exceeds classification threshold")
