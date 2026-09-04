from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class EvidenceCategory(str, Enum):
    IDENTITY = "identity"
    STRUCTURAL = "structural"
    IMAGE = "image"
    CONTENT = "content"
    METADATA = "metadata"


class EvidenceLevel(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class ForensicStatus(str, Enum):
    PASS = "pass"
    SUSPICIOUS = "suspicious"
    INFO = "info"


class ForensicEvidence(BaseModel):
    type: str = Field(..., description="Detector type name (e.g., ela, jpeg, cnn, metadata, text_tampering, copy_move)")
    status: ForensicStatus = Field(ForensicStatus.PASS, description="Status outcome: pass, suspicious, info")
    score: float = Field(0.0, description="Risk or anomaly score between 0.0 and 1.0")
    level: EvidenceLevel = Field(EvidenceLevel.WEAK, description="Evidence significance level: strong, moderate, weak")
    category: EvidenceCategory = Field(EvidenceCategory.IMAGE, description="Evidence category")
    region: Optional[List[int]] = Field(None, description="Bounding box coordinates [x, y, width, height]")
    reason: str = Field(..., description="Human-readable explanation of findings")
