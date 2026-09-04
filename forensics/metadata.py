import os
import numpy as np
from PIL import Image, ExifTags
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class MetadataAnalysisDetector(BaseForensicDetector):
    """
    EXIF Metadata & Software Editing Tag Detector.
    Scans image EXIF tags for signatures of editing applications (Adobe Photoshop, GIMP, Canva, Paint.NET).
    """

    SUSPICIOUS_SOFTWARE = [
        "photoshop", "gimp", "canva", "paint.net", "adobe",
        "pixlr", "snapseed", "lightroom", "editor", "photofiltre"
    ]

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if not os.path.exists(img_path):
            return []

        try:
            with Image.open(img_path) as img:
                exif_data = img._getexif()
                if not exif_data:
                    return []

                software_tag = None
                for tag_id, val in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, "")
                    if tag_name.lower() == "software":
                        software_tag = str(val)
                        break

                if software_tag:
                    soft_lower = software_tag.lower()
                    is_editing_tool = any(tool in soft_lower for tool in self.SUSPICIOUS_SOFTWARE)
                    if is_editing_tool:
                        return [
                            ForensicEvidence(
                                type="metadata",
                                status=ForensicStatus.SUSPICIOUS,
                                score=0.75,
                                level=EvidenceLevel.STRONG,
                                category=EvidenceCategory.METADATA,
                                region=None,
                                reason=f"EXIF metadata indicates image created/edited with software: '{software_tag}'"
                            )
                        ]
        except Exception:
            pass

        return []
