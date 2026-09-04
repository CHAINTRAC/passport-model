from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from schemas.evidence import ForensicEvidence


class BaseForensicDetector(ABC):
    """
    Abstract interface for independent forensic detectors.
    Outputs structured ForensicEvidence objects with location bounding box [x, y, w, h].
    """

    @abstractmethod
    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        pass
