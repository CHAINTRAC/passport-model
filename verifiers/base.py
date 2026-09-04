from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from schemas.response import ValidationItem


class BaseDocumentVerifier(ABC):
    """
    Abstract interface for document-specific domain verifiers.
    Standardized signature: verify(image, ocr_result, metadata) -> Dict[str, ValidationItem]
    """

    @abstractmethod
    def verify(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        ocr_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, ValidationItem]:
        pass
