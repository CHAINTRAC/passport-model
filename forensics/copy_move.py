import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel


class CopyMoveDetector(BaseForensicDetector):
    """
    Copy-Move Forgery Detector using ORB feature keypoints and distance matching.
    Identifies cloned/duplicated text or image patches within the same document.
    """

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if image is None or image.size == 0:
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) < 20:
                return []

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = matcher.knnMatch(descriptors, descriptors, k=2)

            cloned_pairs = []
            for m, n in matches:
                # Filter out self-matches and apply Lowe's ratio test
                if m.distance < 0.6 * n.distance and m.queryIdx != m.trainIdx:
                    pt1 = keypoints[m.queryIdx].pt
                    pt2 = keypoints[m.trainIdx].pt
                    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    if dist > 30:  # Distance threshold to ignore adjacent pixel noise
                        cloned_pairs.append((pt1, pt2))

            has_cloning = len(cloned_pairs) >= 8
            if has_cloning:
                # Compute bounding box covering the cloned points
                pts = [p for pair in cloned_pairs for p in pair]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                region = [min_x, min_y, max_x - min_x, max_y - min_y]

                return [
                    ForensicEvidence(
                        type="copy_move",
                        status=ForensicStatus.SUSPICIOUS,
                        score=0.82,
                        level=EvidenceLevel.STRONG,
                        category=EvidenceCategory.IMAGE,
                        region=region,
                        reason=f"Copy-move forgery detected: {len(cloned_pairs)} duplicate keypoint feature pairs found"
                    )
                ]
        except Exception:
            pass

        return []
