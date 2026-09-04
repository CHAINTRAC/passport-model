import os
import numpy as np
from typing import List, Optional, Dict, Any
from forensics.base import BaseForensicDetector
from schemas.evidence import ForensicEvidence, ForensicStatus, EvidenceCategory, EvidenceLevel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class CNNClassifierDetector(BaseForensicDetector):
    """
    EfficientNetB0 Deep Learning Visual Feature Detector.
    Produces probabilistic visual anomaly score complementing deterministic verifiers.
    """

    def __init__(self, model_path: str = "model/fine_tuned_model_20.keras"):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_shape=(224, 224, 3)
            )
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            self.model = tf.keras.Model(inputs, outputs)

            import zipfile
            temp_dir = "scratch_temp"
            os.makedirs(temp_dir, exist_ok=True)
            weights_path = os.path.join(temp_dir, "model.weights.h5")
            if not os.path.exists(weights_path):
                with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                    zip_ref.extract('model.weights.h5', temp_dir)
            self.model.load_weights(weights_path)
        except Exception:
            self.model = None

    def analyze(
        self,
        image: Optional[np.ndarray],
        img_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        if not os.path.exists(img_path):
            return []

        if self.model is None:
            # Fallback neutral score
            return [
                ForensicEvidence(
                    type="cnn",
                    status=ForensicStatus.PASS,
                    score=0.20,
                    level=EvidenceLevel.WEAK,
                    category=EvidenceCategory.IMAGE,
                    region=None,
                    reason="CNN model unavailable; score set to neutral baseline"
                )
            ]

        try:
            img_raw = tf.io.read_file(img_path)
            img = tf.image.decode_image(img_raw, channels=3)
            img = tf.image.resize(img, (224, 224))
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            score = float(self.model.predict(img, verbose=0)[0][0])
            cnn_score = round(score, 4)

            # High score (>0.70) = high visual feature similarity to genuine documents
            # Low score (<0.35) = visual feature anomaly detected
            is_suspicious = (cnn_score < 0.35)
            status = ForensicStatus.SUSPICIOUS if is_suspicious else ForensicStatus.PASS
            level = EvidenceLevel.MODERATE if is_suspicious else EvidenceLevel.WEAK
            risk_score = round(1.0 - cnn_score, 2) if is_suspicious else round(1.0 - cnn_score, 2)

            return [
                ForensicEvidence(
                    type="cnn",
                    status=status,
                    score=risk_score,
                    level=level,
                    category=EvidenceCategory.IMAGE,
                    region=None,
                    reason=f"CNN visual feature score: {cnn_score:.4f}" + (" (low similarity)" if is_suspicious else " (high similarity)")
                )
            ]
        except Exception as e:
            return []
