import os
import sys
import re
import cv2
import zipfile
import argparse
import shutil
import numpy as np
import pandas as pd
from PIL import Image

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from indian_passport_verifier import IndianPassportVerifier
from aadhaar_verifier import AadhaarVerifier


class DocumentAuthenticityPipeline:
    """
    Production Document Authenticity & Fraud Detection Pipeline
    Integrates multi-layered evidence aggregation:
    1. Input Quality & Pre-Flight Assessor (Blur, Resolution, Glare)
    2. Document Type Router (Passport vs. Aadhaar)
    3. Deep Learning Visual Feature Classifier (EfficientNetB0 cnn_score)
    4. Forensic Engine (Error Level Analysis ela_anomaly)
    5. Document-Specific Rule Engines:
       - Passport: ICAO 9303 Checkdigits, Visual ↔ MRZ Cross-Matching, Page Geometry (No ID-1 ratio)
       - Aadhaar: Verhoeff Checksum (D5 Dihedral Group), ID-1 Geometry, Multi-Stage QR Inspection
    6. Evidence Aggregator & Explainable Verdict System (GENUINE, SUSPICIOUS, FAKE, INSUFFICIENT_IMAGE_QUALITY)
    """

    def __init__(self, model_path="model/fine_tuned_model_20.keras"):
        self.model_path = model_path
        self.model = None
        self.passport_verifier = IndianPassportVerifier()
        self.aadhaar_verifier = AadhaarVerifier()
        self._load_cnn_model()

    def _load_cnn_model(self):
        """Re-builds EfficientNetB0 architecture and loads fine-tuned weights."""
        if not os.path.exists(self.model_path):
            print(f"[WARNING] Model file '{self.model_path}' not found. CNN score will fallback to neutral (0.50).")
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

            # Weight extraction
            temp_dir = "scratch_temp"
            os.makedirs(temp_dir, exist_ok=True)
            weights_path = os.path.join(temp_dir, "model.weights.h5")

            if not os.path.exists(weights_path):
                with zipfile.ZipFile(self.model_path, 'r') as zip_ref:
                    zip_ref.extract('model.weights.h5', temp_dir)

            self.model.load_weights(weights_path)
            print("[INFO] EfficientNetB0 CNN weights loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load CNN weights ({str(e)}). Falling back to neutral CNN score.")
            self.model = None

    def assess_image_quality(self, img_path):
        """
        Pre-Flight Quality Assessment:
        - Image load check
        - Blur detection via Laplacian variance
        - Minimum resolution check
        - Overexposure / Glare check
        """
        reasons = []
        if not os.path.exists(img_path):
            return {'is_sufficient': False, 'reasons': ['File does not exist']}

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {'is_sufficient': False, 'reasons': ['Unreadable image file']}

        h, w, c = img_bgr.shape
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Blur Detection
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurry = blur_score < 25.0
        if is_blurry:
            reasons.append(f"Severe image blur detected (Blur Score: {blur_score:.1f} < 25.0)")

        # 2. Resolution Check
        is_low_res = (w < 320 or h < 220)
        if is_low_res:
            reasons.append(f"Low image resolution ({w}x{h} < 320x220 minimum threshold)")

        # 3. Glare Check (differentiating white document margins from destructive flash glare)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        glare_ratio = float(np.sum((gray > 253) & (sat < 20))) / float(gray.size)
        has_severe_glare = glare_ratio > 0.65
        if has_severe_glare:
            reasons.append(f"High surface glare / overexposure detected ({glare_ratio*100:.1f}% hotspot pixels)")

        is_sufficient = not (is_blurry or is_low_res or has_severe_glare)

        return {
            'is_sufficient': is_sufficient,
            'blur_score': round(blur_score, 2),
            'resolution': f"{w}x{h}",
            'glare_ratio': round(glare_ratio, 3),
            'reasons': reasons
        }

    def predict_cnn_score(self, img_path):
        """Computes CNN visual similarity feature score (0.0 to 1.0)."""
        if self.model is None:
            return 0.50
        try:
            img_raw = tf.io.read_file(img_path)
            img = tf.image.decode_image(img_raw, channels=3)
            img = tf.image.resize(img, (224, 224))
            img = tf.cast(img, tf.float32)
            img = tf.keras.applications.efficientnet.preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            score = float(self.model.predict(img, verbose=0)[0][0])
            return round(score, 4)
        except Exception:
            return 0.50

    def predict_document_authenticity(
        self,
        img_path,
        doc_type="auto",
        doc_number=None,
        mrz_line1=None,
        mrz_line2=None,
        visual_fields=None
    ):
        """
        Master Pipeline Method:
        Runs quality check, document type routing, CNN score extraction, ELA forensic analysis,
        document-specific checksums, and compiles a transparent Evidence Aggregation Table.
        """
        report = {
            'filename': os.path.basename(img_path),
            'img_path': img_path,
            'doc_type': doc_type,
            'verdict': 'GENUINE',
            'risk_score': 0.0,
            'evidence_table': {},
            'reasons': []
        }

        # Step 1: Pre-Flight Image Quality Assessment
        quality_res = self.assess_image_quality(img_path)
        if not quality_res['is_sufficient']:
            report['verdict'] = 'INSUFFICIENT_IMAGE_QUALITY'
            report['reasons'] = quality_res['reasons']
            report['evidence_table']['quality_assessment'] = quality_res
            return report

        report['evidence_table']['quality_assessment'] = quality_res

        # Step 2: Auto-Detect Document Type if requested
        if doc_type.lower() == "auto":
            if mrz_line1 or mrz_line2 or "passport" in img_path.lower():
                doc_type = "passport"
            elif "aadhar" in img_path.lower() or "aadhaar" in img_path.lower():
                doc_type = "aadhar"
            else:
                doc_type = "passport"  # Default fallback
        report['doc_type'] = doc_type.lower()

        # Step 3: CNN Visual Feature Classification
        cnn_score = self.predict_cnn_score(img_path)
        report['evidence_table']['cnn_score'] = cnn_score

        # Step 4: Forensic ELA Anomaly Score
        ela_res = self.passport_verifier.compute_ela_score(img_path) if doc_type == "passport" else self.aadhaar_verifier.compute_ela_tampering_score(img_path)
        report['evidence_table']['ela_forensics'] = ela_res

        # Risk Score Initialization
        risk_score = 0.0
        reasons = []

        # CNN Feature Penalty / Reward
        if cnn_score < 0.30:
            risk_score += 0.35
            reasons.append(f"Low CNN visual similarity score ({cnn_score:.4f})")
        elif cnn_score < 0.50:
            risk_score += 0.15
            reasons.append(f"Moderate CNN visual similarity score ({cnn_score:.4f})")

        # ELA Supporting Forensic Penalty
        ela_var = ela_res.get('ela_variance_score', ela_res.get('ela_variance', 0.0))
        if ela_var > 350.0:
            risk_score += 0.20
            reasons.append(f"Elevated ELA compression variance ({ela_var:.1f} > 350.0 - possible digital editing)")

        # Step 5: Document-Specific Engines
        if doc_type == "passport":
            # Geometry check (MRZ zone & layout)
            geom_res = self.passport_verifier.analyze_passport_geometry(img_path)
            report['evidence_table']['passport_geometry'] = geom_res
            if not geom_res['valid_layout']:
                risk_score += 0.20
                reasons.append("Weak or missing MRZ zone geometry on passport page")

            # MRZ Checksum Validation
            if mrz_line1 and mrz_line2:
                mrz_res = self.passport_verifier.validate_mrz_lines(mrz_line1, mrz_line2)
                report['evidence_table']['mrz_checksums'] = mrz_res
                if not mrz_res['valid_mrz']:
                    risk_score += 0.25
                    reasons.append(f"MRZ Checksum invalid or OCR uncertain ({mrz_res['status']})")

                # Visual ↔ MRZ Cross-Matching
                if visual_fields:
                    cross_res = self.passport_verifier.cross_validate_visual_and_mrz(visual_fields, mrz_res)
                    report['evidence_table']['visual_mrz_cross_match'] = cross_res
                    if cross_res['match_status'] == 'MISMATCH':
                        risk_score += 0.40  # Heavy penalty for field mismatch
                        reasons.append(f"Visual text ↔ MRZ field mismatch: {', '.join(cross_res['mismatches'])}")

            # Passport Number Pattern Check
            if doc_number:
                fmt_res = self.passport_verifier.validate_passport_number_format(doc_number)
                report['evidence_table']['number_format'] = fmt_res
                if not fmt_res['valid_format']:
                    risk_score += 0.25
                    reasons.append(f"Invalid passport number format ('{doc_number}')")

        elif doc_type == "aadhar":
            # Geometry Check: ISO/IEC 7810 ID-1 standard aspect ratio
            geom_res = self.aadhaar_verifier.verify_id1_geometry(img_path)
            report['evidence_table']['id1_geometry'] = geom_res
            if not geom_res['valid_id1_format']:
                risk_score += 0.25
                reasons.append(f"Non-standard card aspect ratio ({geom_res['aspect_ratio']} - expected ID-1 1.50-1.70)")

            # Verhoeff Checksum & Number Format
            if doc_number:
                num_res = self.aadhaar_verifier.validate_aadhaar_number_format(doc_number)
                report['evidence_table']['verhoeff_validation'] = num_res
                if not num_res['valid_aadhaar']:
                    risk_score += 0.35
                    reasons.append(f"Invalid Aadhaar number or failed Verhoeff checksum ({num_res['status']})")

            # Multi-Stage QR Code Inspection
            qr_res = self.aadhaar_verifier.detect_and_read_qr_code(img_path)
            report['evidence_table']['qr_code_analysis'] = qr_res
            if qr_res['status'] == 'QR_UNVERIFIED_OR_DEGRADED':
                risk_score += 0.10  # Minor supporting flag, NOT automatic fake
                reasons.append("QR code missing or unverified (supporting evidence)")

        # Step 6: Verdict Determination based on Aggregated Risk
        report['risk_score'] = round(min(risk_score, 1.0), 2)
        report['reasons'] = reasons

        if report['risk_score'] >= 0.55:
            report['verdict'] = 'FAKE'
        elif report['risk_score'] >= 0.25:
            report['verdict'] = 'SUSPICIOUS'
        else:
            report['verdict'] = 'GENUINE'

        return report

    def process_directory(self, input_dir, doc_type="auto", output_dir="sample_output"):
        """Runs batch evaluation on all images in a directory and exports structured reports."""
        os.makedirs(output_dir, exist_ok=True)
        genuine_dir = os.path.join(output_dir, "genuine")
        suspicious_dir = os.path.join(output_dir, "suspicious")
        fake_dir = os.path.join(output_dir, "fake")
        quality_dir = os.path.join(output_dir, "insufficient_quality")

        for d in [genuine_dir, suspicious_dir, fake_dir, quality_dir]:
            os.makedirs(d, exist_ok=True)

        image_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        results = []
        print("\n" + "=" * 115)
        print(f"       DOCUMENT AUTHENTICITY EVALUATION PIPELINE: '{input_dir}' ({len(image_files)} IMAGES)")
        print("=" * 115)
        print(f"{'FILENAME':<38} | {'TYPE':<10} | {'CNN SCORE':<10} | {'RISK':<6} | {'VERDICT':<22} | {'PRIMARY REASON'}")
        print("=" * 115)

        for f in image_files:
            img_path = os.path.join(input_dir, f)
            res = self.predict_document_authenticity(img_path, doc_type=doc_type)

            verdict = res['verdict']
            cnn_score = res['evidence_table'].get('cnn_score', 0.0)
            risk = res['risk_score']
            primary_reason = res['reasons'][0] if res['reasons'] else "Valid layout & verifiable security features"

            # Copy to corresponding verdict folder
            if verdict == 'GENUINE':
                dest = genuine_dir
            elif verdict == 'SUSPICIOUS':
                dest = suspicious_dir
            elif verdict == 'FAKE':
                dest = fake_dir
            else:
                dest = quality_dir
            
            try:
                if os.path.exists(img_path):
                    shutil.copy2(img_path, os.path.join(dest, f))
            except Exception:
                pass

            results.append({
                'filename': f,
                'doc_type': res['doc_type'],
                'cnn_score': cnn_score,
                'risk_score': risk,
                'verdict': verdict,
                'primary_reason': primary_reason,
                'all_reasons': " | ".join(res['reasons'])
            })

            safe_f = f.encode('ascii', 'replace').decode('ascii')
            print(f"{safe_f:<38} | {res['doc_type'].upper():<10} | {cnn_score:<10.4f} | {risk:<6.2f} | {verdict:<22} | {primary_reason[:35]}")

        print("=" * 115)

        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "authenticity_summary_report.csv")
        df.to_csv(csv_path, index=False)

        print(f"\n[SUMMARY OF PIPELINE EVALUATION]")
        print(f" Total Images Processed      : {len(image_files)}")
        print(f" GENUINE Verdicts           : {sum(df['verdict'] == 'GENUINE')}")
        print(f" SUSPICIOUS Verdicts        : {sum(df['verdict'] == 'SUSPICIOUS')}")
        print(f" FAKE Verdicts              : {sum(df['verdict'] == 'FAKE')}")
        print(f" INSUFFICIENT QUALITY       : {sum(df['verdict'] == 'INSUFFICIENT_IMAGE_QUALITY')}")
        print(f" CSV Summary Report Saved To: '{csv_path}'")
        return df


def main():
    parser = argparse.ArgumentParser(description="Document Authenticity & Fraud Prediction Pipeline")
    parser.add_argument("--image", type=str, help="Path to single image for evaluation")
    parser.add_argument("--dir", type=str, help="Path to directory containing images for batch evaluation")
    parser.add_argument("--doc-type", type=str, default="auto", choices=["auto", "passport", "aadhar"], help="Document type (passport, aadhar, auto)")
    parser.add_argument("--doc-num", type=str, help="Document Number (Passport Number or Aadhaar Number)")
    parser.add_argument("--mrz1", type=str, help="Passport MRZ Line 1")
    parser.add_argument("--mrz2", type=str, help="Passport MRZ Line 2")
    parser.add_argument("--output", type=str, default="sample_output", help="Output directory for reports")

    args = parser.parse_args()

    pipeline = DocumentAuthenticityPipeline()

    if args.image:
        res = pipeline.predict_document_authenticity(
            args.image,
            doc_type=args.doc_type,
            doc_number=args.doc_num,
            mrz_line1=args.mrz1,
            mrz_line2=args.mrz2
        )
        print("\n" + "=" * 80)
        print("               SINGLE DOCUMENT AUTHENTICITY REPORT")
        print("=" * 80)
        print(f"File        : {res['filename']}")
        print(f"Doc Type    : {res['doc_type'].upper()}")
        print(f"Verdict     : {res['verdict']}")
        print(f"Risk Score  : {res['risk_score']}")
        print(f"Reasons     : {res['reasons']}")
        print("\nEvidence Table:")
        for k, v in res['evidence_table'].items():
            print(f"  - {k}: {v}")
        print("=" * 80)
    elif args.dir:
        pipeline.process_directory(args.dir, doc_type=args.doc_type, output_dir=args.output)
    else:
        # Default test on sample folders if available
        if os.path.exists("sample/passport"):
            pipeline.process_directory("sample/passport", doc_type="passport", output_dir="sample_output/passport_results")
        if os.path.exists("sample/aadhar"):
            pipeline.process_directory("sample/aadhar", doc_type="aadhar", output_dir="sample_output/aadhar_results")


if __name__ == "__main__":
    main()
