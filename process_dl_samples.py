import os
import sys
import json
import shutil
import pandas as pd

# Ensure root workspace directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dl_forensic_analyzer import DLForensicAnalyzer
from indian_dl_verifier import IndianDLVerifier


def process_dl_directory(sample_dir: str, output_dir: str):
    """
    Processes Driving Licence images in sample_dir using pure digital forensics (DLForensicAnalyzer),
    classifies them into subdirectories (genuine, suspicious, insufficient_quality) inside output_dir,
    and generates comprehensive CSV & JSON reports.
    """
    analyzer = DLForensicAnalyzer()
    verifier = IndianDLVerifier()

    # Subdirectories according to classification
    genuine_dir = os.path.join(output_dir, "genuine")
    suspicious_dir = os.path.join(output_dir, "suspicious")
    insufficient_dir = os.path.join(output_dir, "insufficient_quality")

    # Clean existing output subfolders for fresh classification run
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for d in [genuine_dir, suspicious_dir, insufficient_dir]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(sample_dir):
        print(f"[ERROR] Sample directory not found: {sample_dir}")
        return

    files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    print(f"[INFO] Running pure forensic classification on {len(files)} image files from {sample_dir}...")

    results = []

    for filename in sorted(files):
        img_path = os.path.join(sample_dir, filename)

        # Run pure digital forensics without relying on filenames or static dimensions
        report = analyzer.analyze_document(img_path)

        verdict = report['verdict']
        if verdict == 'GENUINE':
            target_subfolder = genuine_dir
        elif verdict == 'SUSPICIOUS':
            target_subfolder = suspicious_dir
        else:
            target_subfolder = insufficient_dir

        # Copy image file to classified subfolder
        dest_path = os.path.join(target_subfolder, filename)
        shutil.copy2(img_path, dest_path)

        ev = report.get('evidence', {})
        q = ev.get('quality', {})

        rec = {
            "filename": filename,
            "verdict": verdict,
            "risk_score": report['risk_score'],
            "resolution": q.get('resolution', 'N/A'),
            "aspect_ratio": ev.get('aspect_ratio', 'N/A'),
            "is_id1_geometry": ev.get('is_id1_geometry', False),
            "blur_score": q.get('blur_score', 0.0),
            "ela_variance": ev.get('ela_variance', 0.0),
            "sensor_noise_variance": ev.get('sensor_noise_variance', 0.0),
            "reasons": "; ".join(report['reasons']),
            "classified_folder": os.path.basename(target_subfolder),
            "output_path": os.path.relpath(dest_path, output_dir)
        }
        results.append(rec)

    # Save CSV summary
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "dl_validation_results.csv")
    df.to_csv(csv_path, index=False)

    # Save JSON summary report
    json_path = os.path.join(output_dir, "dl_classification_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "forensic_engine": "DLForensicAnalyzer v1.0",
            "total_processed": len(results),
            "genuine_count": sum(1 for r in results if r["verdict"] == "GENUINE"),
            "suspicious_count": sum(1 for r in results if r["verdict"] == "SUSPICIOUS"),
            "insufficient_quality_count": sum(1 for r in results if r["verdict"] == "INSUFFICIENT_IMAGE_QUALITY"),
            "detailed_results": results
        }, f, indent=2)

    print(f"\n[SUCCESS] Classification complete using pure digital forensics!")
    print(f"  - Genuine folder            : {genuine_dir} ({sum(1 for r in results if r['verdict'] == 'GENUINE')} files)")
    print(f"  - Suspicious folder         : {suspicious_dir} ({sum(1 for r in results if r['verdict'] == 'SUSPICIOUS')} files)")
    print(f"  - Insufficient Quality      : {insufficient_dir} ({sum(1 for r in results if r['verdict'] == 'INSUFFICIENT_IMAGE_QUALITY')} files)")
    print(f"  - CSV Report                : {csv_path}")
    print(f"  - JSON Report               : {json_path}\n")

    return results


if __name__ == "__main__":
    sample_dir = os.path.abspath("sample/DL")
    output_dir = os.path.abspath("sample_output/dl")
    process_dl_directory(sample_dir, output_dir)
