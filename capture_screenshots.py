import os
import shutil
import subprocess
import tempfile
from pathlib import Path

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(EDGE_PATH):
    EDGE_PATH = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

OUT_DIR = Path(r"c:\Users\darsh\Predictive-Machine-Guard\presentation_assets")
ARTIFACT_DIR = Path(r"C:\Users\darsh\.gemini\antigravity\brain\c47ca8d3-f101-4358-ae4c-933e3d3971d1")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

base_uri = Path(r"c:\Users\darsh\Predictive-Machine-Guard\static\index.html").as_uri()

targets = [
    ("output_1_telemetry_kpis.png", f"{base_uri}?tab=tab-twin"),
    ("output_2_models_prescriptions.png", f"{base_uri}?tab=tab-models"),
    ("output_3_preprocessing_shap.png", f"{base_uri}?tab=tab-xai"),
    ("output_4_data_cleaning_audit.png", f"{base_uri}?tab=tab-data"),
]

with tempfile.TemporaryDirectory() as tmp_profile:
    for filename, url in targets:
        out_path = OUT_DIR / filename
        if out_path.exists():
            out_path.unlink()
        subprocess.run([
            EDGE_PATH,
            f"--user-data-dir={tmp_profile}",
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--window-size=1920,1080",
            "--virtual-time-budget=3500",
            f"--screenshot={out_path}",
            url
        ], check=True)
        shutil.copy2(out_path, ARTIFACT_DIR / filename)
        print(f"Captured fresh {filename} ({out_path.stat().st_size} bytes)")
