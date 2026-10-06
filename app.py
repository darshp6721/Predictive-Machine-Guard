"""
Full-Stack Web Application Server for Predictive-Machine Guard (`app.py`).

Provides REST endpoints for:
- Multi-dataset switching (AI4I 2020, NASA CMAPSS Turbofan, Custom Plant Telemetry, User CSV Upload)
- Automated Data Preprocessing Audit (Missing Values Imputed + IQR/Z-Score Outliers Winsorized)
- 7-Model Benchmarking (XGBoost, LightGBM, Random Forest, Logistic Regression + PyTorch ANN, 1D-CNN, BiLSTM)
- Real-time Digital Twin inference + Local SHAP Explainable AI + RUL & Health Index estimation
- AI Maintenance Recommendation Copilot & Counterfactual Parameter Optimizer

Includes a built-in multi-threaded HTTP server (`ThreadingHTTPServer`) so it runs reliably on
Windows systems even when Smart App Control policies restrict `_ssl.pyd`.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    import cgi
    HAS_CGI = True
except ImportError:
    HAS_CGI = False

import pandas as pd

from ml_engine import (
    DATASET_CATALOG,
    PredictiveMaintenanceEngine,
    ensure_all_12_datasets_generated,
    generate_or_load_domain_dataset,
    normalize_custom_dataframe,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Global ML Engine instance + cache of trained engines per (dataset_key, use_smote)
engine = PredictiveMaintenanceEngine()
ENGINE_CACHE: Dict[ tuple[str, bool], PredictiveMaintenanceEngine ] = {}
CUSTOM_UPLOADED_DF: Optional[pd.DataFrame] = None


def _get_dataset_by_key(key: str) -> pd.DataFrame:
    global CUSTOM_UPLOADED_DF
    if key == "custom_plant" and CUSTOM_UPLOADED_DF is not None:
        return CUSTOM_UPLOADED_DF
    if key in DATASET_CATALOG:
        return generate_or_load_domain_dataset(key)
    raise ValueError(f"Unknown dataset_key: {key}")


def initialize_engine() -> None:
    """Pre-generates all 12 industrial benchmark datasets and trains the initial AI4I 2020 pipeline."""
    global engine
    print("[Predictive-Machine Guard] Ensuring all 12 industrial datasets & training 7-model ML/DL suite...")
    ensure_all_12_datasets_generated()
    df_ai4i = generate_or_load_domain_dataset("ai4i2020")
    engine.train_all(df_ai4i, dataset_key="ai4i2020", use_smote=True, max_rows=6000)
    ENGINE_CACHE[("ai4i2020", True)] = engine
    print("[Predictive-Machine Guard] Engine ready with all 12 datasets!")


def api_get_state() -> Dict[str, Any]:
    state = engine.get_dashboard_state()
    state["initial_prediction"] = engine.sample_live_telemetry(inject_anomaly="normal")
    return state


def api_switch_dataset(dataset_key: str, use_smote: bool) -> Dict[str, Any]:
    global engine
    cache_key = (dataset_key, use_smote)
    if cache_key in ENGINE_CACHE and not (dataset_key == "custom_plant" and CUSTOM_UPLOADED_DF is not None):
        engine = ENGINE_CACHE[cache_key]
        state = engine.get_dashboard_state()
    else:
        df = _get_dataset_by_key(dataset_key)
        new_eng = PredictiveMaintenanceEngine()
        state = new_eng.train_all(df, dataset_key=dataset_key, use_smote=use_smote, max_rows=5000)
        engine = new_eng
        ENGINE_CACHE[cache_key] = new_eng
    state["initial_prediction"] = engine.sample_live_telemetry(inject_anomaly="normal")
    return state


def api_sample_rows(limit: int = 14) -> Dict[str, Any]:
    if engine.raw_df is None:
        return {"rows": []}
    df = engine.raw_df
    failures = df[df["Machine failure"] == 1]
    healthy = df[df["Machine failure"] == 0]
    n_fail = min(len(failures), limit // 2)
    n_healthy = min(len(healthy), limit - n_fail)
    sample_df = pd.concat(
        [
            failures.sample(n=n_fail, random_state=42) if n_fail > 0 else pd.DataFrame(),
            healthy.sample(n=n_healthy, random_state=42) if n_healthy > 0 else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    cols_to_keep = (
        ["Type"]
        + engine.raw_sensor_cols
        + ["Machine failure", "Failure Mode", "RUL [min]", "Health Index [%]"]
    )
    cols_to_keep = [c for c in cols_to_keep if c in sample_df.columns]
    return {"rows": sample_df[cols_to_keep].to_dict(orient="records")}


class MachineGuardHTTPHandler(BaseHTTPRequestHandler):
    """High-speed REST & Static File Handler for Predictive-Machine Guard."""

    def log_message(self, format: str, *args: Any) -> None:
        # Keep console output clean
        pass

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File Not Found")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_file(STATIC_DIR / "index.html")
            elif path in ("/privacy", "/privacy.html"):
                self._send_file(STATIC_DIR / "privacy.html")
            elif path in ("/terms", "/terms.html", "/tnc", "/tos"):
                self._send_file(STATIC_DIR / "terms.html")
            elif path == "/favicon.ico":
                self._send_file(STATIC_DIR / "favicon.ico")
            elif path == "/favicon.svg":
                self._send_file(STATIC_DIR / "favicon.svg")
            elif path == "/healthz":
                self._send_json({"status": "ok", "active_dataset": engine.active_dataset_key})
            elif path.startswith("/static/"):
                rel = path.replace("/static/", "", 1)
                self._send_file(STATIC_DIR / rel)
            elif (STATIC_DIR / path.lstrip("/")).exists() and (STATIC_DIR / path.lstrip("/")).is_file():
                self._send_file(STATIC_DIR / path.lstrip("/"))
            elif path == "/api/state":
                self._send_json(api_get_state())
            elif path == "/api/simulate-step":
                anomaly = query.get("inject_anomaly", [None])[0]
                self._send_json(engine.sample_live_telemetry(inject_anomaly=anomaly))
            elif path == "/api/sample-rows":
                limit = int(query.get("limit", ["14"])[0])
                self._send_json(api_sample_rows(limit=limit))
            elif path == "/api/download-dataset":
                if engine.raw_df is None:
                    self.send_error(404, "No dataset loaded")
                    return
                csv_bytes = engine.raw_df.to_csv(index=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f"attachment; filename={engine.active_dataset_key}_cleaned_dataset.csv",
                )
                self.send_header("Content-Length", str(len(csv_bytes)))
                self.end_headers()
                self.wfile.write(csv_bytes)
            else:
                self.send_error(404, "Endpoint Not Found")
        except Exception as exc:
            self._send_json({"detail": str(exc)}, status=500)

    def do_POST(self) -> None:
        global CUSTOM_UPLOADED_DF
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            content_len = int(self.headers.get("Content-Length", 0))
            if path == "/api/switch-dataset":
                raw_body = self.rfile.read(content_len)
                data = json.loads(raw_body.decode("utf-8"))
                ds_key = data.get("dataset_key", "ai4i2020")
                use_smote = bool(data.get("use_smote", True))
                self._send_json(api_switch_dataset(ds_key, use_smote))

            elif path == "/api/predict":
                raw_body = self.rfile.read(content_len)
                data = json.loads(raw_body.decode("utf-8"))
                res = engine.predict_and_explain(
                    sensor_inputs=data.get("sensor_inputs", {}),
                    model_name=data.get("model_name"),
                )
                self._send_json(res)

            elif path == "/api/upload-csv":
                use_smote = query.get("use_smote", ["true"])[0].lower() == "true"
                content_type_header = self.headers.get("Content-Type", "")
                if HAS_CGI and "multipart/form-data" in content_type_header:
                    form = cgi.FieldStorage(
                        fp=self.rfile,
                        headers=self.headers,
                        environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type_header},
                    )
                    file_item = form["file"]
                    filename = getattr(file_item, "filename", "uploaded.csv")
                    raw_bytes = file_item.file.read()
                else:
                    filename = "uploaded.csv"
                    raw_bytes = self.rfile.read(content_len)

                raw_df = pd.read_csv(io.BytesIO(raw_bytes))
                if len(raw_df) < 30:
                    self._send_json(
                        {"detail": "Uploaded CSV must contain at least 30 rows for ML/DL training."},
                        status=400,
                    )
                    return

                norm_df, schema_info = normalize_custom_dataframe(raw_df)
                CUSTOM_UPLOADED_DF = norm_df
                state = engine.train_all(norm_df, dataset_key="custom_plant", use_smote=use_smote)
                state["uploaded_schema_info"] = {"filename": filename, **schema_info}
                state["initial_prediction"] = engine.sample_live_telemetry(inject_anomaly="normal")
                self._send_json(state)
            else:
                self.send_error(404, "Endpoint Not Found")
        except Exception as exc:
            self._send_json({"detail": str(exc)}, status=400)


def run_server(host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    bind_port = port if port is not None else int(os.environ.get("PORT", 8000))
    initialize_engine()
    server = ThreadingHTTPServer((host, bind_port), MachineGuardHTTPHandler)
    print(f"[Predictive-Machine Guard] Live Web Application running at: http://127.0.0.1:{bind_port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()

