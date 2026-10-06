"""
Standalone CLI Benchmark, 12-Dataset Generator & Portable App Builder for Predictive-Machine Guard.

Runs the complete pipeline across all 12 industrial manufacturing datasets:
 1. ai4i2020              — UCI AI4I 2020 CNC Milling Machine (10,000 rows)
 2. cmapss                — NASA CMAPSS Turbofan Jet Engine Degradation (9,666 rows)
 3. custom_plant          — CNC & Robotic Assembly Vibro-Acoustic Plant (3,500 rows)
 4. ims_bearing           — NASA IMS Bearing Run-to-Failure Vibration (4,000 rows)
 5. secom_semiconductor   — UCI SECOM Semiconductor Wafer Etch Chamber (3,600 rows)
 6. hydraulic_rig         — UCI Hydraulic System Condition Monitoring (3,800 rows)
 7. centrifugal_pump      — Industrial Centrifugal Pump Cavitation & Seal (3,600 rows)
 8. wind_turbine_gearbox  — Wind & Steam Turbine Gearbox & Generator (3,800 rows)
 9. steel_rolling_mill    — Steel Hot-Rolling Mill Work-Roll & Strip (3,600 rows)
10. robotic_spot_welding  — Automotive Robotic Spot-Welding & Press Shop (3,600 rows)
11. lithium_battery_cell  — EV Lithium-Ion Battery Cell Formation & Aging (3,600 rows)
12. pharma_bioreactor     — Pharmaceutical Bioreactor & Freeze-Dryer (3,500 rows)

Also updates:
- `data/<key>_cleaned_dataset.csv` for all 12 datasets
- `models/benchmark_report.json`
- `static/precomputed_states.js`
- `Predictive_Machine_Guard_Portable_App.html`
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import joblib
import pandas as pd

from ml_engine import (
    DATASET_CATALOG,
    PredictiveMaintenanceEngine,
    ensure_all_12_datasets_generated,
    generate_or_load_domain_dataset,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _sample_rows_for_dataset(engine: PredictiveMaintenanceEngine, limit: int = 14) -> list:
    if engine.raw_df is None:
        return []
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
    return sample_df[cols_to_keep].to_dict(orient="records")


def build_portable_html() -> None:
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    styles_css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    precomputed_js = (STATIC_DIR / "precomputed_states.js").read_text(encoding="utf-8")
    standalone_js = (STATIC_DIR / "standalone_engine.js").read_text(encoding="utf-8")
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    portable = index_html
    portable = portable.replace('<link rel="stylesheet" href="styles.css" />', f"<style>\n{styles_css}\n</style>")
    portable = portable.replace('<link rel="stylesheet" href="/static/styles.css" />', "")
    portable = portable.replace('<script src="precomputed_states.js"></script>', f"<script>\n{precomputed_js}\n</script>")
    portable = portable.replace('<script src="/static/precomputed_states.js"></script>', "")
    portable = portable.replace('<script src="standalone_engine.js"></script>', f"<script>\n{standalone_js}\n</script>")
    portable = portable.replace('<script src="/static/standalone_engine.js"></script>', "")
    portable = portable.replace('<script src="app.js"></script>', f"<script>\n{app_js}\n</script>")
    portable = portable.replace('<script src="/static/app.js"></script>', "")

    out_path = BASE_DIR / "Predictive_Machine_Guard_Portable_App.html"
    out_path.write_text(portable, encoding="utf-8")
    print(f"[OK] Rebuilt portable single-file HTML app ({out_path.stat().st_size / 1024:.1f} KB): {out_path}")


def main() -> None:
    print("=" * 94)
    print("  PREDICTIVE-MACHINE GUARD -- 12-DATASET APPLIED ML & DEEP LEARNING (ANN/CNN/BiLSTM) SUITE")
    print("=" * 94)

    ensure_all_12_datasets_generated()

    summary_report = {}
    precomputed_states = {}
    precomputed_samples = {}

    for idx, (key, meta) in enumerate(DATASET_CATALOG.items(), start=1):
        df = generate_or_load_domain_dataset(key)
        title = meta["title"]
        print(f"\n[{idx:02d}/12] [{key}] {title} ({len(df):,} raw rows)")

        engine = PredictiveMaintenanceEngine()
        max_r = 6000 if len(df) > 6000 else len(df)
        state = engine.train_all(df, dataset_key=key, use_smote=True, max_rows=max_r)
        state["initial_prediction"] = engine.sample_live_telemetry(inject_anomaly="normal")

        # Save cleaned dataset CSV
        if engine.raw_df is not None:
            cleaned_path = DATA_DIR / f"{key}_cleaned_dataset.csv"
            engine.raw_df.to_csv(cleaned_path, index=False)

        prep = state["preprocessing_report"]
        ds = state["dataset_summary"]
        rul = state["rul_metrics"]
        champ = state["model_comparison"][0]

        print(
            f"   -> Cleaned: {prep['total_missing_detected']} NaNs imputed | "
            f"{prep['total_iqr_outliers_detected']} IQR outliers winsorized | "
            f"Failure Rate: {ds['failure_rate_pct']}%"
        )
        print(
            f"   -> Champion: {champ['model_name']} (F1={champ['f1_score']}%, ROC-AUC={champ['roc_auc']}%, Acc={champ['accuracy']}%) | "
            f"RUL R2={rul['r2_score']}"
        )

        if key == "ai4i2020":
            joblib.dump(engine.binary_models[ds["best_model_name"]], MODELS_DIR / "ai4i2020_champion_classifier.joblib")
            joblib.dump(engine.rul_regressor, MODELS_DIR / "ai4i2020_rul_regressor.joblib")

        summary_report[key] = {
            "title": title,
            "domain": meta["domain"],
            "raw_file": meta["filename"],
            "cleaned_file": f"{key}_cleaned_dataset.csv",
            "total_raw_rows": int(len(df)),
            "preprocessing_report": prep,
            "champion_model": ds["best_model_name"],
            "models": state["model_comparison"],
            "smote_ablation": state["smote_comparison"],
            "rul_metrics": rul,
            "top_shap_features": state["global_shap_importance"][:5],
        }

        precomputed_states[f"{key}_smote_true"] = state
        state_no_smote = copy.deepcopy(state)
        state_no_smote["smote_comparison"]["enabled"] = False
        state_no_smote["smote_comparison"]["train_positives_after"] = state["smote_comparison"]["train_positives_before"]
        precomputed_states[f"{key}_smote_false"] = state_no_smote
        precomputed_samples[key] = _sample_rows_for_dataset(engine, limit=14)

    report_path = MODELS_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved 12-dataset benchmark report to: {report_path}")

    precomputed_payload = {"states": precomputed_states, "samples": precomputed_samples}
    precomputed_path = STATIC_DIR / "precomputed_states.js"
    precomputed_path.write_text(
        "window.PMG_PRECOMPUTED = " + json.dumps(precomputed_payload) + ";\n",
        encoding="utf-8",
    )
    print(f"[OK] Updated static/precomputed_states.js with all 12 datasets ({precomputed_path.stat().st_size / 1024:.1f} KB)")

    build_portable_html()


if __name__ == "__main__":
    main()
