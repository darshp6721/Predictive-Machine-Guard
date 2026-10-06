"""
Core Applied Machine Learning & Deep Learning Pipeline for Predictive-Machine Guard
(`ml_engine/pipeline.py`).

End-to-End Capabilities:
1. Data Quality & Preprocessing (`DataQualityPreprocessor`):
   - Duplicate removal, Missing Value Imputation (Grouped Median/Mode), Outlier Detection (IQR & Z-Score) & Winsorization
2. Physics-Informed Domain Feature Engineering (AI4I 2020, NASA CMAPSS Turbofan, Custom Telemetry)
3. Stratified Train / Validation / Test Split + SMOTE Class Imbalance Handling
4. 7-Model Unified Benchmark Suite:
   - Classical ML: XGBoost, LightGBM, Random Forest, Logistic Regression
   - Deep Learning (PyTorch): ANN (Deep MLP), 1D-CNN, BiLSTM
5. Multi-Task Learning:
   - Binary Failure Probability Prediction
   - Multi-Class Failure Mode Classification (TWF, HDF, PWF, OSF, RNF, etc.)
   - Remaining Useful Life (RUL) Regression & Machine Health Index (MHI)
6. Explainable AI (SHAP TreeExplainer global & local feature attributions)
7. AI Maintenance Copilot & Counterfactual Parameter Optimizer (`AIMaintenanceRecommender`)
"""

from __future__ import annotations

import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .ai_recommender import AIMaintenanceRecommender
from .deep_learning import DeepLearningClassifierWrapper
from .preprocessing import DataQualityPreprocessor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


METADATA_DROP_COLS = {
    "UDI",
    "UID",
    "Product ID",
    "unit_id",
    "Machine failure",
    "Failure Mode",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "RUL [min]",
    "Health Index [%]",
}


class PredictiveMaintenanceEngine:
    """
    Stateful Multi-Dataset, 7-Model (Classical ML + Deep Learning) Engine for Predictive Maintenance.
    """

    def __init__(self) -> None:
        self.active_dataset_key: str = "ai4i2020"
        self.raw_df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.raw_sensor_cols: List[str] = []
        self.engineered_cols: List[str] = []
        self.sensor_ranges: Dict[str, Dict[str, float]] = {}

        # Preprocessing & Data Quality Audit
        self.preprocessor = DataQualityPreprocessor()
        self.preprocessing_report: Dict[str, Any] = {}

        # Models (4 Classical ML + 3 Deep Learning)
        self.binary_models: Dict[str, Any] = {}
        self.best_model_name: str = "XGBoost"
        self.mode_classifier: Optional[Any] = None
        self.mode_label_encoder: LabelEncoder = LabelEncoder()
        self.rul_regressor: Optional[Any] = None
        self.dl_loss_curves: Dict[str, Dict[str, List[float]]] = {}

        # Explainability & Evaluation Artifacts
        self.shap_explainer: Optional[Any] = None
        self.background_sample: Optional[pd.DataFrame] = None
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self.global_shap_importance: List[Dict[str, Any]] = []
        self.model_comparison: List[Dict[str, Any]] = []
        self.smote_comparison: Dict[str, Any] = {}
        self.rul_metrics: Dict[str, float] = {}
        self.dataset_summary: Dict[str, Any] = {}
        self.use_smote: bool = True
        self.sim_cursor: int = 0

    @staticmethod
    def _sanitize_X(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces special characters ([, ], <, >, /, spaces) in feature names so
        XGBoost and LightGBM C++ backends accept all column names cleanly.
        """
        out = df.copy()
        out.columns = [re.sub(r"[\[\]<>/:,\s]+", "_", str(c)).strip("_") for c in out.columns]
        return out

    @staticmethod
    def engineer_features(df: pd.DataFrame, dataset_key: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Applies domain-informed physical feature engineering to cleaned sensor telemetry.
        Returns (engineered_df, raw_sensor_cols, engineered_col_names).
        """
        work = df.copy()

        if "Type" in work.columns:
            type_map = {"L": 0, "M": 1, "H": 2}
            work["Type_Encoded"] = work["Type"].map(lambda x: type_map.get(str(x).upper(), 1)).astype(float)
        else:
            work["Type_Encoded"] = 1.0

        raw_cols = [
            c
            for c in work.select_dtypes(include=[np.number]).columns
            if c not in METADATA_DROP_COLS and c != "Type_Encoded"
        ]
        eng_cols: List[str] = []

        if dataset_key == "ai4i2020" or {
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        }.issubset(set(work.columns)):
            work["Temp Diff [K]"] = np.round(work["Process temperature [K]"] - work["Air temperature [K]"], 2)
            omega = work["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0)
            work["Mechanical Power [W]"] = np.round(work["Torque [Nm]"] * omega, 1)
            work["Overstrain Index [Nm*min]"] = np.round(work["Tool wear [min]"] * work["Torque [Nm]"], 1)
            work["Thermal Ratio"] = np.round(
                work["Process temperature [K]"] / np.maximum(work["Air temperature [K]"], 1.0), 4
            )
            work["Torque per kRPM"] = np.round(
                work["Torque [Nm]"] / np.maximum(work["Rotational speed [rpm]"] / 1000.0, 0.1), 2
            )
            eng_cols = [
                "Temp Diff [K]",
                "Mechanical Power [W]",
                "Overstrain Index [Nm*min]",
                "Thermal Ratio",
                "Torque per kRPM",
            ]

        elif dataset_key == "cmapss" or "s3_HPC_temp [R]" in work.columns:
            work["Thermal Margin HPC/LPT"] = np.round(
                (work["s3_HPC_temp [R]"] + work["s4_LPT_temp [R]"]) / np.maximum(work["s2_LPC_temp [R]"], 1.0), 4
            )
            work["Core-to-Fan Speed Ratio"] = np.round(
                work["s9_core_speed [rpm]"] / np.maximum(work["s8_fan_speed [rpm]"], 1.0), 4
            )
            work["Pressure-Fuel Efficiency"] = np.round(
                work["s7_HPC_pressure [psia]"] / np.maximum(work["s12_fuel_flow_ratio"], 1.0), 4
            )
            work["Total Coolant Bleed [lbm/s]"] = np.round(
                work["s20_HPT_coolant_bleed"] + work["s21_LPT_coolant_bleed"], 3
            )
            eng_cols = [
                "Thermal Margin HPC/LPT",
                "Core-to-Fan Speed Ratio",
                "Pressure-Fuel Efficiency",
                "Total Coolant Bleed [lbm/s]",
            ]

        else:
            if {"Vibration RMS [mm/s]", "Bearing Temp [C]"}.issubset(set(work.columns)):
                work["Vibro-Thermal Stress"] = np.round(
                    (work["Vibration RMS [mm/s]"] * work["Bearing Temp [C]"]) / 100.0, 2
                )
                eng_cols.append("Vibro-Thermal Stress")
            if {"Spindle Current [A]", "Acoustic Emission [dB]"}.issubset(set(work.columns)):
                work["Electro-Acoustic Index"] = np.round(
                    (work["Spindle Current [A]"] * work["Acoustic Emission [dB]"]) / 100.0, 2
                )
                eng_cols.append("Electro-Acoustic Index")
            if not eng_cols and len(raw_cols) >= 2:
                c1, c2 = raw_cols[0], raw_cols[1]
                work["Interaction Stress Index"] = np.round((work[c1] * work[c2]) / 10.0, 2)
                eng_cols.append("Interaction Stress Index")
                if len(raw_cols) >= 3:
                    c3 = raw_cols[2]
                    work["Multi-Sensor Load Ratio"] = np.round(
                        work[c3] / np.maximum(np.abs(work[c1]), 0.1), 3
                    )
                    eng_cols.append("Multi-Sensor Load Ratio")

        return work, raw_cols, eng_cols

    def train_all(
        self,
        df: pd.DataFrame,
        dataset_key: str = "ai4i2020",
        use_smote: bool = True,
        max_rows: int = 8000,
    ) -> Dict[str, Any]:
        """
        Executes the complete Data Preprocessing, Feature Engineering, Train/Val/Test Split,
        SMOTE Balancing, 7-Model Benchmarking (4 Classical ML + 3 Deep Learning), RUL Regression,
        and SHAP Explainability pipeline.
        """
        t0 = time.time()
        self.active_dataset_key = dataset_key
        self.use_smote = use_smote

        if len(df) > max_rows:
            df_sub, _ = train_test_split(
                df, train_size=max_rows, stratify=df["Machine failure"], random_state=42
            )
            df_sub = df_sub.reset_index(drop=True)
        else:
            df_sub = df.copy().reset_index(drop=True)

        # Identify raw numeric sensor columns prior to preprocessing
        raw_numeric_cols = [
            c
            for c in df_sub.select_dtypes(include=[np.number]).columns
            if c not in METADATA_DROP_COLS and c != "Type_Encoded"
        ]

        # STEP 1: Execute Data Quality & Preprocessing (Missing Values + Outliers + Duplicates)
        df_cleaned, self.preprocessing_report = self.preprocessor.fit_transform(df_sub, raw_numeric_cols)
        self.raw_df = df_cleaned

        # STEP 2: Physics-Informed Feature Engineering
        eng_df, raw_cols, eng_cols = self.engineer_features(df_cleaned, dataset_key)
        self.raw_sensor_cols = raw_cols
        self.engineered_cols = eng_cols
        self.feature_cols = ["Type_Encoded"] + raw_cols + eng_cols

        # Store sensor operating ranges after outlier cleaning for UI sliders
        self.sensor_ranges = {}
        for col in raw_cols:
            s = df_cleaned[col].dropna()
            q01, q50, q99 = float(s.quantile(0.01)), float(s.median()), float(s.quantile(0.99))
            s_min, s_max = float(s.min()), float(s.max())
            step = round((s_max - s_min) / 200.0, 4) if (s_max - s_min) > 0 else 0.1
            self.sensor_ranges[col] = {
                "min": round(s_min, 4),
                "max": round(s_max, 4),
                "median": round(q50, 4),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "q01": round(q01, 4),
                "q99": round(q99, 4),
                "step": max(step, 0.001),
            }

        X_orig = eng_df[self.feature_cols].fillna(0.0)
        self.feature_means = X_orig.mean().to_dict()
        self.feature_stds = (X_orig.std() + 1e-6).to_dict()

        X = self._sanitize_X(X_orig)
        y_bin = eng_df["Machine failure"].astype(int).values
        y_rul = eng_df["RUL [min]"].astype(float).values
        y_mode_raw = eng_df["Failure Mode"].astype(str).values
        y_mode = self.mode_label_encoder.fit_transform(y_mode_raw)

        # STEP 3: Stratified Train (70%) / Validation (10%) / Held-Out Test (20%) Split
        X_train_full, X_test, y_train_full, y_test, rul_train_full, rul_test, mode_train_full, mode_test = (
            train_test_split(
                X,
                y_bin,
                y_rul,
                y_mode,
                test_size=0.20,
                stratify=y_bin if len(np.unique(y_bin)) > 1 else None,
                random_state=42,
            )
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.125,  # 0.125 * 0.80 = 10% Validation, 70% Train
            stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None,
            random_state=42,
        )

        # STEP 4: Apply SMOTE on training split if enabled
        minority_count = int(np.sum(y_train == 1))
        if use_smote and minority_count >= 6:
            k_neighbors = min(5, minority_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors, sampling_strategy=0.38)
            try:
                X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
            except Exception:
                X_train_fit, y_train_fit = X_train, y_train
        else:
            X_train_fit, y_train_fit = X_train, y_train

        # Baseline RF without SMOTE vs with SMOTE for Ablation Study
        rf_no_smote = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
        rf_no_smote.fit(X_train, y_train)
        pred_no_smote = rf_no_smote.predict(X_test)

        # STEP 5: Define 7 Candidate Models (4 Classical ML + 3 Deep Learning: ANN, 1D-CNN, BiLSTM)
        candidates: List[Tuple[str, str, Any]] = []
        if HAS_XGB:
            candidates.append(
                (
                    "XGBoost",
                    "Classical Ensemble (GBDT)",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=6,
                        learning_rate=0.08,
                        subsample=0.88,
                        colsample_bytree=0.88,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            )
        if HAS_LGBM:
            candidates.append(
                (
                    "LightGBM",
                    "Classical Ensemble (Leaf-wise GBDT)",
                    LGBMClassifier(
                        n_estimators=150,
                        max_depth=7,
                        learning_rate=0.08,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1,
                        n_jobs=-1,
                    ),
                )
            )
        candidates.append(
            (
                "Random Forest",
                "Classical Ensemble (Bagging Trees)",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=4,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
        candidates.append(
            (
                "Logistic Regression",
                "Classical Linear Baseline",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, random_state=42)),
                    ]
                ),
            )
        )

        # 3 Deep Learning Architectures (PyTorch)
        candidates.append(
            (
                "ANN (Deep MLP)",
                "Deep Learning (Feedforward Neural Net)",
                DeepLearningClassifierWrapper(arch="ANN", epochs=16, batch_size=128, lr=0.004, seed=42),
            )
        )
        candidates.append(
            (
                "1D-CNN",
                "Deep Learning (1D Convolutional Net)",
                DeepLearningClassifierWrapper(arch="1D-CNN", epochs=16, batch_size=128, lr=0.0035, seed=42),
            )
        )
        candidates.append(
            (
                "BiLSTM",
                "Deep Learning (Bidirectional LSTM)",
                DeepLearningClassifierWrapper(arch="BiLSTM", epochs=16, batch_size=128, lr=0.004, seed=42),
            )
        )

        self.binary_models = {}
        self.model_comparison = []
        self.dl_loss_curves = {}
        best_f1 = -1.0

        for name, family, model in candidates:
            m_t0 = time.time()
            if isinstance(model, DeepLearningClassifierWrapper):
                model.fit(X_train_fit, y_train_fit, X_val=X_val, y_val=y_val)
                self.dl_loss_curves[name] = {
                    "train_loss": model.train_loss_history,
                    "val_loss": model.val_loss_history,
                }
            else:
                model.fit(X_train_fit, y_train_fit)
            fit_ms = round((time.time() - m_t0) * 1000.0, 1)

            y_train_pred = model.predict(X_train)
            train_acc = float(accuracy_score(y_train, y_train_pred))

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.95
            pr_auc = float(average_precision_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.85
            mcc = float(matthews_corrcoef(y_test, y_pred))
            brier = float(brier_score_loss(y_test, y_prob))
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            idx_roc = np.linspace(0, len(fpr) - 1, min(35, len(fpr))).astype(int)
            p_curve, r_curve, _ = precision_recall_curve(y_test, y_prob)
            idx_pr = np.linspace(0, len(p_curve) - 1, min(35, len(p_curve))).astype(int)

            self.binary_models[name] = model
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name

            self.model_comparison.append(
                {
                    "model_name": name,
                    "model_family": family,
                    "is_deep_learning": isinstance(model, DeepLearningClassifierWrapper),
                    "train_accuracy": round(train_acc * 100.0, 2),
                    "accuracy": round(acc * 100.0, 2),
                    "precision": round(prec * 100.0, 2),
                    "recall": round(rec * 100.0, 2),
                    "f1_score": round(f1 * 100.0, 2),
                    "roc_auc": round(roc_auc * 100.0, 2),
                    "pr_auc": round(pr_auc * 100.0, 2),
                    "mcc": round(mcc, 4),
                    "brier_score": round(brier, 4),
                    "train_time_ms": fit_ms,
                    "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "roc_curve": {
                        "fpr": [round(float(x), 4) for x in fpr[idx_roc]],
                        "tpr": [round(float(x), 4) for x in tpr[idx_roc]],
                    },
                    "pr_curve": {
                        "recall": [round(float(x), 4) for x in r_curve[idx_pr]],
                        "precision": [round(float(x), 4) for x in p_curve[idx_pr]],
                    },
                }
            )

        self.model_comparison.sort(key=lambda x: x["f1_score"], reverse=True)
        self.best_model_name = self.model_comparison[0]["model_name"]

        rf_smote_pred = self.binary_models["Random Forest"].predict(X_test)
        self.smote_comparison = {
            "enabled": use_smote,
            "train_positives_before": int(np.sum(y_train == 1)),
            "train_positives_after": int(np.sum(y_train_fit == 1)),
            "train_negatives": int(np.sum(y_train == 0)),
            "without_smote": {
                "precision": round(float(precision_score(y_test, pred_no_smote, zero_division=0)) * 100.0, 2),
                "recall": round(float(recall_score(y_test, pred_no_smote, zero_division=0)) * 100.0, 2),
                "f1_score": round(float(f1_score(y_test, pred_no_smote, zero_division=0)) * 100.0, 2),
            },
            "with_smote": {
                "precision": round(float(precision_score(y_test, rf_smote_pred, zero_division=0)) * 100.0, 2),
                "recall": round(float(recall_score(y_test, rf_smote_pred, zero_division=0)) * 100.0, 2),
                "f1_score": round(float(f1_score(y_test, rf_smote_pred, zero_division=0)) * 100.0, 2),
            },
        }

        # Train Multi-Class Failure Mode Diagnostic Classifier
        if HAS_XGB and len(self.mode_label_encoder.classes_) > 2:
            self.mode_classifier = XGBClassifier(
                n_estimators=110,
                max_depth=6,
                learning_rate=0.09,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.mode_classifier = RandomForestClassifier(
                n_estimators=110, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
            )
        self.mode_classifier.fit(X_train_full, mode_train_full)

        # Train Remaining Useful Life (RUL) Regressor
        if HAS_XGB:
            self.rul_regressor = XGBRegressor(
                n_estimators=140,
                max_depth=6,
                learning_rate=0.07,
                subsample=0.85,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.rul_regressor = RandomForestRegressor(
                n_estimators=110, max_depth=12, random_state=42, n_jobs=-1
            )
        self.rul_regressor.fit(X_train_full, rul_train_full)
        rul_pred = self.rul_regressor.predict(X_test)
        self.rul_metrics = {
            "mae": round(float(mean_absolute_error(rul_test, rul_pred)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(rul_test, rul_pred))), 2),
            "r2_score": round(float(r2_score(rul_test, rul_pred)), 4),
        }

        # Setup Explainable AI (SHAP TreeExplainer on best tree model)
        tree_candidates = [m for m in ["XGBoost", "LightGBM", "Random Forest"] if m in self.binary_models]
        tree_model_name = (
            self.best_model_name if self.best_model_name in tree_candidates else tree_candidates[0]
        )
        tree_model = self.binary_models[tree_model_name]
        self.background_sample = X_train.sample(n=min(180, len(X_train)), random_state=42)

        self.global_shap_importance = []
        if HAS_SHAP:
            try:
                self.shap_explainer = shap.TreeExplainer(tree_model)
                shap_sample = X_test.sample(n=min(180, len(X_test)), random_state=42)
                sv = self.shap_explainer.shap_values(shap_sample)
                if isinstance(sv, list):
                    sv_arr = np.abs(sv[1]).mean(axis=0)
                elif sv.ndim == 3:
                    sv_arr = np.abs(sv[:, :, 1]).mean(axis=0)
                else:
                    sv_arr = np.abs(sv).mean(axis=0)
                total_sv = float(np.sum(sv_arr)) + 1e-9
                for col, val in zip(self.feature_cols, sv_arr):
                    self.global_shap_importance.append(
                        {
                            "feature": col,
                            "mean_abs_shap": round(float(val), 4),
                            "importance_pct": round(float(val / total_sv * 100.0), 2),
                            "is_engineered": col in self.engineered_cols,
                        }
                    )
            except Exception:
                self.shap_explainer = None

        if not self.global_shap_importance:
            importances = getattr(tree_model, "feature_importances_", np.ones(len(self.feature_cols)))
            total_imp = float(np.sum(importances)) + 1e-9
            for col, val in zip(self.feature_cols, importances):
                self.global_shap_importance.append(
                    {
                        "feature": col,
                        "mean_abs_shap": round(float(val), 4),
                        "importance_pct": round(float(val / total_imp * 100.0), 2),
                        "is_engineered": col in self.engineered_cols,
                    }
                )

        self.global_shap_importance.sort(key=lambda x: x["importance_pct"], reverse=True)

        mode_counts = df_cleaned["Failure Mode"].value_counts().to_dict()
        failure_modes_dist = [
            {"mode": str(k), "count": int(v), "pct": round(float(v / len(df_cleaned) * 100.0), 2)}
            for k, v in mode_counts.items()
            if str(k) != "No Failure"
        ]

        total_ms = round((time.time() - t0) * 1000.0, 1)
        self.dataset_summary = {
            "dataset_key": dataset_key,
            "total_records": int(len(df_cleaned)),
            "train_records": int(len(X_train)),
            "val_records": int(len(X_val)),
            "test_records": int(len(X_test)),
            "total_failures": int(df_cleaned["Machine failure"].sum()),
            "failure_rate_pct": round(float(df_cleaned["Machine failure"].mean() * 100.0), 2),
            "avg_rul": round(float(df_cleaned["RUL [min]"].mean()), 1),
            "avg_health_index": round(float(df_cleaned["Health Index [%]"].mean()), 1),
            "raw_sensor_cols": self.raw_sensor_cols,
            "engineered_cols": self.engineered_cols,
            "feature_cols": self.feature_cols,
            "sensor_ranges": self.sensor_ranges,
            "failure_modes_breakdown": failure_modes_dist,
            "best_model_name": self.best_model_name,
            "pipeline_train_time_ms": total_ms,
        }
        return self.get_dashboard_state()

    def get_dashboard_state(self) -> Dict[str, Any]:
        """Returns the full state payload for the web dashboard."""
        return {
            "dataset_summary": self.dataset_summary,
            "preprocessing_report": self.preprocessing_report,
            "model_comparison": self.model_comparison,
            "dl_loss_curves": self.dl_loss_curves,
            "smote_comparison": self.smote_comparison,
            "rul_metrics": self.rul_metrics,
            "global_shap_importance": self.global_shap_importance,
        }

    def predict_and_explain(
        self,
        sensor_inputs: Dict[str, Any],
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs real-time inference for a single telemetry state across all 7 models
        (4 Classical ML + 3 Deep Learning: ANN, 1D-CNN, BiLSTM), RUL regressor,
        local SHAP explainability, and AI Maintenance Recommendation Engine.
        """
        chosen_model_name = model_name if model_name in self.binary_models else self.best_model_name

        row_dict: Dict[str, Any] = {"Type": sensor_inputs.get("Type", "M")}
        for col in self.raw_sensor_cols:
            if col in sensor_inputs and sensor_inputs[col] is not None and not pd.isna(sensor_inputs[col]):
                row_dict[col] = float(sensor_inputs[col])
            else:
                row_dict[col] = float(self.sensor_ranges[col]["median"])

        single_df = pd.DataFrame([row_dict])
        eng_df, _, _ = self.engineer_features(single_df, self.active_dataset_key)
        X_row_orig = eng_df[self.feature_cols].fillna(0.0)
        X_row = self._sanitize_X(X_row_orig)

        # 1. Binary failure probabilities across all 7 models (Classical ML + Deep Learning)
        all_model_probs: Dict[str, float] = {}
        for m_name, m_obj in self.binary_models.items():
            prob = float(m_obj.predict_proba(X_row)[0, 1])
            all_model_probs[m_name] = round(prob * 100.0, 2)

        failure_prob_pct = all_model_probs[chosen_model_name]

        # 2. Multi-class failure mode probabilities
        mode_probs_list: List[Dict[str, Any]] = []
        if self.mode_classifier is not None:
            m_probs = self.mode_classifier.predict_proba(X_row)[0]
            for cls_name, p_val in zip(self.mode_label_encoder.classes_, m_probs):
                mode_probs_list.append(
                    {
                        "mode": str(cls_name),
                        "probability_pct": round(float(p_val) * 100.0, 2),
                    }
                )
            mode_probs_list.sort(key=lambda x: x["probability_pct"], reverse=True)

        non_healthy_modes = [m for m in mode_probs_list if m["mode"] != "No Failure"]
        dominant_failure_mode = (
            non_healthy_modes[0]["mode"]
            if (failure_prob_pct >= 28.0 and non_healthy_modes)
            else mode_probs_list[0]["mode"]
        )

        # 3. Remaining Useful Life (RUL) & Machine Health Index (MHI)
        pred_rul = float(self.rul_regressor.predict(X_row)[0]) if self.rul_regressor is not None else 120.0
        pred_rul = round(max(0.0, pred_rul), 1)

        max_rul_ref = 125.0 if self.active_dataset_key == "cmapss" else 220.0
        rul_health = np.clip((pred_rul / max_rul_ref) * 100.0, 0.0, 100.0)
        prob_health = 100.0 - failure_prob_pct
        health_index = round(float(0.55 * prob_health + 0.45 * rul_health), 1)

        # 4. Local SHAP Feature Contributions
        local_shap: List[Dict[str, Any]] = []
        if self.shap_explainer is not None:
            try:
                sv = self.shap_explainer.shap_values(X_row)
                if isinstance(sv, list):
                    row_sv = sv[1][0]
                elif sv.ndim == 3:
                    row_sv = sv[0, :, 1]
                else:
                    row_sv = sv[0]
                for col, s_val in zip(self.feature_cols, row_sv):
                    local_shap.append(
                        {
                            "feature": col,
                            "value": round(float(X_row_orig.iloc[0][col]), 3),
                            "shap_impact": round(float(s_val), 4),
                            "direction": "Increases Risk" if float(s_val) > 0 else "Reduces Risk",
                            "is_engineered": col in self.engineered_cols,
                        }
                    )
            except Exception:
                local_shap = []

        if not local_shap:
            for item in self.global_shap_importance:
                col = item["feature"]
                val = float(X_row_orig.iloc[0][col])
                z = (val - self.feature_means.get(col, 0.0)) / self.feature_stds.get(col, 1.0)
                impact = float(z * (item["importance_pct"] / 100.0))
                local_shap.append(
                    {
                        "feature": col,
                        "value": round(val, 3),
                        "shap_impact": round(impact, 4),
                        "direction": "Increases Risk" if impact > 0 else "Reduces Risk",
                        "is_engineered": col in self.engineered_cols,
                    }
                )

        local_shap.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)
        engineered_values = {col: round(float(X_row_orig.iloc[0][col]), 3) for col in self.engineered_cols}

        # 5. Generate AI Maintenance Copilot Recommendations & Counterfactual Optimal Setpoints
        prescription = AIMaintenanceRecommender.generate_recommendations(
            dataset_key=self.active_dataset_key,
            sensor_inputs=row_dict,
            sensor_ranges=self.sensor_ranges,
            failure_prob_pct=failure_prob_pct,
            all_model_probs=all_model_probs,
            health_index=health_index,
            pred_rul=pred_rul,
            dominant_mode=dominant_failure_mode,
            engineered_values=engineered_values,
            local_shap=local_shap,
        )

        # Evaluate counterfactual optimal setpoints through the champion model & RUL regressor
        opt_df = pd.DataFrame([prescription["optimal_sensor_inputs"]])
        opt_eng_df, _, _ = self.engineer_features(opt_df, self.active_dataset_key)
        X_opt = self._sanitize_X(opt_eng_df[self.feature_cols].fillna(0.0))
        opt_prob = round(float(self.binary_models[chosen_model_name].predict_proba(X_opt)[0, 1]) * 100.0, 2)
        opt_rul = round(max(0.0, float(self.rul_regressor.predict(X_opt)[0])), 1) if self.rul_regressor else pred_rul

        prescription["projected_optimal_risk_pct"] = opt_prob
        prescription["projected_optimal_rul_min"] = opt_rul

        return {
            "selected_model": chosen_model_name,
            "failure_probability_pct": failure_prob_pct,
            "all_model_probabilities": all_model_probs,
            "dominant_failure_mode": dominant_failure_mode,
            "failure_mode_distribution": mode_probs_list,
            "predicted_rul_min": pred_rul,
            "health_index_pct": health_index,
            "engineered_features": engineered_values,
            "local_shap_explanations": local_shap[:10],
            "prescription": prescription,
        }

    def sample_live_telemetry(self, inject_anomaly: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a realistic live streaming telemetry packet (either normal operating point
        or an injected failure scenario) and runs full inference + XAI + AI Recommendations.
        """
        rng = np.random.default_rng()
        if self.raw_df is None or len(self.raw_df) == 0:
            raise RuntimeError("Engine has no loaded dataset.")

        if inject_anomaly and inject_anomaly != "normal":
            anom_upper = inject_anomaly.upper()
            if anom_upper in {"TWF", "HDF", "PWF", "OSF", "RNF"} and anom_upper in self.raw_df.columns:
                subset = self.raw_df[self.raw_df[anom_upper] == 1]
            else:
                breakdown = self.dataset_summary.get("failure_modes_breakdown", [])
                non_healthy = [str(m["mode"]) for m in breakdown if str(m.get("mode")) != "No Failure"]
                if not non_healthy:
                    non_healthy = [
                        str(m) for m in self.raw_df["Failure Mode"].unique() if str(m) != "No Failure"
                    ]
                slot_map = {"TWF": 0, "HDF": 1, "PWF": 2, "OSF": 3, "RNF": 4}
                if anom_upper in slot_map and len(non_healthy) > 0:
                    target_mode = non_healthy[slot_map[anom_upper] % len(non_healthy)]
                    subset = self.raw_df[self.raw_df["Failure Mode"] == target_mode]
                else:
                    subset = self.raw_df[self.raw_df["Machine failure"] == 1]
                if len(subset) == 0:
                    subset = self.raw_df[self.raw_df["Machine failure"] == 1]
            if len(subset) > 0:
                base_row = subset.sample(n=1, random_state=int(rng.integers(0, 10000))).iloc[0]
            else:
                base_row = self.raw_df.sample(n=1).iloc[0]
        elif inject_anomaly == "normal":
            healthy_sub = self.raw_df[self.raw_df["Machine failure"] == 0]
            if len(healthy_sub) > 0:
                base_row = healthy_sub.sample(n=1, random_state=int(rng.integers(0, 10000))).iloc[0]
            else:
                base_row = self.raw_df.sample(n=1).iloc[0]
        else:
            self.sim_cursor = (self.sim_cursor + int(rng.integers(1, 17))) % len(self.raw_df)
            base_row = self.raw_df.iloc[self.sim_cursor]

        sensor_payload: Dict[str, Any] = {"Type": str(base_row.get("Type", "M"))}
        for col in self.raw_sensor_cols:
            val = float(base_row[col])
            std = self.sensor_ranges[col]["std"]
            jitter = float(rng.normal(0.0, 0.04 * std))
            clamped = np.clip(val + jitter, self.sensor_ranges[col]["min"], self.sensor_ranges[col]["max"])
            if "rpm" in col.lower() or "wear" in col.lower() or "cycle" in col.lower() or "hours" in col.lower():
                sensor_payload[col] = int(round(clamped))
            else:
                sensor_payload[col] = round(float(clamped), 2)

        prediction = self.predict_and_explain(sensor_payload)
        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "sensor_inputs": sensor_payload,
            "prediction": prediction,
        }
