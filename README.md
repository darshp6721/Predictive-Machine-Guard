# Predictive-Machine Guard 🛡️⚙️ (v2.0 — Applied ML & Deep Learning)

**End-to-End Industrial Predictive Maintenance Platform featuring Automated Missing/Outlier Preprocessing, Classical ML + PyTorch Deep Learning (ANN, 1D-CNN, BiLSTM), RUL Prognostics, SHAP Explainability, and an AI Maintenance Recommendation Copilot**

---

## 📂 Step-by-Step Code File Guide

| File Path | Role & Key Operations |
| :--- | :--- |
| [`ml_engine/datasets.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/ml_engine/datasets.py) | Generates/loads **AI4I 2020 Milling (10k rows)**, **NASA CMAPSS Turbofan**, and **Custom Vibro-Acoustic Telemetry**, injects realistic raw telemetry missing values (`NaN`) and EMI sensor spikes for cleaning verification, and auto-maps user-uploaded CSV schemas. |
| [`ml_engine/preprocessing.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/ml_engine/preprocessing.py) | **`DataQualityPreprocessor`**: Performs duplicate removal, Type-stratified Median/Mode missing value imputation, and Tukey $1.5 \times \text{IQR}$ + Z-Score ($|Z|>3$) outlier detection with Winsorization clipping. |
| [`ml_engine/deep_learning.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/ml_engine/deep_learning.py) | **`DeepLearningClassifierWrapper` (PyTorch)**: Implements **ANN (Deep MLP)**, **1D-CNN (1D Convolutional Neural Network)**, and **BiLSTM (Bidirectional LSTM)** with `StandardScaler`, class-weighted `BCEWithLogitsLoss`, and epoch train/validation loss tracking. |
| [`ml_engine/ai_recommender.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/ml_engine/ai_recommender.py) | **`AIMaintenanceRecommender`**: Synthesizes 7-model ML/DL predictions, SHAP attributions, and domain physics to generate executive diagnostics, prioritized work orders (`P1`/`P2`/`P3`), spare parts forecasts, and counterfactual optimal sensor setpoints. |
| [`ml_engine/pipeline.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/ml_engine/pipeline.py) | **`PredictiveMaintenanceEngine`**: Orchestrates preprocessing, physics feature engineering, stratified **Train (70%) / Validation (10%) / Test (20%)** splitting, **SMOTE** oversampling, 7-model training, RUL regression, and `TreeSHAP` explainability. |
| [`app.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/app.py) | **FastAPI Backend Server**: Exposes REST API endpoints (`/api/state`, `/api/switch-dataset`, `/api/upload-csv`, `/api/predict`, `/api/simulate-step`, `/api/sample-rows`, `/api/download-dataset`). |
| [`static/index.html`](file:///c:/Users/darsh/Predictive-Machine-Guard/static/index.html), [`static/styles.css`](file:///c:/Users/darsh/Predictive-Machine-Guard/static/styles.css), [`static/app.js`](file:///c:/Users/darsh/Predictive-Machine-Guard/static/app.js) | **4-Tab Interactive Web Application**: Live Digital Twin, 1-Click AI Optimal Setpoint Tuner, 7-Model Benchmark & DL Loss Curves, SHAP Waterfall Explorer, and Data Preprocessing Audit Studio. |
| [`train_and_evaluate.py`](file:///c:/Users/darsh/Predictive-Machine-Guard/train_and_evaluate.py) | Standalone CLI verification script that runs all 3 datasets through preprocessing, 7-model training, RUL regression, and AI recommendation testing, saving artifacts to `models/`. |

---

## 🚀 Quickstart

### 1. Launch the Full-Stack Web Application
```powershell
cd c:\Users\darsh\Predictive-Machine-Guard
.\.venv\Scripts\python.exe app.py
```
Open **http://127.0.0.1:8000** in your browser.

### 2. Run the Multi-Dataset CLI Benchmark Suite
```powershell
cd c:\Users\darsh\Predictive-Machine-Guard
.\.venv\Scripts\python.exe train_and_evaluate.py
```
