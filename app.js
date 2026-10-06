/**
 * Predictive-Machine Guard: Frontend Interactive Controller (v2.0)
 * Supports:
 *  - Automated Data Preprocessing Audit (Missing Values Imputed + IQR/Z-Score Outliers Winsorized)
 *  - 7-Model Unified Suite: 4 Classical ML (XGBoost, LightGBM, RF, LogReg) + 3 PyTorch Deep Learning (ANN, 1D-CNN, BiLSTM)
 *  - Counterfactual Parameter Setpoint Optimizer
 *  - Live Telemetry Stream Simulator & Local/Global TreeSHAP Explainability
 */

const state = {
  theme: localStorage.getItem("pmg_theme") || "dark",
  datasetKey: "ai4i2020",
  useSmote: true,
  selectedModel: "XGBoost",
  selectedType: "M",
  sensorInputs: {},
  sensorRanges: {},
  rawSensorCols: [],
  dashboardData: null,
  lastPrediction: null,
  streamInterval: null,
  streamHistory: [],
};

const MODEL_COLORS = {
  "XGBoost": "#38bdf8",
  "LightGBM": "#10b981",
  "Random Forest": "#f59e0b",
  "Logistic Regression": "#94a3b8",
  "ANN (Deep MLP)": "#14b8a6",
  "1D-CNN": "#f43f5e",
  "BiLSTM": "#fb923c",
};

const loadingOverlay = document.getElementById("loadingOverlay");
const datasetSelect = document.getElementById("datasetSelect");
const modelSelect = document.getElementById("modelSelect");
const smoteToggleBtn = document.getElementById("smoteToggleBtn");
const smoteStatusText = document.getElementById("smoteStatusText");
const csvUploadInput = document.getElementById("csvUploadInput");

document.addEventListener("DOMContentLoaded", async () => {
  applyTheme(state.theme);
  setupTabs();
  setupTopbarEvents();
  setupTwinEvents();
  setupUploadEvents();
  await fetchInitialState();
});

function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem("pmg_theme", theme);
  document.body.setAttribute("data-theme", theme);
  const labelEl = document.getElementById("themeLabel");
  if (labelEl) {
    labelEl.textContent = theme === "light" ? "Theme: Dark" : "Theme: Light";
  }
  drawLiveStreamCanvas();
  if (state.dashboardData) {
    drawRocAndDlLossCurves(state.dashboardData);
  }
}

function showLoading(show, title = "Training Applied ML & Deep Learning Pipeline...") {
  document.getElementById("loadingTitle").textContent = title;
  loadingOverlay.classList.toggle("hidden", !show);
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
      if (btn.dataset.tab === "tab-models" && state.dashboardData) {
        drawRocAndDlLossCurves(state.dashboardData);
      }
    });
  });
}

function setupTopbarEvents() {
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      applyTheme(state.theme === "dark" ? "light" : "dark");
    });
  }

  datasetSelect.addEventListener("change", async (e) => {
    state.datasetKey = e.target.value;
    await switchDatasetOrSmote();
  });

  modelSelect.addEventListener("change", async (e) => {
    state.selectedModel = e.target.value;
    await runSinglePrediction();
  });

  smoteToggleBtn.addEventListener("click", async () => {
    state.useSmote = !state.useSmote;
    smoteToggleBtn.classList.toggle("active", state.useSmote);
    smoteStatusText.textContent = state.useSmote ? "SMOTE ON" : "SMOTE OFF";
    await switchDatasetOrSmote();
  });

  const exportCsvBtn = document.getElementById("exportCsvBtn");
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener("click", async (e) => {
      e.preventDefault();
      try {
        const res = await fetch("/api/download-dataset");
        if (res.ok) {
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `${state.datasetKey}_cleaned_dataset.csv`;
          a.click();
          URL.revokeObjectURL(url);
          return;
        }
      } catch (_) {}
      if (window.PMGStandaloneEngine) {
        const csvText = window.PMGStandaloneEngine.exportCsv(state.datasetKey);
        const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${state.datasetKey}_cleaned_dataset.csv`;
        a.click();
        URL.revokeObjectURL(url);
      }
    });
  }
}

function setupTwinEvents() {
  document.querySelectorAll(".tier-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      document.querySelectorAll(".tier-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.selectedType = btn.dataset.type;
      state.sensorInputs["Type"] = state.selectedType;
      await runSinglePrediction();
    });
  });

  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const scenario = chip.dataset.scenario;
      await injectScenarioStep(scenario);
    });
  });

  // 1-Click AI Counterfactual Parameter Setpoint Optimizer
  document.getElementById("applyAiOptimalBtn").addEventListener("click", async () => {
    if (!state.lastPrediction || !state.lastPrediction.prescription) return;
    const optInputs = state.lastPrediction.prescription.optimal_sensor_inputs;
    if (optInputs) {
      state.sensorInputs = { ...optInputs };
      updateSlidersFromInputs();
      await runSinglePrediction();
    }
  });

  // Live Telemetry Stream toggle
  const streamToggleBtn = document.getElementById("streamToggleBtn");
  streamToggleBtn.addEventListener("click", () => {
    if (state.streamInterval) {
      clearInterval(state.streamInterval);
      state.streamInterval = null;
      streamToggleBtn.textContent = "Start Stream";
      streamToggleBtn.classList.remove("btn-danger-outline");
      streamToggleBtn.classList.add("btn-primary");
    } else {
      streamToggleBtn.textContent = "Pause Stream";
      streamToggleBtn.classList.remove("btn-primary");
      streamToggleBtn.classList.add("btn-danger-outline");
      state.streamInterval = setInterval(() => {
        injectScenarioStep("normal");
      }, 1700);
    }
  });

  document.getElementById("streamSpikeBtn").addEventListener("click", async () => {
    const modes = ["TWF", "HDF", "PWF", "OSF"];
    const pick = modes[Math.floor(Math.random() * modes.length)];
    await injectScenarioStep(pick);
  });

  document.getElementById("clearAlertsBtn").addEventListener("click", () => {
    document.getElementById("liveAlertFeed").innerHTML = "";
  });

  document.getElementById("refreshSamplesBtn").addEventListener("click", () => {
    loadSampleRowsTable();
  });
}

function setupUploadEvents() {
  const browseBtn = document.getElementById("browseCsvBtn");
  const dropzone = document.getElementById("dropzoneArea");

  browseBtn.addEventListener("click", () => csvUploadInput.click());
  csvUploadInput.addEventListener("change", async (e) => {
    if (e.target.files && e.target.files[0]) {
      await uploadCustomCsvFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#10b981";
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "rgba(56, 189, 248, 0.4)";
  });
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "rgba(56, 189, 248, 0.4)";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await uploadCustomCsvFile(e.dataTransfer.files[0]);
    }
  });
}

async function fetchInitialState() {
  showLoading(true, "Initializing Predictive-Machine Guard (ML + PyTorch ANN/CNN/BiLSTM)...");
  try {
    let data = null;
    try {
      const res = await fetch("/api/state");
      if (res.ok) data = await res.json();
    } catch (_) {}
    if (!data && window.PMGStandaloneEngine) {
      data = window.PMGStandaloneEngine.getState(state.datasetKey, state.useSmote);
    }
    if (data) {
      applyDashboardState(data);
      await loadSampleRowsTable();
    }
  } catch (err) {
    console.error("Failed to load initial state:", err);
  } finally {
    showLoading(false);
  }
}

async function switchDatasetOrSmote() {
  showLoading(
    true,
    `Cleaning Outliers/Missing Values & Training 7 ML/DL Models on ${state.datasetKey.toUpperCase()}...`
  );
  try {
    let data = null;
    try {
      const res = await fetch("/api/switch-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_key: state.datasetKey,
          use_smote: state.useSmote,
        }),
      });
      if (res.ok) data = await res.json();
    } catch (_) {}
    if (!data && window.PMGStandaloneEngine) {
      data = window.PMGStandaloneEngine.getState(state.datasetKey, state.useSmote);
    }
    if (data) {
      state.streamHistory = [];
      applyDashboardState(data);
      await loadSampleRowsTable();
    }
  } catch (err) {
    console.error("Error switching dataset:", err);
  } finally {
    showLoading(false);
  }
}

async function uploadCustomCsvFile(file) {
  showLoading(true, `Cleaning & Training 7 ML/DL Models on Uploaded CSV (${file.name})...`);
  try {
    let data = null;
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`/api/upload-csv?use_smote=${state.useSmote}`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        data = await res.json();
      }
    } catch (_) {}

    if (!data && window.PMGStandaloneEngine) {
      const text = await file.text();
      data = window.PMGStandaloneEngine.ingestCustomCsv(text, file.name, state.useSmote);
    }

    if (data) {
      state.datasetKey = "custom_plant";
      datasetSelect.value = "custom_plant";
      state.streamHistory = [];
      applyDashboardState(data);
      await loadSampleRowsTable();

      const banner = document.getElementById("uploadStatusBanner");
      if (data.uploaded_schema_info) {
        banner.classList.remove("hidden");
        banner.textContent = `[OK] Cleaned & Trained '${data.uploaded_schema_info.filename}' (${data.uploaded_schema_info.row_count} rows, Failure Rate: ${data.uploaded_schema_info.failure_rate_pct}%, Target: ${data.uploaded_schema_info.detected_target})`;
      }
    }
  } catch (err) {
    console.error("Upload error:", err);
  } finally {
    showLoading(false);
  }
}

function applyDashboardState(data) {
  state.dashboardData = data;
  const ds = data.dataset_summary;
  const prep = data.preprocessing_report || {};
  state.rawSensorCols = ds.raw_sensor_cols;
  state.sensorRanges = ds.sensor_ranges;
  state.selectedModel = ds.best_model_name;
  modelSelect.value = state.selectedModel;

  // 1. Update Top KPI Strip
  document.getElementById("kpiTotalRecords").textContent = ds.total_records.toLocaleString();
  document.getElementById("kpiSplitMeta").textContent =
    `Train: ${ds.train_records.toLocaleString()} | Val: ${(ds.val_records || 0).toLocaleString()} | Test: ${ds.test_records.toLocaleString()}`;

  document.getElementById("kpiCleanedSummary").textContent =
    `${(prep.total_missing_detected || 0).toLocaleString()} NaNs Imputed`;
  document.getElementById("kpiOutlierSummary").textContent =
    `${(prep.total_iqr_outliers_detected || 0).toLocaleString()} IQR Outliers Winsorized`;

  const bestModelObj = data.model_comparison[0];
  document.getElementById("kpiBestModel").textContent = bestModelObj.model_name;
  document.getElementById("kpiBestModelMetrics").textContent =
    `F1: ${bestModelObj.f1_score}% | ROC-AUC: ${bestModelObj.roc_auc}%`;

  const rul = data.rul_metrics;
  document.getElementById("kpiRulR2").textContent = `R²: ${rul.r2_score}`;
  document.getElementById("kpiRulErrors").textContent = `MAE: ${rul.mae} | RMSE: ${rul.rmse}`;
  document.getElementById("kpiFailureRate").textContent = `${ds.failure_rate_pct}%`;
  document.getElementById("kpiTrainTime").textContent =
    `${ds.feature_cols.length} features • 7 models (${ds.pipeline_train_time_ms} ms)`;

  // 2. Build Sensor Sliders & Dynamic Fault Scenario Buttons
  const scenarioSlots = ["TWF", "HDF", "PWF", "OSF"];
  const ai4iDefaults = {
    TWF: "Tool Wear (TWF)",
    HDF: "Heat Dissipation (HDF)",
    PWF: "Power Fault (PWF)",
    OSF: "Overstrain (OSF)",
  };
  const modeList = (ds.failure_modes_breakdown || []).map((m) => m.mode);
  scenarioSlots.forEach((slot, idx) => {
    const btn = document.querySelector(`.chip-btn[data-scenario="${slot}"]`);
    if (!btn) return;
    if (ds.dataset_key === "ai4i2020" || modeList.length === 0) {
      btn.textContent = ai4iDefaults[slot];
    } else {
      btn.textContent = modeList[idx % modeList.length];
    }
  });

  if (data.initial_prediction) {
    state.sensorInputs = { ...data.initial_prediction.sensor_inputs };
    state.selectedType = state.sensorInputs.Type || "M";
  }
  renderSensorSliders();

  // 3. Apply Initial Prediction
  if (data.initial_prediction) {
    recordStreamPoint(data.initial_prediction);
    renderPredictionOutput(data.initial_prediction.prediction);
  }

  // 4. Render Tab 2 Benchmark Table, Confusion Matrices, SMOTE Ablation & ROC/DL Loss Curves
  renderModelBenchmarkTab(data);

  // 5. Render Tab 3 Global SHAP & Dataset Failure Mode Breakdown
  renderGlobalShapAndModes(data);

  // 6. Render Tab 4 Preprocessing Audit Table
  renderPreprocessingAuditTab(prep);
}

function renderSensorSliders() {
  const container = document.getElementById("sensorSlidersContainer");
  container.innerHTML = "";

  state.rawSensorCols.forEach((col) => {
    const range = state.sensorRanges[col];
    const currentVal = state.sensorInputs[col] !== undefined ? state.sensorInputs[col] : range.median;
    state.sensorInputs[col] = currentVal;

    const item = document.createElement("div");
    item.className = "slider-item";
    item.innerHTML = `
      <div class="slider-top">
        <span class="slider-name">${col}</span>
        <span class="slider-val" id="val-${cssSafe(col)}">${currentVal}</span>
      </div>
      <input
        type="range"
        class="slider-range-input"
        id="input-${cssSafe(col)}"
        min="${range.min}"
        max="${range.max}"
        step="${range.step}"
        value="${currentVal}"
      />
      <div class="slider-bounds">
        <span>Min: ${range.min}</span>
        <span>Median: ${range.median}</span>
        <span>Max: ${range.max}</span>
      </div>
    `;

    const inputEl = item.querySelector("input");
    const valEl = item.querySelector(".slider-val");
    inputEl.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      valEl.textContent = v;
      state.sensorInputs[col] = v;
    });
    inputEl.addEventListener("change", async () => {
      await runSinglePrediction();
    });

    container.appendChild(item);
  });
}

function updateSlidersFromInputs() {
  state.rawSensorCols.forEach((col) => {
    const v = state.sensorInputs[col];
    const inputEl = document.getElementById(`input-${cssSafe(col)}`);
    const valEl = document.getElementById(`val-${cssSafe(col)}`);
    if (inputEl && v !== undefined) inputEl.value = v;
    if (valEl && v !== undefined) valEl.textContent = v;
  });
  document.querySelectorAll(".tier-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.type === (state.sensorInputs.Type || "M"));
  });
}

async function runSinglePrediction() {
  try {
    let pred = null;
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sensor_inputs: state.sensorInputs,
          model_name: state.selectedModel,
        }),
      });
      if (res.ok) pred = await res.json();
    } catch (_) {}
    if (!pred && window.PMGStandaloneEngine) {
      pred = window.PMGStandaloneEngine.predict(state.datasetKey, state.sensorInputs, state.selectedModel);
    }
    if (pred) {
      const packet = {
        timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
        sensor_inputs: { ...state.sensorInputs },
        prediction: pred,
      };
      recordStreamPoint(packet);
      renderPredictionOutput(pred);
    }
  } catch (err) {
    console.error("Prediction error:", err);
  }
}

async function injectScenarioStep(scenario) {
  try {
    let packet = null;
    try {
      const res = await fetch(`/api/simulate-step?inject_anomaly=${encodeURIComponent(scenario)}`);
      if (res.ok) packet = await res.json();
    } catch (_) {}
    if (!packet && window.PMGStandaloneEngine) {
      packet = window.PMGStandaloneEngine.simulateStep(state.datasetKey, scenario, state.selectedModel);
    }
    if (packet) {
      state.sensorInputs = { ...packet.sensor_inputs };
      updateSlidersFromInputs();
      recordStreamPoint(packet);
      renderPredictionOutput(packet.prediction);
    }
  } catch (err) {
    console.error("Simulation step error:", err);
  }
}

function renderPredictionOutput(pred) {
  state.lastPrediction = pred;
  const rec = pred.prescription;

  document.getElementById("activeInferenceSubtitle").textContent =
    `Active Model: ${pred.selected_model} • Classical ML Avg: ${rec.classical_ml_avg_risk_pct}% • Deep Learning (ANN/CNN/BiLSTM) Avg: ${rec.deep_learning_avg_risk_pct}%`;

  // 1. Severity Badge
  const badge = document.getElementById("severityBadge");
  badge.textContent = rec.severity;
  badge.className = `status-badge status-${rec.badge_color}`;

  // 2. Radial Gauges
  const arcLen = 173;
  const failOffset = arcLen - (Math.min(100, Math.max(0, pred.failure_probability_pct)) / 100) * arcLen;
  const failArc = document.getElementById("failureGaugeArc");
  failArc.style.strokeDashoffset = failOffset;
  failArc.style.stroke =
    pred.failure_probability_pct >= 65
      ? "#f43f5e"
      : pred.failure_probability_pct >= 35
      ? "#f59e0b"
      : "#38bdf8";

  const failValEl = document.getElementById("failureProbValue");
  failValEl.textContent = `${pred.failure_probability_pct}%`;
  failValEl.style.color = failArc.style.stroke;

  const healthOffset = arcLen - (Math.min(100, Math.max(0, pred.health_index_pct)) / 100) * arcLen;
  const healthArc = document.getElementById("healthGaugeArc");
  healthArc.style.strokeDashoffset = healthOffset;
  healthArc.style.stroke =
    pred.health_index_pct <= 35 ? "#f43f5e" : pred.health_index_pct <= 65 ? "#f59e0b" : "#10b981";
  document.getElementById("healthIndexValue").textContent = `${pred.health_index_pct}%`;

  // 3. Remaining Useful Life (RUL)
  document.getElementById("rulValue").textContent = pred.predicted_rul_min;
  document.getElementById("rulUnit").textContent = state.datasetKey === "cmapss" ? "cycles" : "min";
  const maxRef = state.datasetKey === "cmapss" ? 125 : 220;
  const rulPct = Math.min(100, Math.max(4, (pred.predicted_rul_min / maxRef) * 100));
  document.getElementById("rulProgressBar").style.width = `${rulPct}%`;
  document.getElementById("dominantModeLabel").textContent = `Mode: ${pred.dominant_failure_mode}`;

  // 4. Engineered Physics Features Grid
  const engGrid = document.getElementById("engineeredFeaturesGrid");
  engGrid.innerHTML = "";
  Object.entries(pred.engineered_features || {}).forEach(([k, v]) => {
    const chip = document.createElement("div");
    chip.className = "eng-chip";
    chip.innerHTML = `
      <span class="eng-chip-label">${k}</span>
      <span class="eng-chip-val">${v}</span>
    `;
    engGrid.appendChild(chip);
  });

  // 5. 7-Model Consensus Bar (4 Classical ML + 3 Deep Learning)
  const consensusGrid = document.getElementById("modelConsensusGrid");
  consensusGrid.innerHTML = "";
  const dlSet = new Set(["ANN (Deep MLP)", "1D-CNN", "BiLSTM"]);
  Object.entries(pred.all_model_probabilities || {}).forEach(([mName, prob]) => {
    const isDl = dlSet.has(mName);
    const card = document.createElement("div");
    card.className = `consensus-card ${mName === pred.selected_model ? "selected" : ""} ${isDl ? "dl-card" : ""}`;
    const cColor = prob >= 65 ? "text-rose" : prob >= 35 ? "text-amber" : isDl ? "text-teal" : "text-cyan";
    card.innerHTML = `
      <div class="consensus-name" title="${mName}">${mName}</div>
      <div class="consensus-val ${cColor}">${prob}%</div>
    `;
    card.addEventListener("click", async () => {
      state.selectedModel = mName;
      modelSelect.value = mName;
      await runSinglePrediction();
    });
    consensusGrid.appendChild(card);
  });

  // 6. Counterfactual Parameter Optimizer
  document.getElementById("prescriptionUrgency").textContent =
    `Urgency Window: ${rec.urgency_window} | ${rec.confidence_tier || ""}`;
  const riskDelta = Number((rec.projected_optimal_risk_pct - pred.failure_probability_pct).toFixed(2));
  const deltaSign = riskDelta > 0 ? "+" : "";
  document.getElementById("avoidedCostValue").textContent =
    `${deltaSign}${riskDelta}% (${pred.failure_probability_pct}% -> ${rec.projected_optimal_risk_pct}%)`;
  document.getElementById("aiExecutiveSummaryBox").textContent = rec.ai_executive_summary || "";

  document.getElementById("projectedOptimizationBadge").textContent =
    `Projected Post-Tuning Risk: ${rec.projected_optimal_risk_pct}% | Projected RUL: ${rec.projected_optimal_rul_min} min`;

  const tuningBody = document.getElementById("aiParameterTuningBody");
  tuningBody.innerHTML = "";
  (rec.parameter_tuning || []).forEach((t) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${t.sensor}</strong></td>
      <td style="font-family:var(--font-mono);">${t.current}</td>
      <td style="font-family:var(--font-mono);" class="text-emerald"><strong>${t.recommended}</strong></td>
      <td style="font-family:var(--font-mono);" class="text-cyan">${t.change}</td>
      <td>${t.rationale}</td>
    `;
    tuningBody.appendChild(tr);
  });

  const diagList = document.getElementById("physicsDiagnosticsList");
  diagList.innerHTML = "";
  (rec.physics_diagnostics || []).forEach((d) => {
    const li = document.createElement("li");
    li.textContent = d;
    diagList.appendChild(li);
  });

  const actList = document.getElementById("recommendedActionsList");
  actList.innerHTML = "";
  (rec.recommended_actions || []).forEach((a) => {
    const li = document.createElement("li");
    li.textContent = a;
    actList.appendChild(li);
  });

  const spareLine = document.getElementById("sparePartsLine");
  spareLine.textContent = `Required Parts: ${(rec.spare_parts_forecast || []).join(" | ")}`;

  // 7. Local SHAP Bars
  renderLocalShapBars(pred.local_shap_explanations, "liveLocalShapBars", 6);
  renderLocalShapBars(pred.local_shap_explanations, "detailedLocalShapContainer", 10);
  renderFailureModeProbs(pred.failure_mode_distribution);
}

function renderLocalShapBars(shapList, containerId, limit = 8) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";

  const items = (shapList || []).slice(0, limit);
  const maxAbs = Math.max(...items.map((x) => Math.abs(x.shap_impact)), 0.01);

  items.forEach((item) => {
    const pct = Math.min(100, Math.max(6, (Math.abs(item.shap_impact) / maxAbs) * 100));
    const isPos = item.shap_impact >= 0;
    const sign = isPos ? "+" : "";
    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <span class="shap-feature-name" title="${item.feature} = ${item.value}">
        ${item.feature}${item.is_engineered ? '<span class="eng-badge">PHYS</span>' : ""}
      </span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${isPos ? "shap-pos" : "shap-neg"}" style="width: ${pct}%;"></div>
      </div>
      <span class="shap-val-text ${isPos ? "text-rose" : "text-emerald"}">${sign}${item.shap_impact.toFixed(3)}</span>
    `;
    container.appendChild(row);
  });
}

function renderFailureModeProbs(modeList) {
  const container = document.getElementById("failureModeProbsContainer");
  if (!container) return;
  container.innerHTML = "";
  (modeList || []).forEach((item) => {
    const pct = Math.min(100, Math.max(3, item.probability_pct));
    const isHealthy = item.mode === "No Failure";
    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <span class="shap-feature-name">${item.mode}</span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${isHealthy ? "shap-neg" : "shap-pos"}" style="width: ${pct}%;"></div>
      </div>
      <span class="shap-val-text">${item.probability_pct}%</span>
    `;
    container.appendChild(row);
  });
}

function renderGlobalShapAndModes(data) {
  const globalContainer = document.getElementById("globalShapContainer");
  globalContainer.innerHTML = "";
  const list = data.global_shap_importance || [];
  const maxPct = Math.max(...list.map((x) => x.importance_pct), 1);

  list.forEach((item) => {
    const widthPct = Math.min(100, Math.max(5, (item.importance_pct / maxPct) * 100));
    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <span class="shap-feature-name">
        ${item.feature}${item.is_engineered ? '<span class="eng-badge">PHYS</span>' : ""}
      </span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill shap-global" style="width: ${widthPct}%;"></div>
      </div>
      <span class="shap-val-text text-cyan">${item.importance_pct}%</span>
    `;
    globalContainer.appendChild(row);
  });

  const dsModesContainer = document.getElementById("datasetFailureModesBreakdown");
  dsModesContainer.innerHTML = "";
  const modes = data.dataset_summary.failure_modes_breakdown || [];
  const maxCount = Math.max(...modes.map((m) => m.count), 1);
  modes.forEach((m) => {
    const w = Math.min(100, Math.max(8, (m.count / maxCount) * 100));
    const row = document.createElement("div");
    row.className = "shap-row";
    row.innerHTML = `
      <span class="shap-feature-name">${m.mode}</span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill shap-pos" style="width: ${w}%;"></div>
      </div>
      <span class="shap-val-text">${m.count} (${m.pct}%)</span>
    `;
    dsModesContainer.appendChild(row);
  });
}

function renderPreprocessingAuditTab(prep) {
  const tbody = document.getElementById("preprocessingAuditTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";
  const audits = prep.column_audits || [];
  audits.forEach((a) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${a.column}</strong></td>
      <td><span class="text-amber">${a.missing_before} (${a.missing_pct}%)</span> → <strong class="text-emerald">${a.missing_after}</strong></td>
      <td>${a.imputation_strategy}</td>
      <td class="text-cyan">${a.iqr_outliers_detected}</td>
      <td class="text-rose">${a.zscore_outliers_detected}</td>
      <td style="font-family:var(--font-mono);font-size:0.73rem;">${a.outlier_treatment}</td>
      <td style="font-family:var(--font-mono);">${a.mean_before} ± ${a.std_before}</td>
      <td style="font-family:var(--font-mono);" class="text-emerald">${a.mean_after} ± ${a.std_after}</td>
    `;
    tbody.appendChild(tr);
  });
}

function recordStreamPoint(packet) {
  const pred = packet.prediction;
  state.streamHistory.push({
    time: packet.timestamp,
    failProb: pred.failure_probability_pct,
    healthIndex: pred.health_index_pct,
    rul: pred.predicted_rul_min,
  });
  if (state.streamHistory.length > 32) {
    state.streamHistory.shift();
  }
  drawLiveStreamCanvas();

  const feed = document.getElementById("liveAlertFeed");
  if (pred.prescription.severity !== "OPTIMAL" || feed.children.length === 0) {
    const alertDiv = document.createElement("div");
    alertDiv.className = `alert-item alert-${pred.prescription.severity}`;
    alertDiv.innerHTML = `
      <span><strong>[${packet.timestamp}] ${pred.prescription.severity}</strong>: ${pred.dominant_failure_mode}</span>
      <span class="text-cyan">Risk: ${pred.failure_probability_pct}% | RUL: ${pred.predicted_rul_min}m</span>
    `;
    feed.prepend(alertDiv);
    while (feed.children.length > 15) {
      feed.removeChild(feed.lastChild);
    }
  }
}

function drawLiveStreamCanvas() {
  const canvas = document.getElementById("liveStreamCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(148, 163, 184, 0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = 20 + ((H - 40) * i) / 4;
    ctx.beginPath();
    ctx.moveTo(36, y);
    ctx.lineTo(W - 12, y);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText(`${100 - i * 25}%`, 4, y + 3);
  }

  const pts = state.streamHistory;
  if (pts.length < 2) return;

  const plotLine = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    pts.forEach((p, idx) => {
      const x = 38 + (idx / Math.max(1, pts.length - 1)) * (W - 52);
      const y = 20 + ((100 - Math.min(100, Math.max(0, p[key]))) / 100) * (H - 40);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  plotLine("healthIndex", "#10b981");
  plotLine("failProb", "#f43f5e");

  const legendColor = state.theme === "light" ? "#0f172a" : "#cbd5e1";
  ctx.fillStyle = "#10b981";
  ctx.fillRect(45, 5, 10, 8);
  ctx.fillStyle = legendColor;
  ctx.font = "11px Inter";
  ctx.fillText("Health Index (%)", 60, 13);

  ctx.fillStyle = "#f43f5e";
  ctx.fillRect(175, 5, 10, 8);
  ctx.fillStyle = legendColor;
  ctx.fillText("Failure Risk (%)", 190, 13);
}

function renderModelBenchmarkTab(data) {
  const tbody = document.getElementById("modelBenchmarkTableBody");
  tbody.innerHTML = "";
  const models = data.model_comparison || [];

  models.forEach((m, idx) => {
    const tr = document.createElement("tr");
    const badge = m.is_deep_learning
      ? '<span class="dl-badge">PYTORCH DL</span>'
      : '<span class="eng-badge">CLASSICAL ML</span>';
    tr.innerHTML = `
      <td>
        <strong style="color:${MODEL_COLORS[m.model_name] || "#38bdf8"}">${m.model_name}</strong>
        ${idx === 0 ? '<span class="eng-badge">CHAMPION</span>' : ""}
      </td>
      <td>${badge} <small style="color:var(--text-secondary);">${m.model_family || ""}</small></td>
      <td><strong>${m.f1_score}%</strong></td>
      <td>${m.roc_auc}%</td>
      <td>${m.pr_auc}%</td>
      <td class="text-emerald">${m.recall}%</td>
      <td>${m.precision}%</td>
      <td>${m.train_accuracy || m.accuracy}%</td>
      <td><strong>${m.accuracy}%</strong></td>
      <td>${m.mcc}</td>
      <td>${m.brier_score}</td>
      <td>${m.train_time_ms} ms</td>
    `;
    tbody.appendChild(tr);
  });

  // Confusion Matrices
  const cmGrid = document.getElementById("confusionMatricesGrid");
  cmGrid.innerHTML = "";
  models.forEach((m) => {
    const cm = m.confusion_matrix;
    const box = document.createElement("div");
    box.className = "cm-box";
    box.innerHTML = `
      <div class="cm-title">${m.model_name} (F1: ${m.f1_score}%)</div>
      <div class="cm-matrix">
        <div class="cm-cell cm-tn">TN: ${cm.tn}</div>
        <div class="cm-cell cm-fp">FP: ${cm.fp}</div>
        <div class="cm-cell cm-fn">FN: ${cm.fn}</div>
        <div class="cm-cell cm-tp">TP: ${cm.tp}</div>
      </div>
    `;
    cmGrid.appendChild(box);
  });

  // SMOTE Ablation & RUL Card
  const sm = data.smote_comparison;
  const rul = data.rul_metrics;
  const smoteBox = document.getElementById("smoteAblationContainer");
  smoteBox.innerHTML = `
    <div class="ablation-cards">
      <div class="ablation-card">
        <div class="cm-title">Without SMOTE (Imbalanced Baseline)</div>
        <p style="font-size:0.75rem;color:var(--text-secondary);margin-bottom:0.35rem;">
          Minority Train Positives: <strong>${sm.train_positives_before}</strong> vs Negatives: <strong>${sm.train_negatives}</strong>
        </p>
        <div style="font-family:var(--font-mono);font-size:0.8rem;">
          Recall: <span class="text-amber">${sm.without_smote.recall}%</span> |
          Precision: ${sm.without_smote.precision}% |
          F1: <strong>${sm.without_smote.f1_score}%</strong>
        </div>
      </div>
      <div class="ablation-card">
        <div class="cm-title">With SMOTE Minority Oversampling</div>
        <p style="font-size:0.75rem;color:var(--text-secondary);margin-bottom:0.35rem;">
          Synthesized Train Positives: <strong class="text-emerald">${sm.train_positives_after}</strong>
        </p>
        <div style="font-family:var(--font-mono);font-size:0.8rem;">
          Recall: <span class="text-emerald">${sm.with_smote.recall}%</span> |
          Precision: ${sm.with_smote.precision}% |
          F1: <strong class="text-cyan">${sm.with_smote.f1_score}%</strong>
        </div>
      </div>
    </div>
    <div class="ablation-card">
      <div class="cm-title">Remaining Useful Life (RUL) Regression Evaluation</div>
      <div style="font-family:var(--font-mono);font-size:0.83rem;display:flex;gap:1.4rem;margin-top:0.25rem;">
        <span>R² Score: <strong class="text-emerald">${rul.r2_score}</strong></span>
        <span>MAE: <strong class="text-cyan">${rul.mae} min</strong></span>
        <span>RMSE: <strong>${rul.rmse} min</strong></span>
      </div>
    </div>
  `;

  drawRocAndDlLossCurves(data);
}

function drawRocAndDlLossCurves(data) {
  drawCurveCanvas("rocChartCanvas", data.model_comparison || [], "roc_curve", "fpr", "tpr");
  drawDlLossCanvas("dlLossChartCanvas", data.dl_loss_curves || {});
}

function drawCurveCanvas(canvasId, models, curveKey, xKey, yKey) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const padL = 44, padR = 18, padT = 18, padB = 46;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const legendColor = state.theme === "light" ? "#0f172a" : "#e2e8f0";

  ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText((1 - i * 0.25).toFixed(2), 6, y + 3);
  }

  ctx.strokeStyle = "rgba(148, 163, 184, 0.35)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT);
  ctx.stroke();
  ctx.setLineDash([]);

  models.forEach((m, idx) => {
    const xs = m[curveKey][xKey];
    const ys = m[curveKey][yKey];
    const color = MODEL_COLORS[m.model_name] || "#38bdf8";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.1;
    ctx.beginPath();
    xs.forEach((xv, i) => {
      const px = padL + xv * plotW;
      const py = padT + (1 - ys[i]) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    const colIdx = idx % 3;
    const rowIdx = Math.floor(idx / 3);
    const lx = padL + colIdx * 155;
    const ly = H - 24 + rowIdx * 12;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 7, 8, 7);
    ctx.fillStyle = legendColor;
    ctx.font = "10px Inter";
    ctx.fillText(`${m.model_name} (${m.roc_auc}%)`, lx + 12, ly);
  });
}

function drawDlLossCanvas(canvasId, lossMap) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const entries = Object.entries(lossMap || {});
  if (!entries.length) return;

  const padL = 44, padR = 18, padT = 18, padB = 40;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const legendColor = state.theme === "light" ? "#0f172a" : "#e2e8f0";

  let maxLoss = 0.8;
  entries.forEach(([, obj]) => {
    (obj.train_loss || []).forEach((v) => { if (v > maxLoss) maxLoss = v; });
    (obj.val_loss || []).forEach((v) => { if (v > maxLoss) maxLoss = v; });
  });

  ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(W - padR, y);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "10px JetBrains Mono";
    ctx.fillText((maxLoss * (1 - i * 0.25)).toFixed(2), 6, y + 3);
  }

  entries.forEach(([name, obj], idx) => {
    const color = MODEL_COLORS[name] || "#14b8a6";
    const tLoss = obj.train_loss || [];
    const vLoss = obj.val_loss || [];

    // Solid line for Train Loss
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.setLineDash([]);
    ctx.beginPath();
    tLoss.forEach((val, i) => {
      const px = padL + (i / Math.max(1, tLoss.length - 1)) * plotW;
      const py = padT + (1 - Math.min(1, val / maxLoss)) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Dashed line for Validation Loss
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    vLoss.forEach((val, i) => {
      const px = padL + (i / Math.max(1, vLoss.length - 1)) * plotW;
      const py = padT + (1 - Math.min(1, val / maxLoss)) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    const lx = padL + idx * 155;
    const ly = H - 14;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 7, 10, 7);
    ctx.fillStyle = legendColor;
    ctx.font = "10px Inter";
    ctx.fillText(`${name} (Train/Val)`, lx + 14, ly);
  });
}

async function loadSampleRowsTable() {
  try {
    let data = null;
    try {
      const res = await fetch("/api/sample-rows?limit=14");
      if (res.ok) data = await res.json();
    } catch (_) {}
    if (!data && window.PMGStandaloneEngine) {
      data = window.PMGStandaloneEngine.getSampleRows(state.datasetKey);
    }
    const rows = (data && data.rows) || [];
    const thead = document.getElementById("sampleRowsThead");
    const tbody = document.getElementById("sampleRowsTbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";
    if (!rows.length) return;

    const cols = Object.keys(rows[0]);
    const trHead = document.createElement("tr");
    trHead.innerHTML = `<th>Action</th>` + cols.map((c) => `<th>${c}</th>`).join("");
    thead.appendChild(trHead);

    rows.forEach((r) => {
      const tr = document.createElement("tr");
      const isFail = Number(r["Machine failure"]) === 1;
      const actionTd = document.createElement("td");
      const btn = document.createElement("button");
      btn.className = `chip-btn ${isFail ? "chip-danger" : "chip-ok"}`;
      btn.textContent = "Load into Twin";
      btn.addEventListener("click", async () => {
        state.sensorInputs = { ...r };
        updateSlidersFromInputs();
        await runSinglePrediction();
        document.querySelector('[data-tab="tab-twin"]').click();
      });
      actionTd.appendChild(btn);
      tr.appendChild(actionTd);

      cols.forEach((c) => {
        const td = document.createElement("td");
        if (c === "Machine failure") {
          td.innerHTML = isFail
            ? `<span class="text-rose" style="font-weight:700;">1 (FAILURE)</span>`
            : `<span class="text-emerald">0 (Healthy)</span>`;
        } else {
          td.textContent = r[c];
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed loading sample rows:", err);
  }
}

function cssSafe(str) {
  return str.replace(/[^a-zA-Z0-9_-]/g, "_");
}
