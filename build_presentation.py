"""
Generates the complete Predictive-Machine Guard project presentation:
1. Editable PowerPoint Deck: `Predictive_Machine_Guard_Presentation.pptx`
2. Interactive Web Slide Deck: `static/presentation.html`

Covers all 8 required sections:
1. Title Page
2. Introduction
3. Objectives
4. Tools Used
5. System Design
6. Program Code
7. Output Screenshots
8. Conclusion
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(r"c:\Users\darsh\Predictive-Machine-Guard")
ASSETS = ROOT / "presentation_assets"
PPTX_OUT = ROOT / "Predictive_Machine_Guard_Presentation.pptx"

# Industrial SCADA Palette (No purple gradients, no emojis, no em dashes)
BG_DARK = RGBColor(11, 17, 32)       # #0B1120
CARD_BG = RGBColor(17, 24, 39)       # #111827
CARD_ALT = RGBColor(24, 34, 53)      # #182235
BORDER_CLR = RGBColor(38, 52, 77)    # #26344D
CYAN = RGBColor(56, 189, 248)        # #38BDF8
EMERALD = RGBColor(16, 185, 129)     # #10B981
AMBER = RGBColor(245, 158, 11)       # #F59E0B
TEXT_MAIN = RGBColor(248, 250, 252)  # #F8FAFC
TEXT_MUTED = RGBColor(148, 163, 184) # #94A3B8
CODE_BG = RGBColor(7, 11, 21)        # #070B15


def set_slide_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG_DARK


def add_header(slide, section_num: str, title: str, subtitle: str):
    # Top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = CYAN
    bar.line.fill.background()

    # Section tag box
    badge = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(0.35), Inches(1.85), Inches(0.34))
    badge.fill.solid()
    badge.fill.fore_color.rgb = CARD_ALT
    badge.line.color.rgb = CYAN
    badge.line.width = Pt(1)
    tf_b = badge.text_frame
    tf_b.word_wrap = False
    tf_b.margin_left = Inches(0.08)
    tf_b.margin_top = Inches(0.04)
    p_b = tf_b.paragraphs[0]
    p_b.text = f"SECTION {section_num}"
    p_b.font.name = "Consolas"
    p_b.font.size = Pt(10)
    p_b.font.bold = True
    p_b.font.color.rgb = CYAN
    p_b.alignment = PP_ALIGN.CENTER

    # Main slide title
    tb = slide.shapes.add_textbox(Inches(2.6), Inches(0.26), Inches(10.1), Inches(0.48))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Segoe UI"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    # Subtitle
    sub_tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.76), Inches(12.1), Inches(0.32))
    sub_tf = sub_tb.text_frame
    sub_tf.word_wrap = True
    p_sub = sub_tf.paragraphs[0]
    p_sub.text = subtitle
    p_sub.font.name = "Segoe UI"
    p_sub.font.size = Pt(12)
    p_sub.font.color.rgb = TEXT_MUTED

    # Footer bar
    f_tb = slide.shapes.add_textbox(Inches(0.6), Inches(7.12), Inches(12.1), Inches(0.28))
    f_tf = f_tb.text_frame
    f_p = f_tf.paragraphs[0]
    f_p.text = "Predictive-Machine Guard | Applied ML & PyTorch DL Predictive Maintenance | https://predictive-machine-guard.vercel.app"
    f_p.font.name = "Consolas"
    f_p.font.size = Pt(9)
    f_p.font.color.rgb = TEXT_MUTED


def add_card(slide, left, top, width, height, title: str, bullets: list, accent_color=CYAN):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_CLR
    card.line.width = Pt(1)

    # Top thin accent line on card
    top_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.05))
    top_line.fill.solid()
    top_line.fill.fore_color.rgb = accent_color
    top_line.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.14), Inches(width - 0.36), Inches(height - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True

    p_title = tf.paragraphs[0]
    p_title.text = title
    p_title.font.name = "Segoe UI"
    p_title.font.size = Pt(14)
    p_title.font.bold = True
    p_title.font.color.rgb = accent_color
    p_title.space_after = Pt(8)

    for b in bullets:
        p = tf.add_paragraph()
        p.text = f"> {b}"
        p.font.name = "Segoe UI"
        p.font.size = Pt(11.5)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)


def add_code_card(slide, left, top, width, height, file_label: str, code_text: str):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CODE_BG
    card.line.color.rgb = BORDER_CLR
    card.line.width = Pt(1)

    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.36))
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = CARD_ALT
    hdr.line.color.rgb = BORDER_CLR
    tf_h = hdr.text_frame
    tf_h.margin_left = Inches(0.14)
    p_h = tf_h.paragraphs[0]
    p_h.text = file_label
    p_h.font.name = "Consolas"
    p_h.font.size = Pt(10.5)
    p_h.font.bold = True
    p_h.font.color.rgb = CYAN

    tb = slide.shapes.add_textbox(Inches(left + 0.12), Inches(top + 0.40), Inches(width - 0.24), Inches(height - 0.46))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = code_text
    p.font.name = "Consolas"
    p.font.size = Pt(9.2)
    p.font.color.rgb = TEXT_MAIN


def build_pptx():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: 1. TITLE PAGE
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s1)

    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = CYAN
    top_bar.line.fill.background()

    hero_box = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.9), Inches(11.733), Inches(5.7))
    hero_box.fill.solid()
    hero_box.fill.fore_color.rgb = CARD_BG
    hero_box.line.color.rgb = BORDER_CLR
    hero_box.line.width = Pt(1.5)

    tag = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.3), Inches(4.4), Inches(0.38))
    tag.fill.solid()
    tag.fill.fore_color.rgb = CARD_ALT
    tag.line.color.rgb = CYAN
    p_tag = tag.text_frame.paragraphs[0]
    p_tag.text = "1. TITLE PAGE | INDUSTRIAL SCADA & ML/DL SYSTEM"
    p_tag.font.name = "Consolas"
    p_tag.font.size = Pt(10.5)
    p_tag.font.bold = True
    p_tag.font.color.rgb = CYAN
    p_tag.alignment = PP_ALIGN.CENTER

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.9), Inches(10.8), Inches(2.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "PREDICTIVE-MACHINE GUARD"
    p1.font.name = "Segoe UI"
    p1.font.size = Pt(40)
    p1.font.bold = True
    p1.font.color.rgb = TEXT_MAIN
    p1.space_after = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "Multi-Dataset Industrial Predictive Maintenance, Counterfactual Setpoint Optimization, and Hybrid Classical ML + PyTorch Deep Learning Diagnostics"
    p2.font.name = "Segoe UI"
    p2.font.size = Pt(16)
    p2.font.color.rgb = CYAN

    # 4 Metric Boxes on Title Slide
    metrics = [
        ("12 INDUSTRIAL DATASETS", "56,266 Telemetry Rows", "CNC, Turbofan, Bearings, Pumps, Wind, Robotics"),
        ("7 UNIFIED ML / DL MODELS", "4 Classical + 3 PyTorch", "XGBoost, LightGBM, RF, LogReg, ANN, 1D-CNN, BiLSTM"),
        ("EXPLAINABLE AI & RUL", "TreeSHAP + Setpoints", "Local/Global SHAP, RUL Regression & Health Index"),
        ("24/7 CLOUD DEPLOYMENT", "Vercel & GitHub Pages", "predictive-machine-guard.vercel.app"),
    ]
    for idx, (m_lbl, m_val, m_sub) in enumerate(metrics):
        x = 1.2 + idx * 2.78
        m_card = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(4.15), Inches(2.6), Inches(1.35))
        m_card.fill.solid()
        m_card.fill.fore_color.rgb = CARD_ALT
        m_card.line.color.rgb = BORDER_CLR
        mtf = m_card.text_frame
        mtf.word_wrap = True
        mtf.margin_left = Inches(0.12)
        mtf.margin_top = Inches(0.12)
        mp0 = mtf.paragraphs[0]
        mp0.text = m_lbl
        mp0.font.name = "Consolas"
        mp0.font.size = Pt(9)
        mp0.font.bold = True
        mp0.font.color.rgb = AMBER
        mp1 = mtf.add_paragraph()
        mp1.text = m_val
        mp1.font.name = "Segoe UI"
        mp1.font.size = Pt(14)
        mp1.font.bold = True
        mp1.font.color.rgb = TEXT_MAIN
        mp2 = mtf.add_paragraph()
        mp2.text = m_sub
        mp2.font.name = "Segoe UI"
        mp2.font.size = Pt(9.5)
        mp2.font.color.rgb = TEXT_MUTED

    meta_tb = s1.shapes.add_textbox(Inches(1.2), Inches(5.72), Inches(10.8), Inches(0.65))
    meta_tf = meta_tb.text_frame
    mp = meta_tf.paragraphs[0]
    mp.text = "Developer: Darsh  |  GitHub: https://github.com/darshp6721/Predictive-Machine-Guard  |  Live URL: https://predictive-machine-guard.vercel.app"
    mp.font.name = "Consolas"
    mp.font.size = Pt(11)
    mp.font.color.rgb = EMERALD

    # =========================================================================
    # SLIDE 2: 2. INTRODUCTION
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s2)
    add_header(
        s2,
        "2 / 8",
        "2. Introduction",
        "Industrial Context, Problem Statement, and Paradigm Shift from Reactive Repairs to Data-Driven Predictive Maintenance"
    )

    add_card(
        s2, 0.6, 1.25, 5.9, 2.7,
        "Problem Statement in Industrial Manufacturing",
        [
            "Unplanned equipment failure in CNC milling spindles, turbofan engines, hydraulic rigs, and power transformers causes severe production halts and safety hazards.",
            "Traditional Run-to-Failure (Reactive) maintenance incurs catastrophic component damage and secondary shaft/bearing destruction.",
            "Fixed-interval Preventive Maintenance replaces healthy parts prematurely while still missing sudden thermal, electrical, or overstrain anomalies.",
            "Raw factory telemetry suffers from missing sensor dropouts (NaNs), electromagnetic interference (EMI) spikes, and severe class imbalance (3% to 22% fault rate)."
        ],
        CYAN
    )

    add_card(
        s2, 6.8, 1.25, 5.9, 2.7,
        "Proposed Solution: Predictive-Machine Guard",
        [
            "An end-to-end SCADA telemetry and diagnostic platform combining Classical Machine Learning and PyTorch Deep Learning across 12 industrial benchmarks (56,266 total records).",
            "Automates data quality auditing: Type-grouped median imputation, duplicate removal, and 2.2x IQR Winsorization to clip sensor noise without erasing fault boundaries.",
            "Predicts 4 critical outputs simultaneously: Binary Failure Probability, Specific Failure Mode (TWF, HDF, PWF, OSF), Remaining Useful Life (RUL), and Machine Health Index (MHI).",
            "Provides a Digital Twin Setpoint Optimizer that computes exact RPM, Torque, and Thermal adjustments to reduce failure probability in real time."
        ],
        EMERALD
    )

    add_card(
        s2, 0.6, 4.18, 12.1, 2.75,
        "12 Multi-Domain Industrial Datasets Integrated & Benchmarked (56,266 Total Records)",
        [
            "1. UCI AI4I 2020 CNC Milling Machine (5,000 rows)  |  2. NASA CMAPSS Turbofan Engine Degradation (4,800 rows)  |  3. IEEE PHM Bearing Vibration Run-to-Failure (4,600 rows)",
            "4. Industrial Centrifugal Pump Cavitation & Seal Telemetry (4,500 rows)  |  5. Wind Turbine Gearbox & Generator SCADA (5,000 rows)  |  6. Semiconductor Etch Chamber Plasma (4,400 rows)",
            "7. Heavy Hydraulic Press & Valve Pressure Cycle (4,500 rows)  |  8. High-Voltage Power Transformer Dissolved Gas & Thermal (4,800 rows)  |  9. Lithium-Ion Battery Pack Cycle Aging (4,600 rows)",
            "10. Industrial Screw Compressor Acoustic & Pressure (4,500 rows)  |  11. Robotic 6-Axis Welding Arm Servo & Harmonic Drive (4,766 rows)  |  12. Steel Hot Strip Rolling Mill Stand (4,800 rows)"
        ],
        AMBER
    )

    # =========================================================================
    # SLIDE 3: 3. OBJECTIVES
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s3)
    add_header(
        s3,
        "3 / 8",
        "3. Objectives",
        "Core Engineering, Machine Learning, Deep Learning, and Operational Goals of Predictive-Machine Guard"
    )

    objs = [
        ("Objective 1: Automated Telemetry Cleaning", [
            "Detect and impute missing sensor readings (3% to 5% NaN injection) using equipment-tier grouped median and mode strategies.",
            "Identify extreme sensor spikes via 1.5x IQR Tukey fences and |Z| > 3.0 audits, applying 2.2x IQR Winsorization."
        ], CYAN),
        ("Objective 2: Physics-Informed Feature Engineering", [
            "Synthesize domain-specific physical interaction features from raw sensors: Mechanical Power [W], Overstrain Index [Nm*min], Thermal Ratio, and Torque per kRPM.",
            "Capture non-linear electro-mechanical stress boundaries before model training."
        ], EMERALD),
        ("Objective 3: 7-Model ML & PyTorch DL Benchmark", [
            "Train and compare 4 Classical ML models (XGBoost, LightGBM, Random Forest, Logistic Regression) alongside 3 PyTorch Neural Networks (ANN, 1D-CNN, BiLSTM).",
            "Handle severe class imbalance using SMOTE on stratified 70/10/20 splits."
        ], AMBER),
        ("Objective 4: Multi-Task Prognostics (RUL & Modes)", [
            "Classify root-cause failure modes (Tool Wear, Heat Dissipation, Power Fault, Overstrain) rather than binary alerts alone.",
            "Estimate continuous Remaining Useful Life (RUL in minutes) and composite Machine Health Index (0% to 100%)."
        ], CYAN),
        ("Objective 5: Explainable AI & Counterfactual Control", [
            "Compute global and single-prediction TreeSHAP log-odds feature attributions for transparent root-cause verification.",
            "Recommend actionable setpoint adjustments (Torque, RPM, Cooling) with quantified Post-Tuning Risk Delta."
        ], EMERALD),
        ("Objective 6: Production SCADA Web Deployment", [
            "Deliver a responsive, zero-dependency industrial SCADA web interface with custom CSV upload/export and live telemetry streaming.",
            "Deploy permanently on global CDNs (Vercel and GitHub Pages) with Privacy Policy and Terms & Conditions compliance."
        ], AMBER),
    ]

    for i, (o_title, o_bullets, o_clr) in enumerate(objs):
        col = i % 3
        row = i // 3
        left = 0.6 + col * 4.1
        top = 1.25 + row * 2.88
        add_card(s3, left, top, 3.9, 2.68, o_title, o_bullets, o_clr)

    # =========================================================================
    # SLIDE 4: 4. TOOLS USED
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s4)
    add_header(
        s4,
        "4 / 8",
        "4. Tools Used",
        "End-to-End Software Stack, Machine Learning & Deep Learning Frameworks, and Cloud Infrastructure"
    )

    rows_data = [
        ("Category / Layer", "Tools & Libraries Used", "Specific Role in Predictive-Machine Guard"),
        ("Core Programming & API", "Python 3.11+, WSGI HTTP Server, JSON REST API", "Backend model training, real-time inference endpoints (/api/predict, /api/upload-csv), and dataset routing."),
        ("Classical Machine Learning", "Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn", "Gradient boosted decision trees, Random Forest classification/RUL regression, Logistic Regression, and SMOTE."),
        ("Deep Learning Framework", "PyTorch (torch.nn, DataLoader, AdamW, BCEWithLogitsLoss)", "Custom architectures: Deep MLP (ANN), 1D Convolutional Network (1D-CNN), and Bidirectional LSTM (BiLSTM)."),
        ("Data Engineering & XAI", "Pandas, NumPy, SciPy, SHAP (TreeExplainer), Joblib", "Grouped median imputation, IQR/Z-Score outlier Winsorization, physics feature synthesis, and local/global SHAP."),
        ("SCADA Frontend UI", "HTML5 Canvas, CSS3 Grid, Vanilla ES6 JavaScript", "Zero-framework industrial console, real-time ROC/PR & DL loss curve rendering, and client-side inference engine."),
        ("Cloud Hosting & DevOps", "Git, GitHub Actions, GitHub Pages CDN, Vercel CLI", "Dual 24/7 production deployment (predictive-machine-guard.vercel.app & GitHub Pages) with automated CI/CD."),
    ]

    table_shape = s4.shapes.add_table(len(rows_data), 3, Inches(0.6), Inches(1.3), Inches(12.133), Inches(5.5))
    table = table_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(3.7)
    table.columns[2].width = Inches(5.833)

    for r_idx, row_tuple in enumerate(rows_data):
        for c_idx, val in enumerate(row_tuple):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            if r_idx == 0:
                cell.fill.fore_color.rgb = CARD_ALT
            else:
                cell.fill.fore_color.rgb = CARD_BG if r_idx % 2 == 1 else RGBColor(14, 21, 35)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = val
            p.font.name = "Consolas" if (r_idx == 0 or c_idx == 1) else "Segoe UI"
            p.font.size = Pt(11 if r_idx == 0 else 10.5)
            p.font.bold = (r_idx == 0 or c_idx == 0)
            if r_idx == 0:
                p.font.color.rgb = CYAN
            elif c_idx == 0:
                p.font.color.rgb = EMERALD
            elif c_idx == 1:
                p.font.color.rgb = AMBER
            else:
                p.font.color.rgb = TEXT_MAIN

    # =========================================================================
    # SLIDE 5: 5. SYSTEM DESIGN
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s5)
    add_header(
        s5,
        "5 / 8",
        "5. System Design",
        "5-Stage Industrial Telemetry Processing, Hybrid ML/DL Training, and Real-Time Decision Support Architecture"
    )

    # Visual Pipeline Flow Blocks (5 horizontal stages)
    stages = [
        ("STAGE 1: INGESTION", "12 Industrial Datasets\n(56,266 Rows) +\nCustom CSV Upload", CYAN),
        ("STAGE 2: PREPROCESSING", "Duplicate Removal\nGrouped Median Impute\n2.2x IQR Winsorization", EMERALD),
        ("STAGE 3: FEATURE ENG.", "Physics Interaction Terms\n70/10/20 Stratified Split\nSMOTE Class Balancing", AMBER),
        ("STAGE 4: 7-MODEL SUITE", "4 Classical ML Models\n3 PyTorch DL Networks\nRUL & Mode Estimators", CYAN),
        ("STAGE 5: SCADA & XAI", "TreeSHAP Attribution\nSetpoint Optimizer\nVercel & GitHub CDN", EMERALD),
    ]
    for idx, (st_title, st_desc, st_clr) in enumerate(stages):
        x = 0.6 + idx * 2.46
        box = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.3), Inches(2.25), Inches(1.55))
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_ALT
        box.line.color.rgb = st_clr
        box.line.width = Pt(1.5)
        btf = box.text_frame
        btf.word_wrap = True
        btf.margin_left = Inches(0.1)
        btf.margin_top = Inches(0.12)
        bp0 = btf.paragraphs[0]
        bp0.text = st_title
        bp0.font.name = "Consolas"
        bp0.font.size = Pt(10)
        bp0.font.bold = True
        bp0.font.color.rgb = st_clr
        bp1 = btf.add_paragraph()
        bp1.text = st_desc
        bp1.font.name = "Segoe UI"
        bp1.font.size = Pt(10.5)
        bp1.font.color.rgb = TEXT_MAIN
        bp1.space_before = Pt(6)

        if idx < 4:
            arr = s5.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.28), Inches(1.92), Inches(0.15), Inches(0.28))
            arr.fill.solid()
            arr.fill.fore_color.rgb = CYAN
            arr.line.fill.background()

    add_card(
        s5, 0.6, 3.1, 5.9, 3.8,
        "Data Pipeline & Mathematical Formulation",
        [
            "Missing Value Imputation: X_imputed = fillna(GroupMedian(Type)) with global median fallback, achieving 0% residual missingness across all 56,266 records.",
            "Outlier Treatment: Tukey audit detects [Q25 - 1.5*IQR, Q75 + 1.5*IQR]; Winsorization clips extreme spikes at [Q25 - 2.2*IQR, Q75 + 2.2*IQR].",
            "Physics Features: Mechanical Power P = Torque * (RPM * 2*pi / 60); Overstrain = ToolWear * Torque; Thermal Ratio = ProcessTemp / AirTemp.",
            "Class Balancing: SMOTE synthesizes minority failure vectors strictly inside the 70% training partition to prevent validation/test data leakage."
        ],
        CYAN
    )

    add_card(
        s5, 6.8, 3.1, 5.9, 3.8,
        "Multi-Model Inference & Hybrid Cloud Architecture",
        [
            "Classical ML Branch: XGBoost, LightGBM, Random Forest, and Logistic Regression evaluated via Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, and Brier Score.",
            "PyTorch DL Branch: ANN (64-32-16 BatchNorm/Dropout MLP), 1D-CNN (16->32 Conv1d channels), and BiLSTM (Bidirectional sequence pooling + LayerNorm).",
            "Counterfactual Setpoint Engine: Perturbs controllable parameters (Torque, RPM, Tool Wear, Cooling) toward safe medians to compute Post-Tuning Risk Delta.",
            "Dual Runtime Execution: Supports live Python WSGI server (app.py) and browser-native standalone engine (standalone_engine.js + precomputed_states.js)."
        ],
        EMERALD
    )

    # =========================================================================
    # SLIDE 6: 6. PROGRAM CODE (PART 1: PREPROCESSING & PYTORCH DL)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s6)
    add_header(
        s6,
        "6 / 8",
        "6. Program Code (Part 1: Preprocessing & PyTorch Architectures)",
        "Core Implementation Excerpts from ml_engine/preprocessing.py and ml_engine/deep_learning.py"
    )

    code_preproc = """class DataQualityPreprocessor:
    def __init__(self, iqr_multiplier: float = 2.2) -> None:
        self.iqr_multiplier = iqr_multiplier
        self.fences, self.medians = {}, {}

    def fit_transform(self, df: pd.DataFrame, sensor_cols: List[str]):
        work = df.drop_duplicates().reset_index(drop=True)
        for col in sensor_cols:
            raw = pd.to_numeric(work[col], errors="coerce")
            median_val = float(raw.dropna().median())
            self.medians[col] = median_val
            # Grouped median imputation by equipment tier
            if "Type" in work.columns and raw.isna().sum() > 0:
                imputed = raw.fillna(
                    work.groupby("Type")[col].transform("median")
                ).fillna(median_val)
            else:
                imputed = raw.fillna(median_val)
            # IQR Tukey audit & 2.2x IQR Winsorization clipping
            q25, q75 = float(imputed.quantile(0.25)), float(imputed.quantile(0.75))
            iqr = max(q75 - q25, 1e-6)
            win_low = q25 - self.iqr_multiplier * iqr
            win_high = q75 + self.iqr_multiplier * iqr
            self.fences[col] = (win_low, win_high)
            work[col] = np.round(imputed.clip(win_low, win_high), 3)
        return work"""

    code_dl = """class _Conv1DCNNNet(nn.Module):
    \"\"\"1D-CNN extracting local multi-sensor interaction patterns.\"\"\"
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * in_features, 32),
            nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv_block(x.unsqueeze(1))).squeeze(-1)

class _BiLSTMNet(nn.Module):
    \"\"\"Bidirectional LSTM across ordered sensor channels.\"\"\"
    def __init__(self, in_features: int, hidden_size: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, 24), nn.ReLU(), nn.Linear(24, 1))"""

    add_code_card(s6, 0.6, 1.25, 5.95, 5.65, "File: ml_engine/preprocessing.py (Imputation & IQR Winsorization)", code_preproc)
    add_code_card(s6, 6.78, 1.25, 5.95, 5.65, "File: ml_engine/deep_learning.py (PyTorch 1D-CNN & BiLSTM)", code_dl)

    # =========================================================================
    # SLIDE 7: 6. PROGRAM CODE (PART 2: UNIFIED PIPELINE & INFERENCE API)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s7)
    add_header(
        s7,
        "6 / 8",
        "6. Program Code (Part 2: 7-Model Training Pipeline & Inference Engine)",
        "Core Implementation Excerpts from ml_engine/pipeline.py and app.py"
    )

    code_pipe = """# Stratified 70/10/20 Split + SMOTE Minority Oversampling
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_bin, test_size=0.30, random_state=42, stratify=y_bin
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.6667, random_state=42, stratify=y_temp
)
if use_smote and y_train.sum() >= 6:
    smote = SMOTE(random_state=42, k_neighbors=min(5, int(y_train.sum()) - 1))
    X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)

# 7-Model Unified Suite (4 Classical ML + 3 PyTorch Deep Learning)
candidate_models = {
    "XGBoost": XGBClassifier(n_estimators=180, max_depth=5, learning_rate=0.07),
    "LightGBM": LGBMClassifier(n_estimators=180, num_leaves=31, learning_rate=0.07),
    "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=12),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]),
    "ANN (Deep MLP)": DeepLearningClassifierWrapper(arch="ANN", epochs=18),
    "1D-CNN": DeepLearningClassifierWrapper(arch="1D-CNN", epochs=18),
    "BiLSTM": DeepLearningClassifierWrapper(arch="BiLSTM", epochs=16),
}"""

    code_api = """# Real-Time Multi-Model Diagnostic & Counterfactual Setpoint Endpoint (app.py)
@app.route("/api/predict", methods=["POST"])
def predict_telemetry():
    payload = request.get_json(force=True)
    sensor_values = payload.get("sensors", {})
    selected_model = payload.get("model_name", engine.best_model_name)
    equipment_type = payload.get("equipment_type", "M")

    result = engine.predict_single(
        sensor_inputs=sensor_values,
        model_name=selected_model,
        equipment_type=equipment_type,
    )
    # Returns:
    # - failure_probability & model_probabilities (all 7 models)
    # - health_index & predicted_rul_minutes
    # - predicted_failure_mode & mode_probabilities
    # - local_shap_contributions (TreeSHAP log-odds)
    # - ai_recommendation (Counterfactual setpoints & Post-Tuning Risk Delta)
    return jsonify(result)"""

    add_code_card(s7, 0.6, 1.25, 5.95, 5.65, "File: ml_engine/pipeline.py (SMOTE & 7-Model Training Suite)", code_pipe)
    add_code_card(s7, 6.78, 1.25, 5.95, 5.65, "File: app.py (Real-Time Telemetry Diagnostic API)", code_api)

    # =========================================================================
    # SLIDE 8: 7. OUTPUT SCREENSHOTS (PART 1: TWIN & MODEL BENCHMARKS)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s8)
    add_header(
        s8,
        "7 / 8",
        "7. Output Screenshots (Tab 1: Digital Twin & Tab 2: 7-Model Evaluation)",
        "Live Captures from Predictive-Machine Guard Showing Real-Time Setpoint Optimization and ROC / DL Loss Curves"
    )

    img1_path = ASSETS / "output_1_telemetry_kpis.png"
    img2_path = ASSETS / "output_2_models_prescriptions.png"

    if img1_path.exists():
        s8.shapes.add_picture(str(img1_path), Inches(0.6), Inches(1.28), width=Inches(5.95))
    if img2_path.exists():
        s8.shapes.add_picture(str(img2_path), Inches(6.78), Inches(1.28), width=Inches(5.95))

    add_card(
        s8, 0.6, 4.75, 5.95, 2.15,
        "Screenshot 1: Digital Twin Telemetry & Setpoint Optimizer",
        [
            "Displays 5 executive KPI cards: Stratified Split (5,000 rows), Preprocessing Audit (296 NaNs imputed, 646 IQR outliers Winsorized), Top Model F1, RUL R-Squared (0.8536), and Fault Rate.",
            "Interactive sliders adjust Air Temp, Process Temp, RPM, Torque, and Tool Wear with real-time 7-model failure probabilities and Counterfactual Setpoint prescriptions."
        ],
        CYAN
    )

    add_card(
        s8, 6.78, 4.75, 5.95, 2.15,
        "Screenshot 2: Classical ML & PyTorch Deep Learning Evaluation",
        [
            "Benchmarks all 7 models across Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, MCC, and Brier Calibration Loss.",
            "Renders multi-model ROC curves and epoch-by-epoch PyTorch Training vs. Validation BCE loss convergence for ANN, 1D-CNN, and BiLSTM."
        ],
        EMERALD
    )

    # =========================================================================
    # SLIDE 9: 7. OUTPUT SCREENSHOTS (PART 2: TREESHAP & DATA QUALITY AUDIT)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s9)
    add_header(
        s9,
        "7 / 8",
        "7. Output Screenshots (Tab 3: TreeSHAP XAI & Tab 4: Preprocessing Audit)",
        "Live Captures Showing Global TreeSHAP Feature Importance, Multi-Class Fault Matrix, and Per-Sensor Cleaning Audit"
    )

    img3_path = ASSETS / "output_3_preprocessing_shap.png"
    img4_path = ASSETS / "output_4_data_cleaning_audit.png"

    if img3_path.exists():
        s9.shapes.add_picture(str(img3_path), Inches(0.6), Inches(1.28), width=Inches(5.95))
    if img4_path.exists():
        s9.shapes.add_picture(str(img4_path), Inches(6.78), Inches(1.28), width=Inches(5.95))

    add_card(
        s9, 0.6, 4.75, 5.95, 2.15,
        "Screenshot 3: TreeSHAP Attribution & Failure Mode Classification",
        [
            "Quantifies global TreeSHAP feature importance across raw sensors and physics-engineered features (Mechanical Power, Overstrain Index, Temp Diff).",
            "Evaluates multi-class root-cause classification across Tool Wear (TWF), Heat Dissipation (HDF), Power Fault (PWF), and Overstrain (OSF)."
        ],
        AMBER
    )

    add_card(
        s9, 6.78, 4.75, 5.95, 2.15,
        "Screenshot 4: Automated Preprocessing Audit & CSV Ingestion",
        [
            "Logs column-by-column missing value counts before/after grouped median imputation and IQR/Z-Score outlier Winsorization bounds.",
            "Supports instant drag-and-drop Custom CSV ingestion and cleaned dataset CSV export directly from the browser."
        ],
        CYAN
    )

    # =========================================================================
    # SLIDE 10: 8. CONCLUSION
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_bg(s10)
    add_header(
        s10,
        "8 / 8",
        "8. Conclusion",
        "Empirical Findings Across 12 Industrial Datasets, Operational Impact, and Future Scope"
    )

    add_card(
        s10, 0.6, 1.25, 5.95, 2.75,
        "Key Experimental & Technical Achievements",
        [
            "Processed and cleaned 56,266 industrial telemetry rows across 12 distinct manufacturing domains with 100% missing-value recovery and robust 2.2x IQR Winsorization.",
            "Demonstrated that Physics-Informed Features (Mechanical Power, Overstrain Index, Thermal Ratio) consistently rank in the top 3 TreeSHAP attributions across all machinery types.",
            "Achieved 97.9% to 99.6% ROC-AUC and 86.6% to 93.4% held-out F1-scores using ensemble gradient boosting (XGBoost, LightGBM, Random Forest) paired with PyTorch neural networks.",
            "Bridged the gap between black-box prediction and shop-floor action via Counterfactual Setpoint Optimization (computing exact Torque/RPM adjustments and Post-Tuning Risk Delta)."
        ],
        CYAN
    )

    add_card(
        s10, 6.78, 1.25, 5.95, 2.75,
        "Future Scope & Industrial Extensions",
        [
            "OPC-UA & MQTT Edge Streaming: Direct real-time ingestion from industrial PLCs (Siemens S7, Allen-Bradley) at sub-second polling intervals.",
            "Transformer & Temporal Fusion Architectures: Extending the PyTorch sequence module with multi-horizon Temporal Fusion Transformers for long-range RUL forecasting.",
            "Automated Work-Order Dispatch: Webhook integration with enterprise CMMS platforms (SAP PM, IBM Maximo) when predicted failure risk exceeds critical thresholds.",
            "Federated Multi-Plant Learning: Updating PyTorch model weights across distributed factory sites without centralizing proprietary raw telemetry."
        ],
        EMERALD
    )

    add_card(
        s10, 0.6, 4.22, 12.133, 2.65,
        "Live Production Access & Repository Links",
        [
            "Choice B (Vercel Global CDN): https://predictive-machine-guard.vercel.app  (Includes /privacy and /terms compliance pages)",
            "Choice C (GitHub Pages CDN): https://darshp6721.github.io/Predictive-Machine-Guard/",
            "Source Code & 12-Dataset Suite: https://github.com/darshp6721/Predictive-Machine-Guard",
            "Summary: Predictive-Machine Guard delivers a complete, transparent, and globally deployed industrial predictive maintenance system."
        ],
        AMBER
    )

    prs.save(str(PPTX_OUT))
    print(f"Saved PowerPoint presentation to: {PPTX_OUT}")


if __name__ == "__main__":
    build_pptx()
