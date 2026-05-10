"""
app.py  –  Diabetes Complication Risk Predictor
================================================
Streamlit frontend for the multi-label classification model.

Deployment: Hugging Face Spaces (Streamlit SDK)
             Upload models/chain_rf.pkl, models/training_metadata.pkl,
             and data/processed/scaler_final.pkl alongside this file.

Usage:
    streamlit run app.py
"""

import os
import pickle
import warnings
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesRisk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Global page background */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Top banner */
.top-banner {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 60%, #1a1f2e 100%);
    border-bottom: 1px solid #21262d;
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
}
.banner-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.banner-sub {
    font-size: 0.95rem;
    color: #8b949e;
    margin-top: 0.3rem;
}

/* Section headers */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58a6ff;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
}

/* Risk card */
.risk-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.risk-card.high  { border-left: 4px solid #f85149; }
.risk-card.medium { border-left: 4px solid #d29922; }
.risk-card.low   { border-left: 4px solid #3fb950; }

.risk-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 0.2rem;
}
.risk-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e6edf3;
    margin-bottom: 0.5rem;
}
.risk-pct {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1;
    margin: 0.4rem 0;
}
.risk-pct.high   { color: #f85149; }
.risk-pct.medium { color: #d29922; }
.risk-pct.low    { color: #3fb950; }

.risk-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.4rem;
}
.badge-high   { background: rgba(248,81,73,0.15);  color: #f85149; }
.badge-medium { background: rgba(210,153,34,0.15); color: #d29922; }
.badge-low    { background: rgba(63,185,80,0.15);  color: #3fb950; }

/* Progress bar wrapper */
.bar-wrap {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.7rem;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}

/* Info box */
.info-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #8b949e;
    line-height: 1.6;
}
.info-box strong { color: #e6edf3; }

/* Disclaimer */
.disclaimer {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.3);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    font-size: 0.8rem;
    color: #d29922;
    margin-top: 2rem;
    line-height: 1.6;
}

/* Override Streamlit's default input/select styling */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}
div[data-testid="stSlider"] .stSlider {
    color: #58a6ff !important;
}

/* Primary button */
.stButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #388bfd !important;
}

/* Column divider */
.divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "scaler_final.pkl")

LABEL_COLS = ["cardiovascular_complication", "kidney_complication", "neuropathy_complication"]
LABEL_NAMES = ["Cardiovascular Disease", "Kidney Disease", "Diabetic Neuropathy"]
LABEL_ICONS = ["❤️", "🫘", "⚡"]

# Tuned thresholds from evaluation phase (from report results)
TUNED_THRESHOLDS = {
    "cardiovascular_complication": 0.45,
    "kidney_complication":         0.25,
    "neuropathy_complication":     0.10,
}


@st.cache_resource
def load_model():
    """Load the best model (ClassifierChain RF) and scaler. Returns (model, scaler, feature_cols, numeric_cols) or None."""
    try:
        meta_path  = os.path.join(MODELS_DIR, "training_metadata.pkl")
        model_path = os.path.join(MODELS_DIR, "chain_rf.pkl")

        if not os.path.exists(model_path) or not os.path.exists(meta_path):
            return None, None, [], []

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        scaler, numeric_cols = None, []
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, "rb") as f:
                scaler_bundle = pickle.load(f)
                scaler      = scaler_bundle.get("scaler")
                numeric_cols = scaler_bundle.get("numeric_cols", [])

        return model, scaler, meta.get("feature_cols", []), numeric_cols

    except Exception as e:
        st.warning(f"Could not load model: {e}. Running in demo mode.")
        return None, None, [], []


model, scaler, feature_cols, numeric_cols = load_model()
MODEL_LOADED = model is not None


# ══════════════════════════════════════════════════════════════════════════════
# Feature Construction
# ══════════════════════════════════════════════════════════════════════════════

VISIT_NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_diagnoses",
    "number_outpatient", "number_emergency", "number_inpatient", "age",
]
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "insulin",
]


def build_feature_vector(inputs: dict, feat_cols: list) -> np.ndarray:
    """
    Reconstruct the exact feature vector expected by the trained model.
    For a single-visit patient: std=0, slope=0, delta=0, min=max=mean=last=value.
    """
    row = {}

    row["num_visits"] = inputs.get("num_visits", 1)

    for col in VISIT_NUMERIC:
        val = inputs.get(col, 0.0)
        row[f"{col}_mean"]  = val
        row[f"{col}_std"]   = 0.0
        row[f"{col}_min"]   = val
        row[f"{col}_max"]   = val
        row[f"{col}_last"]  = val
        row[f"{col}_delta"] = 0.0
        row[f"{col}_slope"] = 0.0

    for col in MEDICATION_COLS:
        row[f"{col}_mean"] = inputs.get(col, 0.0)

    row["gender"]      = inputs.get("gender", 0)
    row["diabetesMed"] = inputs.get("diabetesMed", 0)

    # One-hot for A1Cresult (drop_first=False → all categories kept)
    a1c = inputs.get("A1Cresult", "None")
    for cat in [">7", ">8", "None", "Norm"]:
        row[f"A1Cresult_{cat}"] = 1 if a1c == cat else 0

    # One-hot for max_glu_serum
    glu = inputs.get("max_glu_serum", "None")
    for cat in [">200", ">300", "None", "Norm"]:
        row[f"max_glu_serum_{cat}"] = 1 if glu == cat else 0

    # Align to training feature columns (fill missing with 0)
    if feat_cols:
        vec = np.array([row.get(c, 0.0) for c in feat_cols], dtype=float)
    else:
        vec = np.array(list(row.values()), dtype=float)

    return vec.reshape(1, -1)


def demo_predict(inputs: dict) -> dict:
    """
    Heuristic-based demo predictor when model files are absent.
    Weights the most clinically relevant inputs to produce plausible risk scores.
    """
    age     = inputs.get("age", 50)
    hba1c   = inputs.get("A1Cresult", "None")
    glu     = inputs.get("max_glu_serum", "None")
    n_diag  = inputs.get("number_diagnoses", 3)
    insulin = inputs.get("insulin", 0)
    n_vis   = inputs.get("num_visits", 1)
    time_h  = inputs.get("time_in_hospital", 3)
    n_meds  = inputs.get("num_medications", 5)
    n_emerg = inputs.get("number_emergency", 0)

    # Map categorical inputs
    hba1c_score = {">8": 1.0, ">7": 0.6, "Norm": 0.2, "None": 0.3}.get(hba1c, 0.3)
    glu_score   = {">300": 1.0, ">200": 0.7, "Norm": 0.2, "None": 0.3}.get(glu, 0.3)

    # Cardiovascular (most prevalent ~62%)
    cardio = (
        0.30 + 0.15 * min(age / 80, 1)
        + 0.15 * hba1c_score
        + 0.10 * min(n_diag / 9, 1)
        + 0.10 * glu_score
        + 0.05 * min(n_emerg / 3, 1)
        + 0.05 * min(n_vis / 10, 1)
    )

    # Kidney (~9%)
    kidney = (
        0.03 + 0.25 * hba1c_score
        + 0.20 * glu_score
        + 0.15 * min(age / 80, 1)
        + 0.10 * (insulin / 3)
        + 0.10 * min(n_diag / 9, 1)
        + 0.08 * min(time_h / 14, 1)
    )

    # Neuropathy (~0.8%)
    neuro = (
        0.002 + 0.20 * hba1c_score
        + 0.15 * glu_score
        + 0.12 * (insulin / 3)
        + 0.08 * min(age / 80, 1)
        + 0.06 * min(n_meds / 20, 1)
    )

    # Clip to [0, 1]
    scores = {
        "cardiovascular_complication": float(np.clip(cardio, 0, 0.99)),
        "kidney_complication":         float(np.clip(kidney, 0, 0.99)),
        "neuropathy_complication":     float(np.clip(neuro,  0, 0.99)),
    }
    preds = {
        k: int(v >= TUNED_THRESHOLDS[k]) for k, v in scores.items()
    }
    return scores, preds


def predict(inputs: dict):
    """Route to real model or demo heuristic."""
    if MODEL_LOADED:
        vec = build_feature_vector(inputs, feature_cols)   # shape (1, n_features)
        if scaler is not None and numeric_cols:
            # Scaler was fitted only on numeric cols — apply it only to those indices
            num_idx = [feature_cols.index(c) for c in numeric_cols if c in feature_cols]
            vec[0, num_idx] = scaler.transform(vec[:, num_idx])[0]
        proba = model.predict_proba(vec)   # ClassifierChain → (1, n_labels)
        scores = {
            LABEL_COLS[i]: float(proba[0, i]) for i in range(len(LABEL_COLS))
        }
        preds = {
            k: int(v >= TUNED_THRESHOLDS[k]) for k, v in scores.items()
        }
        return scores, preds
    else:
        return demo_predict(inputs)


# ══════════════════════════════════════════════════════════════════════════════
# UI Helpers
# ══════════════════════════════════════════════════════════════════════════════

def risk_level(prob: float, threshold: float):
    if prob >= threshold + 0.20:
        return "high"
    elif prob >= threshold:
        return "medium"
    else:
        return "low"


def render_risk_card(label_name: str, icon: str, prob: float, predicted: int, threshold: float):
    level = risk_level(prob, threshold)
    pct   = int(prob * 100)

    bar_color = {"high": "#f85149", "medium": "#d29922", "low": "#3fb950"}[level]
    badge_cls = f"badge-{level}"
    badge_txt = {"high": "⚠ HIGH RISK", "medium": "⚡ MODERATE", "low": "✓ LOW RISK"}[level]

    st.markdown(f"""
    <div class="risk-card {level}">
        <div class="risk-label">{icon} Complication Risk</div>
        <div class="risk-name">{label_name}</div>
        <div class="risk-pct {level}">{pct}%</div>
        <div class="bar-wrap">
            <div class="bar-fill" style="width:{pct}%; background:{bar_color};"></div>
        </div>
        <span class="risk-badge {badge_cls}">{badge_txt}</span>
        <span style="font-size:0.75rem; color:#8b949e; margin-left:0.5rem;">
            threshold {int(threshold*100)}%
        </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Layout
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="top-banner">
    <div class="banner-title">🩺 DiabetesRisk AI</div>
    <div class="banner-sub">
        Multi-label complication risk prediction · Cardiovascular · Kidney · Neuropathy
    </div>
</div>
""", unsafe_allow_html=True)

if not MODEL_LOADED:
    st.info(
        "**Demo Mode** — Model files not found in `models/`. "
        "Upload `chain_rf.pkl`, `training_metadata.pkl`, and `scaler_final.pkl` to enable full inference. "
        "Predictions below are heuristic estimates for demonstration purposes.",
        icon="ℹ️",
    )

# ── Two-column layout: inputs | results ───────────────────────────────────────
col_input, col_gap, col_results = st.columns([5, 0.3, 4])

with col_input:
    st.markdown('<div class="section-label">Patient Clinical Data</div>', unsafe_allow_html=True)

    # ── Row 1: Demographics ───────────────────────────────────────────────────
    r1a, r1b, r1c = st.columns(3)
    with r1a:
        age = st.slider("Age", min_value=10, max_value=100, value=58, step=1)
    with r1b:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with r1c:
        num_visits = st.number_input("No. of Past Visits", min_value=1, max_value=50, value=2)

    # ── Row 2: Lab / Clinical ─────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Laboratory & Clinical</div>', unsafe_allow_html=True)

    r2a, r2b = st.columns(2)
    with r2a:
        a1c = st.selectbox("HbA1c Result (A1C)", ["None", "Norm", ">7", ">8"],
                           help="None = not measured")
    with r2b:
        glu = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"],
                           help="None = not measured")

    r3a, r3b, r3c = st.columns(3)
    with r3a:
        num_lab  = st.number_input("Lab Procedures", 1, 132, 44)
    with r3b:
        num_proc = st.number_input("Clinical Procedures", 0, 6, 1)
    with r3c:
        num_diag = st.number_input("Number of Diagnoses", 1, 16, 7)

    # ── Row 3: Hospital utilisation ───────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Hospital Utilisation</div>', unsafe_allow_html=True)

    r4a, r4b, r4c, r4d = st.columns(4)
    with r4a:
        time_hosp = st.number_input("Days in Hospital", 1, 14, 4)
    with r4b:
        n_outpat  = st.number_input("Outpatient Visits", 0, 42, 0)
    with r4c:
        n_emerg   = st.number_input("Emergency Visits", 0, 76, 0)
    with r4d:
        n_inpat   = st.number_input("Inpatient Visits", 0, 21, 0)

    # ── Row 4: Medications ────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Medications</div>', unsafe_allow_html=True)

    m1a, m1b, m1c = st.columns(3)
    med_map = {"Not prescribed": 0, "Steady": 1, "Increased": 2, "Decreased": 3}
    with m1a:
        insulin_val   = st.selectbox("Insulin", list(med_map.keys()))
    with m1b:
        metformin_val = st.selectbox("Metformin", list(med_map.keys()))
    with m1c:
        num_meds = st.number_input("Total Medications", 1, 81, 14)

    m2a, m2b = st.columns(2)
    with m2a:
        diabetes_med = st.checkbox("On diabetes medication?", value=True)
    with m2b:
        med_change   = st.checkbox("Medication change this visit?", value=False)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    predict_btn = st.button("🔍  Predict Complication Risk")


# ══════════════════════════════════════════════════════════════════════════════
# Prediction + Results Panel
# ══════════════════════════════════════════════════════════════════════════════

with col_results:
    st.markdown('<div class="section-label">Risk Assessment</div>', unsafe_allow_html=True)

    if predict_btn:
        inputs = {
            "age":               age,
            "gender":            1 if gender == "Male" else 0,
            "num_visits":        num_visits,
            "A1Cresult":         a1c,
            "max_glu_serum":     glu,
            "num_lab_procedures": num_lab,
            "num_procedures":    num_proc,
            "number_diagnoses":  num_diag,
            "time_in_hospital":  time_hosp,
            "number_outpatient": n_outpat,
            "number_emergency":  n_emerg,
            "number_inpatient":  n_inpat,
            "insulin":           med_map[insulin_val],
            "metformin":         med_map[metformin_val],
            "num_medications":   num_meds,
            "diabetesMed":       int(diabetes_med),
            "change":            int(med_change),
        }

        with st.spinner("Running inference…"):
            scores, preds = predict(inputs)

        # Risk cards
        for i, (lbl, name, icon) in enumerate(zip(LABEL_COLS, LABEL_NAMES, LABEL_ICONS)):
            render_risk_card(name, icon, scores[lbl], preds[lbl], TUNED_THRESHOLDS[lbl])

        # Summary
        flagged = [LABEL_NAMES[i] for i, lbl in enumerate(LABEL_COLS) if preds[lbl]]
        if flagged:
            st.markdown(f"""
            <div class="info-box">
                <strong>⚠ Elevated risk detected</strong><br>
                The model flagged elevated risk for: <strong>{', '.join(flagged)}</strong>.<br>
                These results are intended to support clinical review — not replace it.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <strong>✓ No elevated risk detected</strong><br>
                All three complication risks are below their clinical thresholds.
                Regular monitoring remains important.
            </div>
            """, unsafe_allow_html=True)

        mode_note = "" if MODEL_LOADED else " (demo mode — heuristic)"
        st.markdown(f"""
        <div style="font-size:0.72rem; color:#484f58; margin-top:0.5rem; text-align:right;">
            Model: ClassifierChain RF · Thresholds tuned for Recall ≥ 0.80{mode_note}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color:#484f58;">
            <div style="font-size:3rem; margin-bottom:1rem;">🩺</div>
            <div style="font-size:0.95rem;">
                Fill in the patient's clinical data on the left,<br>
                then click <strong style="color:#58a6ff;">Predict Complication Risk</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Metric reference
    st.markdown("""
    <div class="info-box" style="margin-top: 2rem;">
        <strong>Model Performance (test set)</strong><br>
        <table style="width:100%; font-size:0.82rem; margin-top:0.4rem; border-collapse:collapse;">
            <tr style="color:#8b949e; border-bottom:1px solid #30363d;">
                <td>Metric</td><td style="text-align:right;">Score</td>
            </tr>
            <tr><td>Recall-Macro (tuned)</td><td style="text-align:right; color:#3fb950;"><b>0.878</b></td></tr>
            <tr><td>F1-Macro (tuned)</td><td style="text-align:right;">0.362</td></tr>
            <tr><td>F1-Micro (tuned)</td><td style="text-align:right;">0.495</td></tr>
            <tr><td>ROC-AUC Macro</td><td style="text-align:right;">0.708</td></tr>
        </table>
        <span style="font-size:0.72rem;">Trained on UCI Diabetes 130-US Hospitals dataset (1999–2008)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠ <strong>Clinical Disclaimer</strong><br>
        This tool is for research and educational purposes only. It does not constitute
        medical advice and must not be used as the sole basis for clinical decisions.
        Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)
