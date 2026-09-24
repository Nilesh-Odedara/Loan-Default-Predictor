"""
====================================================
  Loan Default Prediction — Streamlit Application
  B.Tech CSE | Semester 5 | Machine Learning Project
====================================================
Preprocessing facts (from training):
  - Scaler: MinMaxScaler applied ONLY to 9 numerical columns
  - Encoders: LabelEncoder dict with lowercase column-name keys
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ──────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="[LDP]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────
#  CUSTOM CSS
# ──────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]          { font-family: 'Inter', sans-serif; }
.stApp                               { background: linear-gradient(135deg,#0f1117 0%,#1a1d2e 50%,#0f1117 100%); }
[data-testid="stSidebar"]            { background: linear-gradient(180deg,#1a1d2e 0%,#12152a 100%);
                                       border-right: 1px solid rgba(99,179,237,.15); }
.card                                { background: rgba(255,255,255,.04);
                                       border: 1px solid rgba(99,179,237,.15);
                                       border-radius: 16px; padding: 1.5rem 2rem;
                                       margin-bottom: 1.5rem; backdrop-filter: blur(10px); }
.section-title                       { font-size:1.1rem; font-weight:600; color:#63b3ed;
                                       letter-spacing:.04em; text-transform:uppercase;
                                       margin-bottom:1rem; padding-bottom:.4rem;
                                       border-bottom:2px solid rgba(99,179,237,.25); }
.hero-title                          { font-size:2.4rem; font-weight:700;
                                       background:linear-gradient(90deg,#63b3ed,#9f7aea,#f687b3);
                                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                                       background-clip:text; margin-bottom:.3rem; }
.hero-sub                            { color:#a0aec0; font-size:1rem; margin-bottom:2rem; }
.stButton > button                   { width:100%;
                                       background:linear-gradient(135deg,#3182ce,#6b46c1);
                                       color:white; border:none; border-radius:12px;
                                       padding:.85rem 2rem; font-size:1.05rem; font-weight:600;
                                       letter-spacing:.03em; cursor:pointer; transition:all .25s ease; }
.stButton > button:hover             { background:linear-gradient(135deg,#2b6cb0,#553c9a);
                                       transform:translateY(-2px);
                                       box-shadow:0 8px 20px rgba(99,102,241,.35); }
.result-default                      { background:linear-gradient(135deg,rgba(229,62,62,.15),rgba(229,62,62,.05));
                                       border:1px solid rgba(229,62,62,.4);
                                       border-radius:16px; padding:1.8rem 2rem; text-align:center; }
.result-no-default                   { background:linear-gradient(135deg,rgba(72,187,120,.15),rgba(72,187,120,.05));
                                       border:1px solid rgba(72,187,120,.4);
                                       border-radius:16px; padding:1.8rem 2rem; text-align:center; }
.result-label                        { font-size:1.7rem; font-weight:700; margin-bottom:.3rem; }
.result-sub                          { color:#a0aec0; font-size:.95rem; }
.metric-tile                         { background:rgba(255,255,255,.03);
                                       border:1px solid rgba(255,255,255,.08);
                                       border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.metric-tile .metric-val             { font-size:1.6rem; font-weight:700; color:#63b3ed; }
.metric-tile .metric-label           { color:#718096; font-size:.8rem; margin-top:.15rem; }
/* Metric score pill */
.score-pill                          { display:inline-block; border-radius:20px;
                                       padding:.2rem .75rem; font-size:.82rem;
                                       font-weight:600; letter-spacing:.02em; }
.score-high   { background:rgba(72,187,120,.18); color:#68d391; border:1px solid rgba(72,187,120,.3); }
.score-mid    { background:rgba(246,173,85,.15);  color:#f6ad55; border:1px solid rgba(246,173,85,.25); }
.score-low    { background:rgba(229,62,62,.15);   color:#fc8181; border:1px solid rgba(229,62,62,.25); }
/* Confusion matrix */
.cm-table     { border-collapse:collapse; width:100%; text-align:center; }
.cm-table th  { background:rgba(99,179,237,.12); color:#63b3ed; padding:.5rem .8rem;
                font-size:.82rem; letter-spacing:.04em; text-transform:uppercase; }
.cm-table td  { padding:.6rem .8rem; font-size:.9rem; font-weight:600;
                border:1px solid rgba(255,255,255,.06); }
.cm-tp        { background:rgba(72,187,120,.18);  color:#68d391; }
.cm-tn        { background:rgba(99,179,237,.12);  color:#90cdf4; }
.cm-fp        { background:rgba(229,62,62,.14);   color:#fc8181; }
.cm-fn        { background:rgba(246,173,85,.12);  color:#f6ad55; }
/* Bar chart fill */
.bar-wrap     { background:rgba(255,255,255,.05); border-radius:6px;
                height:10px; overflow:hidden; margin-top:.25rem; }
.bar-fill     { height:100%; border-radius:6px;
                background:linear-gradient(90deg,#3182ce,#9f7aea); }
hr            { border-color:rgba(99,179,237,.1) !important; }
label         { color:#cbd5e0 !important; font-size:.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────
MODEL_PATHS = {
    "Logistic Regression":       "models/logistic_regression.pkl",
    "K-Nearest Neighbors (KNN)": "models/knn.pkl",
    "Decision Tree":             "models/decision_tree.pkl",
    "Random Forest":             "models/random_forest.pkl",
}
SCALER_PATH   = "preprocessing/scaler.pkl"
ENCODER_PATH  = "preprocessing/label_encoders.pkl"
METRICS_PATH  = "preprocessing/model_metrics.json"

CATEGORICAL_KEYS = [
    "education","employmenttype","maritalstatus",
    "hasmortgage","hasdependents","loanpurpose","hascosigner",
]
NUMERICAL_COLS = [
    "age","income","loanamount","creditscore","monthsemployed",
    "numcreditlines","interestrate","loanterm","dtiratio",
]
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_KEYS

MODEL_META = {
    "Logistic Regression": {
        "icon":"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><polyline points='23 6 13.5 15.5 8.5 10.5 1 18'/><polyline points='17 6 23 6 23 12'/></svg>","color":"#63b3ed",
        "desc":"A linear probabilistic classifier that models the log-odds of default as a linear combination of input features. Fast, interpretable, and great as a baseline.",
        "pros":"Interpretable · Fast inference · Outputs probabilities",
        "cons":"Cannot capture non-linear relationships",
        "type":"Linear","complexity":"Low","supports_proba":True,
    },
    "K-Nearest Neighbors (KNN)": {
        "icon":"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#9f7aea' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><circle cx='7.5' cy='7.5' r='1.5'/><circle cx='18.5' cy='5.5' r='1.5'/><circle cx='11.5' cy='11.5' r='1.5'/><circle cx='7.5' cy='16.5' r='1.5'/><circle cx='17.5' cy='14.5' r='1.5'/><circle cx='6.5' cy='20.5' r='1.5'/><circle cx='18.5' cy='20.5' r='1.5'/></svg>","color":"#9f7aea",
        "desc":"Classifies each sample by majority vote of its K nearest training neighbors in feature space. Non-parametric with no assumptions on data distribution.",
        "pros":"Simple · No explicit training · Adapts locally",
        "cons":"Slow on large datasets · Sensitive to feature scale",
        "type":"Instance-based","complexity":"Medium","supports_proba":True,
    },
    "Decision Tree": {
        "icon":"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#68d391' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><line x1='6' y1='3' x2='6' y2='15'/><circle cx='18' cy='6' r='3'/><circle cx='6' cy='18' r='3'/><path d='M18 9a9 9 0 0 1-9 9'/></svg>","color":"#68d391",
        "desc":"Recursively partitions the feature space using if-else rules, forming an interpretable tree structure. Handles non-linear relationships naturally.",
        "pros":"Interpretable · Handles non-linearity · No scaling needed",
        "cons":"Prone to overfitting on noisy data",
        "type":"Tree-based","complexity":"Medium","supports_proba":True,
    },
    "Random Forest": {
        "icon":"<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#f6ad55' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><polygon points='12 2 2 7 12 12 22 7 12 2'/><polyline points='2 17 12 22 22 17'/><polyline points='2 12 12 17 22 12'/></svg>","color":"#f6ad55",
        "desc":"An ensemble of decision trees trained on random subsets (bagging). Significantly reduces overfitting and variance compared to a single tree.",
        "pros":"High accuracy · Robust to noise · Feature importance",
        "cons":"Less interpretable · Slow to train on large datasets",
        "type":"Ensemble","complexity":"High","supports_proba":True,
    },
}

# ──────────────────────────────────────────────────
#  CACHED LOADERS
# ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load(SCALER_PATH)

@st.cache_resource(show_spinner=False)
def load_encoders():
    return joblib.load(ENCODER_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None

# ──────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────
def score_class(v: float) -> str:
    if v >= 0.80: return "score-high"
    if v >= 0.60: return "score-mid"
    return "score-low"

def pct(v): return f"{v:.1%}"

def bar(v, color="#3182ce"):
    return (
        f"<div class='bar-wrap'>"
        f"<div class='bar-fill' style='width:{v*100:.1f}%;background:{color};'></div>"
        f"</div>"
    )

def svg(paths, color="#63b3ed", size=18):
    """Build a Lucide-style SVG with single-quoted attrs — safe to embed in Python strings."""
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"width='{size}' height='{size}' viewBox='0 0 24 24' "
        f"fill='none' stroke='{color}' stroke-width='2' "
        f"stroke-linecap='round' stroke-linejoin='round' "
        f"style='vertical-align:middle;display:inline;flex-shrink:0;'>"
        f"{paths}</svg>"
    )

# Lucide icon path library (MIT licensed — https://lucide.dev)
_ICON_PATHS = {
    "check-circle": (
        "<path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/>"
        "<polyline points='22 4 12 14.01 9 11.01'/>"
    ),
    "x-circle": (
        "<circle cx='12' cy='12' r='10'/>"
        "<line x1='15' y1='9' x2='9' y2='15'/>"
        "<line x1='9' y1='9' x2='15' y2='15'/>"
    ),
    "alert-tri": (
        "<path d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z'/>"
        "<line x1='12' y1='9' x2='12' y2='13'/>"
        "<line x1='12' y1='17' x2='12.01' y2='17'/>"
    ),
    "bank": (
        "<rect x='3' y='9' width='18' height='11' rx='1'/>"
        "<path d='M3 9l9-6 9 6'/>"
        "<line x1='9' y1='9' x2='9' y2='20'/>"
        "<line x1='15' y1='9' x2='15' y2='20'/>"
    ),
    "cpu": (
        "<rect x='4' y='4' width='16' height='16' rx='2'/>"
        "<rect x='9' y='9' width='6' height='6'/>"
        "<line x1='9' y1='1' x2='9' y2='4'/><line x1='15' y1='1' x2='15' y2='4'/>"
        "<line x1='9' y1='20' x2='9' y2='23'/><line x1='15' y1='20' x2='15' y2='23'/>"
        "<line x1='20' y1='9' x2='23' y2='9'/><line x1='20' y1='14' x2='23' y2='14'/>"
        "<line x1='1' y1='9' x2='4' y2='9'/><line x1='1' y1='14' x2='4' y2='14'/>"
    ),
    "trending-up": (
        "<polyline points='23 6 13.5 15.5 8.5 10.5 1 18'/>"
        "<polyline points='17 6 23 6 23 12'/>"
    ),
    "scatter": (
        "<circle cx='7.5' cy='7.5' r='1.5'/><circle cx='18.5' cy='5.5' r='1.5'/>"
        "<circle cx='11.5' cy='11.5' r='1.5'/><circle cx='7.5' cy='16.5' r='1.5'/>"
        "<circle cx='17.5' cy='14.5' r='1.5'/><circle cx='6.5' cy='20.5' r='1.5'/>"
        "<circle cx='18.5' cy='20.5' r='1.5'/>"
    ),
    "git-branch": (
        "<line x1='6' y1='3' x2='6' y2='15'/>"
        "<circle cx='18' cy='6' r='3'/><circle cx='6' cy='18' r='3'/>"
        "<path d='M18 9a9 9 0 0 1-9 9'/>"
    ),
    "layers": (
        "<polygon points='12 2 2 7 12 12 22 7 12 2'/>"
        "<polyline points='2 17 12 22 22 17'/>"
        "<polyline points='2 12 12 17 22 12'/>"
    ),
}

def icon(name, color="#63b3ed", size=18):
    """Return an inline Lucide SVG icon string by name."""
    paths = _ICON_PATHS.get(name, _ICON_PATHS["alert-tri"])
    return svg(paths, color, size)

# ──────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0 .5rem;'>"
        "<span style='display:block;text-align:center;'><svg xmlns='http://www.w3.org/2000/svg' width='52' height='52' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><rect x='3' y='9' width='18' height='11' rx='1'/><path d='M3 9l9-6 9 6'/><line x1='9' y1='9' x2='9' y2='20'/><line x1='15' y1='9' x2='15' y2='20'/></svg></span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align:center;color:#63b3ed;font-size:1.25rem;"
        "font-weight:700;margin-bottom:.2rem;'>Loan Default Predictor</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#718096;font-size:.8rem;"
        "margin-bottom:1.5rem;'>B.Tech CSE · Sem 5 · ML Project</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        "<p style='color:#a0aec0;font-size:.78rem;font-weight:600;"
        "letter-spacing:.06em;text-transform:uppercase;margin-bottom:.5rem;'>"
        "<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='#a0aec0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;'><rect x='4' y='4' width='16' height='16' rx='2'/><rect x='9' y='9' width='6' height='6'/><line x1='9' y1='1' x2='9' y2='4'/><line x1='15' y1='1' x2='15' y2='4'/><line x1='9' y1='20' x2='9' y2='23'/><line x1='15' y1='20' x2='15' y2='23'/><line x1='20' y1='9' x2='23' y2='9'/><line x1='20' y1='14' x2='23' y2='14'/><line x1='1' y1='9' x2='4' y2='9'/><line x1='1' y1='14' x2='4' y2='14'/></svg>  Select Model</p>",
        unsafe_allow_html=True,
    )
    selected_model_name = st.selectbox(
        "Model", list(MODEL_PATHS.keys()), label_visibility="collapsed"
    )
    st.markdown("---")

    # Navigation hint
    st.markdown(
        "<p style='color:#a0aec0;font-size:.78rem;font-weight:600;"
        "letter-spacing:.06em;text-transform:uppercase;margin-bottom:.5rem;'>"
        "<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='#a0aec0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;'><path d='M4 19.5A2.5 2.5 0 0 1 6.5 17H20'/><path d='M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z'/></svg>  About</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#718096;font-size:.83rem;line-height:1.6;'>"
        "This application predicts loan default risk using four pre-trained ML models. "
        "Scroll down to see full <strong style='color:#63b3ed;'>model performance metrics</strong>, "
        "confusion matrices, and comparison charts."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        "<p style='color:#a0aec0;font-size:.78rem;font-weight:600;"
        "letter-spacing:.06em;text-transform:uppercase;margin-bottom:.6rem;'>"
        "<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='#a0aec0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;'><circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/></svg>  Available Models</p>",
        unsafe_allow_html=True,
    )
    metrics = load_metrics()
    for m in MODEL_PATHS:
        is_active = m == selected_model_name
        color = "#63b3ed" if is_active else "#4a5568"
        active = icon("check-circle","#63b3ed",13) + " " if is_active else "<span style='color:#4a5568;'>&bull;</span> "
        acc_tag = ""
        if metrics and m in metrics:
            acc_val = metrics[m]["accuracy"]
            acc_tag = f" <span style='color:#718096;font-size:.72rem;'>({acc_val:.1%})</span>"
        st.markdown(
            f"<div style='color:{color};font-size:.83rem;padding:.25rem 0;'>"
            f"{active}{m}{acc_tag}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        "<p style='color:#4a5568;font-size:.72rem;text-align:center;'>"
        "© 2025 ML Project · All rights reserved</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────
#  HERO HEADER
# ──────────────────────────────────────────────────
st.markdown(
    "<div class='hero-title'><span style='display:inline-block;vertical-align:middle;margin-right:.4rem;'><svg xmlns='http://www.w3.org/2000/svg' width='34' height='34' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><rect x='3' y='9' width='18' height='11' rx='1'/><path d='M3 9l9-6 9 6'/><line x1='9' y1='9' x2='9' y2='20'/><line x1='15' y1='9' x2='15' y2='20'/></svg></span> Loan Default Prediction</div>"
    "<div class='hero-sub'>Enter applicant information below and click "
    "<strong>Predict</strong> to get an instant risk assessment. "
    "Scroll down for full <strong>model performance analytics</strong>.</div>",
    unsafe_allow_html=True,
)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
for col, val, label in zip(
    [col_m1, col_m2, col_m3, col_m4],
    ["255,347", "16", "4", "Binary"],
    ["Total Records", "Input Features", "Trained Models", "Classification Task"],
):
    col.markdown(
        f"<div class='metric-tile'>"
        f"<div class='metric-val'>{val}</div>"
        f"<div class='metric-label'>{label}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
#  LOAD PREPROCESSING OBJECTS
# ──────────────────────────────────────────────────
try:
    scaler           = load_scaler()
    encoders         = load_encoders()
    preprocessing_ok = True
except Exception as e:
    st.error(f"Could not load preprocessing files: {e}")
    preprocessing_ok = False

# ──────────────────────────────────────────────────
#  INPUT FORM — Applicant Information
# ──────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><circle cx='12' cy='8' r='4'/><path d='M20 21a8 8 0 0 0-16 0'/></svg> Applicant Information</div>", unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1,
                          help="Applicant's age in years")
with r1c2:
    income = st.number_input("Annual Income ($)", min_value=0, max_value=10_000_000,
                              value=60_000, step=1_000, help="Gross annual income in USD")
with r1c3:
    education = st.selectbox("Education Level",
                             options=["Bachelor's","High School","Master's","PhD"],
                             help="Highest education level attained")
with r1c4:
    marital_status = st.selectbox("Marital Status",
                                  options=["Divorced","Married","Single"])

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
with r2c1:
    employment_type = st.selectbox("Employment Type",
                                   options=["Full-time","Part-time","Self-employed","Unemployed"])
with r2c2:
    months_employed = st.number_input("Months Employed", min_value=0, max_value=600,
                                       value=60, step=1)
with r2c3:
    has_dependents = st.selectbox("Has Dependents?", options=["No","Yes"])
with r2c4:
    has_mortgage = st.selectbox("Has Mortgage?", options=["No","Yes"])

st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
#  INPUT FORM — Financial Details
# ──────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><rect x='1' y='4' width='22' height='16' rx='2'/><line x1='1' y1='10' x2='23' y2='10'/></svg> Financial Details</div>", unsafe_allow_html=True)

r3c1, r3c2, r3c3, r3c4 = st.columns(4)
with r3c1:
    loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=1_000_000,
                                   value=25_000, step=500)
with r3c2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0,
                                     value=8.5, step=0.1, format="%.1f")
with r3c3:
    loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360,
                                 value=60, step=6)
with r3c4:
    loan_purpose = st.selectbox("Loan Purpose",
                                options=["Auto","Business","Education","Home","Other"])

r4c1, r4c2, r4c3, r4c4 = st.columns(4)
with r4c1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850,
                                    value=650, step=1, help="FICO score (300-850)")
with r4c2:
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0,
                                        max_value=50, value=3, step=1)
with r4c3:
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0,
                                 value=0.35, step=0.01, format="%.2f",
                                 help="Debt-to-Income ratio (0.00 - 1.00)")
with r4c4:
    has_cosigner = st.selectbox("Has Co-Signer?", options=["No","Yes"])

st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────
#  PREPROCESSING HELPERS
# ──────────────────────────────────────────────────
def build_raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age":[age],"income":[income],"loanamount":[loan_amount],
        "creditscore":[credit_score],"monthsemployed":[months_employed],
        "numcreditlines":[num_credit_lines],"interestrate":[interest_rate],
        "loanterm":[loan_term],"dtiratio":[dti_ratio],
        "education":[education],"employmenttype":[employment_type],
        "maritalstatus":[marital_status],"hasmortgage":[has_mortgage],
        "hasdependents":[has_dependents],"loanpurpose":[loan_purpose],
        "hascosigner":[has_cosigner],
    })

def preprocess_input(raw_df: pd.DataFrame) -> np.ndarray:
    df = raw_df.copy()
    for key in CATEGORICAL_KEYS:
        if key in encoders:
            le  = encoders[key]
            val = df[key].iloc[0]
            df[key] = int(le.transform([val])[0]) if val in le.classes_ else 0
        else:
            df[key] = 0
    num_df     = df[NUMERICAL_COLS].astype(float)
    num_scaled = scaler.transform(num_df)
    scaled_dict = dict(zip(NUMERICAL_COLS, num_scaled[0]))
    cat_dict    = {key: df[key].iloc[0] for key in CATEGORICAL_KEYS}
    combined    = {**scaled_dict, **cat_dict}
    return np.array([[combined[col] for col in FEATURE_COLS]])

# ──────────────────────────────────────────────────
#  PREDICTION SECTION
# ──────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><circle cx='11' cy='11' r='8'/><line x1='21' y1='21' x2='16.65' y2='16.65'/></svg> Run Prediction</div>", unsafe_allow_html=True)

col_btn, col_note = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("Predict Loan Default")
with col_note:
    st.markdown(
        f"<div style='color:#718096;font-size:.85rem;padding-top:.85rem;'>"
        f"Using model: <strong style='color:#63b3ed;'>{selected_model_name}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

if predict_clicked:
    if not preprocessing_ok:
        st.error("Cannot predict — preprocessing files failed to load.")
    else:
        with st.spinner("Running inference …"):
            try:
                raw_df     = build_raw_df()
                X          = preprocess_input(raw_df)
                model      = load_model(MODEL_PATHS[selected_model_name])
                prediction = int(model.predict(X)[0])
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]

                st.markdown("<br>", unsafe_allow_html=True)
                if prediction == 1:
                    st.markdown(
                        "<div class='result-default'>"
                        "<div class='result-label' style='color:#fc8181;'>"
                        "<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='#fc8181' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><path d='m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z'/><line x1='12' y1='9' x2='12' y2='13'/><line x1='12' y1='17' x2='12.01' y2='17'/></svg> Loan will likely be <u>Default</u></div>"
                        "<div class='result-sub'>The model predicts a "
                        "<strong>HIGH</strong> risk of loan default.</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='result-no-default'>"
                        "<div class='result-label' style='color:#68d391;'>"
                        "<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='#68d391' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><polyline points='22 4 12 14.01 9 11.01'/></svg> Loan will likely be <u>No Default</u></div>"
                        "<div class='result-sub'>The model predicts a "
                        "<strong>LOW</strong> risk of loan default.</div></div>",
                        unsafe_allow_html=True,
                    )

                if proba is not None:
                    p_default    = float(proba[1])
                    p_no_default = float(proba[0])
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><polyline points='22 12 18 12 15 21 9 3 6 12 2 12'/></svg> Prediction Probabilities</div>",
                                unsafe_allow_html=True)
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown(
                            f"<div class='metric-tile'>"
                            f"<div class='metric-val' style='color:#fc8181;'>{p_default:.1%}</div>"
                            f"<div class='metric-label'>Probability of Default</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.progress(p_default)
                    with pc2:
                        st.markdown(
                            f"<div class='metric-tile'>"
                            f"<div class='metric-val' style='color:#68d391;'>{p_no_default:.1%}</div>"
                            f"<div class='metric-label'>Probability of No Default</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.progress(p_no_default)

                with st.expander("<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><rect x='9' y='2' width='6' height='4' rx='1'/><path d='M8 4H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-2'/></svg> View Submitted Input Data"):
                    display_df = pd.DataFrame({
                        "Feature": [
                            "Age","Income ($)","Loan Amount ($)","Credit Score",
                            "Months Employed","Num Credit Lines","Interest Rate (%)",
                            "Loan Term (months)","DTI Ratio","Education",
                            "Employment Type","Marital Status","Has Mortgage",
                            "Has Dependents","Loan Purpose","Has Co-Signer",
                        ],
                        "Value": [
                            age,income,loan_amount,credit_score,
                            months_employed,num_credit_lines,interest_rate,
                            loan_term,dti_ratio,education,
                            employment_type,marital_status,has_mortgage,
                            has_dependents,loan_purpose,has_cosigner,
                        ],
                    })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════
#  MODEL PERFORMANCE DASHBOARD
# ══════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-title' style='font-size:1.9rem;'><span style='display:inline-block;vertical-align:middle;margin-right:.4rem;'><svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><line x1='12' y1='20' x2='12' y2='10'/><line x1='18' y1='20' x2='18' y2='4'/><line x1='6' y1='20' x2='6' y2='16'/></svg></span> Model Performance Dashboard</div>"
    "<div class='hero-sub'>Evaluation metrics computed on the held-out test set "
    "(20% of 255,347 records · random_state=42)</div>",
    unsafe_allow_html=True,
)

metrics = load_metrics()

if metrics is None:
    st.warning(
        "Model metrics file not found (`preprocessing/model_metrics.json`). "
        "Run the evaluation script to generate it."
    )
else:
    # ── 1. SUMMARY COMPARISON TABLE ──────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><rect x='3' y='3' width='7' height='7'/><rect x='14' y='3' width='7' height='7'/><rect x='14' y='14' width='7' height='7'/><rect x='3' y='14' width='7' height='7'/></svg> All Models — Side-by-Side Comparison</div>",
                unsafe_allow_html=True)

    # Build comparison dataframe
    rows = []
    metric_keys = ["accuracy","precision","recall","f1","roc_auc"]
    metric_labels = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    for mname, mdata in metrics.items():
        row = {"Model": mname}
        for k in metric_keys:
            row[k] = mdata.get(k)
        rows.append(row)

    comp_df = pd.DataFrame(rows).set_index("Model")

    # Render as styled HTML table
    def render_comparison_table(df):
        col_labels = {"accuracy":"Accuracy","precision":"Precision",
                      "recall":"Recall","f1":"F1-Score","roc_auc":"ROC-AUC"}
        html = "<table style='width:100%;border-collapse:collapse;'>"
        html += "<thead><tr>"
        html += "<th style='text-align:left;padding:.6rem 1rem;color:#63b3ed;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em;background:rgba(99,179,237,.08);border-radius:8px 0 0 0;'>Model</th>"
        for k,lab in col_labels.items():
            html += f"<th style='text-align:center;padding:.6rem .8rem;color:#63b3ed;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em;background:rgba(99,179,237,.08);'>{lab}</th>"
        html += "</tr></thead><tbody>"

        # Find best per column
        bests = {}
        for k in col_labels:
            vals = [r[k] for r in rows if r[k] is not None]
            bests[k] = max(vals) if vals else None

        for row in rows:
            mname = row["Model"]
            meta  = MODEL_META.get(mname, {})
            icon  = meta.get("icon","")
            color = meta.get("color","#63b3ed")
            html += "<tr>"
            html += (f"<td style='padding:.65rem 1rem;border-bottom:1px solid rgba(255,255,255,.05);'>"
                     f"<span style='font-size:1rem;'>{icon}</span> "
                     f"<strong style='color:{color};'>{mname}</strong></td>")
            for k in col_labels:
                val = row.get(k)
                if val is None:
                    cell = "<span style='color:#4a5568;'>N/A</span>"
                else:
                    cls   = score_class(val)
                    crown = " &#9733;" if val == bests.get(k) else ""
                    cell  = f"<span class='score-pill {cls}'>{pct(val)}{crown}</span>"
                html += (f"<td style='text-align:center;padding:.65rem .8rem;"
                         f"border-bottom:1px solid rgba(255,255,255,.05);'>{cell}</td>")
            html += "</tr>"

        html += "</tbody></table>"
        return html

    st.markdown(render_comparison_table(rows), unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#4a5568;font-size:.75rem;margin-top:.6rem;'>"
        "&#9733; = Best score in that metric &nbsp;|&nbsp; "
        "<span class='score-pill score-high'>≥80%</span> &nbsp;"
        "<span class='score-pill score-mid'>60–80%</span> &nbsp;"
        "<span class='score-pill score-low'>&lt;60%</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. METRIC BAR CHARTS (Plotly) ────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><line x1='12' y1='20' x2='12' y2='10'/><line x1='18' y1='20' x2='18' y2='4'/><line x1='6' y1='20' x2='6' y2='16'/></svg> Metric Comparison — Bar Charts</div>",
                unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        model_names  = list(metrics.keys())
        short_names  = [n.replace("K-Nearest Neighbors ","KNN\n") for n in model_names]
        bar_colors   = [MODEL_META[n]["color"] for n in model_names]
        metric_pairs = [
            ("accuracy","Accuracy"),("precision","Precision"),
            ("recall","Recall"),("f1","F1-Score"),("roc_auc","ROC-AUC"),
        ]

        fig = make_subplots(
            rows=1, cols=5,
            subplot_titles=[lab for _,lab in metric_pairs],
            shared_yaxes=True,
        )
        for ci, (key, label) in enumerate(metric_pairs, start=1):
            vals = [metrics[n].get(key, 0) or 0 for n in model_names]
            fig.add_trace(
                go.Bar(
                    x=short_names, y=vals,
                    marker_color=bar_colors,
                    text=[f"{v:.1%}" for v in vals],
                    textposition="outside",
                    showlegend=False,
                    name=label,
                ),
                row=1, col=ci,
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(color="#a0aec0", size=11),
            height=380,
            margin=dict(l=10,r=10,t=50,b=30),
            yaxis=dict(range=[0,1.05], tickformat=".0%",
                       gridcolor="rgba(255,255,255,.05)"),
        )
        fig.update_xaxes(tickfont=dict(size=10))
        fig.update_annotations(font_color="#63b3ed")

        for ax in ["yaxis2","yaxis3","yaxis4","yaxis5"]:
            fig.update_layout(**{ax: dict(range=[0,1.05], showticklabels=False,
                                          gridcolor="rgba(255,255,255,.05)")})

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback: HTML bars
        metric_display = [("accuracy","Accuracy"),("precision","Precision"),
                          ("recall","Recall"),("f1","F1-Score"),("roc_auc","ROC-AUC")]
        for key, label in metric_display:
            st.markdown(f"<p style='color:#a0aec0;font-size:.85rem;font-weight:600;"
                        f"margin-bottom:.3rem;'>{label}</p>", unsafe_allow_html=True)
            cols = st.columns(len(metrics))
            for i, (mname, mdata) in enumerate(metrics.items()):
                val = mdata.get(key)
                color = MODEL_META.get(mname,{}).get("color","#63b3ed")
                with cols[i]:
                    if val is not None:
                        st.markdown(
                            f"<div style='font-size:.75rem;color:#718096;'>{mname.split()[0]}</div>"
                            f"<div style='font-size:1rem;font-weight:700;color:{color};'>{pct(val)}</div>"
                            + bar(val, color),
                            unsafe_allow_html=True,
                        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. INDIVIDUAL MODEL CARDS WITH CONFUSION MATRIX ──
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><polygon points='12 2 2 7 12 12 22 7 12 2'/><polyline points='2 17 12 22 22 17'/><polyline points='2 12 12 17 22 12'/></svg> Individual Model Details & Confusion Matrix</div>",
                unsafe_allow_html=True)

    # Tab labels must be plain text — st.tabs() does not support HTML/SVG
    _TAB_LABELS = {
        "Logistic Regression":       "Logistic Regression",
        "K-Nearest Neighbors (KNN)": "KNN",
        "Decision Tree":             "Decision Tree",
        "Random Forest":             "Random Forest",
    }
    tabs = st.tabs([_TAB_LABELS.get(n, n) for n in metrics])
    for tab, (mname, mdata) in zip(tabs, metrics.items()):
        with tab:
            meta = MODEL_META.get(mname, {})
            color = meta.get("color","#63b3ed")

            # ── Overfitting alert ──────────────────────────────────
            train_acc = mdata.get("train_accuracy")
            test_acc  = mdata.get("accuracy", 0)
            if train_acc is not None and train_acc >= 0.99 and (train_acc - test_acc) > 0.05:
                gap = train_acc - test_acc
                st.markdown(
                    f"<div style='background:rgba(246,173,85,.12);border:1px solid rgba(246,173,85,.35);"
                    f"border-radius:10px;padding:.7rem 1rem;margin-bottom:.8rem;font-size:.83rem;'>"
                    + icon("alert-tri","#f6ad55",15)
                    + f" <strong style='color:#f6ad55;'>Overfitting Detected</strong> &mdash; "
                    f"Training accuracy: <strong>{train_acc:.1%}</strong> vs "
                    f"Test accuracy: <strong>{test_acc:.1%}</strong> "
                    f"(gap: <strong style='color:#fc8181;'>{gap:.1%}</strong>). "
                    f"This model memorised the training data and generalises poorly.</div>",
                    unsafe_allow_html=True,
                )

            # ── 6 metric tiles ─────────────────────────────────────
            tc1,tc2,tc3,tc4,tc5,tc6 = st.columns(6)
            tile_defs = [
                (tc1, "Train Acc",  mdata.get("train_accuracy"), "#4a9eca"),
                (tc2, "Test Acc",   mdata.get("accuracy"),       "#63b3ed"),
                (tc3, "Precision",  mdata.get("precision"),      "#9f7aea"),
                (tc4, "Recall",     mdata.get("recall"),         "#f6ad55"),
                (tc5, "F1-Score",   mdata.get("f1"),             "#68d391"),
                (tc6, "ROC-AUC",    mdata.get("roc_auc"),        "#f687b3"),
            ]
            for tcol, tlabel, tval, tcolor in tile_defs:
                with tcol:
                    disp = pct(tval) if tval is not None else "N/A"
                    st.markdown(
                        f"<div class='metric-tile'>"
                        f"<div class='metric-val' style='color:{tcolor};'>{disp}</div>"
                        f"<div class='metric-label'>{tlabel}</div></div>"
                        + (bar(tval, tcolor) if tval else ""),
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            left, right = st.columns([1, 1])

            # Confusion Matrix
            with left:
                cm = mdata.get("confusion_matrix", {})
                tn = cm.get("tn",0); fp = cm.get("fp",0)
                fn = cm.get("fn",0); tp = cm.get("tp",0)
                total = tn + fp + fn + tp

                st.markdown(
                    "<p style='color:#a0aec0;font-size:.85rem;font-weight:600;margin-bottom:.6rem;'>"
                    "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#a0aec0' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><rect x='3' y='3' width='7' height='7'/><rect x='14' y='3' width='7' height='7'/><rect x='14' y='14' width='7' height='7'/><rect x='3' y='14' width='7' height='7'/></svg> Confusion Matrix</p>",
                    unsafe_allow_html=True,
                )
                cm_html = f"""
                <table class='cm-table'>
                  <thead>
                    <tr>
                      <th style='background:transparent;border:none;'></th>
                      <th>Predicted: No Default</th>
                      <th>Predicted: Default</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <th>Actual: No Default</th>
                      <td class='cm-tn'>TN = {tn:,}<br><span style='font-size:.72rem;font-weight:400;color:#718096;'>({tn/total:.1%})</span></td>
                      <td class='cm-fp'>FP = {fp:,}<br><span style='font-size:.72rem;font-weight:400;color:#718096;'>({fp/total:.1%})</span></td>
                    </tr>
                    <tr>
                      <th>Actual: Default</th>
                      <td class='cm-fn'>FN = {fn:,}<br><span style='font-size:.72rem;font-weight:400;color:#718096;'>({fn/total:.1%})</span></td>
                      <td class='cm-tp'>TP = {tp:,}<br><span style='font-size:.72rem;font-weight:400;color:#718096;'>({tp/total:.1%})</span></td>
                    </tr>
                  </tbody>
                </table>
                """
                st.markdown(cm_html, unsafe_allow_html=True)
                st.markdown(
                    f"<p style='color:#4a5568;font-size:.72rem;margin-top:.5rem;'>"
                    f"Test set: {total:,} samples &nbsp;|&nbsp; "
                    f"Default rate: {mdata.get('default_rate',0):.1%}</p>",
                    unsafe_allow_html=True,
                )

            # Model description + stats
            with right:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);"
                    f"border-radius:12px;padding:1.2rem;'>"
                    f"<div style='font-size:1.3rem;margin-bottom:.3rem;'>{meta.get('icon','')} "
                    f"<strong style='color:{color};'>{mname}</strong></div>"
                    f"<div style='color:#a0aec0;font-size:.83rem;line-height:1.65;margin-bottom:.8rem;'>"
                    f"{meta.get('desc','')}</div>"
                    f"<div style='display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.7rem;'>"
                    f"<span style='background:rgba(99,179,237,.1);color:#63b3ed;border-radius:6px;"
                    f"padding:.2rem .7rem;font-size:.75rem;'>Type: {meta.get('type','')}</span>"
                    f"<span style='background:rgba(159,122,234,.1);color:#9f7aea;border-radius:6px;"
                    f"padding:.2rem .7rem;font-size:.75rem;'>Complexity: {meta.get('complexity','')}</span>"
                    f"<span style='background:rgba(72,187,120,.1);color:#68d391;border-radius:6px;"
                    f"padding:.2rem .7rem;font-size:.75rem;'>Proba: {'Yes' if meta.get('supports_proba') else 'No'}</span>"
                    f"</div>"
                    f"<div style='font-size:.8rem;margin-bottom:.3rem;'>"
                    f"<span style='color:#68d391;'><svg xmlns='http://www.w3.org/2000/svg' width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='#68d391' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><path d='M22 11.08V12a10 10 0 1 1-5.93-9.14'/><polyline points='22 4 12 14.01 9 11.01'/></svg> Strengths:</span> "
                    f"<span style='color:#a0aec0;'>{meta.get('pros','')}</span></div>"
                    f"<div style='font-size:.8rem;'>"
                    f"<span style='color:#fc8181;'><svg xmlns='http://www.w3.org/2000/svg' width='13' height='13' viewBox='0 0 24 24' fill='none' stroke='#fc8181' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><circle cx='12' cy='12' r='10'/><line x1='15' y1='9' x2='9' y2='15'/><line x1='9' y1='9' x2='15' y2='15'/></svg> Weaknesses:</span> "
                    f"<span style='color:#a0aec0;'>{meta.get('cons','')}</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Derived stats
                if total > 0:
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    st.markdown("<br>", unsafe_allow_html=True)
                    ds1, ds2 = st.columns(2)
                    with ds1:
                        st.markdown(
                            f"<div class='metric-tile'>"
                            f"<div class='metric-val' style='color:#90cdf4;'>{specificity:.1%}</div>"
                            f"<div class='metric-label'>Specificity (TNR)</div></div>",
                            unsafe_allow_html=True,
                        )
                    with ds2:
                        st.markdown(
                            f"<div class='metric-tile'>"
                            f"<div class='metric-val' style='color:#f6ad55;'>{sensitivity:.1%}</div>"
                            f"<div class='metric-label'>Sensitivity (TPR)</div></div>",
                            unsafe_allow_html=True,
                        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 4. DATASET INFO ──────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='#63b3ed' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' style='vertical-align:middle;display:inline;flex-shrink:0;'><line x1='12' y1='17' x2='12' y2='22'/><path d='M5 17h14v-1.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V6h1a2 2 0 0 0 0-4H8a2 2 0 0 0 0 4h1v4.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24z'/></svg> Project Notes & Dataset Info</div>",
                unsafe_allow_html=True)

    n1, n2, n3 = st.columns(3)
    with n1:
        st.info(
            "**Binary Classification Task**\n\n"
            "The model predicts one of two classes:\n"
            "- `0` → **No Default** (loan repaid)\n"
            "- `1` → **Default** (loan not repaid)\n\n"
            "Train/Test split: **80% / 20%** · `random_state=42`"
        )
    with n2:
        st.warning(
            "**Imbalanced Dataset**\n\n"
            "- Default: **29,653** records (**11.6%**)\n"
            "- No Default: **225,694** records (**88.4%**)\n\n"
            "This bias means **Precision, Recall & F1** matter "
            "more than plain Accuracy."
        )
    with n3:
        st.success(
            "**Metric Glossary**\n\n"
            "- **Precision** = TP / (TP + FP)\n"
            "- **Recall** = TP / (TP + FN)\n"
            "- **F1** = harmonic mean of P & R\n"
            "- **ROC-AUC** = area under ROC curve\n"
            "- **Specificity** = TN / (TN + FP)"
        )

    st.markdown(
        "<div style='color:#4a5568;font-size:.78rem;margin-top:.5rem;text-align:center;'>"
        "All models loaded from pre-trained <code>.pkl</code> files. "
        "No retraining occurs in this application.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
