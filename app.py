"""
app.py
------
Streamlit web application for Customer Churn Prediction.

Run with:
    streamlit run app.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background: #0d1117; }

    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a2332 0%, #0d2137 50%, #1a1a2e 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(56,139,253,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e6edf3;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(56,139,253,0.15);
        color: #388bfd;
        border: 1px solid rgba(56,139,253,0.3);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 1rem;
        font-family: 'DM Mono', monospace;
    }

    /* Section cards */
    .section-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        font-family: 'DM Mono', monospace;
    }

    /* Result boxes */
    .result-churn {
        background: linear-gradient(135deg, #3d1c1c, #2d1515);
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, #1c3d2a, #152d1e);
        border: 1px solid #3fb950;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .result-prob {
        font-size: 0.9rem;
        color: #8b949e;
        margin-top: 0.4rem;
        font-family: 'DM Mono', monospace;
    }

    /* Metric pills */
    .metric-pill {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        color: #e6edf3;
        margin: 4px;
    }

    /* Override Streamlit widgets */
    .stSelectbox > div > div,
    .stSlider > div,
    .stNumberInput > div > div > input {
        background: #21262d !important;
        border-color: #30363d !important;
        color: #e6edf3 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #388bfd, #1a6fd8) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #21262d !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and feature names from disk."""
    try:
        with open("model/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError:
        return None, None, None


model, scaler, feature_names = load_artifacts()

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FIELDS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def build_and_predict(customer: dict) -> dict:
    """Build feature row, scale numerics, predict."""
    data = customer.copy()

    # Binary encode
    for f in BINARY_FIELDS:
        if f in data:
            data[f] = 1 if str(data[f]).lower() == "yes" else 0

    df = pd.DataFrame([data])
    existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # Align to training columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    return {"prediction": pred, "label": "Churn" if pred == 1 else "Not Churn", "probability": prob}


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">ML · BINARY CLASSIFICATION</div>
    <h1 class="hero-title">📡 Customer Churn Predictor</h1>
    <p class="hero-subtitle">Predict whether a telecom customer is likely to leave — powered by Random Forest</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️  Model not found. Run `python src/train.py` first to train and save the model.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 About")
    st.markdown("""
    This app predicts **customer churn** for a telecom company.

    **How it works:**
    1. Fill in customer details
    2. Click **Predict**
    3. View churn risk instantly

    **Model:** Random Forest
    **Dataset:** Telco Customer Churn (~7k records)
    """)

    model_type = type(model).__name__
    st.markdown("---")
    st.markdown(f"""
    <div class="metric-pill">🤖 {model_type}</div>
    <div class="metric-pill">📊 30 features</div>
    <div class="metric-pill">🎯 ~78% accuracy</div>
    """, unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### ✏️ Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-title">👤 Demographics</div>', unsafe_allow_html=True)
    gender          = st.selectbox("Gender",         ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Has Partner",    ["Yes", "No"])
    dependents      = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure          = st.slider("Tenure (months)", 1, 72, 12)

with col2:
    st.markdown('<div class="section-title">📶 Services</div>', unsafe_allow_html=True)
    phone_service   = st.selectbox("Phone Service",      ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines",     ["No", "Yes", "No phone service"])
    internet        = st.selectbox("Internet Service",   ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security",    ["No", "Yes", "No internet service"])
    online_backup   = st.selectbox("Online Backup",      ["No", "Yes", "No internet service"])
    device_prot     = st.selectbox("Device Protection",  ["No", "Yes", "No internet service"])
    tech_support    = st.selectbox("Tech Support",       ["No", "Yes", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV",       ["No", "Yes", "No internet service"])
    streaming_mov   = st.selectbox("Streaming Movies",   ["No", "Yes", "No internet service"])

with col3:
    st.markdown('<div class="section-title">💳 Billing & Contract</div>', unsafe_allow_html=True)
    contract        = st.selectbox("Contract Type",  ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                       float(tenure * monthly_charges), step=1.0)

# ── Predict button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("🔮  Predict Churn", use_container_width=True)

if predict_btn:
    customer = {
        "gender":           gender,
        "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_mov,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    with st.spinner("Analysing customer profile…"):
        result = build_and_predict(customer)

    # ── Result display ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction Result")

    r_col1, r_col2, r_col3 = st.columns([2, 1, 1])

    with r_col1:
        pct = result["probability"] * 100
        if result["prediction"] == 1:
            st.markdown(f"""
            <div class="result-churn">
                <p class="result-label" style="color:#f85149;">⚠️ Likely to Churn</p>
                <p class="result-prob">Churn probability: {pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <p class="result-label" style="color:#3fb950;">✅ Likely to Stay</p>
                <p class="result-prob">Churn probability: {pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    with r_col2:
        st.metric("Churn Risk", f"{pct:.1f}%")

    with r_col3:
        retention = 100 - pct
        st.metric("Retention Score", f"{retention:.1f}%")

    # Risk level
    st.markdown("<br>", unsafe_allow_html=True)
    if pct < 30:
        st.success("🟢 **Low Risk** — Customer shows strong retention signals.")
    elif pct < 60:
        st.warning("🟡 **Medium Risk** — Consider proactive retention offers.")
    else:
        st.error("🔴 **High Risk** — Immediate intervention recommended.")

    # Progress bar
    st.markdown(f"**Churn Probability Gauge**")
    st.progress(result["probability"])
