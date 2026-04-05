"""
predict.py
----------
Prediction module: load the saved model + scaler and produce
churn probabilities for new customer records.

Usage (standalone):
    python src/predict.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model

# Numeric columns that need scaling (must match preprocessing.py)
NUMERIC_COLS   = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_FIELDS  = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


def build_input_df(customer: dict, feature_names: list) -> pd.DataFrame:
    """
    Convert a raw customer dict into a one-row DataFrame that matches
    the feature columns the model was trained on.

    Steps:
      1. Convert Yes/No fields → 1/0
      2. One-Hot encode categorical fields
      3. Align columns to training feature names (fill missing dummies with 0)
    """
    data = customer.copy()

    # Binary encode
    for field in BINARY_FIELDS:
        if field in data:
            data[field] = 1 if str(data[field]).strip().lower() == "yes" else 0

    df = pd.DataFrame([data])

    # One-hot encode
    existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # Align to training columns (add missing dummies as 0)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    return df


def predict_churn(customer: dict) -> dict:
    """
    Predict churn for a single customer record.

    Parameters
    ----------
    customer : dict with raw field values

    Returns
    -------
    dict with keys:
        'prediction'  : int   (1 = Churn, 0 = Not Churn)
        'label'       : str   ("Churn" | "Not Churn")
        'probability' : float (probability of churn, 0–1)
    """
    model         = load_model("model.pkl")
    scaler        = load_model("scaler.pkl")
    feature_names = load_model("feature_names.pkl")

    df = build_input_df(customer.copy(), feature_names)

    # Scale numeric columns
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])

    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction":  prediction,
        "label":       "Churn" if prediction == 1 else "Not Churn",
        "probability": round(probability, 4),
    }


def predict_batch(records: list) -> list:
    """Run predict_churn on a list of customer dicts."""
    return [predict_churn(rec) for rec in records]


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_customer = {
        "gender":           "Male",
        "SeniorCitizen":    0,
        "Partner":          "Yes",
        "Dependents":       "No",
        "tenure":           12,
        "PhoneService":     "Yes",
        "MultipleLines":    "No",
        "InternetService":  "Fiber optic",
        "OnlineSecurity":   "No",
        "OnlineBackup":     "No",
        "DeviceProtection": "No",
        "TechSupport":      "No",
        "StreamingTV":      "Yes",
        "StreamingMovies":  "Yes",
        "Contract":         "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod":    "Electronic check",
        "MonthlyCharges":   85.00,
        "TotalCharges":     1020.00,
    }

    result = predict_churn(sample_customer)
    print("\n── Prediction Result ─────────────────────────────────")
    print(f"  Prediction  : {result['label']}")
    print(f"  Probability : {result['probability']*100:.1f}% chance of churn")
    print("──────────────────────────────────────────────────────\n")
