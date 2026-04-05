"""
preprocessing.py
----------------
Handles all data cleaning, encoding, and feature engineering steps
for the Customer Churn Prediction pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Constants ────────────────────────────────────────────────────────────────

# Simple Yes/No columns → 1/0
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]

# Multi-class categoricals → One-Hot Encoded
CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# Numeric columns to standardise
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Columns to drop (IDs, non-predictive)
DROP_COLS = ["customerID"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data from the given file path."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix missing / malformed values.
    - TotalCharges: coerce to numeric, fill NaN with median.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    print(f"[handle_missing_values] Fixed {n_missing} missing TotalCharges value(s).")
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yes/No columns to 1/0 integers."""
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    print(f"[encode_binary_columns] Encoded {len(BINARY_COLS)} binary column(s).")
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to multi-class categorical columns."""
    df = df.copy()
    existing = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=existing, drop_first=True)
    print(f"[one_hot_encode] Encoded {len(existing)} column(s). DataFrame now has {df.shape[1]} columns.")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier / leakage columns."""
    df = df.copy()
    cols = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols, inplace=True)
    print(f"[drop_unused_columns] Dropped: {cols}.")
    return df


def scale_numeric_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit: bool = True,
) -> tuple:
    """
    Standardise numeric columns (zero mean, unit variance).

    Parameters
    ----------
    df     : Input DataFrame (already encoded).
    scaler : Pre-fitted scaler; if None a new one is created.
    fit    : If True, fit the scaler on df; otherwise only transform.

    Returns
    -------
    (scaled_df, scaler)
    """
    df = df.copy()
    existing = [c for c in NUMERIC_COLS if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
    if fit:
        df[existing] = scaler.fit_transform(df[existing])
    else:
        df[existing] = scaler.transform(df[existing])
    print(f"[scale_numeric_features] Scaled {len(existing)} numeric column(s).")
    return df, scaler


def preprocess(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline: clean → encode → scale → split X / y.

    Returns
    -------
    X      : Feature DataFrame
    y      : Target Series  (1 = Churn, 0 = Retained)
    scaler : Fitted StandardScaler (save this for inference)
    """
    df = handle_missing_values(df)
    df = encode_binary_columns(df)
    df = drop_unused_columns(df)
    df = one_hot_encode(df)
    df, scaler = scale_numeric_features(df, scaler=scaler, fit=fit_scaler)

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    print(f"[preprocess] Final shapes — X: {X.shape}, y: {y.shape}")
    return X, y, scaler


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw = load_data("data/data.csv")
    X, y, scaler = preprocess(raw)
    print("\nSample feature names:", list(X.columns[:6]))
    print(X.head(2))
