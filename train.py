"""
train.py
--------
Train Logistic Regression and Random Forest, compare results,
save the best model, and produce evaluation plots.

Usage:
    python src/train.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_data, preprocess
from src.utils import (
    save_model,
    print_metrics,
    plot_churn_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
)


def train():
    """Full end-to-end training pipeline."""

    # ── 1. Load & Preprocess ──────────────────────────────────────────────────
    print("\n══ Step 1 · Load & Preprocess ══════════════════════════")
    raw_df = load_data("data/data.csv")

    # Show churn distribution before any processing
    target_binary = raw_df["Churn"].map({"Yes": 1, "No": 0})
    plot_churn_distribution(target_binary)

    X, y, scaler = preprocess(raw_df, fit_scaler=True)
    feature_names = list(X.columns)

    # ── 2. Train / Test Split ─────────────────────────────────────────────────
    print("\n══ Step 2 · Train / Test Split ═════════════════════════")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    # ── 3. Train Models ───────────────────────────────────────────────────────
    print("\n══ Step 3 · Train Models ════════════════════════════════")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    results = {}   # name → accuracy
    trained = {}   # name → fitted model

    for name, model in models.items():
        print(f"\n  ▸ Training {name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = print_metrics(name, y_test, y_pred)
        results[name] = acc
        trained[name] = model

    # ── 4. Compare & Choose Best ──────────────────────────────────────────────
    print("\n══ Step 4 · Model Comparison ════════════════════════════")
    plot_model_comparison(results)

    best_name  = max(results, key=results.get)
    best_model = trained[best_name]
    print(f"\n  ✓ Best model → {best_name}  ({results[best_name]*100:.2f}% accuracy)")

    # ── 5. Detailed Evaluation ────────────────────────────────────────────────
    print("\n══ Step 5 · Detailed Evaluation ═════════════════════════")
    y_best_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_best_pred, best_name)

    # Feature importance (always show Random Forest's importances)
    rf_model = trained.get("Random Forest")
    if rf_model:
        plot_feature_importance(rf_model, feature_names, top_n=15)

    # ── 6. Save Artefacts ─────────────────────────────────────────────────────
    print("\n══ Step 6 · Save Artefacts ══════════════════════════════")
    save_model(best_model,   "model.pkl")
    save_model(scaler,       "scaler.pkl")
    save_model(feature_names, "feature_names.pkl")

    print("\n  All artefacts saved to model/")
    print("  Training complete ✓\n")

    return best_model, scaler, feature_names, results


if __name__ == "__main__":
    train()
