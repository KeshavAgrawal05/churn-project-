"""
utils.py
--------
Shared utility functions: model persistence, metrics printing,
and all visualisation helpers (confusion matrix, feature importance, etc.)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Directory resolution ──────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(obj, filename: str = "model.pkl") -> str:
    """Persist any Python object to model/<filename>."""
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[save_model] Saved → {path}")
    return path


def load_model(filename: str = "model.pkl"):
    """Load a pickled object from model/<filename>."""
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefact not found: {path}\n"
            "  → Run `python src/train.py` first."
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[load_model] Loaded ← {path}")
    return obj


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(name: str, y_true, y_pred) -> float:
    """Print accuracy + full classification report; return accuracy float."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  Model    : {name}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["Not Churn", "Churn"]))
    return acc


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_churn_distribution(y: pd.Series, save_path: str = None) -> None:
    """Bar chart showing churn vs. not-churn counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    labels = ["Not Churn (0)", "Churn (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts.values):
        pct = count / len(y) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Customer Churn Distribution", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path: str = None) -> None:
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names: list, top_n: int = 15, save_path: str = None) -> None:
    """
    Horizontal bar chart of the top-N feature importances.
    Works with any sklearn estimator that has .feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        print("[plot_feature_importance] Model has no feature_importances_. Skipping.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(top)))
    top.plot(kind="barh", ax=ax, color=colors, edgecolor="white")

    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")
    plt.close()


def plot_model_comparison(results: dict, save_path: str = None) -> None:
    """Bar chart comparing accuracy of multiple models."""
    names  = list(results.keys())
    accs   = [v * 100 for v in results.values()]
    colors = ["#3498db", "#9b59b6", "#e67e22", "#1abc9c"][: len(names)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {save_path}")
    plt.close()
