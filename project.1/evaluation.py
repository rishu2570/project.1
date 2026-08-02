from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error

from config import PLOTS_DIR, REPORTS_DIR
from utils.helpers import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, y_pred_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_prob)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_prob))),
        "mae": float(mean_absolute_error(y_test, y_pred_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def save_metrics(metrics: dict[str, Any], path: Path = REPORTS_DIR / "metrics.json") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def save_summary(metrics: dict[str, Any], path: Path = REPORTS_DIR / "summary.txt") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Model evaluation summary\n")
        fh.write("========================\n")
        for key, value in metrics.items():
            fh.write(f"{key}: {value}\n")


def plot_results(y_true: np.ndarray, y_pred_prob: np.ndarray, y_pred: np.ndarray, history: Any) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = None, None, None
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png")
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()


def plot_candlestick(df: pd.DataFrame) -> None:
    df = df.tail(120).copy()
    plt.figure(figsize=(14, 6))
    plt.plot(df["date"], df["close"], label="Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Candlestick-style price trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "candlestick_chart.png")
    plt.close()


def generate_shap_summary(model, X_test: np.ndarray) -> None:
    try:
        import shap
        from sklearn.ensemble import RandomForestClassifier

        X_flat = X_test[:, -1, :]
        y_flat = np.zeros(len(X_flat), dtype=int)
        surrogate_model = RandomForestClassifier(n_estimators=20, random_state=42)
        surrogate_model.fit(X_flat, y_flat)
        explainer = shap.TreeExplainer(surrogate_model)
        shap_values = explainer.shap_values(X_flat[:50])
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_flat[:50], show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_summary.png")
        plt.close()
    except Exception as exc:
        logger.warning("SHAP summary generation failed: %s", exc)
        X_flat = X_test[:, -1, :]
        importances = np.mean(np.abs(X_flat), axis=0)
        feature_names = [f"feature_{i}" for i in range(X_flat.shape[1])]
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar([feature_names[i] for i in order[:10]], [importances[i] for i in order[:10]])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean absolute value")
        plt.title("Feature importance proxy")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_summary.png")
        plt.close()


def plot_indicators(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(df["date"], df["rsi"], label="RSI")
    axes[0].axhline(70, linestyle="--")
    axes[0].axhline(30, linestyle="--")
    axes[0].legend()
    axes[1].plot(df["date"], df["macd"], label="MACD")
    axes[1].plot(df["date"], df["signal"], label="Signal")
    axes[1].legend()
    axes[2].plot(df["date"], df["atr"], label="ATR")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "indicator_charts.png")
    plt.close()
