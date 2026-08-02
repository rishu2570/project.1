from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import MODEL_PATH, RAW_DATA_PATH, REPORTS_DIR, SCALER_PATH, SUMMARY_PATH
from evaluation import evaluate_model, plot_results, plot_candlestick, plot_indicators, save_metrics, save_summary, generate_shap_summary
from model import load_model, train_model
from preprocess import prepare_dataset
from utils.data_utils import download_market_data, init_db, save_accuracy
from utils.helpers import get_logger

logger = get_logger(__name__)


def train_pipeline(symbol: str = "^NSEI") -> dict[str, Any]:
    init_db()
    df = download_market_data(symbol=symbol)
    X_train, X_test, y_train, y_test, scaler, processed_df = prepare_dataset(df)
    model, metrics, history = train_model(X_train, y_train, X_test, y_test)
    model.save(MODEL_PATH)
    import joblib
    joblib.dump(scaler, SCALER_PATH)
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    metrics = evaluate_model(model, X_test, y_test, y_pred_prob, y_pred)
    save_metrics(metrics)
    save_summary(metrics)
    plot_results(y_test, y_pred_prob, y_pred, history)
    plot_candlestick(processed_df)
    plot_indicators(processed_df)
    generate_shap_summary(model, X_test[:100])
    save_accuracy(metrics)
    return {"metrics": metrics, "model_path": str(MODEL_PATH), "scaler_path": str(SCALER_PATH)}
