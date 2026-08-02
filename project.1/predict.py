from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from config import MODEL_PATH, RAW_DATA_PATH, SCALER_PATH, SEQUENCE_LENGTH
from indicators import add_technical_indicators
from preprocess import build_sequences
from utils.data_utils import download_market_data, save_prediction, save_request
from utils.helpers import get_logger

logger = get_logger(__name__)


def predict_movement(symbol: str = "^NSEI", source: str = "web") -> dict[str, Any]:
    save_request(symbol, source)
    df = download_market_data(symbol=symbol)
    df = add_technical_indicators(df)
    feature_columns = [
        "open", "high", "low", "close", "adj_close", "volume",
        "ma_20", "ema_12", "ema_26", "macd", "signal", "rsi",
        "bb_upper", "bb_lower", "stoch_k", "stoch_d", "atr", "obv", "daily_returns"
    ]
    df = df.dropna(subset=feature_columns + ["target"]).copy()
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_columns)
    scaler = joblib.load(SCALER_PATH)
    scaled = scaler.transform(df[feature_columns])
    seq = build_sequences(scaled, SEQUENCE_LENGTH)[-1:]
    model = tf.keras.models.load_model(MODEL_PATH)
    prob = float(model.predict(seq, verbose=0)[0][0])
    prediction = "UP" if prob >= 0.5 else "DOWN"
    save_prediction(symbol, prediction, prob)
    latest = df.iloc[-1]
    return {
        "symbol": symbol,
        "prediction": prediction,
        "confidence": round(prob, 4),
        "close_price": float(latest["close"]),
        "volume": float(latest["volume"]),
        "rsi": float(latest["rsi"]),
        "macd": float(latest["macd"]),
        "atr": float(latest["atr"]),
        "obv": float(latest["obv"]),
        "daily_returns": float(latest["daily_returns"]),
    }
