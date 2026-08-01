from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import SEQUENCE_LENGTH, TEST_SIZE, RANDOM_STATE
from indicators import add_technical_indicators
from utils.helpers import get_logger

logger = get_logger(__name__)


def prepare_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, pd.DataFrame]:
    df = add_technical_indicators(df)
    feature_columns = [
        "open", "high", "low", "close", "adj_close", "volume",
        "ma_20", "ema_12", "ema_26", "macd", "signal", "rsi",
        "bb_upper", "bb_lower", "stoch_k", "stoch_d", "atr", "obv", "daily_returns"
    ]
    df = df.dropna(subset=feature_columns + ["target"]).copy()
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_columns + ["target"])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_columns])
    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled)):
        X.append(scaled[i - SEQUENCE_LENGTH:i])
        y.append(df["target"].iloc[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test, scaler, df


def build_sequences(features: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    return np.array([features[i - sequence_length:i] for i in range(sequence_length, len(features))], dtype=np.float32)
