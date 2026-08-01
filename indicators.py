from __future__ import annotations

import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")

    close = df["close"].astype(float)
    df["ma_20"] = close.rolling(window=20).mean()
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    rolling_mean = close.rolling(window=20).mean()
    rolling_std = close.rolling(window=20).std()
    df["bb_upper"] = rolling_mean + 2 * rolling_std
    df["bb_lower"] = rolling_mean - 2 * rolling_std
    low = df["low"].astype(float)
    high = df["high"].astype(float)
    k_percent = 100 * ((close - low.rolling(window=14).min()) / (high.rolling(window=14).max() - low.rolling(window=14).min()).replace(0, np.nan))
    df["stoch_k"] = k_percent
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
    df["atr"] = np.maximum(
        (high - low),
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    ).rolling(window=14).mean()
    df["obv"] = np.where(close.diff() > 0, df["volume"].astype(float), np.where(close.diff() < 0, -df["volume"].astype(float), 0)).cumsum()
    df["daily_returns"] = close.pct_change()
    df["volume"] = df["volume"].astype(float)
    df["target"] = np.where(close.shift(-1) > close, 1, 0)
    return df
