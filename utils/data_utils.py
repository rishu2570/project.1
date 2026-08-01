import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from config import DEFAULT_END_DATE, DEFAULT_START_DATE, DEFAULT_SYMBOL, RAW_DATA_PATH, DB_PATH
from utils.helpers import get_logger

logger = get_logger(__name__)


def download_market_data(symbol: str = DEFAULT_SYMBOL, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> pd.DataFrame:
    logger.info("Downloading market data for %s", symbol)
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df.empty:
            raise ValueError("No data returned")
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index()
        columns = []
        for col in df.columns:
            if isinstance(col, tuple):
                col = col[0]
            columns.append(str(col).replace(" ", "_").lower())
        df.columns = columns
        if "date" not in df.columns and "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        if "adj_close" not in df.columns and "close" in df.columns:
            df["adj_close"] = df["close"]
        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        if "volume" not in df.columns and "vol" in df.columns:
            df = df.rename(columns={"vol": "volume"})
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df.to_csv(RAW_DATA_PATH, index=False)
        return df
    except Exception as exc:
        logger.warning("Falling back to synthetic data: %s", exc)
        return create_synthetic_dataset(symbol=symbol)


def create_synthetic_dataset(symbol: str = DEFAULT_SYMBOL, rows: int = 2500) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)
    base = 18000 + np.arange(rows) * 0.9 + rng.normal(0, 30, rows)
    close = np.maximum(base, 1000)
    open_ = close + rng.normal(0, 5, rows)
    high = np.maximum(open_, close) + rng.uniform(3, 7, rows)
    low = np.minimum(open_, close) - rng.uniform(3, 7, rows)
    volume = rng.integers(2_000_000, 10_000_000, rows)
    date = pd.date_range(start=DEFAULT_START_DATE, periods=rows, freq="D")
    df = pd.DataFrame({"date": date, "open": open_, "high": high, "low": low, "close": close, "adj_close": close, "volume": volume})
    df.to_csv(RAW_DATA_PATH, index=False)
    return df


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            accuracy REAL NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1_score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(symbol: str, prediction: str, confidence: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (symbol, prediction, confidence, created_at) VALUES (?, ?, ?, datetime('now'))",
        (symbol, prediction, confidence),
    )
    conn.commit()
    conn.close()


def save_request(symbol: str, source: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO user_requests (symbol, source, created_at) VALUES (?, ?, datetime('now'))",
        (symbol, source),
    )
    conn.commit()
    conn.close()


def save_accuracy(metrics: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO model_accuracy (accuracy, precision, recall, f1_score, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
        (metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]),
    )
    conn.commit()
    conn.close()


def get_history() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT symbol, prediction, confidence, created_at FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()
    return [{"symbol": row[0], "prediction": row[1], "confidence": row[2], "created_at": row[3]} for row in rows]
