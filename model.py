from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from config import BATCH_SIZE, EPOCHS, MODEL_PATH, PATIENCE, SAVED_MODELS_DIR
from utils.helpers import get_logger

logger = get_logger(__name__)


def build_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling1D(pool_size=2),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> tuple[tf.keras.Model, dict[str, Any], Any]:
    model = build_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(filepath=str(MODEL_PATH), save_best_only=True, monitor="val_loss"),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics, history


def load_model(path: str | Path = MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(path)
