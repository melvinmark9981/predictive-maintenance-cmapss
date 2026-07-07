"""LSTM RUL model: build, train, predict, and artifact persistence."""
from __future__ import annotations

import json
import os

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_rul.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
SHAP_PATH = os.path.join(MODELS_DIR, "shap_importance.csv")


def build_lstm(window: int, n_features: int):
    """A compact stacked-LSTM regressor for Remaining Useful Life."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def predict_rul(model, X: np.ndarray) -> np.ndarray:
    """Predict RUL, clipped at 0 (negative remaining life is meaningless)."""
    preds = model.predict(X, verbose=0).ravel()
    return np.clip(preds, 0, None)


def save_artifacts(model, scaler, meta: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def artifacts_exist() -> bool:
    return all(os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, META_PATH))


def load_artifacts():
    """Return ``(model, scaler, meta)`` from disk."""
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    return model, scaler, meta
