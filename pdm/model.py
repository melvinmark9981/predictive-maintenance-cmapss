"""LSTM RUL model: build (Keras, training-only), predict, and artifact persistence.

The trained Keras model is exported to **ONNX** and the deployed Streamlit app
loads it via ``onnxruntime`` instead of TensorFlow. This keeps the runtime
dependency footprint to `onnxruntime` (which ships wheels for the newest
Python releases) rather than `tensorflow`, which lags behind on Python-version
support and can fail to install on hosts that default to a very recent Python
(e.g. Streamlit Community Cloud). TensorFlow is still required to *train*.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "lstm_rul.keras")   # training artifact (not used by the app)
ONNX_PATH = os.path.join(MODELS_DIR, "lstm_rul.onnx")     # deployed-app artifact
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
SHAP_PATH = os.path.join(MODELS_DIR, "shap_importance.csv")


def build_lstm(window: int, n_features: int):
    """A compact stacked-LSTM regressor for Remaining Useful Life. Training only."""
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
    """Predict RUL, clipped at 0. Works with a Keras model or an onnxruntime session."""
    if hasattr(model, "get_inputs"):  # onnxruntime.InferenceSession
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        preds = model.run([output_name], {input_name: X.astype("float32")})[0].ravel()
    else:  # Keras model (training/evaluation only)
        preds = model.predict(X, verbose=0).ravel()
    return np.clip(preds, 0, None)


def export_onnx(model, window: int, n_features: int) -> None:
    """Convert a trained Keras model to ONNX for TensorFlow-free deployment.

    ``tf2onnx.convert.from_keras`` fails to introspect Keras 3 output tensors
    (see tf2onnx#2319), so we go through an intermediate SavedModel export and
    the ``tf2onnx`` CLI, which handles Keras 3 models correctly.
    """
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp)
        result = subprocess.run(
            [
                sys.executable, "-m", "tf2onnx.convert",
                "--saved-model", tmp,
                "--output", ONNX_PATH,
                "--opset", "13",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"tf2onnx conversion failed:\n{result.stderr}")


def save_artifacts(model, scaler, meta: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    export_onnx(model, meta["window"], len(meta["feature_cols"]))
    joblib.dump(scaler, SCALER_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def artifacts_exist() -> bool:
    return all(os.path.exists(p) for p in (ONNX_PATH, SCALER_PATH, META_PATH))


def load_artifacts():
    """Return ``(session, scaler, meta)`` from disk — onnxruntime, no TensorFlow needed."""
    import onnxruntime as ort

    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    return session, scaler, meta
