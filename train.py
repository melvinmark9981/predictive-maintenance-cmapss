"""Train the LSTM RUL model on CMAPSS FD001 (real if present, else synthetic).

Usage:  python train.py

Saves to ``models/``:  lstm_rul.keras, scaler.joblib, meta.json, shap_importance.csv
The Streamlit app loads these artifacts — no training happens at deploy time.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TF logging

import numpy as np
import pandas as pd

from pdm import RUL_CAP, WINDOW
from pdm.data import get_data
from pdm.explain import shap_importance
from pdm.features import (
    add_rul, fit_scaler, make_training_sequences, select_features, test_matrix,
)
from pdm.model import build_lstm, predict_rul, save_artifacts


def main(epochs: int = 20, seed: int = 42) -> None:
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)

    train_df, test_df, test_rul, source = get_data(seed=seed)
    print(f"Data source: {source}  |  train rows: {len(train_df)}  test units: {test_df['unit'].nunique()}")

    train_df = add_rul(train_df, RUL_CAP)
    feature_cols = select_features(train_df)
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    scaler = fit_scaler(train_df, feature_cols)
    X, y = make_training_sequences(train_df, feature_cols, scaler, WINDOW)
    print(f"Training sequences: {X.shape}")

    model = build_lstm(WINDOW, len(feature_cols))
    from tensorflow.keras.callbacks import EarlyStopping
    model.fit(
        X, y, validation_split=0.1, epochs=epochs, batch_size=256,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=2,
    )

    # Evaluate on the held-out test units (one last-window per unit).
    X_test, unit_ids = test_matrix(test_df, feature_cols, scaler, WINDOW)
    preds = predict_rul(model, X_test)
    y_true = test_rul.to_numpy(dtype="float32")[: len(preds)].clip(max=RUL_CAP)
    rmse = float(np.sqrt(np.mean((preds - y_true) ** 2)))
    mae = float(np.mean(np.abs(preds - y_true)))
    print(f"\nTest RMSE: {rmse:.2f} cycles  |  MAE: {mae:.2f} cycles")

    # SHAP feature importance (with permutation fallback).
    imp_df, method = shap_importance(model, X, X_test, feature_cols)
    imp_df.to_csv(os.path.join(os.path.dirname(__file__), "models", "shap_importance.csv"), index=False)
    print(f"Importance method: {method}")
    print(imp_df.head(8).to_string(index=False))

    save_artifacts(model, scaler, {
        "window": WINDOW,
        "rul_cap": RUL_CAP,
        "feature_cols": feature_cols,
        "data_source": source,
        "test_rmse": round(rmse, 2),
        "test_mae": round(mae, 2),
        "importance_method": method,
        "seed": seed,
    })
    print("\nSaved artifacts to models/.")


if __name__ == "__main__":
    main()
