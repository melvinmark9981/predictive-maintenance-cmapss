"""SHAP feature importance for the LSTM RUL model.

Primary path: ``shap.GradientExplainer`` on the LSTM directly (works well with
TF/Keras deep models). If SHAP cannot run in a given environment, we fall back
to model-agnostic **permutation importance** on the same LSTM so the dashboard
always has an importance ranking. The method actually used is recorded in meta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_importance(model, X_background: np.ndarray, X_sample: np.ndarray,
                    feature_cols: list[str]) -> tuple[pd.DataFrame, str]:
    """Return ``(importance_df, method)`` ranked by mean |contribution|."""
    try:
        import shap

        bg = X_background[np.random.default_rng(0).choice(
            len(X_background), size=min(80, len(X_background)), replace=False)]
        explainer = shap.GradientExplainer(model, bg)
        values = explainer.shap_values(X_sample)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 4:          # (samples, window, features, outputs)
            values = values[..., 0]
        # Mean absolute SHAP across samples and time steps -> per feature.
        importance = np.abs(values).mean(axis=(0, 1))
        method = "SHAP (GradientExplainer)"
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[explain] SHAP unavailable ({exc!r}); using permutation importance.")
        importance = _permutation_importance(model, X_sample)
        method = "Permutation importance (SHAP fallback)"

    df = (
        pd.DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df, method


def _permutation_importance(model, X: np.ndarray) -> np.ndarray:
    """Increase in RMSE when each feature's time series is shuffled."""
    from .model import predict_rul

    rng = np.random.default_rng(0)
    baseline = predict_rul(model, X)
    n_features = X.shape[2]
    scores = np.zeros(n_features)
    for f in range(n_features):
        permuted = X.copy()
        order = rng.permutation(len(X))
        permuted[:, :, f] = permuted[order, :, f]
        preds = predict_rul(model, permuted)
        scores[f] = np.sqrt(np.mean((preds - baseline) ** 2))
    return scores
