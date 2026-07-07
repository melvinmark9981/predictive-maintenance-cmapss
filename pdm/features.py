"""Feature engineering: RUL labelling, sensor selection, scaling and windowing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import RUL_CAP, WINDOW
from .data import SENSOR_NAMES, SETTING_NAMES


def add_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add a piecewise-linear RUL column (capped) to a run-to-failure frame."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


def select_features(train_df: pd.DataFrame, var_threshold: float = 1e-4) -> list[str]:
    """Keep operating settings + sensors that actually vary (drop flat ones)."""
    candidates = SETTING_NAMES + SENSOR_NAMES
    stds = train_df[candidates].std()
    return [c for c in candidates if stds[c] > var_threshold]


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    return MinMaxScaler().fit(train_df[feature_cols].to_numpy())


def scale(df: pd.DataFrame, feature_cols: list[str], scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[feature_cols].to_numpy())


def make_training_sequences(
    train_df: pd.DataFrame, feature_cols: list[str], scaler: MinMaxScaler,
    window: int = WINDOW,
):
    """Sliding windows over each training unit -> (X, y) for supervised RUL."""
    xs, ys = [], []
    for _, unit in train_df.groupby("unit"):
        feats = scaler.transform(unit[feature_cols].to_numpy())
        rul = unit["RUL"].to_numpy()
        if len(unit) < window:
            continue
        for start in range(len(unit) - window + 1):
            xs.append(feats[start : start + window])
            ys.append(rul[start + window - 1])
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="float32")


def last_window(unit_df: pd.DataFrame, feature_cols: list[str], scaler: MinMaxScaler,
                window: int = WINDOW) -> np.ndarray:
    """The most recent ``window`` cycles for one unit (front-padded if short)."""
    feats = scaler.transform(unit_df[feature_cols].to_numpy())
    if len(feats) >= window:
        seq = feats[-window:]
    else:
        pad = np.repeat(feats[:1], window - len(feats), axis=0)
        seq = np.vstack([pad, feats])
    return seq.astype("float32")


def test_matrix(test_df: pd.DataFrame, feature_cols: list[str], scaler: MinMaxScaler,
                window: int = WINDOW):
    """One last-window sequence per test unit -> (X, unit_ids)."""
    xs, units = [], []
    for uid, unit in test_df.groupby("unit"):
        xs.append(last_window(unit, feature_cols, scaler, window))
        units.append(int(uid))
    return np.asarray(xs, dtype="float32"), units
