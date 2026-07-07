"""CMAPSS data loading.

The NASA CMAPSS turbofan dataset (subset **FD001**) has a fixed 26-column,
space-separated layout:

    unit  cycle  op1 op2 op3  s1 s2 ... s21

This module loads the real Kaggle/NASA files when present in ``data/`` and
otherwise generates a **realistic synthetic dataset with the identical schema**
so the app and training pipeline run out-of-the-box.

Kaggle dataset: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
Drop ``train_FD001.txt``, ``test_FD001.txt`` and ``RUL_FD001.txt`` into ``data/``
to train/evaluate on the real thing — everything downstream is unchanged.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

INDEX_NAMES = ["unit", "cycle"]
SETTING_NAMES = ["op1", "op2", "op3"]
SENSOR_NAMES = [f"s{i}" for i in range(1, 22)]
COLUMNS = INDEX_NAMES + SETTING_NAMES + SENSOR_NAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def real_files_present(data_dir: str = DATA_DIR) -> bool:
    return all(
        os.path.exists(os.path.join(data_dir, f))
        for f in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
    )


def load_fd001(data_dir: str = DATA_DIR):
    """Load the real FD001 train/test frames and the test RUL targets."""
    def _read(name):
        df = pd.read_csv(os.path.join(data_dir, name), sep=r"\s+", header=None)
        df = df.iloc[:, : len(COLUMNS)]
        df.columns = COLUMNS
        return df

    train = _read("train_FD001.txt")
    test = _read("test_FD001.txt")
    rul = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"), sep=r"\s+", header=None
    ).iloc[:, 0]
    rul.name = "RUL"
    return train, test, rul


def get_data(data_dir: str = DATA_DIR, seed: int = 42):
    """Return ``(train_df, test_df, test_rul, source)``.

    ``source`` is ``"real"`` or ``"synthetic"`` for display in the UI.
    """
    if real_files_present(data_dir):
        train, test, rul = load_fd001(data_dir)
        return train, test, rul, "real"
    train, test, rul = generate_synthetic(seed)
    return train, test, rul, "synthetic"


# ---------------------------------------------------------------------------
# Synthetic CMAPSS-like generator
# ---------------------------------------------------------------------------
# Sensors that are (near-)constant in real FD001, kept constant here too so the
# feature-selection logic behaves identically on synthetic and real data.
_CONSTANT_SENSORS = {1, 5, 6, 10, 16, 18, 19}

# Degradation horizon: sensors sit at a healthy plateau while remaining life
# exceeds this, then drift monotonically over the final ~DEGRADATION_HORIZON
# cycles. Set just above RUL_CAP so sensor state is a consistent function of the
# (capped) RUL target across units — which is what makes RUL learnable.
DEGRADATION_HORIZON = 130


def _sensor_profile(rng: np.random.Generator):
    """Random baseline, degradation direction/magnitude and noise per sensor."""
    profile = {}
    for i in range(1, 22):
        base = rng.uniform(20, 600)
        if i in _CONSTANT_SENSORS:
            profile[i] = (base, 0.0, 0.0)  # exactly constant -> dropped by feature selection
        else:
            direction = rng.choice([-1.0, 1.0])
            drift = direction * rng.uniform(0.08, 0.25) * base
            noise = abs(base) * rng.uniform(0.05, 0.09) + 0.3
            profile[i] = (base, drift, noise)
    return profile


def generate_synthetic(seed: int = 42, n_train: int = 100, n_test: int = 100):
    """Generate a CMAPSS-format dataset with realistic degradation.

    Each unit runs from a healthy state (health index h=1) to failure (h=0);
    non-constant sensors drift with ``h`` plus noise. Test units are truncated
    before failure and the remaining life is returned as the RUL target.
    """
    rng = np.random.default_rng(seed)
    profile = _sensor_profile(rng)

    def _unit_frame(unit_id: int, life: int, truncate_at: int | None):
        n = truncate_at or life
        cycles = np.arange(1, n + 1)
        # Per-unit variation (manufacturing spread) so the sensor->RUL mapping is
        # noisy, not exact — this keeps the task realistically hard.
        horizon = int(rng.integers(110, 151))
        curve = rng.uniform(1.0, 1.6)
        # Health from *remaining life*: h=1 while far from failure, ramping to 0
        # over the final `horizon` cycles.
        remaining = life - cycles
        h = np.clip(remaining / horizon, 0.0, 1.0) ** curve

        cols = {"unit": unit_id, "cycle": cycles}
        for name, v in zip(SETTING_NAMES, (0.0, 0.0, 100.0)):
            cols[name] = float(v)  # single operating condition -> constant
        for i in range(1, 22):
            base, drift, noise = profile[i]
            unit_offset = rng.normal(0, noise * 0.3)          # small per-unit bias
            drift_scale = 1.0 + rng.normal(0, 0.10)           # per-unit severity
            cols[f"s{i}"] = (
                base + unit_offset
                + drift * drift_scale * (1.0 - h)
                + rng.normal(0, noise, len(cycles))
            )
        return pd.DataFrame(cols)

    # Training units: full run-to-failure trajectories.
    train_frames = []
    for u in range(1, n_train + 1):
        life = int(rng.integers(130, 340))
        train_frames.append(_unit_frame(u, life, None))
    train = pd.concat(train_frames, ignore_index=True)

    # Test units: truncated, with the held-out remaining life as the target.
    test_frames, ruls = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(130, 340))
        truncate_at = int(rng.integers(30, life - 10))
        test_frames.append(_unit_frame(u, life, truncate_at))
        ruls.append(life - truncate_at)
    test = pd.concat(test_frames, ignore_index=True)
    rul = pd.Series(ruls, name="RUL")

    return train, test, rul
