"""Predictive Maintenance (NASA CMAPSS) — data, features, LSTM model and explainability."""
import os

# On Windows, importing scikit-learn before TensorFlow loads a conflicting
# OpenMP runtime that breaks TF's native DLL init. Preloading TensorFlow here —
# before any submodule imports sklearn — guarantees the safe order everywhere.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:  # pragma: no cover - best effort; harmless if TF is absent
    import tensorflow as _tf  # noqa: F401
except Exception:  # noqa: BLE001
    _tf = None

__all__ = ["data", "features", "model", "explain"]

WINDOW = 30          # sequence length fed to the LSTM
RUL_CAP = 125        # piecewise-linear RUL clip (standard for CMAPSS)
