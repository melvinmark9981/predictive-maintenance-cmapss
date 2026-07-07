# 🔧 Predictive Maintenance — Turbofan Engine RUL (NASA CMAPSS)

An **Industry 4.0 / smart-manufacturing** dashboard that predicts the
**Remaining Useful Life (RUL)** of turbofan engines from multivariate sensor
streams using an **LSTM** deep-learning model, and turns those predictions into
**failure-risk alerts** and maintenance actions — with **SHAP** explainability.

Built with **Python + Streamlit + TensorFlow/Keras + SHAP + Plotly**, trained on
the **NASA CMAPSS** turbofan degradation dataset (subset FD001).

> ⚠️ The repo ships a model trained on a **synthetic CMAPSS-format dataset** so
> it runs and deploys with zero setup. Drop the real Kaggle FD001 files into
> `data/` and rerun `python train.py` to train on the real data — see
> [Using the real dataset](#-using-the-real-nasa-cmapss-dataset).

---

## 🏭 Industry 4.0 / smart manufacturing context

Traditional maintenance is either **reactive** (fix it after it breaks — costly
downtime) or **preventive** (fixed schedules — wasteful, replaces healthy
parts). **Predictive maintenance (PdM)** is the Industry 4.0 answer: use IoT
sensor data + machine learning to forecast *when* a machine will fail and act
**just in time**.

This dashboard demonstrates a full PdM loop for a fleet of jet engines:

- **Condition monitoring** — 21 simulated sensors (temperatures, pressures,
  speeds, flows) streamed per operating cycle.
- **RUL prognostics** — an LSTM reads the recent sensor window and predicts how
  many cycles remain before failure.
- **Failure-risk alerts** — engines are triaged into High / Medium / Low risk so
  planners know where to send technicians *first*.
- **Explainability** — SHAP shows which sensors drive each prediction, building
  the trust engineers need to act on a model's output.

The same pattern generalises to any smart factory asset — CNC spindles, pumps,
turbines, HVAC, robotics — wherever sensor telemetry and downtime cost meet.

---

## ✨ Features

- **LSTM RUL model** (stacked LSTM + dropout) trained on windowed sensor
  sequences with a piecewise-linear RUL target (capped at 125 cycles).
- **Fleet Overview** — KPIs, a ranked predicted-RUL chart colour-coded by risk,
  and an actionable maintenance-alert list.
- **Engine Deep-Dive** — per-engine predicted vs actual RUL, a **predicted-RUL
  trajectory** over the engine's life, and interactive **sensor-trend charts**.
- **SHAP feature importance** — which sensors most influence the prediction
  (computed offline via `GradientExplainer`, with a permutation fallback).
- **Adjustable risk thresholds**, dark professional UI, CSV-free deploy.
- **Deploy-ready**: the model is trained offline and committed; the app only
  *loads* it — no training on Streamlit Cloud.

---

## 🚀 Quick start (local)

Requires Python 3.9+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py            # a pre-trained model is already in models/
```

Opens at <http://localhost:8501>.

### Retrain the model (optional)

```bash
pip install -r requirements-train.txt
python train.py
```

This trains the LSTM, evaluates it (RMSE/MAE on held-out engines), computes SHAP
importance, and writes artifacts to `models/`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo (including `models/`) to **GitHub**.
2. At <https://share.streamlit.io> → **New app** → pick the repo/branch.
3. Main file path `app.py` → **Deploy**.

`requirements.txt` uses `tensorflow-cpu` and excludes training-only deps (SHAP),
keeping the deploy lean. CPU-only — no GPU required for training *or* inference.

---

## 🧠 How it works

| Stage | What happens |
|-------|--------------|
| **Data** (`pdm/data.py`) | Loads real FD001 `.txt` files if present, else generates a realistic synthetic CMAPSS dataset (26-column schema, degradation driven by remaining life). |
| **Features** (`pdm/features.py`) | Piecewise-linear RUL labels, drops constant sensors, MinMax scaling, sliding-window sequences (window = 30). |
| **Model** (`pdm/model.py`) | Keras `LSTM(64) → LSTM(32) → Dense`, MSE loss, Adam. |
| **Explain** (`pdm/explain.py`) | SHAP `GradientExplainer` over the LSTM → per-sensor importance (permutation fallback). |
| **App** (`app.py`) | Loads artifacts, predicts RUL for the fleet, renders overview / deep-dive / explainability. |

**Reference performance** (synthetic FD001-style data): test **RMSE ≈ 9 cycles**,
**MAE ≈ 7 cycles**. Expect real FD001 to land around RMSE 13–20 depending on
tuning.

---

## 📂 Project structure

```
.
├── app.py                    # Streamlit dashboard
├── train.py                  # Offline training + SHAP → models/
├── pdm/
│   ├── __init__.py           # constants + TF/sklearn import-order fix
│   ├── data.py               # CMAPSS loader + synthetic generator
│   ├── features.py           # RUL labels, scaling, windowing
│   ├── model.py              # LSTM build / predict / persistence
│   └── explain.py            # SHAP feature importance
├── models/                   # committed artifacts (model, scaler, meta, importance)
├── data/                     # drop real FD001 files here (git-ignored)
├── requirements.txt          # runtime deps
├── requirements-train.txt    # + SHAP for retraining
└── README.md
```

---

## 📚 Using the real NASA CMAPSS dataset

See [`data/README.md`](data/README.md). Download FD001 from
[Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps), place the three
`.txt` files in `data/`, and run `python train.py`.

## 📄 License

MIT — free to use, modify and build upon. CMAPSS data © NASA (public domain).
