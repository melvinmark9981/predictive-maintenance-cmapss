"""Predictive Maintenance — RUL dashboard for NASA CMAPSS turbofan engines.

Loads a pre-trained LSTM (no training at runtime) and presents a fleet health
overview, per-engine sensor trends & RUL trajectory, failure-risk alerts, and
SHAP feature importance.

Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# NOTE: importing `pdm` first preloads TensorFlow before scikit-learn (Windows
# OpenMP DLL-order fix) — see pdm/__init__.py.
from pdm.data import get_data
from pdm.features import last_window, test_matrix
from pdm.model import SHAP_PATH, artifacts_exist, load_artifacts, predict_rul

st.set_page_config(page_title="Predictive Maintenance — CMAPSS RUL", page_icon="\U0001F527", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      [data-testid="stMetric"] {background:#161b22;border:1px solid #30363d;border-radius:12px;padding:14px 16px;}
      [data-testid="stMetricLabel"] {opacity:.8;}
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_COLORS = {"High": "#e5534b", "Medium": "#e3b341", "Low": "#2ea043"}


# ---------------------------------------------------------------------------
# Data / model loading (cached once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model & data…")
def load_bundle():
    model, scaler, meta = load_artifacts()
    feature_cols = meta["feature_cols"]
    window = meta["window"]

    train_df, test_df, test_rul, source = get_data(seed=meta.get("seed", 42))

    X_test, unit_ids = test_matrix(test_df, feature_cols, scaler, window)
    preds = predict_rul(model, X_test)

    current_cycle = test_df.groupby("unit")["cycle"].max().reindex(unit_ids).to_numpy()
    fleet = pd.DataFrame({
        "unit": unit_ids,
        "current_cycle": current_cycle,
        "pred_rul": np.round(preds, 1),
        "true_rul": test_rul.to_numpy()[: len(unit_ids)],
    })

    importance = pd.read_csv(SHAP_PATH) if SHAP_PATH else pd.DataFrame()
    return {
        "model": model, "scaler": scaler, "meta": meta,
        "feature_cols": feature_cols, "window": window,
        "test_df": test_df, "fleet": fleet, "importance": importance,
        "source": source,
    }


def risk_of(rul: float, high: float, med: float) -> str:
    if rul < high:
        return "High"
    if rul < med:
        return "Medium"
    return "Low"


def unit_trajectory(model, unit_df, feature_cols, scaler, window):
    """Predicted RUL at each cycle (expanding last-window) for one engine."""
    feats = scaler.transform(unit_df[feature_cols].to_numpy())
    seqs, end_cycles = [], []
    for end in range(window, len(feats) + 1):
        seqs.append(feats[end - window : end])
        end_cycles.append(int(unit_df["cycle"].iloc[end - 1]))
    if not seqs:
        return [], []
    return end_cycles, predict_rul(model, np.asarray(seqs, dtype="float32"))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def fleet_overview(fleet: pd.DataFrame, meta: dict, high: float, med: float) -> None:
    fleet = fleet.copy()
    fleet["risk"] = [risk_of(r, high, med) for r in fleet["pred_rul"]]
    n_high = int((fleet["risk"] == "High").sum())
    n_med = int((fleet["risk"] == "Medium").sum())

    c = st.columns(5)
    c[0].metric("Engines monitored", len(fleet))
    c[1].metric("\U0001F534 High risk", n_high)
    c[2].metric("\U0001F7E1 Medium risk", n_med)
    c[3].metric("Median predicted RUL", f"{fleet['pred_rul'].median():.0f} cyc")
    c[4].metric("Model test RMSE", f"{meta.get('test_rmse', float('nan'))} cyc")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Predicted Remaining Useful Life by engine")
        order = fleet.sort_values("pred_rul")
        fig = go.Figure(go.Bar(
            x=order["pred_rul"], y=order["unit"].astype(str), orientation="h",
            marker_color=[RISK_COLORS[r] for r in order["risk"]],
            hovertemplate="Engine %{y}: %{x:.0f} cycles<extra></extra>",
        ))
        fig.add_vline(x=high, line_dash="dash", line_color=RISK_COLORS["High"])
        fig.add_vline(x=med, line_dash="dash", line_color=RISK_COLORS["Medium"])
        fig.update_layout(
            template="plotly_dark", height=560, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Predicted RUL (cycles)", yaxis_title="Engine",
            yaxis=dict(showticklabels=len(fleet) <= 40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("\U0001F6A8 Maintenance alerts")
        at_risk = fleet[fleet["risk"] != "Low"].sort_values("pred_rul")
        if at_risk.empty:
            st.success("✅ No engines below the medium-risk threshold.")
        else:
            for row in at_risk.itertuples():
                icon = "\U0001F534" if row.risk == "High" else "\U0001F7E1"
                action = "Schedule immediate inspection" if row.risk == "High" else "Plan maintenance soon"
                st.markdown(
                    f"{icon} **Engine {row.unit}** — predicted RUL "
                    f"**{row.pred_rul:.0f} cycles** · {action}"
                )


def engine_deepdive(bundle: dict, high: float, med: float) -> None:
    fleet, test_df = bundle["fleet"], bundle["test_df"]
    feature_cols, window = bundle["feature_cols"], bundle["window"]
    importance = bundle["importance"]

    unit = st.selectbox("Select engine", fleet["unit"].tolist())
    row = fleet[fleet["unit"] == unit].iloc[0]
    unit_df = test_df[test_df["unit"] == unit].sort_values("cycle")
    risk = risk_of(row["pred_rul"], high, med)

    c = st.columns(4)
    c[0].metric("Current cycle", int(row["current_cycle"]))
    c[1].metric("Predicted RUL", f"{row['pred_rul']:.0f} cyc")
    c[2].metric("Actual RUL", f"{row['true_rul']:.0f} cyc",
                delta=f"{row['pred_rul'] - row['true_rul']:+.0f} vs actual")
    c[3].metric("Risk level", risk)

    if risk == "High":
        st.error(f"\U0001F534 **High failure risk** — Engine {unit} predicted to fail in ~{row['pred_rul']:.0f} cycles. Schedule immediate inspection.")
    elif risk == "Medium":
        st.warning(f"\U0001F7E1 **Medium risk** — Engine {unit} degrading; plan maintenance within ~{row['pred_rul']:.0f} cycles.")
    else:
        st.success(f"\U0001F7E2 **Healthy** — Engine {unit} has ~{row['pred_rul']:.0f} cycles of useful life remaining.")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted RUL trajectory")
        cycles, traj = unit_trajectory(bundle["model"], unit_df, feature_cols, bundle["scaler"], window)
        if len(cycles):
            tfig = go.Figure(go.Scatter(x=cycles, y=traj, mode="lines", line=dict(color="#58a6ff", width=3)))
            tfig.update_layout(
                template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Cycle", yaxis_title="Predicted RUL (cycles)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(tfig, use_container_width=True)
        else:
            st.info("Not enough cycles for a trajectory (need at least one full window).")

    with right:
        st.subheader("Sensor trends")
        top_sensors = (
            importance["feature"].head(4).tolist()
            if not importance.empty else feature_cols[:4]
        )
        chosen = st.multiselect("Sensors", feature_cols, default=top_sensors)
        if chosen:
            melted = unit_df.melt(id_vars="cycle", value_vars=chosen, var_name="sensor", value_name="reading")
            sfig = px.line(melted, x="cycle", y="reading", color="sensor", template="plotly_dark")
            sfig.update_layout(
                height=340, margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.3),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(sfig, use_container_width=True)


def explainability(bundle: dict) -> None:
    meta, importance = bundle["meta"], bundle["importance"]
    c = st.columns(3)
    c[0].metric("Test RMSE", f"{meta.get('test_rmse', float('nan'))} cyc")
    c[1].metric("Test MAE", f"{meta.get('test_mae', float('nan'))} cyc")
    c[2].metric("Features used", len(bundle["feature_cols"]))

    st.subheader("Feature importance")
    st.caption(f"Method: **{meta.get('importance_method', 'n/a')}** · "
               "which sensors most drive the RUL prediction.")
    if importance.empty:
        st.info("No importance file found. Run `python train.py` to generate it.")
        return
    imp = importance.sort_values("importance").tail(14)
    fig = px.bar(imp, x="importance", y="feature", orientation="h", template="plotly_dark",
                 color="importance", color_continuous_scale="Blues")
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False,
        xaxis_title="Mean |contribution| to predicted RUL", yaxis_title="Sensor",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("\U0001F527 Predictive Maintenance — Turbofan Engine RUL")
    st.caption(
        "LSTM-based Remaining Useful Life prediction on the NASA CMAPSS dataset — "
        "an Industry 4.0 / smart-manufacturing condition-monitoring demo."
    )

    if not artifacts_exist():
        st.error(
            "No trained model found in `models/`. Run **`python train.py`** first "
            "to train the LSTM and generate the artifacts."
        )
        st.stop()

    bundle = load_bundle()

    st.sidebar.title("⚙️ Risk thresholds")
    high = st.sidebar.slider("High risk if RUL below (cycles)", 5, 60, 30, 5)
    med = st.sidebar.slider("Medium risk if RUL below (cycles)", high + 5, 120, max(70, high + 5), 5)
    st.sidebar.divider()
    src = bundle["source"]
    st.sidebar.info(
        f"Data source: **{src.upper()}**\n\n"
        + ("Real NASA CMAPSS FD001 files detected in `data/`."
           if src == "real"
           else "Synthetic CMAPSS-format data (drop the real FD001 files into "
                "`data/` and rerun `train.py` to use the real dataset).")
    )

    tab1, tab2, tab3 = st.tabs(
        ["\U0001F6E9️ Fleet Overview", "\U0001F50D Engine Deep-Dive", "\U0001F4CA Model & Explainability"]
    )
    with tab1:
        fleet_overview(bundle["fleet"], bundle["meta"], high, med)
    with tab2:
        engine_deepdive(bundle, high, med)
    with tab3:
        explainability(bundle)


if __name__ == "__main__":
    main()
