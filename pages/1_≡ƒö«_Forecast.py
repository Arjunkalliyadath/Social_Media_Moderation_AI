"""
Predictive Intelligence Module
Per-organization 6-month (or custom horizon) forecasts with confidence
bands, growth projections, and a reliability score.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

from config import get_config
from data_utils import DataValidationError, get_volume_df, load_data
from theme import inject_theme, render_footer, render_header

config = get_config()

st.set_page_config(page_title="Forecast — TrustLens", page_icon="🔮", layout="wide", initial_sidebar_state="expanded")
inject_theme()
render_header(
    "🔮 Predictive Enforcement Intelligence",
    "<strong>Multi-Month Forecasting</strong> • <strong>Trend Projection</strong> • <strong>Confidence Modeling</strong>",
    ["📈 Linear Regression", "🎯 Trend Analysis", "🧮 Reliability Scoring"],
)

try:
    df, report, source_label = load_data(st.session_state.get("uploaded_file"))
except (DataValidationError, FileNotFoundError) as e:
    st.error(f"❌ {e}")
    st.stop()

volume_df = get_volume_df(df)
if volume_df.empty:
    st.error("No volume data available to forecast.")
    st.stop()

top_orgs = (
    volume_df.groupby("organization")["normalized_value"].sum().sort_values(ascending=False).head(4).index.tolist()
)
all_orgs = sorted(volume_df["organization"].unique())

col_a, col_b = st.columns([3, 1])
with col_a:
    selected_orgs = st.multiselect("Select organizations to forecast", all_orgs, default=top_orgs)
with col_b:
    horizon = st.slider("Horizon (months)", 3, 12, config.FORECAST_MONTHS)

if not selected_orgs:
    st.warning("Select at least one organization above.")
    st.stop()

fig = go.Figure()
growth_summary = []
uncertainty_summary = []

for org in selected_orgs:
    org_df = (
        volume_df[volume_df["organization"] == org]
        .groupby("date")["normalized_value"].sum()
        .reset_index().sort_values("date")
    )
    org_df = org_df.dropna()
    org_df = org_df[org_df["normalized_value"] > 0]

    if len(org_df) < config.MIN_POINTS_FOR_FORECAST:
        st.warning(f"Not enough data points to forecast **{org}** (need at least {config.MIN_POINTS_FOR_FORECAST}, have {len(org_df)}).")
        continue

    org_df["time_index"] = np.arange(len(org_df))
    X, y = org_df[["time_index"]].to_numpy(), org_df["normalized_value"].to_numpy()

    model = LinearRegression().fit(X, y)
    future_index = np.arange(len(org_df), len(org_df) + horizon)
    future_dates = pd.date_range(org_df["date"].iloc[-1], periods=horizon + 1, freq="MS")[1:]
    future_predictions = model.predict(future_index.reshape(-1, 1))
    # Forecasts are trend projections and can't go below zero for a count metric.
    future_predictions = np.clip(future_predictions, 0, None)

    full_dates = pd.concat([org_df["date"], pd.Series(future_dates)])
    full_values = np.concatenate([y, future_predictions])

    fig.add_trace(go.Scatter(x=full_dates, y=full_values, mode="lines", name=f"{org} Forecast"))

    residuals = y - model.predict(X)
    std_dev = np.std(residuals)
    upper = future_predictions + (1.96 * std_dev)
    lower = np.clip(future_predictions - (1.96 * std_dev), 0, None)

    fig.add_trace(go.Scatter(x=future_dates, y=upper, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates, y=lower, fill="tonexty", mode="lines",
                              line=dict(width=0), opacity=0.2, name=f"{org} Confidence"))

    last_actual = y[-1]
    future_avg = np.mean(future_predictions)
    growth = ((future_avg - last_actual) / last_actual * 100) if last_actual else 0

    # Reliability score: 100 minus the coefficient of variation of the forecast,
    # guarded against a zero/negative denominator (flat or near-zero forecasts).
    if future_avg > 0:
        reliability_score = max(100 - (std_dev / future_avg * 100), 0)
    else:
        reliability_score = 0.0

    growth_summary.append({"Organization": org, "Predicted Growth (%)": round(growth, 2),
                            "Predicted Avg (Next Period)": round(future_avg, 0)})
    uncertainty_summary.append({"Organization": org, "Forecast Reliability Score": round(reliability_score, 2)})

fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="#E5E7EB"), legend=dict(font=dict(color="#E5E7EB")))
st.plotly_chart(fig, width="stretch")

growth_df = pd.DataFrame(growth_summary)
uncertainty_df = pd.DataFrame(uncertainty_summary)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Forecast Growth Comparison")
    st.dataframe(growth_df, width="stretch", hide_index=True)
with col2:
    st.subheader("📉 Forecast Reliability Scores")
    st.dataframe(uncertainty_df, width="stretch", hide_index=True)

if not growth_df.empty and not uncertainty_df.empty:
    leader = growth_df.sort_values(by="Predicted Growth (%)", ascending=False).iloc[0]["Organization"]
    most_reliable = uncertainty_df.sort_values(by="Forecast Reliability Score", ascending=False).iloc[0]["Organization"]
    st.success(
        f"**{leader}** is projected to show the strongest growth. "
        f"**{most_reliable}** demonstrates the highest forecast reliability."
    )
    st.caption(
        "Reliability score = 100 − coefficient of variation of the forecast residuals. "
        "It reflects how noisy the historical data was, not certainty about the future — "
        "treat these projections as directional, not guaranteed."
    )
else:
    st.info("No organization in the current selection has enough history to forecast.")

csv_bytes = growth_df.to_csv(index=False).encode("utf-8") if not growth_df.empty else b""
if csv_bytes:
    st.download_button("⬇️ Download Forecast Summary (CSV)", data=csv_bytes, file_name="trustlens_forecast_summary.csv", mime="text/csv")

st.markdown("---")
if st.button("⬅ Back to Dashboard"):
    st.switch_page("app.py")

render_footer()
