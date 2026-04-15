"""
Predictive Intelligence Module
Advanced forecasting and trend projection for enforcement data.

This page provides 6-month forecasts with confidence intervals and
comparative analysis across platforms.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from config import get_config

# -------------------------------------------------
# PAGE CONFIG & THEME
# -------------------------------------------------
config = get_config()

st.set_page_config(
    page_title="Predictive Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Theme CSS
THEME_CSS = """
<style>
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f28 100%);
        color: #e8eef2;
    }
    
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1f28 0%, #252d38 100%);
    }
    
    .main h1, .main h2 {
        color: #00d4ff;
    }
    
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 102, 255, 0.1);
    }
    
    .badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

st.title("🔮 Predictive Enforcement Intelligence")

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='color: #7f8a9c; font-size: 1.1rem;'>
        <strong>6-Month Forecasting</strong> • <strong>Trend Projection</strong> • <strong>Confidence Modeling</strong>
    </p>
    <p style='color: #5a6476; font-size: 0.95rem; margin-top: 0.5rem;'>
        Advanced predictive analytics with machine learning confidence intervals
    </p>
    <div style='margin-top: 1rem;'>
        <span class="badge">📈 Linear Regression</span>
        <span class="badge" style='margin-left: 0.5rem;'>🎯 Trend Analysis</span>
        <span class="badge" style='margin-left: 0.5rem;'>⚡ Real-time Updates</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_path, "preprocessed_enforcement_data.csv")

df = pd.read_csv(file_path)
df["date"] = pd.to_datetime(df["date"])

valid_actions = [
    "Content Actioned",
    "Content Removed",
    "Removed",
    "Total Accounts Banned",
    "Total Accounts Suspended"
]

df = df[df["action_as_per_source"].isin(valid_actions)]

# -------------------------------------------------
# SELECT TOP 4 ORGANIZATIONS
# -------------------------------------------------
top_orgs = (
    df.groupby("organization")["standard_value"]
    .sum()
    .sort_values(ascending=False)
    .head(4)
    .index.tolist()
)

selected_orgs = st.multiselect(
    "Select Organizations",
    top_orgs,
    default=top_orgs
)

fig = go.Figure()
growth_summary = []
uncertainty_summary = []

# -------------------------------------------------
# FORECAST LOOP (Regression Based)
# -------------------------------------------------
for org in selected_orgs:

    org_df = (
        df[df["organization"] == org]
        .groupby("date")["standard_value"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    if len(org_df) < 6:
        st.warning(f"Not enough data to forecast {org}.")
        continue

    org_df = org_df.dropna()
    org_df = org_df[org_df["standard_value"] > 0]

    if len(org_df) < 6:
        continue

    # Create time index
    org_df["time_index"] = np.arange(len(org_df))

    X = org_df[["time_index"]]
    y = org_df["standard_value"]

    model = LinearRegression()
    model.fit(X, y)

    # Predict future
    future_months = 6
    future_index = np.arange(len(org_df), len(org_df) + future_months)

    future_dates = pd.date_range(
        org_df["date"].iloc[-1],
        periods=future_months + 1,
        freq="MS"
    )[1:]

    future_predictions = model.predict(future_index.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": future_predictions
    })

    # Combine past + future
    full_dates = pd.concat([org_df["date"], forecast_df["date"]])
    full_values = np.concatenate([y, future_predictions])

    # Add forecast line
    fig.add_trace(go.Scatter(
        x=full_dates,
        y=full_values,
        mode="lines",
        name=f"{org} Forecast"
    ))

    # Confidence Band (std-based)
    residuals = y - model.predict(X)
    std_dev = np.std(residuals)

    upper = future_predictions + (1.96 * std_dev)
    lower = future_predictions - (1.96 * std_dev)

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower,
        fill="tonexty",
        mode="lines",
        line=dict(width=0),
        opacity=0.2,
        name=f"{org} Confidence"
    ))

    # Growth %
    last_actual = y.iloc[-1]
    future_avg = np.mean(future_predictions)

    growth = (
        ((future_avg - last_actual) / last_actual * 100)
        if last_actual != 0 else 0
    )

    reliability_score = max(100 - (std_dev / future_avg * 100), 0)

    growth_summary.append({
        "Organization": org,
        "Predicted Growth (%)": round(growth, 2),
        "Predicted Avg (Next 6M)": round(future_avg, 0)
    })

    uncertainty_summary.append({
        "Organization": org,
        "Forecast Reliability Score": round(reliability_score, 2)
    })

# -------------------------------------------------
# STYLE
# -------------------------------------------------
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E5E7EB"),
    legend=dict(font=dict(color="#E5E7EB"))
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# SUMMARY TABLES
# -------------------------------------------------
growth_df = pd.DataFrame(growth_summary)
uncertainty_df = pd.DataFrame(uncertainty_summary)

st.subheader("📊 Forecast Growth Comparison")
st.dataframe(growth_df, use_container_width=True)

st.subheader("📉 Forecast Reliability Scores")
st.dataframe(uncertainty_df, use_container_width=True)

if not growth_df.empty and not uncertainty_df.empty:

    leader = growth_df.sort_values(
        by="Predicted Growth (%)",
        ascending=False
    ).iloc[0]["Organization"]

    most_reliable = uncertainty_df.sort_values(
        by="Forecast Reliability Score",
        ascending=False
    ).iloc[0]["Organization"]

    st.success(
        f"{leader} is projected to show the strongest growth. "
        f"{most_reliable} demonstrates the highest forecast reliability."
    )

st.markdown("---")

if st.button("⬅ Back to Dashboard"):
    st.switch_page("app.py")