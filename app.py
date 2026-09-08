"""
TrustLens — Social Media Enforcement Intelligence
Main dashboard: KPIs, trends, rankings, anomaly detection, a quick forecast,
and an offline AI analytics assistant.

Author: rebuilt & hardened by Claude for the original project author.
License: MIT
"""

import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from config import get_config
from data_utils import (
    DataValidationError,
    get_rate_df,
    get_volume_df,
    load_data,
    safe_pct_change,
)
from theme import inject_theme, render_footer, render_header

config = get_config()

st.set_page_config(
    page_title=f"{config.APP_NAME} — {config.APP_TAGLINE}",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()
render_header(
    f"🛡️ {config.APP_NAME}",
    "<strong>Multi-Platform Monitoring</strong> • <strong>Comparative Analytics</strong> • "
    "<strong>Risk Intelligence</strong><br>"
    "<span style='font-size:0.9rem;color:#5a6476;'>Unit-normalized enforcement analytics with "
    "anomaly detection, forecasting, and an offline AI assistant.</span>",
    ["🤖 ML-Powered", "📈 Predictive", "🔍 Unit-Normalized", "🔒 100% Offline"],
)

# -------------------------------------------------------------------------
# DATA SOURCE
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Data Source")
    source_choice = st.radio(
        "Use", ["Sample dataset", "Upload my own CSV"], label_visibility="collapsed"
    )
    uploaded_file = None
    if source_choice == "Upload my own CSV":
        uploaded_file = st.file_uploader(
            "CSV with at least: date, organization, action_as_per_source, standard_value",
            type=["csv"],
        )
        st.caption(
            "Optional columns `units`, `topic`, and `proactive_flag` unlock the "
            "unit-normalization, Category Insights, and Proactive Detection pages."
        )
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
    else:
        # Switching back to the sample dataset clears any previously uploaded
        # file so every page (this one and the others) stays in sync.
        st.session_state["uploaded_file"] = None

try:
    if source_choice == "Upload my own CSV" and uploaded_file is None:
        st.info("⬆️ Upload a CSV in the sidebar to begin, or switch back to the sample dataset.")
        st.stop()
    df, report, source_label = load_data(st.session_state.get("uploaded_file"))
except DataValidationError as e:
    st.error(f"❌ {e}")
    st.stop()
except (FileNotFoundError, pd.errors.ParserError) as e:
    st.error(f"❌ Could not load data: {e}")
    st.stop()

if df.empty:
    st.error("❌ The dataset has no usable rows after cleaning (check date formatting).")
    st.stop()

volume_actions_present = sorted(
    set(df["action_as_per_source"].unique()) & set(config.VOLUME_ACTIONS)
)
if not volume_actions_present:
    st.error(
        "❌ None of the recognized volume action types "
        f"({', '.join(config.VOLUME_ACTIONS)}) were found in `action_as_per_source`."
    )
    st.stop()

# -------------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------------
st.sidebar.header("📅 Time Range")
min_date, max_date = df["date"].min(), df["date"].max()
start_date = st.sidebar.date_input("Start date", min_date.date(), min_value=min_date.date(), max_value=max_date.date())
end_date = st.sidebar.date_input("End date", max_date.date(), min_value=min_date.date(), max_value=max_date.date())

if start_date > end_date:
    st.sidebar.error("Start date must be on or before end date.")
    st.stop()

st.sidebar.header("🏢 Organizations")
all_orgs = sorted(df["organization"].unique())
selected_orgs = st.sidebar.multiselect("Select organizations", all_orgs, default=all_orgs)

st.sidebar.header("⚙️ Action Types")
selected_actions = st.sidebar.multiselect(
    "Volume actions to include", volume_actions_present, default=volume_actions_present
)

with st.sidebar.expander("🧪 Advanced settings"):
    contamination = st.slider(
        "Anomaly sensitivity (Isolation Forest contamination)",
        0.02, 0.25, config.ISOLATION_FOREST_CONTAMINATION, 0.01,
        help="Higher = more points flagged as anomalies.",
    )
    forecast_months = st.slider("Quick forecast horizon (months)", 3, 12, config.FORECAST_MONTHS)
    smooth_trend = st.checkbox("Smooth trend with 3-month rolling average", value=False)

if not selected_orgs or not selected_actions:
    st.warning("Select at least one organization and one action type from the sidebar.")
    st.stop()

mask = (
    (df["date"] >= pd.to_datetime(start_date))
    & (df["date"] <= pd.to_datetime(end_date))
    & (df["organization"].isin(selected_orgs))
)
filtered_df = df[mask]
volume_df = get_volume_df(filtered_df)
volume_df = volume_df[volume_df["action_as_per_source"].isin(selected_actions)]
rate_df = get_rate_df(filtered_df)

if volume_df.empty:
    st.error("No volume data available for the current filters. Try widening the date range or selections.")
    st.stop()

# -------------------------------------------------------------------------
# AGGREGATION (all on normalized_value, never raw standard_value)
# -------------------------------------------------------------------------
monthly = (
    volume_df.groupby(["date", "organization"])["normalized_value"].sum().reset_index()
)
overall = volume_df.groupby("date")["normalized_value"].sum().reset_index()
overall = overall.sort_values("date").reset_index(drop=True)

if smooth_trend and len(overall) >= 3:
    overall["smoothed"] = overall["normalized_value"].rolling(3, min_periods=1).mean()

# -------------------------------------------------------------------------
# ANOMALY DETECTION
# -------------------------------------------------------------------------
if len(overall) >= config.MIN_POINTS_FOR_ANOMALY:
    iso_model = IsolationForest(contamination=contamination, random_state=config.ISOLATION_FOREST_RANDOM_STATE)
    overall["anomaly"] = iso_model.fit_predict(overall[["normalized_value"]])
else:
    overall["anomaly"] = 1
overall["anomaly_flag"] = overall["anomaly"].map({1: "Normal", -1: "Anomaly"})

# -------------------------------------------------------------------------
# TREND (regression-based)
# -------------------------------------------------------------------------
overall["time_index"] = np.arange(len(overall))
trend_model = LinearRegression().fit(
    overall[["time_index"]].to_numpy(), overall["normalized_value"].to_numpy()
)
slope = trend_model.coef_[0]
trend = "Increasing 📈" if slope > 0 else ("Decreasing 📉" if slope < 0 else "Stable ➖")

growth_pct = safe_pct_change(overall.iloc[-1]["normalized_value"], overall.iloc[0]["normalized_value"])
latest_value = overall.iloc[-1]["normalized_value"]
average_value = overall["normalized_value"].mean()

# -------------------------------------------------------------------------
# ORGANIZATION RANKING
# -------------------------------------------------------------------------
org_totals = volume_df.groupby("organization")["normalized_value"].sum().sort_values(ascending=False)
highest_org, highest_value = org_totals.idxmax(), org_totals.max()
lowest_org, lowest_value = org_totals.idxmin(), org_totals.min()

# -------------------------------------------------------------------------
# KPI SECTION
# -------------------------------------------------------------------------
st.subheader("📌 Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Period", f"{latest_value:,.0f}")
c2.metric("Average per Period", f"{average_value:,.0f}")
c3.metric("Trend", trend)
c4.metric("Growth %", f"{growth_pct:.2f}%" if growth_pct is not None else "N/A")

if report.rows_excluded_from_volume or report.typo_units_fixed:
    st.caption(
        f"ℹ️ Figures above are unit-normalized (thousands/millions converted to absolute counts). "
        f"{report.rows_excluded_from_volume} rate/percentage rows were kept out of these totals. "
        "See the **Data Quality** page for the full methodology."
    )

# -------------------------------------------------------------------------
# TREND CHART
# -------------------------------------------------------------------------
st.subheader("📈 Enforcement Trend Over Time")
fig = px.line(monthly, x="date", y="normalized_value", color="organization", markers=True,
              labels={"normalized_value": "Enforcement volume (normalized)", "date": "Date"})
if smooth_trend and "smoothed" in overall.columns:
    fig.add_trace(go.Scatter(x=overall["date"], y=overall["smoothed"], mode="lines",
                              name="Overall (3-mo avg)", line=dict(color="#00d4ff", width=3, dash="dot")))
fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="#E5E7EB"), legend=dict(font=dict(color="#E5E7EB")))
st.plotly_chart(fig, width="stretch")

# -------------------------------------------------------------------------
# RANKING
# -------------------------------------------------------------------------
st.subheader("🏆 Organization Ranking")
rc1, rc2 = st.columns([2, 1])
with rc1:
    rank_fig = px.bar(
        org_totals.reset_index().rename(columns={"normalized_value": "Total Enforcement"}),
        x="organization", y="Total Enforcement", color="organization",
        labels={"organization": "Organization"},
    )
    rank_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#E5E7EB"), showlegend=False)
    st.plotly_chart(rank_fig, width="stretch")
with rc2:
    st.dataframe(
        org_totals.reset_index().rename(columns={"normalized_value": "Total Enforcement"}),
        width="stretch", hide_index=True,
    )

# -------------------------------------------------------------------------
# ANOMALIES
# -------------------------------------------------------------------------
st.subheader("🚨 Detected Anomalies")
anomalies = overall[overall["anomaly_flag"] == "Anomaly"]
anomaly_fig = go.Figure()
anomaly_fig.add_trace(go.Scatter(x=overall["date"], y=overall["normalized_value"], mode="lines+markers",
                                  name="Overall volume", line=dict(color="#0088dd")))
if not anomalies.empty:
    anomaly_fig.add_trace(go.Scatter(x=anomalies["date"], y=anomalies["normalized_value"], mode="markers",
                                      name="Anomaly", marker=dict(color="#ff3232", size=12, symbol="x")))
anomaly_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#E5E7EB"), legend=dict(font=dict(color="#E5E7EB")))
st.plotly_chart(anomaly_fig, width="stretch")

if anomalies.empty:
    st.info("No anomalies detected in the current selection.")
else:
    st.dataframe(
        anomalies[["date", "normalized_value", "anomaly_flag"]].rename(columns={"normalized_value": "Volume"}),
        width="stretch", hide_index=True,
    )

# -------------------------------------------------------------------------
# QUICK FORECAST
# -------------------------------------------------------------------------
st.subheader(f"🔮 Quick {forecast_months}-Month Forecast")
st.caption("A deeper, per-organization forecast with confidence bands lives on the **Forecast** page.")

future_index = np.arange(len(overall), len(overall) + forecast_months)
future_dates = pd.date_range(overall["date"].iloc[-1], periods=forecast_months + 1, freq="MS")[1:]
future_predictions = trend_model.predict(future_index.reshape(-1, 1))
forecast_df = pd.DataFrame({"date": future_dates, "forecast": future_predictions})

forecast_fig = px.line(forecast_df, x="date", y="forecast",
                        title=f"Predicted Enforcement Volume (Next {forecast_months} Months)")
forecast_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"))
st.plotly_chart(forecast_fig, width="stretch")

# -------------------------------------------------------------------------
# OFFLINE AI ANALYTICS ASSISTANT (chat UI, offline intent matching)
# -------------------------------------------------------------------------
st.subheader("🤖 AI Analytics Assistant")
st.caption("Fully offline keyword-based Q&A — no API calls, no data leaves this app.")


def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["trend", "increase", "decrease", "growth", "rising", "falling"]):
        return "trend"
    if any(w in q for w in ["highest", "top", "maximum", "best", "dominant", "leader"]):
        return "highest"
    if any(w in q for w in ["lowest", "minimum", "worst", "bottom"]):
        return "lowest"
    if any(w in q for w in ["average", "mean", "baseline"]):
        return "average"
    if any(w in q for w in ["compare", "difference", "versus", " vs"]):
        return "comparison"
    if any(w in q for w in ["forecast", "future", "predict", "projection"]):
        return "forecast"
    if any(w in q for w in ["volatile", "volatility", "unstable", "risk", "fluctuation"]):
        return "volatility"
    if any(w in q for w in ["topic", "category", "violation", "content type"]):
        return "topic"
    if any(w in q for w in ["proactive", "detection rate", "caught before"]):
        return "proactive"
    if any(w in q for w in ["anomaly", "anomalies", "unusual", "spike"]):
        return "anomaly"
    return "general"


def generate_ai_response(question: str) -> str:
    intent = detect_intent(question)

    if len(volume_df) > 1:
        org_std = volume_df.groupby("organization")["normalized_value"].std().dropna()
        most_volatile_org = org_std.idxmax() if not org_std.empty else "N/A"
    else:
        most_volatile_org = "N/A"

    if intent == "trend":
        direction = "increasing" if slope > 0 else "decreasing"
        change_txt = f"{abs(growth_pct):.2f}%" if growth_pct is not None else "an undetermined amount (starting value was zero)"
        return f"📈 The overall enforcement trend is **{direction}**, a change of {change_txt} over the selected period."
    if intent == "highest":
        return f"🏆 **{highest_org}** leads with {highest_value:,.0f} total enforcement actions (normalized)."
    if intent == "lowest":
        return f"📉 **{lowest_org}** records the lowest total at {lowest_value:,.0f}."
    if intent == "average":
        return f"📊 The average per-period enforcement volume is **{average_value:,.0f}**."
    if intent == "comparison":
        lines = "\n".join([f"- **{org}**: {value:,.0f}" for org, value in org_totals.items()])
        return "📊 Platform comparison (normalized totals):\n\n" + lines
    if intent == "forecast":
        direction = "increase" if slope > 0 else "decrease"
        return f"🔮 The regression model projects enforcement volume will **{direction}** over the next {forecast_months} months."
    if intent == "volatility":
        return f"⚠️ **{most_volatile_org}** is the most volatile organization in the current selection (highest standard deviation)."
    if intent == "topic":
        if "topic" in volume_df.columns and volume_df["topic"].nunique() > 1:
            top_topic = volume_df.groupby("topic")["normalized_value"].sum().idxmax()
            return f"🗂️ The most enforced violation category is **{top_topic}**. See the Category Insights page for the full breakdown."
        return "🗂️ This dataset doesn't include a `topic` column, so category breakdowns aren't available."
    if intent == "proactive":
        if not rate_df.empty:
            avg_rate = rate_df["standard_value"].mean()
            top_org = rate_df.groupby("organization")["standard_value"].mean().idxmax()
            return (f"🛡️ Average proactive detection rate in this selection is **{avg_rate:.1f}%**, "
                    f"led by **{top_org}**. Full breakdown on the Proactive Detection page.")
        return "🛡️ No 'Proactive Rate' rows are present in this dataset."
    if intent == "anomaly":
        if anomalies.empty:
            return "✅ No anomalies were flagged in the current selection."
        dates_txt = ", ".join(d.strftime("%b %Y") for d in anomalies["date"])
        return f"🚨 {len(anomalies)} anomaly period(s) detected: {dates_txt}."
    return ("🧠 Ask me about **trend**, **growth**, **highest/lowest** performers, **platform comparison**, "
            "**volatility**, **anomalies**, **topics/categories**, **proactive detection**, or **forecast** projections.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

example_questions = [
    "📈 Show me the trend",
    "🏆 Who's the top platform?",
    "🛡️ Most proactive platform?",
    "⚠️ Who's most volatile?",
    "🔮 What's the forecast?",
]
picked = st.pills("Quick questions", example_questions, key="quick_pills")

for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

typed_question = st.chat_input("Ask about trend, rankings, volatility, topics, proactive rates, forecasts…")

new_question = typed_question
if picked and picked != st.session_state.get("_last_pill"):
    new_question = picked
    st.session_state["_last_pill"] = picked

if new_question:
    st.session_state.chat_history.append(("user", new_question))
    response = generate_ai_response(new_question)
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()

# -------------------------------------------------------------------------
# EXPORTS
# -------------------------------------------------------------------------
st.subheader("📤 Export")


def generate_pdf() -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CustomTitle", parent=styles["Heading1"], fontSize=22,
                                  textColor=colors.HexColor("#00d4ff"), spaceAfter=12, alignment=1)
    heading_style = ParagraphStyle("CustomHeading", parent=styles["Heading2"], fontSize=14,
                                    textColor=colors.HexColor("#0066ff"), spaceAfter=10)

    elements = [
        Paragraph(f"🛡️ {config.APP_NAME} — Enforcement Intelligence Report", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph("📊 Executive Summary", heading_style),
        Paragraph(f"<b>Trend:</b> {trend}", styles["Normal"]),
        Paragraph(f"<b>Growth Rate:</b> {growth_pct:.2f}%" if growth_pct is not None else "<b>Growth Rate:</b> N/A", styles["Normal"]),
        Paragraph(f"<b>Latest Value:</b> {latest_value:,.0f}", styles["Normal"]),
        Paragraph(f"<b>Average per Period:</b> {average_value:,.0f}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph("🏆 Top Performers", heading_style),
        Paragraph(f"<b>Highest:</b> {highest_org} ({highest_value:,.0f})", styles["Normal"]),
        Paragraph(f"<b>Lowest:</b> {lowest_org} ({lowest_value:,.0f})", styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph("📈 Analysis Details", heading_style),
        Paragraph(f"<b>Time Period:</b> {start_date} to {end_date}", styles["Normal"]),
        Paragraph(f"<b>Organizations Analyzed:</b> {len(selected_orgs)}", styles["Normal"]),
        Paragraph(f"<b>Records Analyzed (volume rows):</b> {len(volume_df):,}", styles["Normal"]),
        Paragraph(f"<b>Data Source:</b> {source_label}", styles["Normal"]),
    ]
    doc.build(elements)
    buffer.seek(0)
    return buffer


ec1, ec2 = st.columns(2)
with ec1:
    if st.button("📄 Generate PDF Report"):
        st.session_state["pdf_buffer"] = generate_pdf()
    if st.session_state.get("pdf_buffer"):
        st.download_button("⬇️ Download PDF Report", data=st.session_state["pdf_buffer"],
                            file_name="TrustLens_Enforcement_Report.pdf", mime="application/pdf")
with ec2:
    csv_bytes = volume_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data (CSV)", data=csv_bytes,
                        file_name="trustlens_filtered_data.csv", mime="text/csv")

render_footer()
