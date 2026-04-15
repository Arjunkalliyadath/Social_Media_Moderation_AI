import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Enforcement Intelligence System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Driven Enforcement Intelligence System")
st.markdown("Multi-Platform Monitoring • Comparative Analytics • Risk Intelligence")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "preprocessed_enforcement_data.csv")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# -------------------------------------------------
# FILTER VALID ACTIONS
# -------------------------------------------------
valid_actions = [
    "Content Actioned",
    "Content Removed",
    "Removed",
    "Total Accounts Banned",
    "Total Accounts Suspended"
]

df = df[df["action_as_per_source"].isin(valid_actions)]

# -------------------------------------------------
# SIDEBAR FILTER
# -------------------------------------------------
st.sidebar.header("📅 Time Range")

min_date = df["date"].min()
max_date = df["date"].max()

start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

filtered_df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

organizations = sorted(filtered_df["organization"].unique())

selected_orgs = st.sidebar.multiselect(
    "Select Organizations",
    organizations,
    default=organizations
)

if not selected_orgs:
    st.warning("Select at least one organization.")
    st.stop()

filtered_df = filtered_df[
    filtered_df["organization"].isin(selected_orgs)
]

# -------------------------------------------------
# AGGREGATION
# -------------------------------------------------
monthly = (
    filtered_df.groupby(["date", "organization"])["standard_value"]
    .sum()
    .reset_index()
)

overall = (
    filtered_df.groupby("date")["standard_value"]
    .sum()
    .reset_index()
)

if overall.empty:
    st.error("No data available.")
    st.stop()

# -------------------------------------------------
# ANOMALY DETECTION
# -------------------------------------------------
if len(overall) > 6:
    iso_model = IsolationForest(contamination=0.08, random_state=42)
    overall["anomaly"] = iso_model.fit_predict(overall[["standard_value"]])
else:
    overall["anomaly"] = 1

overall["anomaly_flag"] = overall["anomaly"].map({1: "Normal", -1: "Anomaly"})

# -------------------------------------------------
# TREND CALCULATION (Regression Based)
# -------------------------------------------------
overall_sorted = overall.sort_values("date").reset_index(drop=True)
overall_sorted["time_index"] = np.arange(len(overall_sorted))

X_trend = overall_sorted[["time_index"]]
y_trend = overall_sorted["standard_value"]

trend_model = LinearRegression()
trend_model.fit(X_trend, y_trend)

slope = trend_model.coef_[0]

if slope > 0:
    trend = "Increasing 📈"
elif slope < 0:
    trend = "Decreasing 📉"
else:
    trend = "Stable ➖"

growth_pct = (
    (overall_sorted.iloc[-1]["standard_value"] -
     overall_sorted.iloc[0]["standard_value"])
    / overall_sorted.iloc[0]["standard_value"]
) * 100

latest_value = overall_sorted.iloc[-1]["standard_value"]
average_value = overall_sorted["standard_value"].mean()

# -------------------------------------------------
# ORGANIZATION RANKING
# -------------------------------------------------
org_totals = (
    filtered_df.groupby("organization")["standard_value"]
    .sum()
    .sort_values(ascending=False)
)

highest_org = org_totals.idxmax()
highest_value = org_totals.max()

lowest_org = org_totals.idxmin()
lowest_value = org_totals.min()

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.subheader("📌 Key Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Month", f"{latest_value:,.0f}")
c2.metric("Average Monthly", f"{average_value:,.0f}")
c3.metric("Trend", trend)
c4.metric("Growth %", f"{growth_pct:.2f}%")

# -------------------------------------------------
# TREND CHART
# -------------------------------------------------
st.subheader("📈 Monthly Enforcement Trend")

fig = px.line(
    monthly,
    x="date",
    y="standard_value",
    color="organization",
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# RANKING TABLE
# -------------------------------------------------
st.subheader("🏆 Organization Ranking")

st.dataframe(
    org_totals.reset_index().rename(
        columns={"standard_value": "Total Enforcement"}
    ),
    use_container_width=True
)

# -------------------------------------------------
# ANOMALIES
# -------------------------------------------------
st.subheader("🚨 Detected Anomalies")

st.dataframe(
    overall[overall["anomaly_flag"] == "Anomaly"],
    use_container_width=True
)

# -------------------------------------------------
# FORECAST
# -------------------------------------------------
st.subheader("🔮 6-Month Forecast")

future_months = 6
future_index = np.arange(len(overall_sorted), len(overall_sorted) + future_months)

future_dates = pd.date_range(
    overall_sorted["date"].iloc[-1],
    periods=future_months + 1,
    freq="MS"
)[1:]

future_predictions = trend_model.predict(future_index.reshape(-1, 1))

forecast_df = pd.DataFrame({
    "date": future_dates,
    "forecast": future_predictions
})

forecast_fig = px.line(
    forecast_df,
    x="date",
    y="forecast",
    title="Predicted Enforcement Trend (Next 6 Months)"
)

st.plotly_chart(forecast_fig, use_container_width=True)

# -------------------------------------------------
# OFFLINE AI-LIKE ANALYTICS ENGINE
# -------------------------------------------------
st.subheader("🤖 AI Analytics Assistant (Offline Mode)")

def detect_intent(query):
    query = query.lower()

    if any(w in query for w in ["trend", "increase", "decrease", "growth", "rising", "falling"]):
        return "trend"

    if any(w in query for w in ["highest", "top", "maximum", "best", "dominant"]):
        return "highest"

    if any(w in query for w in ["lowest", "minimum", "worst", "bottom"]):
        return "lowest"

    if any(w in query for w in ["average", "mean", "baseline"]):
        return "average"

    if any(w in query for w in ["compare", "difference", "versus", "vs"]):
        return "comparison"

    if any(w in query for w in ["forecast", "future", "predict", "projection"]):
        return "forecast"

    if any(w in query for w in ["volatile", "volatility", "unstable", "risk", "fluctuation"]):
        return "volatility"

    return "general"


def generate_ai_response(question):
    intent = detect_intent(question)

    # Volatility
    if len(filtered_df) > 1:
        org_std = (
            filtered_df.groupby("organization")["standard_value"]
            .std()
            .dropna()
        )
        most_volatile_org = org_std.idxmax() if not org_std.empty else "N/A"
    else:
        most_volatile_org = "N/A"

    if intent == "trend":
        direction = "increasing" if growth_pct > 0 else "decreasing"
        return (
            f"📈 The overall enforcement trend is {direction}. "
            f"There is a {abs(growth_pct):.2f}% change over the selected period."
        )

    elif intent == "highest":
        return f"🏆 {highest_org} leads with {highest_value:,.0f} total enforcement actions."

    elif intent == "lowest":
        return f"📉 {lowest_org} records the lowest total at {lowest_value:,.0f}."

    elif intent == "average":
        return f"📊 The average monthly enforcement volume is {average_value:,.0f}."

    elif intent == "comparison":
        comparison_text = "\n".join(
            [f"{org}: {value:,.0f}" for org, value in org_totals.items()]
        )
        return "📊 Platform Comparison:\n\n" + comparison_text

    elif intent == "forecast":
        future_direction = "increase" if slope > 0 else "decrease"
        return f"🔮 Forecast indicates enforcement is likely to {future_direction} over the next 6 months."

    elif intent == "volatility":
        return f"⚠ The most volatile organization is {most_volatile_org}, showing higher enforcement fluctuations."

    else:
        return (
            "🧠 You can ask about trend, growth, highest/lowest performers, "
            "platform comparison, volatility risk, or forecast projections."
        )


question = st.text_input("Ask any question about the dataset...")

if question:
    response = generate_ai_response(question)
    st.success(response)

# -------------------------------------------------
# PDF EXPORT
# -------------------------------------------------
def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("AI Enforcement Intelligence Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"Trend: {trend}", styles["Normal"]))
    elements.append(Paragraph(f"Growth Percentage: {growth_pct:.2f}%", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer

st.download_button(
    "⬇ Download PDF Report",
    generate_pdf(),
    file_name="Enforcement_Report.pdf",
    mime="application/pdf"
)