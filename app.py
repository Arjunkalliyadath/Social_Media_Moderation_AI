"""
AI-Driven Enforcement Intelligence System
A comprehensive data science application for multi-platform enforcement analytics,
anomaly detection, and predictive forecasting.

Author: Data Science Professional
License: MIT
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import os
from config import get_config

# -------------------------------------------------
# PAGE CONFIG & THEME CONFIGURATION
# -------------------------------------------------
config = get_config()

st.set_page_config(
    page_title="AI Enforcement Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PROFESSIONAL DATA SCIENCE THEME (DARK MODE) ============
THEME_CSS = """
<style>
    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f28 100%);
        color: #e8eef2;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1f28 0%, #252d38 100%);
        border-right: 2px solid rgba(0, 102, 255, 0.2);
    }
    
    /* Typography */
    .main h1 {
        color: #00d4ff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 212, 255, 0.1);
        letter-spacing: -0.5px;
    }
    
    .main h2 {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 102, 255, 0.3);
        padding-bottom: 0.5rem;
    }
    
    .main h3 {
        color: #7fd3ff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Cards and Containers */
    .metric-card {
        background: linear-gradient(135deg, #1a1f28 0%, #252d38 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 25px rgba(0, 102, 255, 0.2);
        transform: translateY(-2px);
    }
    
    /* Plotly Charts */
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 102, 255, 0.1);
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 102, 255, 0.05);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0066ff 0%, #0088dd 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background-color: #1a1f28 !important;
        color: #e8eef2 !important;
        border: 1px solid rgba(0, 102, 255, 0.2) !important;
        border-radius: 6px !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: rgba(0, 200, 100, 0.1) !important;
        border-left: 4px solid #00c864 !important;
    }
    
    .stError {
        background-color: rgba(255, 50, 50, 0.1) !important;
        border-left: 4px solid #ff3232 !important;
    }
    
    .stWarning {
        background-color: rgba(255, 150, 0, 0.1) !important;
        border-left: 4px solid #ff9600 !important;
    }
    
    /* Info Messages */
    .stInfo {
        background-color: rgba(0, 150, 255, 0.1) !important;
        border-left: 4px solid #0096ff !important;
    }
    
    /* Custom Badge */
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
    
    /* Column divider */
    .divider {
        border-right: 2px solid rgba(0, 102, 255, 0.2);
    }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

st.title("📊 AI Enforcement Intelligence System")

# Header with elegant description
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='color: #7f8a9c; font-size: 1.1rem;'>
        <strong>Multi-Platform Monitoring</strong> • <strong>Comparative Analytics</strong> • <strong>Risk Intelligence</strong>
    </p>
    <p style='color: #5a6476; font-size: 0.95rem; margin-top: 0.5rem;'>
        Advanced AI-powered enforcement data analysis with anomaly detection & predictive forecasting
    </p>
    <div style='margin-top: 1rem;'>
        <span class="badge">🤖 ML-Powered</span>
        <span class="badge" style='margin-left: 0.5rem;'>📈 Predictive</span>
        <span class="badge" style='margin-left: 0.5rem;'>🔍 Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# DATA LOADING WITH ERROR HANDLING
# -------------------------------------------------
@st.cache_data
def load_data():
    """
    Load enforcement data from CSV file with caching.
    
    Returns:
        pd.DataFrame: Loaded and preprocessed dataframe
        
    Raises:
        FileNotFoundError: If data file is not found
        pd.errors.ParserError: If CSV parsing fails
    """
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, config.DATA_PATH)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_cols = ["date", "action_as_per_source", "organization", "standard_value"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df["date"] = pd.to_datetime(df["date"])
        return df
        
    except FileNotFoundError as e:
        st.error(f"❌ Data Loading Error: {str(e)}")
        st.stop()
    except pd.errors.ParserError as e:
        st.error(f"❌ CSV Parsing Error: {str(e)}")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Data Validation Error: {str(e)}")
        st.stop()

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
# PDF EXPORT WITH ENHANCED FORMATTING
# -------------------------------------------------
def generate_pdf():
    """
    Generate comprehensive PDF report with analysis results.
    
    Returns:
        BytesIO: PDF document as bytes buffer
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#00d4ff'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#7fd3ff'),
        spaceAfter=10
    )
    
    # Report header
    elements.append(Paragraph("🤖 AI ENFORCEMENT INTELLIGENCE REPORT", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Executive Summary
    elements.append(Paragraph("📊 Executive Summary", heading_style))
    elements.append(Paragraph(f"<b>Trend:</b> {trend}", styles['Normal']))
    elements.append(Paragraph(f"<b>Growth Rate:</b> {growth_pct:.2f}%", styles['Normal']))
    elements.append(Paragraph(f"<b>Latest Value:</b> {latest_value:,.0f}", styles['Normal']))
    elements.append(Paragraph(f"<b>Average Monthly:</b> {average_value:,.0f}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Top performers
    elements.append(Paragraph("🏆 Top Performers", heading_style))
    elements.append(Paragraph(f"<b>Highest Performer:</b> {highest_org} ({highest_value:,.0f})", styles['Normal']))
    elements.append(Paragraph(f"<b>Lowest Performer:</b> {lowest_org} ({lowest_value:,.0f})", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Analysis details
    elements.append(Paragraph("📈 Analysis Details", heading_style))
    elements.append(Paragraph(f"<b>Time Period:</b> {df['date'].min().date()} to {df['date'].max().date()}", styles['Normal']))
    elements.append(Paragraph(f"<b>Organizations Analyzed:</b> {len(selected_orgs)}", styles['Normal']))
    elements.append(Paragraph(f"<b>Total Records:</b> {len(filtered_df):,}", styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.download_button(
    "⬇ Download PDF Report",
    generate_pdf(),
    file_name="Enforcement_Report.pdf",
    mime="application/pdf"
)