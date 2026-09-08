"""
Category Insights
Breaks enforcement volume down by violation category (the `topic` column),
which the original dashboard collected but never displayed.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import DataValidationError, get_volume_df, load_data
from theme import inject_theme, render_footer, render_header

st.set_page_config(page_title="Category Insights — TrustLens", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
inject_theme()
render_header(
    "🧭 Category Insights",
    "<strong>Violation Categories</strong> • <strong>Organization Breakdown</strong> • <strong>Topic Trends</strong>",
    ["🗂️ 40+ Categories", "🔥 Heatmap", "📈 Topic Trend"],
)

try:
    df, report, source_label = load_data(st.session_state.get("uploaded_file"))
except (DataValidationError, FileNotFoundError) as e:
    st.error(f"❌ {e}")
    st.stop()

volume_df = get_volume_df(df)
if volume_df.empty or volume_df["topic"].nunique() <= 1:
    st.info(
        "This dataset doesn't include a usable `topic` column, so category "
        "breakdowns aren't available. Everything else still works from the Dashboard page."
    )
    st.stop()

# Exclude the catch-all "All" topic (a rollup row in the source data) from
# category-level comparisons so it doesn't dwarf every real category.
category_df = volume_df[volume_df["topic"].str.lower() != "all"].copy()

all_orgs = sorted(category_df["organization"].unique())
selected_orgs = st.multiselect("Organizations", all_orgs, default=all_orgs)
category_df = category_df[category_df["organization"].isin(selected_orgs)]

if category_df.empty:
    st.warning("Select at least one organization.")
    st.stop()

st.subheader("🔝 Most-Enforced Violation Categories")
topic_totals = (
    category_df.groupby("topic")["normalized_value"].sum().sort_values(ascending=False).head(15)
)
fig = px.bar(
    topic_totals.reset_index().rename(columns={"normalized_value": "Total Enforcement"}),
    x="Total Enforcement", y="topic", orientation="h",
    labels={"topic": "Violation category"},
)
fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                   font=dict(color="#E5E7EB"), yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")

st.subheader("🔥 Category × Organization Heatmap")
top_n_topics = topic_totals.head(12).index.tolist()
pivot = (
    category_df[category_df["topic"].isin(top_n_topics)]
    .pivot_table(index="topic", columns="organization", values="normalized_value", aggfunc="sum", fill_value=0)
)
heat_fig = px.imshow(
    pivot, aspect="auto", color_continuous_scale="Blues",
    labels=dict(color="Enforcement volume"),
)
heat_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"))
st.plotly_chart(heat_fig, width="stretch")

st.subheader("📈 Trend for a Specific Category")
chosen_topic = st.selectbox("Choose a violation category", sorted(category_df["topic"].unique()))
topic_trend = (
    category_df[category_df["topic"] == chosen_topic]
    .groupby(["date", "organization"])["normalized_value"].sum().reset_index()
)
if topic_trend.empty:
    st.info("No data for this category in the current selection.")
else:
    trend_fig = px.line(topic_trend, x="date", y="normalized_value", color="organization", markers=True,
                         labels={"normalized_value": f"{chosen_topic} — enforcement volume"})
    trend_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#E5E7EB"), legend=dict(font=dict(color="#E5E7EB")))
    st.plotly_chart(trend_fig, width="stretch")

with st.expander("📋 Full category breakdown table"):
    full_table = category_df.groupby(["topic", "organization"])["normalized_value"].sum().reset_index()
    full_table = full_table.rename(columns={"normalized_value": "Total Enforcement"}).sort_values(
        "Total Enforcement", ascending=False
    )
    st.dataframe(full_table, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download Category Breakdown (CSV)",
        data=full_table.to_csv(index=False).encode("utf-8"),
        file_name="trustlens_category_breakdown.csv",
        mime="text/csv",
    )

st.markdown("---")
if st.button("⬅ Back to Dashboard"):
    st.switch_page("app.py")

render_footer()
