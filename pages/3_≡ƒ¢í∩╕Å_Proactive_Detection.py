"""
Proactive Detection
Surfaces the "Proactive Rate" rows, which make up roughly 30% of the source
dataset but were never read anywhere in the original app.

Proactive detection rate = the share of enforced content a platform caught
with its own systems *before* anyone reported it. It's a genuinely
different metric from enforcement volume (a percentage, not a count), so
it gets its own page rather than being mixed into the volume totals.
"""

import plotly.express as px
import streamlit as st

from data_utils import DataValidationError, get_rate_df, load_data
from theme import inject_theme, render_footer, render_header

st.set_page_config(page_title="Proactive Detection — TrustLens", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
inject_theme()
render_header(
    "🛡️ Proactive Detection Intelligence",
    "<strong>Caught-Before-Reported Rate</strong> • <strong>Platform Comparison</strong> • <strong>Category Breakdown</strong>",
    ["📊 Percentage Metric", "🏆 Platform Ranking", "🗂️ By Category"],
)

try:
    df, report, source_label = load_data(st.session_state.get("uploaded_file"))
except (DataValidationError, FileNotFoundError) as e:
    st.error(f"❌ {e}")
    st.stop()

rate_df = get_rate_df(df)
if rate_df.empty:
    st.info(
        "This dataset has no `Proactive Rate` rows in `action_as_per_source`, "
        "so proactive-detection analysis isn't available for it."
    )
    st.stop()

st.caption(
    "These figures come from the `Proactive Rate` rows, kept separate from every count-based "
    "chart elsewhere in the app because a percentage can't be summed with a volume."
)

all_orgs = sorted(rate_df["organization"].unique())
selected_orgs = st.multiselect("Organizations", all_orgs, default=all_orgs)
rate_df = rate_df[rate_df["organization"].isin(selected_orgs)]

if rate_df.empty:
    st.warning("Select at least one organization.")
    st.stop()

avg_rate = rate_df["standard_value"].mean()
org_avg = rate_df.groupby("organization")["standard_value"].mean().sort_values(ascending=False)
top_org, bottom_org = org_avg.idxmax(), org_avg.idxmin()

c1, c2, c3 = st.columns(3)
c1.metric("Average Proactive Rate", f"{avg_rate:.1f}%")
c2.metric("Most Proactive Platform", top_org, f"{org_avg.max():.1f}%")
c3.metric("Least Proactive Platform", bottom_org, f"{org_avg.min():.1f}%")

st.subheader("🏆 Average Proactive Rate by Platform")
fig = px.bar(org_avg.reset_index().rename(columns={"standard_value": "Avg Proactive Rate (%)"}),
             x="organization", y="Avg Proactive Rate (%)", color="organization")
fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E5E7EB"), showlegend=False)
st.plotly_chart(fig, width="stretch")

st.subheader("📈 Proactive Rate Over Time")
trend = rate_df.groupby(["date", "organization"])["standard_value"].mean().reset_index()
trend_fig = px.line(trend, x="date", y="standard_value", color="organization", markers=True,
                     labels={"standard_value": "Proactive rate (%)"})
trend_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         font=dict(color="#E5E7EB"), legend=dict(font=dict(color="#E5E7EB")))
st.plotly_chart(trend_fig, width="stretch")

if "topic" in rate_df.columns and rate_df["topic"].nunique() > 1:
    st.subheader("🗂️ Proactive Rate by Violation Category")
    topic_rate = (
        rate_df[rate_df["topic"].str.lower() != "all"]
        .groupby("topic")["standard_value"].mean().sort_values(ascending=False).head(15)
    )
    topic_fig = px.bar(
        topic_rate.reset_index().rename(columns={"standard_value": "Avg Proactive Rate (%)"}),
        x="Avg Proactive Rate (%)", y="topic", orientation="h",
    )
    topic_fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#E5E7EB"), yaxis=dict(autorange="reversed"))
    st.plotly_chart(topic_fig, width="stretch")

with st.expander("📋 Full proactive-rate data"):
    table = rate_df[["date", "organization", "topic", "standard_value"]].rename(
        columns={"standard_value": "Proactive Rate (%)"}
    ).sort_values("date")
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("⬇️ Download Proactive Rate Data (CSV)", data=table.to_csv(index=False).encode("utf-8"),
                        file_name="trustlens_proactive_rate.csv", mime="text/csv")

st.markdown("---")
if st.button("⬅ Back to Dashboard"):
    st.switch_page("app.py")

render_footer()
