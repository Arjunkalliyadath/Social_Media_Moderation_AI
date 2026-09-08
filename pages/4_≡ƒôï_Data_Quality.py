"""
Data Quality & Methodology
Documents exactly how the raw CSV is cleaned before any chart or model sees
it, and shows why that matters with a concrete before/after comparison.
"""

import pandas as pd
import streamlit as st

from config import get_config
from data_utils import DataValidationError, get_volume_df, load_data
from theme import inject_theme, render_footer, render_header

config = get_config()

st.set_page_config(page_title="Data Quality — TrustLens", page_icon="📋", layout="wide", initial_sidebar_state="expanded")
inject_theme()
render_header(
    "📋 Data Quality & Methodology",
    "<strong>Unit Normalization</strong> • <strong>Cleaning Report</strong> • <strong>Full Transparency</strong>",
    ["🔬 Reproducible", "🧹 Auditable", "📥 Exportable"],
)

try:
    df, report, source_label = load_data(st.session_state.get("uploaded_file"))
except (DataValidationError, FileNotFoundError) as e:
    st.error(f"❌ {e}")
    st.stop()

st.markdown(f"**Current data source:** `{source_label}`")

st.subheader("🧹 Cleaning Report")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total rows read", f"{report.total_rows:,}")
c2.metric("Rows with unparseable dates (dropped)", f"{report.rows_missing_date:,}")
c3.metric("Mislabeled unit strings fixed", f"{report.typo_units_fixed:,}")
c4.metric("Rate rows excluded from volume totals", f"{report.rows_excluded_from_volume:,}")

c5, c6, c7 = st.columns(3)
c5.metric("Usable volume rows", f"{report.volume_rows:,}")
c6.metric("Proactive-rate rows", f"{report.rate_rows:,}")
c7.metric("Organizations covered", f"{len(report.organizations)}")

if report.date_min is not None:
    st.caption(f"Date range: {report.date_min.date()} → {report.date_max.date()}")

st.markdown("---")
st.subheader("⚠️ Why unit normalization matters")
st.markdown(
    """
The source data reports `standard_value` in **mixed units within the same column**:
absolute counts, thousands, millions, and percentages — occasionally with a typo
(`"value in percenatge"`). A dashboard that sums that column directly will silently
produce nonsense: a platform reporting in *millions* looks a thousand times smaller
than one reporting in *absolute numbers*, and stray percentage rows get added on top
of raw counts.

TrustLens converts every volume figure onto one absolute scale before anything is
summed, ranked, charted, or forecast, and keeps percentage-type rows (like
`Proactive Rate`) completely separate — see the **Proactive Detection** page.

The table below shows the same organization totals computed the naive way
(summing `standard_value` as-is) versus the corrected way (summing the
unit-normalized value), using whatever data source is currently loaded.
"""
)

volume_mask = df["action_as_per_source"].isin(config.VOLUME_ACTIONS)
raw_totals = df[volume_mask].groupby("organization")["standard_value"].sum().sort_values(ascending=False)
normalized_totals = get_volume_df(df).groupby("organization")["normalized_value"].sum().sort_values(ascending=False)

compare = pd.DataFrame({
    "Naive sum (raw standard_value)": raw_totals,
    "Corrected sum (unit-normalized)": normalized_totals,
}).fillna(0).sort_values("Corrected sum (unit-normalized)", ascending=False)

st.dataframe(compare.style.format("{:,.0f}"), width="stretch")
st.caption(
    "Notice how the ranking itself can change once units are corrected — a platform "
    "reporting in millions can go from looking smallest to largest."
)

st.markdown("---")
st.subheader("🔍 Missing Values by Column")
na_counts = df.isna().sum()
na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
if na_counts.empty:
    st.success("No missing values in any column.")
else:
    st.dataframe(na_counts.rename("Missing values").reset_index().rename(columns={"index": "Column"}),
                 width="stretch", hide_index=True)

st.markdown("---")
st.subheader("📥 Export the Cleaned Dataset")
st.caption("Includes the `normalized_value` and `is_rate_metric` columns TrustLens computes during cleaning.")
st.download_button(
    "⬇️ Download Cleaned Dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="trustlens_cleaned_dataset.csv",
    mime="text/csv",
)

st.markdown("---")
if st.button("⬅ Back to Dashboard"):
    st.switch_page("app.py")

render_footer()
