"""
Shared visual theme for every page of TrustLens.

Keeping the CSS and header markup in one place means the five pages of the
app can never visually drift apart, and a palette change only needs to
happen once.
"""

import streamlit as st

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
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 212, 255, 0.1);
        letter-spacing: -0.5px;
    }

    .main h2 {
        color: #00d4ff;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 102, 255, 0.3);
        padding-bottom: 0.5rem;
    }

    .main h3 {
        color: #7fd3ff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Cards */
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
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #0066ff 0%, #0088dd 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover, .stDownloadButton>button:hover {
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

    /* Alerts */
    .stSuccess { background-color: rgba(0, 200, 100, 0.1) !important; border-left: 4px solid #00c864 !important; }
    .stError   { background-color: rgba(255, 50, 50, 0.1) !important; border-left: 4px solid #ff3232 !important; }
    .stWarning { background-color: rgba(255, 150, 0, 0.1) !important; border-left: 4px solid #ff9600 !important; }
    .stInfo    { background-color: rgba(0, 150, 255, 0.1) !important; border-left: 4px solid #0096ff !important; }

    /* Badge */
    .badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin: 0 0.25rem;
    }

    .footer-note {
        text-align: center;
        color: #5a6476;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0, 102, 255, 0.15);
    }
</style>
"""


def inject_theme():
    """Apply the shared dark theme CSS to the current page."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badges: list[str]):
    """Render the standard TrustLens page header with title + badge row."""
    st.title(title)
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='color: #7f8a9c; font-size: 1.05rem;'>{subtitle}</p>
            <div style='margin-top: 1rem;'>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="footer-note">
            TrustLens · Offline analytics, nothing leaves your machine · Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )
