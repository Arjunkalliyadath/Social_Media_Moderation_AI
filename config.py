"""
Configuration module for TrustLens.

Centralizes environment settings, data-cleaning rules, and ML parameters so
every page in the app reads from a single source of truth instead of
duplicating constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


class Config:
    """Base configuration shared by every environment."""

    # ---- Application metadata -------------------------------------------
    APP_NAME = "TrustLens"
    APP_TAGLINE = "Social Media Enforcement Intelligence"
    APP_VERSION = "2.0.0"

    # ---- Data settings -----------------------------------------------------
    DATA_PATH = os.getenv("DATA_PATH", "preprocessed_enforcement_data.csv")

    # Columns that MUST exist in any dataset (uploaded or bundled) for the
    # app to function at all.
    REQUIRED_COLUMNS = ["date", "organization", "action_as_per_source", "standard_value"]

    # Optional columns that unlock extra features when present.
    OPTIONAL_COLUMNS = ["units", "topic", "proactive_flag", "enforcement_type"]

    # Action types that represent countable enforcement *volume* (as opposed
    # to rates/percentages). Only these are summed for totals, trends,
    # rankings, anomaly detection and forecasting.
    VOLUME_ACTIONS = [
        "Content Actioned",
        "Content Removed",
        "Removed",
        "Total Accounts Banned",
        "Total Accounts Suspended",
    ]

    # Action type that represents the proactive-detection rate metric.
    PROACTIVE_RATE_ACTION = "Proactive Rate"

    # Multipliers used to bring every volume figure onto the same absolute
    # scale before it is aggregated. Anything not listed here (e.g. a
    # percentage unit) is treated as non-additive and excluded from volume
    # totals -- see data_utils.compute_normalized_value().
    UNIT_MULTIPLIERS = {
        "value in absolute number": 1,
        "value in thousands": 1_000,
        "value in millions": 1_000_000,
    }

    # Known data-entry typos in the source `units` column, mapped to their
    # corrected spelling before any matching happens.
    UNIT_TYPO_FIXES = {
        "value in percenatge": "value in percentage",
    }

    # ---- API Keys (never hardcode secrets) ---------------------------------
    API_KEY = os.getenv("API_KEY", None)

    # ---- ML model settings --------------------------------------------------
    ISOLATION_FOREST_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", 0.08))
    ISOLATION_FOREST_RANDOM_STATE = 42
    FORECAST_MONTHS = int(os.getenv("FORECAST_MONTHS", 6))
    MIN_POINTS_FOR_ANOMALY = 7
    MIN_POINTS_FOR_FORECAST = 6

    # ---- Theme --------------------------------------------------------------
    STREAMLIT_THEME = "dark"
    PRIMARY_COLOR = "#00d4ff"
    ACCENT_COLOR = "#0066ff"
    BG_DARK = "#0f1419"
    BG_PANEL = "#1a1f28"
    TEXT_COLOR = "#e8eef2"
    MUTED_TEXT = "#7f8a9c"

    @classmethod
    def validate(cls):
        """Validate that the bundled sample dataset is present on disk."""
        base_path = Path(__file__).parent
        data_file = base_path / cls.DATA_PATH
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        return True


class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"


def get_config():
    """Return the configuration matching the ENVIRONMENT variable."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    if env == "production":
        return ProductionConfig()
    return DevelopmentConfig()
