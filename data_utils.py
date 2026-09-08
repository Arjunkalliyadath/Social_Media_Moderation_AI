"""
Data loading, validation, and cleaning for TrustLens.

This module is the single place where the raw enforcement CSV becomes an
analysis-ready dataframe. It is imported by every page so the cleaning
logic (and any future fix to it) only has to live in one place.

Why this module exists (the short version):
The source data reports `standard_value` in mixed units -- "absolute
number", "thousands", "millions", and "percentage" -- inside the SAME
column, sometimes with a typo ("value in percenatge"). The original app
summed that column directly across organizations, which silently produced
meaningless totals (a platform reporting in "millions" looked 1000x
smaller than one reporting in "absolute numbers", and a few mislabeled
percentage rows got added on top of raw counts). Every function below
exists to make that impossible to get wrong again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from config import get_config

config = get_config()


class DataValidationError(Exception):
    """Raised when an uploaded/bundled CSV is missing required columns."""


@dataclass
class CleaningReport:
    """A transparent record of every adjustment made to the raw data."""

    total_rows: int = 0
    typo_units_fixed: int = 0
    rows_missing_date: int = 0
    rows_excluded_from_volume: int = 0
    volume_rows: int = 0
    rate_rows: int = 0
    organizations: list = field(default_factory=list)
    date_min: Optional[pd.Timestamp] = None
    date_max: Optional[pd.Timestamp] = None


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"A valid file needs at least: {', '.join(config.REQUIRED_COLUMNS)}."
        )


def _clean_units(df: pd.DataFrame) -> pd.Series:
    """Lower-case, strip, and fix known typos in the `units` column."""
    if "units" not in df.columns:
        # No units column at all -> assume every value is already an
        # absolute count (best-effort default for user-uploaded data).
        return pd.Series(["value in absolute number"] * len(df), index=df.index)

    cleaned = df["units"].astype(str).str.strip().str.lower()
    cleaned = cleaned.replace(config.UNIT_TYPO_FIXES)
    return cleaned


def compute_normalized_value(df: pd.DataFrame, units_clean: pd.Series) -> pd.Series:
    """
    Convert `standard_value` onto one consistent absolute scale.

    Rows whose unit is a rate/percentage (not additive with counts) become
    NaN here on purpose -- they must never enter a SUM alongside counts.
    They still exist in the dataframe for rate-specific analysis (see
    get_rate_df).
    """
    multiplier = units_clean.map(config.UNIT_MULTIPLIERS)
    return df["standard_value"] * multiplier


@st.cache_data(show_spinner=False)
def _read_and_clean(file_bytes: bytes, source_label: str) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Core cleaning pipeline, cached on the raw file bytes so the cache
    correctly invalidates whenever the underlying file actually changes
    (including when a user uploads a new CSV).
    """
    import io

    df = pd.read_csv(io.BytesIO(file_bytes))
    validate_columns(df)

    report = CleaningReport(total_rows=len(df))

    # --- dates -------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    report.rows_missing_date = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"]).copy()

    # --- units + normalized volume ------------------------------------------
    raw_units = df["units"].astype(str).str.strip().str.lower() if "units" in df.columns else pd.Series([""] * len(df))
    units_clean = _clean_units(df)
    report.typo_units_fixed = int((raw_units != units_clean).sum()) if "units" in df.columns else 0

    df["units_clean"] = units_clean
    df["normalized_value"] = compute_normalized_value(df, units_clean)
    df["is_rate_metric"] = units_clean.str.contains("percent", na=False)

    # --- tidy optional columns ------------------------------------------------
    for col in ["organization", "action_as_per_source", "topic", "enforcement_type", "proactive_flag"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "topic" not in df.columns:
        df["topic"] = "Unspecified"

    df["organization"] = df["organization"].astype(str).str.strip()
    df["action_as_per_source"] = df["action_as_per_source"].astype(str).str.strip()

    report.rows_excluded_from_volume = int(
        df["action_as_per_source"].isin(config.VOLUME_ACTIONS).sum()
        - df[df["action_as_per_source"].isin(config.VOLUME_ACTIONS)]["normalized_value"].notna().sum()
    )
    report.volume_rows = int(
        (df["action_as_per_source"].isin(config.VOLUME_ACTIONS) & df["normalized_value"].notna()).sum()
    )
    report.rate_rows = int((df["action_as_per_source"] == config.PROACTIVE_RATE_ACTION).sum())
    report.organizations = sorted(df["organization"].dropna().unique().tolist())
    if len(df):
        report.date_min = df["date"].min()
        report.date_max = df["date"].max()

    return df, report


def load_data(uploaded_file=None) -> tuple[pd.DataFrame, CleaningReport, str]:
    """
    Load either the bundled sample dataset or a user-uploaded CSV.

    Returns (dataframe, cleaning_report, source_label).
    """
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        source_label = uploaded_file.name
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, config.DATA_PATH)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bundled data file not found: {file_path}")
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        source_label = config.DATA_PATH

    df, report = _read_and_clean(file_bytes, source_label)
    return df, report, source_label


def get_volume_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that represent a countable enforcement volume (safe to SUM)."""
    mask = df["action_as_per_source"].isin(config.VOLUME_ACTIONS) & df["normalized_value"].notna()
    return df[mask].copy()


def get_rate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that represent the proactive-detection RATE (a percentage, never summed)."""
    mask = df["action_as_per_source"] == config.PROACTIVE_RATE_ACTION
    return df[mask].copy()


def safe_pct_change(new: float, old: float) -> Optional[float]:
    """Percent change that returns None instead of raising/inf on a zero baseline."""
    if old in (0, None) or pd.isna(old):
        return None
    return (new - old) / old * 100
