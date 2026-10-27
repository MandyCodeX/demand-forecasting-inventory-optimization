"""Calendar, Holiday, and Promotion Feature Engineering.

Extracts temporal patterns, cyclical trigonometric encodings, event indicators,
and state-specific Supplemental Nutrition Assistance Program (SNAP) benefits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_and_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich DataFrame with calendar, seasonal, event, and SNAP features."""
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Temporal components
    df["day_of_month"] = df["date"].dt.day.astype(np.int16)
    df["day_of_week"] = df["date"].dt.weekday.astype(np.int8)  # 0=Monday, 6=Sunday
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["year"] = df["date"].dt.year.astype(np.int16)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(np.int8)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    # Cyclical trigonometric encodings (smoothes boundary transitions e.g. Dec to Jan)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0).astype(np.float32)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype(np.float32)

    # Events & Holidays indicator
    if "event_name_1" in df.columns:
        df["is_event_1"] = df["event_name_1"].notna().astype(np.int8)
    else:
        df["is_event_1"] = 0

    if "event_name_2" in df.columns:
        df["is_event_2"] = df["event_name_2"].notna().astype(np.int8)
    else:
        df["is_event_2"] = 0

    df["is_event"] = ((df["is_event_1"] == 1) | (df["is_event_2"] == 1)).astype(np.int8)

    # Event types one-hot flags
    for event_type in ["Sporting", "Cultural", "National", "Religious"]:
        col_name = f"event_type_{event_type}"
        cond1 = (
            df["event_type_1"] == event_type
            if "event_type_1" in df.columns
            else pd.Series(False, index=df.index)
        )
        cond2 = (
            df["event_type_2"] == event_type
            if "event_type_2" in df.columns
            else pd.Series(False, index=df.index)
        )
        df[col_name] = (cond1 | cond2).astype(np.int8)

    # State-aware SNAP food stamp active indicator
    # When state_id is available, map the relevant state's SNAP flag
    if "state_id" in df.columns:
        snap_conditions = [
            df["state_id"] == "CA",
            df["state_id"] == "TX",
            df["state_id"] == "WI",
        ]
        snap_choices = [
            df["snap_CA"] if "snap_CA" in df.columns else 0,
            df["snap_TX"] if "snap_TX" in df.columns else 0,
            df["snap_WI"] if "snap_WI" in df.columns else 0,
        ]
        df["snap_active"] = np.select(snap_conditions, snap_choices, default=0).astype(np.int8)
    else:
        df["snap_active"] = 0

    return df
