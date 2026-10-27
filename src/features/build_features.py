"""Feature Engineering Pipeline for M5 Demand Forecasting.

Generates lags, rolling statistics, price dynamics, promotions,
and categorical encodings with strict protection against data leakage.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .calendar_features import add_calendar_and_event_features

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """End-to-end feature creation and preprocessing pipeline."""

    CATEGORICAL_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        base_lag: int = 1,
    ) -> None:
        self.lags = lags or [7, 14, 21, 28]
        self.rolling_windows = rolling_windows or [7, 14, 28]
        self.base_lag = base_lag
        self.encoders: Dict[str, LabelEncoder] = {}
        self.price_stats: Optional[pd.DataFrame] = None
        self.fitted: bool = False
        self.feature_columns: List[str] = []

    def fit_price_stats(self, df: pd.DataFrame) -> None:
        """Compute reference price statistics per store and item to detect discounts."""
        if "sell_price" in df.columns:
            stats = (
                df.groupby(["store_id", "item_id"])["sell_price"]
                .agg(
                    price_max="max",
                    price_min="min",
                    price_mean="mean",
                    price_std="std",
                )
                .reset_index()
            )
            stats["price_std"] = stats["price_std"].fillna(0.0)
            self.price_stats = stats

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge historical price stats and compute markdown & discount indicators."""
        df = df.copy()
        if self.price_stats is not None and "sell_price" in df.columns:
            df = df.merge(self.price_stats, on=["store_id", "item_id"], how="left")
            df["price_max"] = df["price_max"].fillna(df["sell_price"])
            df["price_mean"] = df["price_mean"].fillna(df["sell_price"])

            # Normalized price (1.0 = regular full price, <1.0 = discount)
            df["price_norm"] = np.where(
                df["price_max"] > 0,
                df["sell_price"] / df["price_max"],
                1.0,
            ).astype(np.float32)

            # Markdown discount fraction (e.g. 0.20 = 20% off)
            df["price_discount"] = (1.0 - df["price_norm"]).clip(lower=0.0).astype(np.float32)
            df["is_discounted"] = (df["price_discount"] > 0.05).astype(np.int8)
        else:
            df["price_norm"] = 1.0
            df["price_discount"] = 0.0
            df["is_discounted"] = 0
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute historical demand lags grouped by individual series."""
        df = df.copy()
        df = df.sort_values(by=["id", "date"]).reset_index(drop=True)

        for lag in self.lags:
            col_name = f"lag_{lag}"
            df[col_name] = df.groupby("id")["sales"].shift(lag).astype(np.float32)

        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean, std, min, and max over lagged series to prevent leakage.

        We compute rolling metrics on shift(base_lag) so the current step's value
        is strictly excluded from the window.
        """
        df = df.copy()
        df = df.sort_values(by=["id", "date"]).reset_index(drop=True)

        for w in self.rolling_windows:
            grouped = df.groupby("id")["sales"]
            shifted = grouped.shift(self.base_lag)

            # Rolling Mean
            df[f"roll_mean_{w}"] = (
                shifted.groupby(df["id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
            ).astype(np.float32)

            # Rolling Std
            df[f"roll_std_{w}"] = (
                shifted.groupby(df["id"]).transform(lambda s: s.rolling(w, min_periods=2).std())
            ).fillna(0.0).astype(np.float32)

            # Rolling Min & Max for short window (7 days)
            if w == 7:
                df["roll_min_7"] = (
                    shifted.groupby(df["id"]).transform(lambda s: s.rolling(7, min_periods=1).min())
                ).astype(np.float32)
                df["roll_max_7"] = (
                    shifted.groupby(df["id"]).transform(lambda s: s.rolling(7, min_periods=1).max())
                ).astype(np.float32)

        return df

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit encoders and price baselines on training set."""
        logger.info("Fitting FeaturePipeline on dataset...")
        self.fit_price_stats(df)

        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.encoders[col] = le

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Apply all feature engineering transformations to input DataFrame."""
        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fit before calling transform().")

        df = df.copy()

        # 1. Calendar & Event Features
        df = add_calendar_and_event_features(df)

        # 2. Price and Discount Features
        df = self.add_price_features(df)

        # 3. Lags
        df = self.add_lag_features(df)

        # 4. Rolling Statistics
        df = self.add_rolling_features(df)

        # 5. Encode Categoricals
        for col, le in self.encoders.items():
            if col in df.columns:
                # Handle unseen categories gracefully
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                df[f"{col}_encoded"] = df[col].astype(str).map(mapping).fillna(-1).astype(np.int32)

        if drop_na:
            max_lag = max(self.lags + self.rolling_windows)
            # Retain rows with valid engineered features
            req_cols = [f"lag_{l}" for l in self.lags] + [f"roll_mean_{w}" for w in self.rolling_windows]
            df = df.dropna(subset=req_cols).reset_index(drop=True)

        # Record engineered numeric & categorical feature names
        exclude = {
            "id",
            "d",
            "date",
            "sales",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "weekday",
            "wm_yr_wk",
        }
        for c in self.CATEGORICAL_COLS:
            exclude.add(c)

        self.feature_columns = [c for c in df.columns if c not in exclude]
        return df

    def fit_transform(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """Fit on DataFrame and return transformed features."""
        return self.fit(df).transform(df, drop_na=drop_na)
