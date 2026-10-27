"""M5 Hierarchical Data Loader & Preprocessing Pipeline.

Loads, validates, unpivots, and joins M5 datasets (calendar, prices, sales).
Builds multi-level hierarchical aggregation tables for Darts & LightGBM.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class M5DataLoader:
    """Loads and transforms M5 Walmart dataset files into structured analytical datasets."""

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/sample",
        sales_filename: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sales_filename = sales_filename
        self.calendar: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.DataFrame] = None
        self.sales_wide: Optional[pd.DataFrame] = None
        self.sales_long: Optional[pd.DataFrame] = None

    def _resolve_sales_file(self) -> Path:
        """Find the appropriate sales file in the data directory."""
        if self.sales_filename:
            return self.data_dir / self.sales_filename

        candidates = [
            "sales_train_evaluation.csv",
            "sales_train_validation.csv",
            "sales_train_sample.csv",
        ]
        for c in candidates:
            p = self.data_dir / c
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No sales file found in {self.data_dir}. Expected one of: {candidates}"
        )

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load calendar, prices, and wide sales files."""
        calendar_path = self.data_dir / "calendar.csv"
        prices_path = self.data_dir / "sell_prices.csv"
        sales_path = self._resolve_sales_file()

        logger.info(f"Loading calendar from {calendar_path}")
        self.calendar = pd.read_csv(calendar_path)
        self.calendar["date"] = pd.to_datetime(self.calendar["date"])

        logger.info(f"Loading prices from {prices_path}")
        self.prices = pd.read_csv(prices_path)

        logger.info(f"Loading sales from {sales_path}")
        self.sales_wide = pd.read_csv(sales_path)

        return self.calendar, self.prices, self.sales_wide

    def melt_sales(
        self,
        max_days: Optional[int] = None,
        item_sample: Optional[List[str]] = None,
        store_sample: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert wide sales DataFrame (d_1...d_N) to long format (date, series, sales)."""
        if self.sales_wide is None:
            self.load_raw_data()

        df = self.sales_wide
        if item_sample is not None:
            df = df[df["item_id"].isin(item_sample)]
        if store_sample is not None:
            df = df[df["store_id"].isin(store_sample)]

        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        d_cols = [c for c in df.columns if c.startswith("d_")]

        if max_days is not None:
            # Keep only the last `max_days` days for fast processing
            d_cols = d_cols[-max_days:]

        melted = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=d_cols,
            var_name="d",
            value_name="sales",
        )
        melted["sales"] = melted["sales"].astype(np.float32)
        return melted

    def build_merged_dataset(
        self,
        max_days: Optional[int] = None,
        item_sample: Optional[List[str]] = None,
        store_sample: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create fully joined tabular dataset with calendar and prices."""
        if self.calendar is None or self.prices is None or self.sales_wide is None:
            self.load_raw_data()

        melted = self.melt_sales(max_days=max_days, item_sample=item_sample, store_sample=store_sample)

        # Merge calendar
        merged = melted.merge(
            self.calendar,
            on="d",
            how="left",
        )

        # Merge sell prices on store_id, item_id, wm_yr_wk
        merged = merged.merge(
            self.prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )

        # Fill missing sell prices with item median
        if merged["sell_price"].isnull().any():
            median_prices = merged.groupby("item_id")["sell_price"].transform("median")
            merged["sell_price"] = merged["sell_price"].fillna(median_prices).fillna(1.99)

        # Ensure correct date sorting
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values(by=["id", "date"]).reset_index(drop=True)

        self.sales_long = merged
        return merged

    def get_hierarchical_aggregations(
        self, df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """Aggregate sales across all M5 hierarchy levels.

        Levels:
          - 'total': Nationwide aggregate
          - 'state': CA, TX, WI
          - 'store': 10 stores
          - 'cat': FOODS, HOUSEHOLD, HOBBIES
          - 'state_cat': State x Category
          - 'store_cat': Store x Category
          - 'item_store': Bottom level (store x item)
        """
        data = df if df is not None else self.sales_long
        if data is None:
            data = self.build_merged_dataset()

        aggregations: Dict[str, pd.DataFrame] = {}

        # 1. Total Nationwide
        aggregations["total"] = (
            data.groupby("date")["sales"].sum().reset_index().rename(columns={"sales": "demand"})
        )
        aggregations["total"]["series_id"] = "Total"

        # 2. State Level
        state_agg = data.groupby(["state_id", "date"])["sales"].sum().reset_index()
        state_agg = state_agg.rename(columns={"sales": "demand", "state_id": "series_id"})
        aggregations["state"] = state_agg

        # 3. Store Level
        store_agg = data.groupby(["store_id", "date"])["sales"].sum().reset_index()
        store_agg = store_agg.rename(columns={"sales": "demand", "store_id": "series_id"})
        aggregations["store"] = store_agg

        # 4. Category Level
        cat_agg = data.groupby(["cat_id", "date"])["sales"].sum().reset_index()
        cat_agg = cat_agg.rename(columns={"sales": "demand", "cat_id": "series_id"})
        aggregations["cat"] = cat_agg

        # 5. Store x Category Level
        store_cat_agg = data.groupby(["store_id", "cat_id", "date"])["sales"].sum().reset_index()
        store_cat_agg["series_id"] = store_cat_agg["store_id"] + "_" + store_cat_agg["cat_id"]
        store_cat_agg = store_cat_agg.rename(columns={"sales": "demand"})
        aggregations["store_cat"] = store_cat_agg

        # 6. Bottom level (Store x Item)
        item_store = data[["id", "store_id", "item_id", "cat_id", "date", "sales"]].copy()
        item_store = item_store.rename(columns={"sales": "demand", "id": "series_id"})
        aggregations["item_store"] = item_store

        return aggregations

    def split_train_val(
        self,
        df: pd.DataFrame,
        horizon_days: int = 28,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into train and validation sets by date."""
        max_date = df["date"].max()
        cutoff_date = max_date - pd.Timedelta(days=horizon_days)

        train_df = df[df["date"] <= cutoff_date].copy()
        val_df = df[df["date"] > cutoff_date].copy()

        logger.info(
            f"Train split: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} rows)"
        )
        logger.info(
            f"Validation split: {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df)} rows)"
        )

        return train_df, val_df
