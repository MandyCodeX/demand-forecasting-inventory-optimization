"""M5 Sample Data Generator.

Generates a representative, schema-faithful hierarchical sample of the Walmart M5 dataset.
Preserves hierarchy: Nationwide -> State -> Store -> Category -> Department -> Item.
Includes calendar events, SNAP promotions, and dynamic weekly prices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class M5SampleGenerator:
    """Generates synthetic yet realistic Walmart M5 hierarchical sales data."""

    STATES: List[str] = ["CA", "TX", "WI"]
    STORES_BY_STATE: Dict[str, List[str]] = {
        "CA": ["CA_1", "CA_2", "CA_3", "CA_4"],
        "TX": ["TX_1", "TX_2", "TX_3"],
        "WI": ["WI_1", "WI_2", "WI_3"],
    }
    CATEGORIES: List[str] = ["FOODS", "HOUSEHOLD", "HOBBIES"]
    DEPTS_BY_CAT: Dict[str, List[str]] = {
        "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
        "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
        "HOBBIES": ["HOBBIES_1", "HOBBIES_2"],
    }

    def __init__(
        self,
        num_items_per_dept: int = 2,
        num_days: int = 365,
        start_date: str = "2015-01-01",
        random_seed: int = 42,
    ) -> None:
        self.num_items_per_dept = num_items_per_dept
        self.num_days = num_days
        self.start_date = pd.to_datetime(start_date)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def generate_calendar(self) -> pd.DataFrame:
        """Generate calendar DataFrame conforming to M5 calendar.csv schema."""
        dates = pd.date_range(start=self.start_date, periods=self.num_days, freq="D")
        df = pd.DataFrame({"date": dates})
        df["d"] = [f"d_{i + 1}" for i in range(len(df))]
        
        # Walmart year week (format: e.g. 11501 for Year 2015 Week 01)
        year_short = (df["date"].dt.year % 100) + 100
        week_num = df["date"].dt.isocalendar().week
        df["wm_yr_wk"] = year_short * 100 + week_num
        
        # Walmart weekday: Saturday is 1, Friday is 7
        # Python weekday: Mon=0, Sun=6. Sat=5 -> 1, Sun=6 -> 2, Mon=0 -> 3...
        df["wday"] = (df["date"].dt.weekday + 2) % 7 + 1
        df["weekday"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year

        # Events & Holidays
        df["event_name_1"] = None
        df["event_type_1"] = None
        df["event_name_2"] = None
        df["event_type_2"] = None

        # Inject some famous retail events
        for idx, row in df.iterrows():
            d = row["date"]
            if d.month == 1 and d.day == 1:
                df.at[idx, "event_name_1"] = "NewYear"
                df.at[idx, "event_type_1"] = "National"
            elif d.month == 2 and d.day == 14:
                df.at[idx, "event_name_1"] = "ValentinesDay"
                df.at[idx, "event_type_1"] = "Cultural"
            elif d.month == 7 and d.day == 4:
                df.at[idx, "event_name_1"] = "IndependenceDay"
                df.at[idx, "event_type_1"] = "National"
            elif d.month == 10 and d.day == 31:
                df.at[idx, "event_name_1"] = "Halloween"
                df.at[idx, "event_type_1"] = "Cultural"
            elif d.month == 11 and d.day == 26:
                df.at[idx, "event_name_1"] = "Thanksgiving"
                df.at[idx, "event_type_1"] = "National"
            elif d.month == 12 and d.day == 25:
                df.at[idx, "event_name_1"] = "Christmas"
                df.at[idx, "event_type_1"] = "National"

        # SNAP food stamp promotion days (first 10-15 days of month per state rules)
        df["snap_CA"] = (df["date"].dt.day.between(1, 10)).astype(int)
        df["snap_TX"] = (df["date"].dt.day.between(5, 15)).astype(int)
        df["snap_WI"] = (df["date"].dt.day.between(2, 12)).astype(int)

        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df

    def generate_items_catalog(self) -> pd.DataFrame:
        """Create metadata catalog of all products across departments."""
        items = []
        for cat, depts in self.DEPTS_BY_CAT.items():
            for dept in depts:
                for i in range(1, self.num_items_per_dept + 1):
                    item_id = f"{dept}_{i:03d}"
                    items.append({
                        "item_id": item_id,
                        "dept_id": dept,
                        "cat_id": cat,
                    })
        return pd.DataFrame(items)

    def generate_sell_prices(
        self, calendar_df: pd.DataFrame, items_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate sell_prices.csv matching M5 schema."""
        weeks = calendar_df["wm_yr_wk"].unique()
        all_stores = [s for stores in self.STORES_BY_STATE.values() for s in stores]

        records = []
        base_prices = {
            "FOODS": 2.98,
            "HOUSEHOLD": 7.49,
            "HOBBIES": 14.99,
        }

        for _, item in items_df.iterrows():
            item_id = item["item_id"]
            cat_id = item["cat_id"]
            item_base_price = base_prices[cat_id] + self.rng.uniform(-1.0, 2.5)
            item_base_price = max(0.99, round(item_base_price, 2))

            for store in all_stores:
                # Add slight store variation
                store_mult = 1.0 + (hash(store) % 7 - 3) * 0.02
                for wk in weeks:
                    # occasional promotional markdown
                    promo = self.rng.choice([1.0, 1.0, 1.0, 0.85, 0.75], p=[0.7, 0.15, 0.05, 0.07, 0.03])
                    price = round(item_base_price * store_mult * promo, 2)
                    records.append({
                        "store_id": store,
                        "item_id": item_id,
                        "wm_yr_wk": wk,
                        "sell_price": price,
                    })

        return pd.DataFrame(records)

    def generate_sales_train(
        self, calendar_df: pd.DataFrame, items_df: pd.DataFrame, prices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate sales_train_sample.csv in wide M5 format (d_1 to d_N)."""
        all_stores = []
        for state, stores in self.STORES_BY_STATE.items():
            for store in stores:
                all_stores.append((state, store))

        rows = []
        day_cols = [f"d_{i + 1}" for i in range(self.num_days)]
        
        # Precompute day weights
        wday_weights = {
            1: 1.35,  # Sat
            2: 1.30,  # Sun
            3: 0.90,  # Mon
            4: 0.88,  # Tue
            5: 0.92,  # Wed
            6: 0.95,  # Thu
            7: 1.15,  # Fri
        }
        wday_series = calendar_df["wday"].map(wday_weights).values
        event_bump = calendar_df["event_name_1"].notna().astype(float).values * 0.45

        for state, store in all_stores:
            snap_col = f"snap_{state}"
            snap_series = calendar_df[snap_col].values * 0.20
            
            # State/store base multiplier
            store_factor = 1.0 + (hash(store) % 5 - 2) * 0.08

            for _, item in items_df.iterrows():
                item_id = item["item_id"]
                dept_id = item["dept_id"]
                cat_id = item["cat_id"]
                series_id = f"{item_id}_{store}_evaluation"

                # Base demand according to category
                base_demand = {"FOODS": 8.0, "HOUSEHOLD": 3.5, "HOBBIES": 1.5}[cat_id]
                
                # Demand with Poisson-like noise, seasonality, events, and snap
                lam = base_demand * store_factor * wday_series * (1.0 + event_bump + snap_series)
                sales = self.rng.poisson(lam=lam)
                
                # occasional zero sales (intermittent demand)
                zero_mask = self.rng.random(len(sales)) < (0.15 if cat_id == "HOBBIES" else 0.04)
                sales[zero_mask] = 0

                row = {
                    "id": series_id,
                    "item_id": item_id,
                    "dept_id": dept_id,
                    "cat_id": cat_id,
                    "store_id": store,
                    "state_id": state,
                }
                for d_col, val in zip(day_cols, sales):
                    row[d_col] = int(val)
                rows.append(row)

        return pd.DataFrame(rows)

    def generate_all(self, output_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and save calendar.csv, sell_prices.csv, sales_train_sample.csv."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        logger.info("Generating M5 hierarchical sample dataset...")
        calendar_df = self.generate_calendar()
        items_df = self.generate_items_catalog()
        prices_df = self.generate_sell_prices(calendar_df, items_df)
        sales_df = self.generate_sales_train(calendar_df, items_df, prices_df)

        calendar_path = out_path / "calendar.csv"
        prices_path = out_path / "sell_prices.csv"
        sales_path = out_path / "sales_train_sample.csv"

        calendar_df.to_csv(calendar_path, index=False)
        prices_df.to_csv(prices_path, index=False)
        sales_df.to_csv(sales_path, index=False)

        logger.info(
            f"Successfully generated M5 sample dataset in {out_path}: "
            f"{len(sales_df)} series, {self.num_days} days."
        )
        return calendar_df, prices_df, sales_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = M5SampleGenerator(num_items_per_dept=2, num_days=365)
    gen.generate_all("data/sample")
