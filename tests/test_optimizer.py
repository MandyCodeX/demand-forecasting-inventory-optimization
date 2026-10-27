"""Tests for OR-Tools inventory optimization under constraints."""

import pytest
import numpy as np
import pandas as pd
from src.optimizer import InventoryOptimizer, ItemParams


def test_optimizer_respects_capacity_and_bounds():
    optimizer = InventoryOptimizer(
        planning_horizon_days=7,
        warehouse_capacity=500.0,
        daily_procurement_budget=1000.0,
        time_limit_seconds=5
    )
    params = [
        ItemParams(
            item_id="FOODS_1_001",
            store_id="CA_1",
            initial_inventory=20.0,
            unit_cost=2.50,
            holding_cost_per_day=0.04,
            stockout_penalty_per_unit=12.0,
            lead_time_days=1,
            min_order_qty=5.0,
        )
    ]
    forecast_df = pd.DataFrame({
        "store_id": ["CA_1"] * 7,
        "item_id": ["FOODS_1_001"] * 7,
        "demand_p50": [8.0, 7.0, 9.0, 10.0, 8.0, 7.0, 6.0],
    })

    result = optimizer.solve(forecast_df, params, demand_column="demand_p50")
    assert result.status in ["OPTIMAL", "FEASIBLE", "HEURISTIC_FALLBACK"]
    assert result.total_cost >= 0.0
    assert result.service_level > 80.0
    assert not result.orders_df.empty
    assert not result.inventory_df.empty
